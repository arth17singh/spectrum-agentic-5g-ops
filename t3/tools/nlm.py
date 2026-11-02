"""The NLM behind `nlm___ask`: our tuned OTel model on a SageMaker endpoint, or a labelled stand-in.

- `NLM_ENDPOINT` set: `sagemaker-runtime.invoke_endpoint` (boto3 signs it with SigV4).
  `NLM_PAYLOAD` picks the body: `messages` (default; OpenAI-style chat body, which the LMI/vLLM
  and TGI containers accept) or `tgi` (`{"inputs", "parameters"}`).
- else `NLM_STANDIN_MODEL` set: Bedrock `converse` with that model id, labelled a stand-in.
- else: an error saying no NLM is configured.
Every answer carries `answered_by {kind, id, version, stand_in}` so nobody mistakes the stand-in for our model.
Clients are created lazily, so importing this module makes no network call.
"""
from __future__ import annotations

import json
import os
from typing import Callable

SYSTEM = "You are a radio access network engineer. Answer in at most 120 words."


def _client_factory(service: str):
    import boto3

    return boto3.client(service)


class NLM:
    def __init__(self, env: dict[str, str] | None = None, client_factory: Callable[[str], object] | None = None):
        self.env = dict(os.environ if env is None else env)
        self.client_factory = client_factory or _client_factory

    def describe(self) -> dict:
        if self.env.get("NLM_ENDPOINT"):
            return {"kind": "sagemaker-endpoint", "id": self.env["NLM_ENDPOINT"],
                    "version": self.env.get("NLM_MODEL_VERSION", "v0"),
                    "model_label": self.env.get("NLM_MODEL_LABEL", "team13 tuned OTel model"), "stand_in": False}
        if self.env.get("NLM_STANDIN_MODEL"):
            return {"kind": "bedrock-standin", "id": self.env["NLM_STANDIN_MODEL"],
                    "version": self.env.get("NLM_MODEL_VERSION", "standin"),
                    "model_label": "stand-in: NOT our tuned model", "stand_in": True}
        return {"kind": "none", "id": None, "version": None, "stand_in": None}

    def ask(self, question: str) -> dict:
        who = self.describe()
        system = self.env.get("NLM_SYSTEM") or SYSTEM
        max_tokens = int(self.env.get("NLM_MAX_TOKENS", "400"))
        if who["kind"] == "sagemaker-endpoint":
            answer = self._sagemaker(who["id"], system, question, max_tokens)
        elif who["kind"] == "bedrock-standin":
            answer = self._bedrock(who["id"], system, question, max_tokens)
        else:
            return {"error": "No NLM is configured (set NLM_ENDPOINT, or NLM_STANDIN_MODEL for a stand-in).",
                    "answered_by": who}
        return {"answer": answer, "answered_by": who}

    def _sagemaker(self, endpoint: str, system: str, question: str, max_tokens: int) -> str:
        if self.env.get("NLM_PAYLOAD", "messages") == "tgi":
            body = {"inputs": f"{system}\n\nQuestion: {question}\nAnswer:",
                    "parameters": {"max_new_tokens": max_tokens, "temperature": 0.2, "return_full_text": False}}
        else:
            body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
                    "max_tokens": max_tokens, "temperature": 0.2}
        resp = self.client_factory("sagemaker-runtime").invoke_endpoint(
            EndpointName=endpoint, ContentType="application/json", Accept="application/json",
            Body=json.dumps(body).encode())
        raw = resp["Body"].read()
        return parse_generation(json.loads(raw))

    def _bedrock(self, model_id: str, system: str, question: str, max_tokens: int) -> str:
        resp = self.client_factory("bedrock-runtime").converse(
            modelId=model_id, system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": question}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2})
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def parse_generation(data) -> str:
    """Text from the common SageMaker LLM container response shapes."""
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        if data.get("choices"):
            c = data["choices"][0]
            return ((c.get("message") or {}).get("content") or c.get("text") or "").strip()
        for key in ("generated_text", "generation", "outputs"):
            if isinstance(data.get(key), str):
                return data[key].strip()
    raise ValueError(f"unrecognised NLM response: {str(data)[:200]}")
