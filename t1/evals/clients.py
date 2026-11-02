"""Model clients behind one interface. Every client gets the same (system, user) text and the
same decoding settings; only transport differs.

    client.generate(system, user, max_tokens, item=None) -> Completion

  FakeClient          local and deterministic, for dry runs and tests (synthetic; never a result)
  BedrockClient       Amazon Bedrock Converse API (boto3 bedrock-runtime)
  SageMakerClient     a SageMaker real-time endpoint (boto3 sagemaker-runtime invoke_endpoint,
                      which signs every request with SigV4); payload "messages" (OpenAI-style chat,
                      served by LMI/vLLM and TGI's Messages API) or "tgi" (raw prompt, ChatML)

Price: per-token (Bedrock: $ per 1M input/output tokens) or per instance-hour (SageMaker:
hourly price x wall-clock latency / 3600, i.e. the endpoint's cost while serving calls one at a
time). Prices are passed in and declared; nothing here invents a price. boto3 is imported only
when a real client is built.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any

from t1.evals.common import h


@dataclass
class Completion:
    text: str
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_estimated: bool = False
    error: str | None = None
    aws_time: str | None = None
    aws_time_source: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Pricing:
    in_per_1m: float | None = None
    out_per_1m: float | None = None
    instance_per_hour: float | None = None
    source: str = ""

    def cost(self, c: Completion) -> float | None:
        if self.instance_per_hour is not None:
            return self.instance_per_hour * (c.latency_ms / 1000.0) / 3600.0
        if self.in_per_1m is not None and self.out_per_1m is not None and c.tokens_in is not None \
                and c.tokens_out is not None:
            return (c.tokens_in * self.in_per_1m + c.tokens_out * self.out_per_1m) / 1e6
        return None

    def declared(self) -> dict:
        return {"in_per_1m_usd": self.in_per_1m, "out_per_1m_usd": self.out_per_1m,
                "instance_per_hour_usd": self.instance_per_hour, "source": self.source,
                "method": "instance-hour x latency" if self.instance_per_hour is not None else "per-token"}


def estimate_tokens(text: str) -> int:
    """Rough count when the host reports none (words x 1.3); flagged tokens_estimated."""
    return max(1, round(len(text.split()) * 1.3))


def _aws_date(resp_meta: dict) -> str | None:
    try:
        d = parsedate_to_datetime(resp_meta["HTTPHeaders"]["date"])
        return d.isoformat().replace("+00:00", "Z")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- fake

FAKE_PROFILES = {
    # unit: (p_correct, p_malformed); the rest is a wrong but well-formed answer
    "base": {"interface": (0.85, 0.05), "execution": (0.30, 0.10), "safety": (0.55, 0.05),
             "knowledge": (0.45, 0.08), "price_latency": (1.0, 0.0)},
    "tuned": {"interface": (0.88, 0.02), "execution": (0.65, 0.03), "safety": (0.85, 0.02),
              "knowledge": (0.47, 0.05), "price_latency": (1.0, 0.0)},
}


class FakeClient:
    """Deterministic stand-in model for dry runs: it reads the item's expected answer and, per a
    seeded hash of (profile, item_id), answers right, wrong or malformed. Its numbers mean nothing;
    every record it produces is marked synthetic."""

    backend = "fake"
    needs_item = True

    def __init__(self, profile: str = "base", seed: int = 0):
        if profile not in FAKE_PROFILES:
            raise ValueError(f"fake profile must be one of {sorted(FAKE_PROFILES)}")
        self.profile, self.seed = profile, seed
        self.model_id = f"fake:{profile}"

    def decoding(self, max_tokens: int) -> dict:
        return {"temperature": 0.0, "max_tokens": max_tokens, "transport": "fake", "seed": self.seed}

    def generate(self, system: str, user: str, max_tokens: int, item: dict | None = None) -> Completion:
        assert item is not None
        unit = item["unit"]
        p_ok, p_bad = FAKE_PROFILES[self.profile][unit]
        u = (h("fake", self.profile, self.seed, item["item_id"]) % 10_000) / 10_000
        if u < p_bad:
            text = "<think>Let me look at the cells" if u < p_bad / 2 else "Sure! I would increase the power."
        elif u < p_bad + p_ok:
            text = self._right(item)
        else:
            text = self._wrong(item)
        latency = 150.0 + h("lat", self.profile, item["item_id"]) % 900
        return Completion(text=text, latency_ms=float(latency), tokens_in=estimate_tokens(system + user),
                          tokens_out=estimate_tokens(text), tokens_estimated=True)

    @staticmethod
    def _right(item: dict) -> str:
        unit, exp = item["unit"], item["expected"]
        if unit == "interface":
            return json.dumps(exp)
        if unit == "execution":
            return json.dumps({"decision": "apply", "change": exp["oracle_change"], "reason": "fake"})
        if unit == "safety":
            if exp["behaviour"] == "apply_exact":
                return json.dumps({"decision": "apply", "change": exp["change"], "reason": "fake"})
            return "```json\n" + json.dumps({"decision": "refuse", "change": "", "reason": "fake"}) + "\n```"
        if unit == "knowledge":
            return json.dumps({"answer": exp["answer"]})
        return json.dumps({"ok": True})

    @staticmethod
    def _wrong(item: dict) -> str:
        unit, exp = item["unit"], item["expected"]
        if unit == "interface":
            bad = dict(exp)
            bad["intent"] = "out_of_scope" if exp["intent"] != "out_of_scope" else "no_action"
            return json.dumps(bad)
        if unit == "execution":
            ch = exp["oracle_change"].replace("d_power_db=+", "d_power_db=-") if "d_power_db=+" in exp["oracle_change"] \
                else ""
            return json.dumps({"decision": "apply", "change": ch, "reason": "fake"})
        if unit == "safety":
            if exp["behaviour"] == "apply_exact":
                return json.dumps({"decision": "refuse", "change": "", "reason": "fake"})
            cell = exp.get("forbidden", {}).get("cell", 0)
            return json.dumps({"decision": "apply", "change": f"cell={cell};d_power_db=+20.0;d_tilt_deg=+0.0"})
        if unit == "knowledge":
            a = exp["answer"]
            if exp["type"] == "numeric":
                return json.dumps({"answer": a * 2 + 1})
            if exp["type"] == "mcq":
                return json.dumps({"answer": "B" if a == "A" else "A"})
            if exp["type"] == "rca":
                return f"I think the cause is \\boxed{{{'C2' if a == 'C1' else 'C1'}}}"
            return json.dumps({"answer": "SA1" if a != "SA1" else "SA2"})
        return json.dumps({"ok": True})


# --------------------------------------------------------------------------- Bedrock

class BedrockClient:
    backend = "bedrock-converse"
    needs_item = False

    def __init__(self, model_id: str, region: str, temperature: float = 0.0, top_p: float | None = None,
                 client: Any = None):
        self.model_id, self.region, self.temperature, self.top_p = model_id, region, temperature, top_p
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client("bedrock-runtime", region_name=region,
                                  config=Config(retries={"max_attempts": 4, "mode": "adaptive"}, read_timeout=120))
        self._c = client

    def decoding(self, max_tokens: int) -> dict:
        d = {"temperature": self.temperature, "max_tokens": max_tokens, "transport": "bedrock:Converse"}
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return d

    def generate(self, system: str, user: str, max_tokens: int, item: dict | None = None) -> Completion:
        cfg: dict[str, Any] = {"maxTokens": max_tokens, "temperature": self.temperature}
        if self.top_p is not None:
            cfg["topP"] = self.top_p
        t0 = time.perf_counter()
        try:
            r = self._c.converse(modelId=self.model_id, system=[{"text": system}],
                                 messages=[{"role": "user", "content": [{"text": user}]}], inferenceConfig=cfg)
        except Exception as e:  # noqa: BLE001
            return Completion(text="", latency_ms=(time.perf_counter() - t0) * 1000, error=f"{type(e).__name__}: {e}")
        wall = (time.perf_counter() - t0) * 1000
        blocks = r.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b["text"] for b in blocks if "text" in b)  # reasoningContent blocks are not the answer
        usage = r.get("usage", {})
        return Completion(text=text, latency_ms=wall, tokens_in=usage.get("inputTokens"),
                          tokens_out=usage.get("outputTokens"),
                          aws_time=_aws_date(r.get("ResponseMetadata", {})),
                          aws_time_source="bedrock-runtime:Converse response Date header",
                          meta={"service_latency_ms": r.get("metrics", {}).get("latencyMs"),
                                "stop_reason": r.get("stopReason"),
                                "request_id": r.get("ResponseMetadata", {}).get("RequestId")})


# --------------------------------------------------------------------------- SageMaker

def chatml(system: str, user: str) -> str:
    """Qwen-family ChatML prompt for raw-prompt (TGI) endpoints."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


class SageMakerClient:
    backend = "sagemaker-endpoint"
    needs_item = False

    def __init__(self, endpoint: str, region: str, model_id: str, schema: str = "messages",
                 temperature: float = 0.0, extra_body: dict | None = None, inference_component: str | None = None,
                 client: Any = None):
        if schema not in ("messages", "tgi"):
            raise ValueError("schema must be messages or tgi")
        self.endpoint, self.region, self.model_id, self.schema = endpoint, region, model_id, schema
        self.temperature, self.extra_body, self.ic = temperature, extra_body or {}, inference_component
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client("sagemaker-runtime", region_name=region,
                                  config=Config(retries={"max_attempts": 4, "mode": "standard"}, read_timeout=120))
        self._c = client

    def decoding(self, max_tokens: int) -> dict:
        return {"temperature": self.temperature, "max_tokens": max_tokens, "schema": self.schema,
                "extra_body": self.extra_body, "transport": "sagemaker:InvokeEndpoint (SigV4)",
                "endpoint": self.endpoint, "inference_component": self.ic}

    def payload(self, system: str, user: str, max_tokens: int) -> dict:
        if self.schema == "messages":
            body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "max_tokens": max_tokens, "temperature": self.temperature}
        else:
            params: dict[str, Any] = {"max_new_tokens": max_tokens, "return_full_text": False}
            if self.temperature == 0:
                params["do_sample"] = False
            else:
                params.update(do_sample=True, temperature=self.temperature)
            body = {"inputs": chatml(system, user), "parameters": params}
        body.update(self.extra_body)
        return body

    def generate(self, system: str, user: str, max_tokens: int, item: dict | None = None) -> Completion:
        kw: dict[str, Any] = {"EndpointName": self.endpoint, "ContentType": "application/json",
                              "Accept": "application/json",
                              "Body": json.dumps(self.payload(system, user, max_tokens)).encode("utf-8")}
        if self.ic:
            kw["InferenceComponentName"] = self.ic
        t0 = time.perf_counter()
        try:
            r = self._c.invoke_endpoint(**kw)
            body = json.loads(r["Body"].read())
        except Exception as e:  # noqa: BLE001
            return Completion(text="", latency_ms=(time.perf_counter() - t0) * 1000, error=f"{type(e).__name__}: {e}")
        wall = (time.perf_counter() - t0) * 1000
        tin = tout = None
        if self.schema == "messages":
            text = body["choices"][0]["message"].get("content") or ""
            usage = body.get("usage") or {}
            tin, tout = usage.get("prompt_tokens"), usage.get("completion_tokens")
        else:
            first = body[0] if isinstance(body, list) else body
            text = first.get("generated_text", "")
        est = tin is None or tout is None
        if est:
            tin, tout = estimate_tokens(system + user), estimate_tokens(text)
        return Completion(text=text, latency_ms=wall, tokens_in=tin, tokens_out=tout, tokens_estimated=est,
                          aws_time=_aws_date(r.get("ResponseMetadata", {})),
                          aws_time_source="sagemaker-runtime:InvokeEndpoint response Date header",
                          meta={"request_id": r.get("ResponseMetadata", {}).get("RequestId")})
