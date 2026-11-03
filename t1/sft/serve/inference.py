"""SageMaker PyTorch-inference handler for team13's tuned NLM (t1.sft.train_lora merged/, Qwen3, bf16).

It ships as code/inference.py next to the merged model files (t1/sft/serve/deploy.py builds that prefix);
the PyTorch inference DLC calls model_fn / input_fn / predict_fn / output_fn.

Request (application/json), either shape:
  {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...],
   "max_tokens": int, "temperature": float, "top_p": float}
      what W3's t3/tools/nlm.py (nlm___ask) and t1/evals clients.SageMakerClient("messages") send; extra
      keys (e.g. chat_template_kwargs) are accepted and ignored: thinking is always OFF;
  {"inputs": str, "parameters": {"max_new_tokens", "temperature", "top_p"}}
      a raw prompt (TGI shape), used verbatim.
Prompt: the model's own chat template, tokenizer.apply_chat_template(messages, add_generation_prompt=True,
enable_thinking=False): it ends "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n", byte-identical to the
eval harness's prompt and to how t1.sft.train_lora trained the model. temperature 0 (or absent) is greedy.
Response: OpenAI chat.completion shape, which nlm.parse_generation and the eval client both read:
  {"object": "chat.completion", "model", "choices": [{"index": 0, "message": {"role": "assistant",
   "content"}, "finish_reason": "stop"|"length"}], "usage": {...}, "team13": {request_sha256,
   prompt_sha256, latency_ms, thinking: "off", model_label, train_run_id}}
Every call prints one line to the endpoint's CloudWatch log (/aws/sagemaker/Endpoints/<endpoint>):
  team13-nlm-call {"request_sha256", "prompt_sha256", "response_sha256" (of the exact body returned),
                   "latency_ms", "prompt_tokens", "completion_tokens", "finish_reason", ...}
The endpoint's data capture (deploy.py) stores every request and response body in S3 as well.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid

THINK_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_NEW_TOKENS_CAP = int(os.environ.get("NLM_MAX_NEW_TOKENS_CAP", "1024"))
DEFAULT_MAX_NEW_TOKENS = 512
LOG_MARK = "team13-nlm-call"


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


def model_fn(model_dir, context=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    probe = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    if not probe.endswith(THINK_OFF_SUFFIX):  # the template must render thinking off as the eval does
        raise RuntimeError(f"chat template does not render thinking off as expected: {probe[-60:]!r}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
    eos = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if i is not None]
    info = {"model_label": os.environ.get("NLM_MODEL_LABEL", "team13 tuned OTel-LLM-1.7B-IT (LoRA merged)"),
            "train_run_id": os.environ.get("NLM_TRAIN_RUN_ID"), "device": device}
    print(f"team13-nlm-ready {json.dumps(info, sort_keys=True)}", flush=True)
    return {"tok": tok, "model": model, "device": device, "eos": eos, **info}


def input_fn(request_body, request_content_type="application/json"):
    if request_content_type and "json" not in request_content_type:
        raise ValueError(f"content type {request_content_type!r}: send application/json")
    raw = request_body if isinstance(request_body, (bytes, bytearray)) else str(request_body).encode("utf-8")
    req = json.loads(raw)
    if not isinstance(req, dict) or not (isinstance(req.get("messages"), list) or isinstance(req.get("inputs"), str)):
        raise ValueError("expected {\"messages\": [...]} or {\"inputs\": \"...\"}")
    return {"req": req, "request_sha256": _sha(bytes(raw))}


def build_prompt(tok, req: dict) -> tuple[str, dict]:
    """(prompt text, generation settings) for one request; thinking is always off for messages."""
    if isinstance(req.get("messages"), list):
        msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in req["messages"]]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        params = req
        max_new = req.get("max_tokens", req.get("max_new_tokens"))
        mode = "messages"
    else:
        prompt = req["inputs"]
        params = req.get("parameters") or {}
        max_new = params.get("max_new_tokens")
        mode = "inputs"
    max_new = min(int(max_new or DEFAULT_MAX_NEW_TOKENS), MAX_NEW_TOKENS_CAP)
    temperature = float(params.get("temperature") or 0.0)
    gen = {"max_new_tokens": max_new, "do_sample": temperature > 0}
    if temperature > 0:
        gen["temperature"] = temperature
        if params.get("top_p") is not None:
            gen["top_p"] = float(params["top_p"])
    return prompt, {"gen": gen, "mode": mode}


def predict_fn(data, m):
    import torch

    t0 = time.perf_counter()
    tok, model = m["tok"], m["model"]
    prompt, cfg = build_prompt(tok, data["req"])
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(m["device"])
    with torch.inference_mode():
        out = model.generate(**enc, **cfg["gen"], eos_token_id=m["eos"],
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else m["eos"][0])
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    finish = "stop" if new and new[-1] in m["eos"] else "length"
    text = tok.decode(new, skip_special_tokens=True)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    meta = {"request_sha256": data["request_sha256"], "prompt_sha256": _sha(prompt), "latency_ms": latency_ms,
            "thinking": "off", "mode": cfg["mode"], "model_label": m["model_label"],
            "train_run_id": m["train_run_id"]}
    body = json.dumps({"id": f"team13-{uuid.uuid4().hex[:16]}", "object": "chat.completion",
                       "model": m["model_label"],
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                    "finish_reason": finish}],
                       "usage": {"prompt_tokens": int(enc["input_ids"].shape[1]), "completion_tokens": len(new),
                                 "total_tokens": int(enc["input_ids"].shape[1]) + len(new)},
                       "team13": meta}, ensure_ascii=False)
    log = {**meta, "response_sha256": _sha(body), "prompt_tokens": int(enc["input_ids"].shape[1]),
           "completion_tokens": len(new), "finish_reason": finish, "max_new_tokens": cfg["gen"]["max_new_tokens"],
           "temperature": cfg["gen"].get("temperature", 0.0)}
    print(f"{LOG_MARK} {json.dumps(log, sort_keys=True)}", flush=True)
    return {"body": body}


def output_fn(prediction, accept="application/json"):
    return prediction["body"], "application/json"
