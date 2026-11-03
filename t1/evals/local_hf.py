"""In-process model client for SageMaker jobs: `--client local-hf`.

Loads a model from a local directory (a SageMaker input channel), optionally applies a LoRA
adapter, and generates with the model's own chat template. Base and tuned go through this one code
path, so "same items, same prompt, same parser" holds by construction.

Prompt format (binding, also for the SFT data builder):
  messages = [{"role": "system", "content": <prompt.txt ### SYSTEM>},
              {"role": "user",   "content": <prompt.txt ### TASK section, filled>}]
  text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)      # Qwen3 template, thinking OFF
Decoding: greedy (do_sample=False), batch size 1, max_new_tokens per unit, no sampling knobs.

Latency is the wall-clock time of one whole call on the job's instance: chat template,
tokenisation, generate and decode, with CUDA synchronised before and after. Warm-up calls run
first and are not recorded. Cost per call = instance on-demand $/h x latency / 3600 (Pricing in
clients.py); the price is passed in and declared, never looked up here.

transformers, torch and peft are imported only when a real model is loaded, so the rest of the
harness and its tests run without them.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from t1.evals.clients import Completion

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
RESOURCE_CONFIG = Path("/opt/ml/input/config/resourceconfig.json")
WARMUP_USER = 'Reply with exactly this JSON object and nothing else: {"ok": true}'


def sagemaker_instance_type() -> str | None:
    """The job's instance type as SageMaker reports it inside the container, if available."""
    try:
        return json.loads(RESOURCE_CONFIG.read_text()).get("current_instance_type")
    except Exception:  # noqa: BLE001 - not in a SageMaker job
        return None


def dir_manifest(path: str | os.PathLike) -> dict[str, str]:
    """sha256 of every file under a model or adapter directory (relative path -> sha256)."""
    root = Path(path)
    out = {}
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        out[p.relative_to(root).as_posix()] = h.hexdigest()
    return out


def manifest_sha256(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _load(model_dir: str, adapter_dir: str | None, merge: bool, dtype: str, device: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge:
            model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return tok, model, device, str(torch_dtype).replace("torch.", "")


class LocalHFClient:
    backend = "local-hf"
    needs_item = False

    def __init__(self, model_dir: str, model_id: str, *, adapter_dir: str | None = None, merge_adapter: bool = True,
                 dtype: str = "auto", device: str | None = None, warmup: int = 3, instance_type: str | None = None,
                 hash_files: bool = True, components: tuple[Any, Any, str, str] | None = None):
        """`components` = (tokenizer, model, device, dtype) injects a ready model (tests)."""
        self.model_dir, self.adapter_dir, self.model_id = model_dir, adapter_dir, model_id
        self.merge_adapter, self.n_warmup = merge_adapter, warmup
        self.tok, self.model, self.device, self.dtype = components or _load(model_dir, adapter_dir, merge_adapter,
                                                                           dtype, device)
        self.instance_type = instance_type or sagemaker_instance_type()
        self.files: dict[str, Any] = {}
        if hash_files and model_dir and Path(model_dir).is_dir():
            m = dir_manifest(model_dir)
            self.files["model_dir_sha256"] = manifest_sha256(m)
            self.files["model_files"] = m
            if adapter_dir:
                a = dir_manifest(adapter_dir)
                self.files["adapter_dir_sha256"] = manifest_sha256(a)
                self.files["adapter_files"] = a
        template = getattr(self.tok, "chat_template", None) or ""
        self.chat_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
        probe = [{"role": "user", "content": "x"}]
        on = self.tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        off = self.tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.thinking_flag_honoured = on != off
        self.warmup_ms: list[float] = []

    # -- prompt ---------------------------------------------------------------------------------

    def render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            **CHAT_TEMPLATE_KWARGS)

    def decoding(self, max_tokens: int) -> dict:
        import importlib.metadata as md

        def ver(p: str) -> str | None:
            try:
                return md.version(p)
            except md.PackageNotFoundError:
                return None

        return {
            "transport": "in-process (local-hf)", "temperature": 0.0, "do_sample": False, "batch_size": 1,
            "max_tokens": max_tokens, "chat_template": "model's own (tokenizer.apply_chat_template)",
            "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS), "chat_template_sha256": self.chat_template_sha256,
            "thinking_flag_honoured": self.thinking_flag_honoured,
            "add_generation_prompt": True, "messages": "system=prompt.txt SYSTEM, user=prompt.txt TASK",
            "truncation": False, "model_max_length_ignored": getattr(self.tok, "model_max_length", None),
            "max_position_embeddings": getattr(getattr(self.model, "config", None), "max_position_embeddings", None),
            "dtype": self.dtype, "device": self.device, "gpu": self._gpu_name(),
            "instance_type": self.instance_type, "adapter": bool(self.adapter_dir),
            "adapter_merged": bool(self.adapter_dir) and self.merge_adapter,
            "warmup_calls": self.n_warmup,
            "latency_method": "wall clock per call: template + tokenise + generate + decode, CUDA synchronised",
            "versions": {"transformers": ver("transformers"), "torch": ver("torch"), "peft": ver("peft"),
                         "python": platform.python_version()},
            **{k: v for k, v in self.files.items() if k.endswith("_sha256")},
        }

    def file_manifests(self) -> dict:
        return dict(self.files)

    # -- calls ----------------------------------------------------------------------------------

    def _gpu_name(self) -> str | None:
        try:
            import torch

            return torch.cuda.get_device_name(0) if str(self.device).startswith("cuda") else None
        except Exception:  # noqa: BLE001
            return None

    def _sync(self) -> None:
        if str(self.device).startswith("cuda"):
            import torch

            torch.cuda.synchronize()

    def warmup(self, system: str, max_tokens: int = 16) -> None:
        for _ in range(self.n_warmup):
            self.warmup_ms.append(self.generate(system, WARMUP_USER, max_tokens).latency_ms)

    def generate(self, system: str, user: str, max_tokens: int, item: dict | None = None) -> Completion:
        import torch

        self._sync()
        t0 = time.perf_counter()
        try:
            prompt = self.render(system, user)
            # never truncate: tokenizer_config model_max_length (1100 for the OTel base) is not a model limit
            enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            n_in = int(enc["input_ids"].shape[1])
            mpe = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
            if isinstance(mpe, int) and n_in + max_tokens > mpe:
                raise ValueError(f"prompt {n_in} + {max_tokens} new tokens exceeds the model's {mpe} positions")
            pad =self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
            with torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, num_beams=1,
                                          temperature=None, top_p=None, top_k=None, pad_token_id=pad)
            new = out[0, n_in:]
            text = self.tok.decode(new, skip_special_tokens=True)
            self._sync()
        except Exception as e:  # noqa: BLE001 - a failed call is recorded, not fatal
            self._sync()
            return Completion(text="", latency_ms=(time.perf_counter() - t0) * 1000, error=f"{type(e).__name__}: {e}")
        latency = (time.perf_counter() - t0) * 1000
        n_out = int(new.shape[0])
        return Completion(text=text, latency_ms=latency, tokens_in=n_in, tokens_out=n_out, tokens_estimated=False,
                          meta={"hit_max_tokens": n_out >= max_tokens})
