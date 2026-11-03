"""Zero overlap between SFT examples and the frozen Track 1 eval items, by content hashes.

Seed ranges alone are not enough: the 4-cell twin is small and discrete, so different seeds can
render the same state, and template text can collide. Every eval item and every SFT example gets
the same content hashes (HASH_METHOD), and an SFT example is dropped if ANY of them matches an eval
item. Within the SFT set, an example whose answer-free hash was already seen is dropped too.

Hashes (sha256 over canonical JSON, keys sorted, compact, UTF-8):
  item_hash    {method, task, inputs, answer}: the item as a whole, normalised (the coordinator's
               "rendered state + task + expected answer")
  input_hash   {method, task, inputs}: the same without the answer, so an identical problem with a
               different label still counts as overlap (stricter than item_hash)
  prompt_hash  t1.evals.prompting.input_sha256(system, user): the exact prompt text, as the eval
               harness records it in every item record
  exec_state   (scenario, seed, step) of an execution item; any match counts
Normalisation: Unicode NFKC, lower case, every run of whitespace collapsed to one space, stripped.
`task` is the prompt.txt TASK section (interface, execution, safety, knowledge_*). For execution
the operator note is NOT part of `inputs` (two items with the same state are the same problem).
Answers: interface {intent, cells, parameter, direction, amount, kpi}; execution and safety
{decision, change} with the canonical change string (eval safety items that expect a refusal, or a
refusal-or-in-limit change, normalise to the refusal); knowledge {answer}.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

HASH_METHOD = "t1-sft-overlap-v1"
EVAL_UNITS = ("interface", "execution", "safety", "knowledge", "price_latency")
REASONS = ("item_hash", "input_hash", "prompt_hash", "exec_state")


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(s)).lower().split())


def _canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canon_json(obj).encode("utf-8")).hexdigest()


def norm_inputs(template: str, inp: dict[str, Any]) -> dict[str, str]:
    """The task inputs that make an item what it is, normalised (execution: the state only)."""
    if template == "execution":
        return {"state": norm_text(inp.get("state"))}
    return {k: norm_text(v) for k, v in sorted(inp.items())}


def _norm_change(text: Any) -> str:
    from t1.evals import canon as C

    try:
        return C.canon(text if text is not None else "")
    except C.ChangeError:
        return norm_text(text)


def norm_answer(template: str, answer: dict[str, Any] | None) -> Any:
    """The answer in one shape for eval items and SFT examples (None when there is none)."""
    if answer is None:
        return None
    if template == "interface":
        cells = answer.get("cells") or []
        amount = answer.get("amount")
        return {"intent": norm_text(answer.get("intent")), "cells": sorted(int(c) for c in cells),
                "parameter": norm_text(answer.get("parameter")) or None,
                "direction": norm_text(answer.get("direction")) or None,
                "amount": None if amount is None else float(amount), "kpi": norm_text(answer.get("kpi")) or None}
    if template in ("execution", "safety"):
        decision = norm_text(answer.get("decision"))
        change = _norm_change(answer.get("change")) if decision == "apply" else ""
        return {"decision": decision, "change": change}
    return {"answer": norm_text(answer.get("answer"))}


def eval_answer(item: dict[str, Any]) -> dict[str, Any] | None:
    """The expected answer of a frozen eval item, in the shape the model must reply with."""
    exp = item.get("expected") or {}
    unit = item["unit"]
    if unit == "interface":
        return {f: exp.get(f) for f in ("intent", "cells", "parameter", "direction", "amount", "kpi")}
    if unit == "execution":
        return {"decision": "apply", "change": exp.get("oracle_change", "")}
    if unit == "safety":
        if exp.get("behaviour") == "apply_exact":
            return {"decision": "apply", "change": exp.get("change", "")}
        return {"decision": "refuse", "change": ""}
    if unit == "knowledge":
        return {"answer": exp.get("answer")}
    return None  # price_latency: no correctness answer


def hashes(template: str, inp: dict[str, Any], answer: dict[str, Any] | None) -> dict[str, str]:
    inputs = norm_inputs(template, inp)
    return {"item_hash": _sha({"m": HASH_METHOD, "task": template, "inputs": inputs,
                               "answer": norm_answer(template, answer)}),
            "input_hash": _sha({"m": HASH_METHOD, "task": template, "inputs": inputs})}


def prompt_hash(template: str, inp: dict[str, Any]) -> str:
    from t1.evals import prompting

    system, user = prompting.render({"template": template, "input": inp})
    return prompting.input_sha256(system, user)


def exec_state_key(meta: dict[str, Any]) -> tuple[str, int, int] | None:
    if "scenario" not in meta or "seed" not in meta or "prefix" not in meta:
        return None
    return (str(meta["scenario"]), int(meta["seed"]), len(meta["prefix"]))


# --------------------------------------------------------------------------- the eval side

@dataclass
class EvalIndex:
    item_hash: dict[str, str] = field(default_factory=dict)
    input_hash: dict[str, str] = field(default_factory=dict)
    prompt_hash: dict[str, str] = field(default_factory=dict)
    exec_state: dict[tuple, str] = field(default_factory=dict)
    exec_seeds: list[int] = field(default_factory=list)
    n_items: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    manifest_sha256: str | None = None
    prompt_sha256: str | None = None

    def summary(self) -> dict[str, Any]:
        return {"n_items": self.n_items, "n_items_total": sum(self.n_items.values()),
                "distinct": {"item_hash": len(self.item_hash), "input_hash": len(self.input_hash),
                             "prompt_hash": len(self.prompt_hash), "exec_state": len(self.exec_state)},
                "exec_seed_min": min(self.exec_seeds) if self.exec_seeds else None,
                "exec_seed_max": max(self.exec_seeds) if self.exec_seeds else None,
                "files_sha256": self.files, "manifest_sha256": self.manifest_sha256,
                "prompt_sha256": self.prompt_sha256}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_items(items: Iterable[dict[str, Any]], index: EvalIndex | None = None) -> EvalIndex:
    idx = index or EvalIndex()
    for it in items:
        tpl, inp, iid = it["template"], it["input"], it["item_id"]
        hs = hashes(tpl, inp, eval_answer(it))
        idx.item_hash.setdefault(hs["item_hash"], iid)
        idx.input_hash.setdefault(hs["input_hash"], iid)
        idx.prompt_hash.setdefault(prompt_hash(tpl, inp), iid)
        key = exec_state_key(it.get("meta") or {}) if it["unit"] == "execution" else None
        if key is not None:
            idx.exec_state.setdefault(key, iid)
            idx.exec_seeds.append(key[1])
        idx.n_items[it["unit"]] = idx.n_items.get(it["unit"], 0) + 1
    return idx


def build_eval_index(items_dir: str | Path) -> EvalIndex:
    """Index every frozen eval item (all five files, dev and test). Each file's sha256 must match
    items/MANIFEST.json, so the index is of the declared item sets and nothing else."""
    items_dir = Path(items_dir)
    manifest_bytes = (items_dir / "MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    idx = EvalIndex(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    prompt_sha256=manifest.get("prompt", {}).get("sha256"))
    for unit in EVAL_UNITS:
        p = items_dir / f"{unit}.jsonl"
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        want = manifest["units"][unit]["sha256"]
        if sha != want:
            raise ValueError(f"{p.name} sha256 {sha} != MANIFEST {want}: not the frozen item set")
        idx.files[p.name] = sha
        index_items(read_jsonl(p), idx)
    return idx


# --------------------------------------------------------------------------- the SFT side

def example_hashes(ex: dict[str, Any]) -> dict[str, Any]:
    out = hashes(ex["template"], ex["input"], ex["answer"])
    out["prompt_hash"] = prompt_hash(ex["template"], ex["input"])
    out["exec_state"] = exec_state_key(ex.get("meta") or {}) if ex["unit"] == "execution" else None
    return out


def matches(h: dict[str, Any], idx: EvalIndex) -> list[str]:
    """Which hashes of one example hit the eval index (empty: no overlap)."""
    hit = []
    if h["item_hash"] in idx.item_hash:
        hit.append("item_hash")
    if h["input_hash"] in idx.input_hash:
        hit.append("input_hash")
    if h["prompt_hash"] in idx.prompt_hash:
        hit.append("prompt_hash")
    if h.get("exec_state") is not None and tuple(h["exec_state"]) in idx.exec_state:
        hit.append("exec_state")
    return hit


def filter_examples(examples: list[dict[str, Any]], idx: EvalIndex) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop eval overlaps, then within-SFT duplicates (by input_hash, first kept). Deterministic in
    the order given. Each kept example carries its hashes in ex["hashes"]."""
    kept: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    rep: dict[str, Any] = {"hash_method": HASH_METHOD, "generated": len(examples), "dropped_eval_overlap": 0,
                           "dropped_eval_by_hash": {r: 0 for r in REASONS}, "dropped_duplicate": 0,
                           "duplicate_answer_conflicts": 0, "kept": 0, "by_unit": {}, "eval_overlap_examples": []}
    for ex in examples:
        u = rep["by_unit"].setdefault(ex["unit"], {"generated": 0, "dropped_eval_overlap": 0, "dropped_duplicate": 0,
                                                   "kept": 0})
        u["generated"] += 1
        h = example_hashes(ex)
        hit = matches(h, idx)
        if hit:
            rep["dropped_eval_overlap"] += 1
            u["dropped_eval_overlap"] += 1
            for r in hit:
                rep["dropped_eval_by_hash"][r] += 1
            if len(rep["eval_overlap_examples"]) < 20:
                first = hit[0]
                key = h[first] if first != "exec_state" else tuple(h[first])
                rep["eval_overlap_examples"].append({"key": ex.get("key"), "unit": ex["unit"], "hashes": hit,
                                                     "eval_item": getattr(idx, first)[key]})
            continue
        if h["input_hash"] in seen:
            rep["dropped_duplicate"] += 1
            u["dropped_duplicate"] += 1
            if seen[h["input_hash"]] != h["item_hash"]:
                rep["duplicate_answer_conflicts"] += 1
            continue
        seen[h["input_hash"]] = h["item_hash"]
        ex = dict(ex)
        ex["hashes"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()}
        kept.append(ex)
        u["kept"] += 1
    rep["kept"] = len(kept)
    return kept, rep


def check_disjoint(rows: Iterable[dict[str, Any]], idx: EvalIndex) -> dict[str, Any]:
    """Independent re-check of written rows: recompute every hash from the stored template, input
    and answer, and count hits per hash and within-set duplicates. eval_overlap_count must be 0."""
    counts = {r: 0 for r in REASONS}
    dup, n, rows_hit, seen = 0, 0, 0, set()
    max_seed = None
    for row in rows:
        n += 1
        h = example_hashes(row)
        hit = matches(h, idx)
        rows_hit += bool(hit)
        for r in hit:
            counts[r] += 1
        if h["input_hash"] in seen:
            dup += 1
        seen.add(h["input_hash"])
        seed = (row.get("meta") or {}).get("seed")
        if row["unit"] == "execution" and seed is not None:
            max_seed = seed if max_seed is None else max(max_seed, seed)
    return {"hash_method": HASH_METHOD, "rows_checked": n, "hits_by_hash": counts, "eval_overlap_count": rows_hit,
            "within_set_duplicates": dup, "max_execution_seed": max_seed}
