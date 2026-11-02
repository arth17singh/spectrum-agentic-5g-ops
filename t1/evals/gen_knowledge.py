"""Knowledge items: a frozen subset of GSMA ot-lite (public, MIT, not gated) at a pinned revision.

Source: https://huggingface.co/datasets/GSMA/ot-lite, revision REVISION, the eight test_*.json
files, each checked against its pinned sha256 before use. The frozen jsonl carries the full
question text, so the evaluation never depends on the hub again.

Groups (reported separately in results):
  core       teleqna 200, teletables 50, oranbench 50, sixg_bench 50 (multiple choice, letter
             answers; ot-lite answers are 0-based indexes into `choices`, verified: max = len-1),
             3gpp_tsg 50 (working-group label)
  telelogs   all 100 (root cause C1-C8)       } the OTel models' training data comes from the
  telemath   all 100 (numeric, rel tol 1%)    } same families, so these are contamination-prone
srsranbench is left out: 124 of its 150 answers are choice A, so "always A" scores 83%.
Selection inside a config: the N items with the smallest sha256(config, index, SEED).
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from t1.evals.common import h, item

UNIT = "knowledge"
SEED = "knowledge-v1"
DATASET = "GSMA/ot-lite"
REVISION = "1c0f2eac3ad0baa29704b147a95fea283b2906c7"
LICENSE = "MIT"
FILES = {
    "teleqna": ("test_teleqna.json", "3bbb71301dd7a2228a8f42d4ca48c100fd57cbbb80b2e22749d5efe86ae31b5a"),
    "teletables": ("test_teletables.json", "d08065834a41cf295242816594963a9ec3f59699c0d26e8bfaabbe6445647b73"),
    "oranbench": ("test_oranbench.json", "458be8c1962041196908aabb94838eae82633efac56c56d114c6982883e81fe8"),
    "sixg_bench": ("test_sixg_bench.json", "b86cff25bb490a5c10a8e22b34d0473fbe6e6b04566f8187942bd6302ba5a5da"),
    "3gpp_tsg": ("test_3gpp_tsg.json", "a7ae12665448bc35dd656629daea640a02fb96a7fe32e400f6a95055c65be20b"),
    "telelogs": ("test_telelogs.json", "9d1eea3a86d39cc9ab792bca0f69edba8703e5681583e70ae82f6ef49b531bf9"),
    "telemath": ("test_telemath.json", "5d4c64ae79d3cdec50c28a4d5cb36b098ae982bd5301aec0337358ec9db4453a"),
}
TAKE = {"teleqna": 200, "teletables": 50, "oranbench": 50, "sixg_bench": 50, "3gpp_tsg": 50,
        "telelogs": 100, "telemath": 100}
GROUP = {"telelogs": "telelogs", "telemath": "telemath"}
MCQ = ("teleqna", "teletables", "oranbench", "sixg_bench")
NUMERIC_REL_TOL = 0.01
LETTERS = "ABCDEFGH"


class SourceUnavailable(RuntimeError):
    pass


def url(fname: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{fname}"


def fetch(cache_dir: Path) -> dict[str, list]:
    """Load (downloading if needed) every source file, verifying the pinned sha256."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for config, (fname, sha) in FILES.items():
        p = cache_dir / fname
        if not p.is_file():
            try:
                with urllib.request.urlopen(url(fname), timeout=60) as r:  # noqa: S310 - pinned https URL
                    p.write_bytes(r.read())
            except Exception as e:  # noqa: BLE001
                raise SourceUnavailable(f"cannot download {url(fname)}: {e}") from e
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != sha:
            raise SourceUnavailable(f"{fname}: sha256 {got} != pinned {sha}")
        out[config] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _choices(config: str, choices: list[str]) -> str:
    lines = []
    for i, ch in enumerate(choices):
        text = re.sub(r"^\s*\d+\.\s*", "", ch) if config == "oranbench" else ch
        lines.append(f"{LETTERS[i]}. {text}")
    return "\n".join(lines)


def _one(config: str, idx: int, rec: dict) -> dict:
    iid = f"kn-{config}-{idx:04d}"
    meta = {"config": config, "source_index": idx, "dataset": DATASET, "revision": REVISION,
            "group": GROUP.get(config, "core"), "gen": SEED}
    if config in MCQ:
        return item(UNIT, iid, "knowledge_mcq",
                    {"question": rec["question"], "choices": _choices(config, rec["choices"])},
                    {"type": "mcq", "answer": LETTERS[int(rec["answer"])], "n_choices": len(rec["choices"])}, meta)
    if config == "3gpp_tsg":
        return item(UNIT, iid, "knowledge_label", {"question": rec["question"]},
                    {"type": "label", "answer": rec["answer"]}, meta)
    if config == "telelogs":
        return item(UNIT, iid, "knowledge_rca", {"question": rec["question"]},
                    {"type": "rca", "answer": rec["answer"].strip().upper()}, meta)
    return item(UNIT, iid, "knowledge_numeric", {"question": rec["question"]},
                {"type": "numeric", "answer": float(rec["answer"]), "rel_tol": NUMERIC_REL_TOL}, meta)


def generate(source: dict[str, list]) -> list[dict]:
    out = []
    for config, n in TAKE.items():
        recs = source[config]
        order = sorted(range(len(recs)), key=lambda i: (h(config, i, SEED), i))[:n]
        for idx in sorted(order):
            out.append(_one(config, idx, recs[idx]))
    return out
