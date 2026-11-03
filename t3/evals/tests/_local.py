"""A small incident suite built on THIS machine, for tests that replay incidents.

The frozen suite of record is frozen on the runner's x86_64 image; the twin is a float32 GP, so its
KPIs replay to 1e-6 only there. Tests that replay incidents use incidents built here instead, with the
same generator, so they pass on any platform.
"""

from __future__ import annotations

import functools
import hashlib
import tempfile
from pathlib import Path

from t3.evals import incidents as I
from t3.evals.graders import load_targets

N_LOCAL = 16  # candidates 0..15 cover both presets, both tiers and several templates


@functools.lru_cache(maxsize=1)
def local_suite() -> tuple[Path, list[dict], str]:
    targets = load_targets()
    kept = []
    for i in range(N_LOCAL):
        inc, _ = I.build(I.candidate(i), targets)
        if inc is not None:
            kept.append(inc)
    path = Path(tempfile.mkdtemp(prefix="t3-evals-local-suite-")) / "incidents-local.jsonl"
    data = I.to_bytes(kept)
    path.write_bytes(data)
    return path, kept, hashlib.sha256(data).hexdigest()
