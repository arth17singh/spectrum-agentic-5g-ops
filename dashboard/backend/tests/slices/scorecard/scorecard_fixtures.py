"""Shared by the scorecard tests: a stand-in for RecordsReader that reads the synthetic fixtures
(every record carries synthetic: true) laid out like the evidence store, plus the golden expectation.

GOLDEN VALUES, computed by hand (the tests compare these exactly, they do not recompute them):

  syn-t1-acc        target accuracy >= 0.85
                    measured 17 correct of 20 = 17 / 20 = 0.85; 0.85 >= 0.85 is TRUE (equal passes) -> pass
                    baseline run measured 10 of 20 = 0.50; gain = 0.85 / 0.50 = 1.7, interval [1.4, 2.0], paired n 20
  syn-t3-deny       target deny_rate >= 0.95
                    measured 39 of 40 = 0.975; 0.975 >= 0.95 -> pass
                    baseline run measured 30 of 40 = 0.75; gain = 0.975 / 0.75 = 1.3, interval [1.1, 1.5], paired n 40
  syn-t3-pause      target first_refusal_s <= 60
                    measured 75; 75 <= 60 is FALSE -> fail (stored passed=false agrees)
  syn-t3-trace      declared (== 1, 12 items) and no result file -> "not checked"
  syn-t3-retention  target retention >= 0.8
                    measured 0.45; 0.45 >= 0.8 is FALSE -> recomputed fail,
                    but the stored result says passed=true -> "inconsistent"

  track-1: 1 declared, 1 passed.
  track-3: 4 declared = 1 passed (deny) + 1 failed (pause) + 1 not checked (trace) + 1 inconsistent (retention).
  totals : 5 declared = 2 passed + 1 failed + 1 not checked + 1 inconsistent; all 5 rows synthetic.
  orphan : syn-orphan has a result and no declaration -> listed, never counted.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

GOLDEN_STATES = {
    "syn-t1-acc": "pass",
    "syn-t3-deny": "pass",
    "syn-t3-pause": "fail",
    "syn-t3-trace": "not checked",
    "syn-t3-retention": "inconsistent",
}


def load(kind: str, run_id: str):
    p = FIXTURES / kind / f"{run_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


class FakeReader:
    """Same surface as spectrum.slices._shared.records.RecordsReader, over the fixture files."""

    def __init__(self, declared=None, results=None, runs=None, n_items=None):
        self._d = declared
        self._r = results
        self._runs = runs
        self._n_items = n_items

    def runs(self):
        if self._runs is not None:
            return list(self._runs)
        ids = {p.stem for k in ("declared", "results") for p in (FIXTURES / k).glob("*.json")}
        return sorted(ids)

    def declared(self, run_id):
        return load("declared", run_id) if self._d is None else self._d.get(run_id)

    def result(self, run_id):
        return load("results", run_id) if self._r is None else self._r.get(run_id)

    def records(self, run_id, kind):
        return [{"kind": kind, "i": i} for i in range(self._n_items or 0)]

    def all_records(self, kind):
        return []

    def model(self):
        return None

    def source(self):
        return {"kind": "local", "uri": "fixtures", "synthetic": True}
