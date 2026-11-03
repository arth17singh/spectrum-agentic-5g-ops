"""A stand-in for spectrum.slices._shared.records.RecordsReader, reading the committed fixtures.

Method names follow the build conventions (runs, declared, result, records, all_records, model, source).
Argument shapes beyond the names are an assumption: result(run_id), records(run_id, kind),
all_records(kind), model(model_id, version).
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


class FakeReader:
    def __init__(self, root: Path = FIXTURES, only_runs: set[str] | None = None) -> None:
        self.root = root
        self.only = only_runs

    def runs(self) -> list[str]:
        ids = sorted(p.stem for p in (self.root / "declared").glob("*.json"))
        return [r for r in ids if self.only is None or r in self.only]

    def declared(self, run_id: str):
        p = self.root / "declared" / f"{run_id}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def result(self, run_id: str):
        p = self.root / "results" / f"{run_id}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def records(self, run_id: str, kind: str) -> list[dict]:
        return load_jsonl(self.root / "records" / run_id / f"{kind}.jsonl")

    def all_records(self, kind: str) -> list[dict]:
        out: list[dict] = []
        for run_id in self.runs():
            out += self.records(run_id, kind)
        return out

    def model(self, model_id: str, version: str):
        p = self.root / "models" / model_id / f"{version}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def source(self):
        return {"kind": "fixtures"}


def all_of(kind: str) -> list[dict]:
    return FakeReader().all_records(kind)


def all_results() -> list[dict]:
    r = FakeReader()
    return [x for x in (r.result(i) for i in r.runs()) if x]
