"""(Re)generate the SYNTHETIC evidence fixtures in tests/fixtures/evidence/. None of these runs are real.

    uv run python tests/make_evidence_fixtures.py
"""

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "fixtures" / "evidence"
R1, R2, R3 = "2026-11-02T00-00-fixture-01", "2026-11-02T00-00-fixture-02", "2026-11-02T00-00-fixture-03"


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def write(rel: str, text: str) -> None:
    p = BASE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def main() -> None:
    decls = {
        R1: {"unit": "knowledge", "target": {"metric": "exact_match", "value": 0.70},
             "test_set_sha256": h("synthetic test set A"), "model_id": "fixture-model-1.7b",
             "grader": "rule-based-parser-v2"},
        R2: {"unit": "reasoning", "target": {"metric": "accuracy", "value": 0.65},
             "test_set_sha256": h("synthetic test set B"), "model_id": "fixture-model-8b",
             "grader": "rule-based-parser-v2"},
    }
    for rid, d in decls.items():
        write(f"declared/{rid}.json", json.dumps({"run_id": rid, **d, "note": "SYNTHETIC FIXTURE"}, indent=2) + "\n")
    # malformed on purpose: target.value missing, short test-set hash, no grader
    write(f"declared/{R3}.json", json.dumps({"run_id": R3, "unit": "knowledge", "target": {"metric": "exact_match"},
                                            "test_set_sha256": "3a7bd3e2", "model_id": "fixture-model-0.6b"},
                                           indent=2) + "\n")
    for rid, n in ((R1, 3), (R2, 2)):
        for i in range(1, n + 1):
            write(f"outputs/{rid}/item-{i:04d}.json",
                  json.dumps({"item": i, "output": f"synthetic answer {i}", "correct": i % 2 == 1}) + "\n")
    write(f"logs/{R1}/run.log", "SYNTHETIC FIXTURE LOG\nstarted\nfinished\n")
    write("judging.json", json.dumps({"run_ids": [R1, R2]}, indent=2) + "\n")
    write("README.md", "Synthetic evidence fixtures for tests and screenshots only. None of these runs are real.\n")


if __name__ == "__main__":
    main()
