"""Every Twin view fixture record is shaped like a real one: it passes W1's team13_contracts.validate_record.

The fixtures are marked synthetic: true (so the view never shows them as real), but apart from that flag they
must be records the real writers could produce, including twin_state, twin.preset and approved_by/approved_at.
"""

import json
import sys
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"
CONTRACTS = Path(__file__).resolve().parents[5] / "contracts"
sys.path.insert(0, str(CONTRACTS))  # W1's package; not installed in the backend env

tc = pytest.importorskip("team13_contracts")


def _records():
    out = []
    for p in sorted(FIX.rglob("*.json")):
        out.append((str(p.relative_to(FIX)), json.loads(p.read_text())))
    for p in sorted(FIX.rglob("*.jsonl")):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip():
                out.append((f"{p.relative_to(FIX)}#{n}", json.loads(line)))
    return out


RECORDS = _records()


def test_fixture_set_covers_every_kind_the_view_reads():
    kinds = {r["kind"] for _, r in RECORDS}
    assert {"change", "decision", "item", "declared"} <= kinds
    changes = [r for _, r in RECORDS if r["kind"] == "change"]
    assert changes and all(r.get("twin_state") and r["twin"].get("preset") for r in changes)
    assert any(r.get("approved_by") and r.get("approved_at") for r in changes)


@pytest.mark.parametrize("where,rec", RECORDS, ids=[w for w, _ in RECORDS])
def test_every_fixture_record_passes_the_contract(where, rec):
    assert rec["synthetic"] is True, f"{where}: fixtures must stay synthetic"
    tc.validate_record(rec)  # raises on any contract violation
