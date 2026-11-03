"""tools/check_unit.py part e replays the intervals t3/evals/run.py records (fake S3, no AWS).

Real runs are produced with this lane's own code (run.py on a locally built suite), copied into a fake
evidence bucket, and checked by the independent checker: details.gain must carry the full
paired-bootstrap-v1 spec and replay exactly.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "tools", ROOT / "tools" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

check_unit = pytest.importorskip("check_unit")
evfake = pytest.importorskip("evfake")

from t3.evals import run as R  # noqa: E402
from t3.evals.graders import load_targets  # noqa: E402
from t3.evals.tests._local import local_suite  # noqa: E402

BUCKET, PREFIX = "fixture-bucket", "evidence/"
SUITE, INCS, DIGEST = local_suite()
SPEC_KEYS = {"method", "seed", "resamples", "level", "pair_key", "score_field", "baseline_run_id",
             "baseline_model_id", "cluster_field", "subset", "numpy_version", "value", "ci_low", "ci_high", "paired_n"}


class Bucket:
    def __init__(self):
        self.aws = evfake.FakeAws(now="2026-11-03T06:00:00Z")
        self.t = datetime(2026, 11, 3, 5, 0, 0, tzinfo=timezone.utc)

    def upload_run(self, root: Path, run_id: str) -> None:
        """declared first, then the items, then the result, each a second later (S3 LastModified order)."""
        for rel in (f"declared/{run_id}.json", f"records/{run_id}/item.jsonl", f"results/{run_id}.json"):
            self.t += timedelta(seconds=1)
            self.aws.s3.put(BUCKET, PREFIX + rel, (root / rel).read_bytes(), self.t.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def part_e(self, run_id: str):
        ev = check_unit.S3Evidence(self.aws.client("s3"), f"s3://{BUCKET}/{PREFIX}")
        ck = check_unit.Checker(run_id, ev, self.aws, None, poll_s=0.0, now=self.aws.now())
        ck.load()
        ck.decl_time = ck.ev.time(ck.decl_rel)
        ck.part_e()
        return ck


def _text(ck) -> str:
    return "\n".join(ck.parts["e"].lines)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("ev")
    kw = dict(allow_dirty=True, suite=SUITE)
    R.run("no_action", d, [], run_id="t-na", **kw)
    R.run("rule_baseline", d, ["t-na"], run_id="t-rb", **kw)
    R.run("config_revert", d, ["t-na"], run_id="t-cr", **kw)
    R.run("config_revert", d, ["t-na"], run_id="t-crm", subset="masked", **kw)
    R.run("swarm", d, ["t-cr", "t-rb"], run_id="t-sw", **kw)  # dry-run fake, every tier
    R.run("swarm", d, ["t-cr", "t-rb"], run_id="t-swm", subset="masked", **kw)
    b = Bucket()
    for rid in ("t-na", "t-rb", "t-cr", "t-crm", "t-sw", "t-swm"):
        b.upload_run(d, rid)
    return d, b


def _result(d: Path, rid: str) -> dict:
    return json.loads((d / "results" / f"{rid}.json").read_text())


def test_gain_records_the_full_spec_from_targets_json(runs):
    d, _ = runs
    boot = load_targets()["bootstrap"]
    res = _result(d, "t-cr")
    g = res["details"]["gain"]
    assert SPEC_KEYS <= set(g)
    assert (g["method"], g["seed"], g["resamples"], g["level"]) == ("paired-bootstrap-v1", boot["seed"],
                                                                    boot["resamples"], boot["ci"])
    assert g["baseline_run_id"] == "t-na" and g["baseline_model_id"] == "no_action"
    assert g["pair_key"] == "item_id" and g["score_field"] == "score" and g["cluster_field"] is None
    assert g["subset"] is None and res["value"] == g["ci_low"]
    for entry in res["details"]["gains"]:
        assert SPEC_KEYS <= set(entry) and entry["success_rate_gain"]["score_field"] == "passed"


@pytest.mark.parametrize("rid", ["t-rb", "t-cr", "t-crm"])
def test_check_unit_part_e_passes_on_the_gain_baselines(runs, rid):
    _, b = runs
    ck = b.part_e(rid)
    assert ck.parts["e"].state == "PASS", _text(ck)
    assert "gain_ci_low_vs_no_action = " in _text(ck) and "max |difference| 0.0e+00" in _text(ck)


def test_swarm_primary_subset_only_when_the_run_spans_every_tier(runs):
    d, b = runs
    every = _result(d, "t-sw")["details"]["gain"]
    masked_only = _result(d, "t-swm")["details"]["gain"]
    assert every["subset"] == {"field": "extra.incident.tier", "equals": "masked"}
    assert masked_only["subset"] is None
    assert every["baseline_run_id"] == "t-cr" and every["baseline_model_id"] == "config-revert-median-v1"
    assert every["value"] == pytest.approx(masked_only["value"]) and every["paired_n"] == masked_only["paired_n"]
    for rid in ("t-sw", "t-swm"):
        ck = b.part_e(rid)
        assert ck.parts["e"].state == "PASS", _text(ck)
        assert "gain_ci_low_vs_config_revert_masked = " in _text(ck)
    assert "subset extra.incident.tier='masked'" in _text(b.part_e("t-sw"))


def test_no_action_needs_no_interval(runs):
    _, b = runs
    ck = b.part_e("t-na")
    assert ck.parts["e"].state == "PASS", _text(ck)
