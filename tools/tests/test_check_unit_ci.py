"""check_unit part e: paired-bootstrap targets (gain_ci_low_*) replayed from the item records.

The producers here are verbatim copies of the lanes' own bootstrap code (W10 t3/evals/stats.py, W11
silos/nn_core.py), so an exact match shows that check_unit's independent replay is the same computation."""

import hashlib
import json
from typing import Any

import numpy as np
import pytest

import check_unit
from evfake import FakeAws

BUCKET, PREFIX = "fixture-bucket", "evidence/"
TS = hashlib.sha256(b"frozen suite").hexdigest()
SHA = "d65cb863db084d9cd20c80e48d47e49d2e9fd1b4"


# ---------------------------------------------------------------------------- the producers' own code

def w10_paired_bootstrap(candidate: dict, baseline: dict, resamples: int, seed: int, ci: float) -> dict:
    """t3/evals/stats.py (W10), unchanged."""
    ids = sorted(set(candidate) & set(baseline))
    d = np.array([candidate[i] - baseline[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {"value": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "paired_n": len(ids)}


def w11_paired_bootstrap(diff, groups, n_boot: int, seed: int, level: float = 0.95) -> dict:
    """silos/nn_core.py paired_bootstrap (W11), cluster part, unchanged."""
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    g_sum = np.bincount(inv, weights=diff, minlength=len(uniq))
    g_n = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = g_sum[idx].sum(axis=1) / g_n[idx].sum(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [a, 1.0 - a])
    return {"mean": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


# ---------------------------------------------------------------------------- fixture builder

def env(kind: str, run_id: str) -> dict:
    return {"schema_version": "1", "kind": kind, "run_id": run_id, "team": "team13", "created_at": "2026-11-03T03:00:00.000Z",
            "aws_time": None, "aws_time_source": None, "synthetic": True}


def item(run_id: str, iid: str, score: float, passed: bool, **extra) -> dict:
    return {**env("item", run_id), "item_id": iid, "input_sha256": "0" * 64, "output_sha256": "1" * 64,
            "score": score, "passed": passed, "grader": "g", "backend": "b", "extra": extra}


class Store:
    def __init__(self):
        self.aws = FakeAws(now="2026-11-03T04:00:00Z")
        self.t = 0

    def put(self, rel: str, body: bytes) -> None:
        self.t += 1
        self.aws.s3.put(BUCKET, PREFIX + rel, body, f"2026-11-03T03:{self.t // 60:02d}:{self.t % 60:02d}Z")

    def run(self, run_id: str, items: list[dict], target: dict, model_id: str, result: dict | None = None,
            baseline_run_id: str | None = None, test_set=TS) -> None:
        decl = {**env("declared", run_id), "unit": "t3.evals.recovery", "track": "track-3", "target": target,
                "n_items": len(items), "test_set_sha256": test_set, "model_id": model_id, "grader": "g",
                "code_sha": SHA, "region": "us-east-2", "declared_at": "2026-11-03T03:00:00.000Z"}
        if baseline_run_id:
            decl["baseline_run_id"] = baseline_run_id
        self.put(f"declared/{run_id}.json", json.dumps(decl).encode())
        self.put(f"records/{run_id}/item.jsonl", b"".join(json.dumps(i).encode() + b"\n" for i in items))
        if result is not None:
            res = {**env("result", run_id), "unit": "t3.evals.recovery", "metric": target["metric"],
                   "target": target, "n_items": len(items), "checked_at": "2026-11-03T03:30:00.000Z",
                   "checker": {"script": "x", "code_sha": SHA}, **result}
            self.put(f"results/{run_id}.json", json.dumps(res).encode())

    def check(self, run_id: str) -> check_unit.Checker:
        ev = check_unit.S3Evidence(self.aws.client("s3"), f"s3://{BUCKET}/{PREFIX}")
        ck = check_unit.Checker(run_id, ev, self.aws, None, poll_s=0.0, now=self.aws.now())
        ck.load()
        ck.decl_time = ck.ev.time(ck.decl_rel)
        ck.part_e()
        return ck


def text(ck) -> str:
    return "\n".join(ck.parts["e"].lines)


SEED, R, LEVEL = 20261102, 10000, 0.95
TARGET = {"metric": "gain_ci_low_vs_no_action", "op": ">", "value": 0.0, "unit": "recovery"}
rng0 = np.random.default_rng(7)
BASE = {f"inc-{k:03d}": float(v) for k, v in enumerate(rng0.normal(0.0, 1e-9, 24))}
CAND = {f"inc-{k:03d}": float(v) for k, v in enumerate(rng0.uniform(-0.2, 0.9, 24))}


def spec(g: dict, **over) -> dict[str, Any]:
    s = {"method": "paired-bootstrap-v1", "seed": SEED, "resamples": R, "level": LEVEL, "pair_key": "item_id",
         "score_field": "score", "baseline_run_id": "t.base", "cluster_field": None, "subset": None, **g}
    s.update(over)
    return s


def build(store: Store, gain: dict | None = None, value: float | None = None, base_test_set=TS,
          cand_items: dict | None = None) -> None:
    cand_items = cand_items or CAND
    store.run("t.base", [item("t.base", i, s, False) for i, s in BASE.items()],
              {"metric": "mean_score", "op": "<=", "value": 0.05, "unit": "recovery"}, "no_action",
              result={"value": float(np.mean(list(BASE.values()))), "passed": True, "details": {}},
              test_set=base_test_set)
    g = w10_paired_bootstrap(cand_items, BASE, R, SEED, LEVEL)
    gain = spec(g) if gain is None else gain
    v = g["ci_low"] if value is None else value
    store.run("t.cand", [item("t.cand", i, s, s > 0.3) for i, s in cand_items.items()], TARGET, "config-revert-median-v1",
              result={"value": v, "passed": v > 0, "details": {"gain": gain, "gains": [gain]}}, baseline_run_id="t.base")


# ---------------------------------------------------------------------------- tests

def test_w10_gain_ci_low_replays_exactly():
    s = Store()
    build(s)
    ck = s.check("t.cand")
    assert ck.parts["e"].state == "PASS", text(ck)
    out = text(ck)
    assert "gain_ci_low_vs_no_action = " in out and "max |difference| 0.0e+00" in out
    assert "baseline t.base (declared model 'no_action')" in out and "24 pairs on item_id" in out


def test_w10_today_names_the_missing_fields():
    s = Store()
    g = w10_paired_bootstrap(CAND, BASE, R, SEED, LEVEL)
    build(s, gain={**g, "baseline_run_id": "t.base"})  # what W10 records today
    ck = s.check("t.cand")
    assert ck.parts["e"].state == "CANNOT CHECK"
    assert ("details.gain does not record method, seed, resamples, level, pair_key, score_field, cluster_field, "
            "subset") in text(ck)


def test_a_changed_bound_fails_with_the_difference():
    s = Store()
    g = w10_paired_bootstrap(CAND, BASE, R, SEED, LEVEL)
    build(s, gain=spec({**g, "ci_low": g["ci_low"] + 1e-6}), value=g["ci_low"] + 1e-6)
    ck = s.check("t.cand")
    assert ck.parts["e"].state == "FAIL" and "difference -1.000e-06" in text(ck)


def test_another_seed_does_not_replay():
    s = Store()
    g = w10_paired_bootstrap(CAND, BASE, R, SEED, LEVEL)
    build(s, gain=spec(g, seed=SEED + 1))
    assert s.check("t.cand").parts["e"].state == "FAIL"


def test_baseline_on_another_test_set_fails():
    s = Store()
    build(s, base_test_set=hashlib.sha256(b"other").hexdigest())
    ck = s.check("t.cand")
    assert ck.parts["e"].state == "FAIL" and "not the same items" in text(ck)


def test_unknown_method_cannot_be_replayed():
    s = Store()
    g = w10_paired_bootstrap(CAND, BASE, R, SEED, LEVEL)
    build(s, gain=spec(g, method="paired bootstrap, percentile 95%"))
    ck = s.check("t.cand")
    assert ck.parts["e"].state == "CANNOT CHECK" and "replays only 'paired-bootstrap-v1'" in text(ck)


def test_subset_on_the_candidate_items():
    """The swarm's primary gain: masked-tier items only, against a baseline over every tier."""
    s = Store()
    tiers = {i: ("masked" if k % 2 else "visible") for k, i in enumerate(CAND)}
    s.run("t.base", [item("t.base", i, sc, False) for i, sc in BASE.items()], TARGET | {"metric": "mean_score"},
          "config-revert-median-v1", result={"value": float(np.mean(list(BASE.values()))), "passed": True, "details": {}})
    masked = {i: sc for i, sc in CAND.items() if tiers[i] == "masked"}
    g = w10_paired_bootstrap(masked, BASE, R, SEED, LEVEL)
    gain = spec(g, subset={"field": "extra.incident.tier", "equals": "masked"})
    s.run("t.cand", [item("t.cand", i, sc, sc > 0.3, incident={"tier": tiers[i]}) for i, sc in CAND.items()],
          {**TARGET, "metric": "gain_ci_low_vs_config_revert_masked"}, "swarm",
          result={"value": g["ci_low"], "passed": g["ci_low"] > 0, "details": {"gain": gain}}, baseline_run_id="t.base")
    ck = s.check("t.cand")
    assert ck.parts["e"].state == "PASS", text(ck)
    assert "12 pairs" in text(ck) and "subset extra.incident.tier='masked'" in text(ck)


def test_w11_cluster_bootstrap_beside_a_plain_target():
    """Track 2 compare: items hold the per-row difference; clusters are extra.seed; target mean_score > 0."""
    s = Store()
    r = np.random.default_rng(3)
    rows = [(f"row-{k:04d}", float(d), int(k // 5)) for k, d in enumerate(r.normal(0.02, 0.05, 60))]
    ids = sorted(i for i, _, _ in rows)
    by = {i: (d, c) for i, d, c in rows}
    b = w11_paired_bootstrap([by[i][0] for i in ids], np.array([by[i][1] for i in ids]), 2000, SEED)
    gain = {"method": "paired-bootstrap-v1", "seed": SEED, "resamples": 2000, "level": 0.95, "pair_key": "item_id",
            "score_field": "score", "baseline_run_id": None, "cluster_field": "extra.seed", "subset": None,
            "value": b["mean"], "ci_low": b["ci_lo"], "ci_high": b["ci_hi"], "paired_n": len(ids)}
    mean = float(np.mean([d for _, d, _ in rows]))
    s.run("t.cmp", [item("t.cmp", i, d, d > 0, seed=c) for i, d, c in rows],
          {"metric": "mean_score", "op": ">", "value": 0, "unit": "scaled-MAE"}, "x vs y",
          result={"value": mean, "passed": mean > 0, "details": {"gain": gain}})
    ck = s.check("t.cmp")
    out = text(ck)
    assert ck.parts["e"].state == "PASS", out
    assert "details.gain interval" in out and "clusters extra.seed (12)" in out and "max |difference| 0.0e+00" in out
    gain["ci_low"] += 1e-3  # a recorded interval that does not replay fails part e even beside a plain target
    s2 = Store()
    s2.run("t.cmp", [item("t.cmp", i, d, d > 0, seed=c) for i, d, c in rows],
           {"metric": "mean_score", "op": ">", "value": 0, "unit": "scaled-MAE"}, "x vs y",
           result={"value": mean, "passed": mean > 0, "details": {"gain": gain}})
    assert s2.check("t.cmp").parts["e"].state == "FAIL"


def test_w12_style_interval_is_noted_not_replayed():
    s = Store()
    gain = {"value": 0.1, "ci_low": 0.0, "ci_high": 0.2, "paired_n": 24, "metric": "pass_rate",
            "method": "paired bootstrap, percentile 95%", "resamples": 2000, "seed": SEED, "baseline_run_id": "t.base"}
    s.run("t.t1", [item("t.t1", i, sc, sc > 0.3) for i, sc in CAND.items()],
          {"metric": "pass_rate", "op": ">=", "value": 0.1, "unit": "ratio"}, "tuned",
          result={"value": sum(1 for v in CAND.values() if v > 0.3) / len(CAND), "passed": True,
                  "details": {"gain": gain}})
    ck = s.check("t.t1")
    assert ck.parts["e"].state == "PASS"
    assert "details.gain does not record level, pair_key, score_field, cluster_field, subset" in text(ck)


@pytest.mark.parametrize("metric,expect", [("gain_ci_low_vs_no_action", True), ("ci_high", True),
                                           ("mean_score", False), ("p95_latency_ms", False), ("gain_cil", False)])
def test_ci_metric_names(metric, expect):
    assert bool(check_unit.CI_METRIC_RE.search(metric)) is expect
