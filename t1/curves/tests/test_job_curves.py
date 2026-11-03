"""t1/curves/job_curves.py offline: tiny fake artifacts, a fake CloudWatch Logs client, a local evlib store."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "contracts")]

from t1.curves import job_curves as jc  # noqa: E402
from tools import evlib  # noqa: E402

HOST = "twin-secret-host.example.invalid"
SFT_RUN = "t1.sft.train-20261103T034500Z-9d4f"
RL_TRAIN = "t1.rl.train-20261103T023805Z-a09e"
RL_EVAL = "t1.rl.eval-20261103T043115Z-bf52"
JOB = "t1-rl-train-20261103T023805Z-a09e-20261103-023807"
RUN = "t1.curves-20261103T060000Z-aaaa"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeLogs:
    """GetLogEvents paging like CloudWatch: pages of `page` events, an empty page first, and the last
    forward token repeated. FilterLogEvents / StartQuery do not exist (the job role cannot call them)."""

    def __init__(self, lines, page=7, stream=JOB + "/algo-1-1", deny=None):
        self.lines, self.page, self.stream, self.deny = lines, page, stream, deny
        self.calls = 0

    def describe_log_streams(self, **kw):
        if self.deny:
            raise self.deny
        assert kw["logStreamNamePrefix"] == JOB
        return {"logStreams": [{"logStreamName": self.stream}]}

    def get_log_events(self, **kw):
        self.calls += 1
        assert kw["startFromHead"] is True and kw["logStreamName"] == self.stream
        pos = int(kw["nextToken"][2:]) if kw.get("nextToken") else -1  # f/<index> tokens; first page is empty
        if pos < 0:
            return {"events": [], "nextForwardToken": "f/0"}
        chunk = self.lines[pos:pos + self.page]
        nxt = pos + len(chunk)
        return {"events": [{"message": m} for m in chunk], "nextForwardToken": f"f/{nxt}"}


def rl_lines(rewards, host=HOST):
    lines = [f'{{"ts": "x", "level": "INFO", "message": "POST https://{host}/step 200"}}']
    for i, r in enumerate(rewards, 1):
        lines += [f"METRIC timesteps={i * 1024}", f"METRIC episode_reward_mean={r!r}", "METRIC episode_length_mean=24.0",
                  "METRIC coverage_mean=0.7", f'{{"message": "GET https://{host}/state 200", "note": "METRIC x=1"}}',
                  f"METRIC trailing=1 https://{host}"]
    lines += [f"httpcore.ReadError: [Errno 104] Connection reset by peer {host}"]
    return lines


def make_env(tmp_path, sft_train=5, sft_dev=2, rl_n=4, tamper=False, sums_wrong=False, rl_rewards=None,
             job_status="Failed", with_eval=True, n_items=None):
    train = [{"step": i, "loss": 2.0 / i, "elapsed_s": i} for i in range(1, sft_train + 1)]
    dev = [{"step": s, "loss": 1.5 / (s + 1), "by_unit": {"execution": 1.0, "interface": 0.5}} for s in range(0, sft_dev * 100, 100)]
    metrics = json.dumps({"run_id": SFT_RUN, "train_loss_curve": train, "dev_loss_curve": dev, "steps": sft_train,
                          "hyperparameters": {"epochs": 2, "lr": 0.0002}}, sort_keys=True).encode()
    mdir, sdir = tmp_path / "sft_metrics", tmp_path / "sft_sums"
    mdir.mkdir(), sdir.mkdir()
    (mdir / "train_metrics.json").write_bytes(metrics + (b" " if tamper else b""))
    sums = (f"{'0' * 64}  adapter/x.bin\n{sha(metrics)}  train_metrics.json\n").encode()
    (sdir / "SHA256SUMS").write_bytes(sums + (b"#\n" if sums_wrong else b""))
    src = {
        "schema": "t1-curves-sources-v1",
        "downsample": {"max_points": 500, "rule": "uniform stride, first and last kept"},
        "sft": {"run_id": SFT_RUN, "x": "step", "metrics_key": f"models/m/{SFT_RUN}/train_metrics.json",
                "sums_key": f"models/m/{SFT_RUN}/SHA256SUMS", "sha256sums_sha256": sha(sums),
                "sums_member": "train_metrics.json",
                "series": {"train_loss": {"field": "train_loss_curve", "x": "step", "y": "loss"},
                           "dev_loss": {"field": "dev_loss_curve", "x": "step", "y": "loss"}},
                "expect_points": {"train_loss": sft_train, "dev_loss": sft_dev}},
        "rl": {"train_run_id": RL_TRAIN, "train_job_name": JOB, "log_group": "/aws/sagemaker/TrainingJobs",
               "x": "timesteps", "series": {"mean_episode_reward": {"metric": "episode_reward_mean"}},
               "eval_run_id": RL_EVAL, "expect_points": {"mean_episode_reward": rl_n}},
    }
    sp = tmp_path / "sources.json"
    sp.write_bytes(json.dumps(src, indent=1).encode())
    total = min(sft_train, 500) + sft_dev + rl_n
    decl = {"run_id": RUN, "unit": jc.UNIT, "code_sha": "a" * 40, "test_set_sha256": sha(sp.read_bytes()),
            "n_items": n_items or total, "synthetic": False,
            "target": {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"}}
    dp = tmp_path / "declared.json"
    dp.write_text(json.dumps(decl))
    ev = tmp_path / "evidence_read"
    store = evlib.LocalStore(ev)
    evlib.append_records(RL_TRAIN, "job", [{"job_name": JOB, "job_kind": "training", "status": job_status,
                                            "arn": "arn:aws:sagemaker:us-east-2:123456789012:training-job/" + JOB,
                                            "region": "us-east-2", "aws_start": "2026-11-03T02:38:10.547Z",
                                            "aws_end": "2026-11-03T04:19:28.470Z", "billable_s": 6034,
                                            "est_cost_usd": 1.9, "instance_type": "ml.g6.xlarge"}], store=store)
    if with_eval:
        res = {"run_id": RL_EVAL, "unit": "t1.rl.eval", "metric": "pass_rate", "value": 5 / 6, "passed": False,
               "target": {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"}, "n_items": 6,
               "checker": {"script": "x", "code_sha": "b" * 40},
               "details": {"checkpoint_steps": 50000, "items": {"UPLIFT": False, "SAFETY": True}}}
        evlib.write_result(res, store=store)
    logs = FakeLogs(rl_lines(rl_rewards or [-4.5 + i for i in range(rl_n)]))
    out = tmp_path / "evidence"
    argv = ["--run-id", RUN, "--sources", str(sp), "--sft-metrics", str(mdir), "--sft-sums", str(sdir),
            "--evidence-dir", str(out), "--evidence-read", str(ev), "--declared", str(dp),
            "--summary-dir", str(tmp_path / "summary")]
    return argv, logs, out


def result_of(out):
    return evlib.read_result(RUN, evlib.LocalStore(out)), evlib.read_records(RUN, "item", evlib.LocalStore(out))


def test_happy_path_shape_and_recompute(tmp_path):
    argv, logs, out = make_env(tmp_path)
    assert jc.main(argv, logs=logs) == 0
    res, items = result_of(out)
    assert len(items) == res["n_items"] == 11 and res["passed"] and res["value"] == 1.0
    assert evlib.summarize_items(items, "pass_rate") == 1.0  # what check_unit recomputes
    d = res["details"]
    assert set(d["sft"]) >= {"run_id", "x", "series", "source"} and d["sft"]["x"] == "step"
    assert d["sft"]["series"]["train_loss"][0] == [1, 2.0] and len(d["sft"]["series"]["train_loss"]) == 5
    assert d["sft"]["series"]["dev_loss"][1][0] == 100
    assert d["rl"]["x"] == "timesteps" and d["rl"]["train_run_id"] == RL_TRAIN
    assert d["rl"]["series"]["mean_episode_reward"] == [[1024, -4.5], [2048, -3.5], [3072, -2.5], [4096, -1.5]]
    assert d["rl"]["crashed_at_timesteps"] == 4096 and d["rl"]["served_checkpoint"] == "50k"
    assert d["rl"]["eval_run_id"] == RL_EVAL and d["rl"]["eval"]["passed"] is False
    assert d["rl"]["train_job_status"] == "Failed" and "series_reasons" in d["rl"]
    assert d["downsample"]["max_points"] == 500
    ids = [i["item_id"] for i in items]
    assert len(set(ids)) == len(ids) and "rl.mean_episode_reward:t=1024" in ids and "sft.train_loss:step=1" in ids
    dev = next(i for i in items if i["item_id"] == "sft.dev_loss:step=100")
    assert dev["extra"]["by_unit"] == {"execution": 1.0, "interface": 0.5}
    assert json.loads((tmp_path / "summary" / "curves_summary.json").read_text())["result"]["passed"] is True


def test_twin_address_never_leaves_the_log(tmp_path, capsys):
    argv, logs, out = make_env(tmp_path)
    assert jc.main(argv, logs=logs) == 0
    blob = capsys.readouterr().out
    for p in list(out.rglob("*")) + list((tmp_path / "summary").rglob("*")):
        if p.is_file():
            blob += p.read_text()
    assert HOST not in blob and "Connection reset" not in blob


def test_not_crashed_when_job_completed(tmp_path):
    argv, logs, out = make_env(tmp_path, job_status="Completed")
    jc.main(argv, logs=logs)
    d = result_of(out)[0]["details"]
    assert d["rl"]["crashed_at_timesteps"] is None and d["rl"]["crash_note"] is None


def test_no_eval_result_leaves_checkpoint_null(tmp_path):
    argv, logs, out = make_env(tmp_path, with_eval=False)
    jc.main(argv, logs=logs)
    d = result_of(out)[0]["details"]["rl"]
    assert d["served_checkpoint"] is None and d["eval"] is None


def test_downsample_rule():
    pts = [(i, float(i)) for i in range(1, 1201)]
    kept = jc.downsample(pts, 500)
    assert len(kept) == 500 and kept[0] == pts[0] and kept[-1] == pts[-1]
    xs = [x for x, _ in kept]
    assert xs == sorted(set(xs))
    assert jc.downsample(pts[:500], 500) == pts[:500] and jc.downsample(pts[:3], 500) == pts[:3]


def test_long_series_is_thinned_in_evidence(tmp_path):
    argv, logs, out = make_env(tmp_path, sft_train=1200, sft_dev=2, rl_n=3)
    # 1200 points found as declared, 500 recorded: n_items counts the records a complete run writes
    assert json.loads((tmp_path / "declared.json").read_text())["n_items"] == 500 + 2 + 3
    jc.main(argv, logs=logs)
    res, items = result_of(out)
    pts = res["details"]["sft"]["series"]["train_loss"]
    assert len(pts) == 500 and pts[0][0] == 1 and pts[-1][0] == 1200
    assert res["details"]["downsample"]["by_series"]["sft.train_loss"] == {"raw_points": 1200, "kept": 500, "applied": True}


def test_sft_checksum_mismatch_is_null_with_reason_and_rl_still_recorded(tmp_path):
    argv, logs, out = make_env(tmp_path, tamper=True)
    assert jc.main(argv, logs=logs) == 0
    res, items = result_of(out)
    s = res["details"]["sft"]
    assert s["series"] == {"train_loss": None, "dev_loss": None}
    assert "differs from SHA256SUMS" in s["series_reasons"]["train_loss"]
    assert res["details"]["rl"]["series"]["mean_episode_reward"] is not None
    assert not res["passed"] and res["value"] < 1.0
    assert {i["item_id"] for i in items if not i["passed"]} == {"sft.train_loss:incomplete", "sft.dev_loss:incomplete"}


def test_wrong_sums_file_is_refused_as_a_series(tmp_path):
    argv, logs, out = make_env(tmp_path, sums_wrong=True)
    jc.main(argv, logs=logs)
    s = result_of(out)[0]["details"]["sft"]
    assert s["series"]["train_loss"] is None and "model record" in s["series_reasons"]["dev_loss"]


def test_logs_access_denied_nulls_rl_only(tmp_path):
    class Denied(Exception):
        response = {"Error": {"Code": "AccessDeniedException"}}

    argv, _, out = make_env(tmp_path)
    logs = FakeLogs([], deny=Denied("not allowed"))
    assert jc.main(argv, logs=logs) == 0
    res, _ = result_of(out)
    rl = res["details"]["rl"]
    assert rl["series"]["mean_episode_reward"] is None and "AccessDeniedException" in rl["series_reasons"]["mean_episode_reward"]
    assert rl["crashed_at_timesteps"] is None
    assert res["details"]["sft"]["series"]["train_loss"] is not None and not res["passed"]


def test_short_series_gets_an_incomplete_item(tmp_path):
    argv, logs, out = make_env(tmp_path, rl_n=4)
    d = json.loads((tmp_path / "declared.json").read_text())
    src = json.loads((tmp_path / "sources.json").read_text())
    src["rl"]["expect_points"]["mean_episode_reward"] = 6  # declared 6, the log has 4
    (tmp_path / "sources.json").write_text(json.dumps(src))
    d["test_set_sha256"] = sha((tmp_path / "sources.json").read_bytes())
    d["n_items"] = 5 + 2 + 6
    (tmp_path / "declared.json").write_text(json.dumps(d))
    jc.main(argv, logs=logs)
    res, items = result_of(out)
    assert not res["passed"]
    bad = next(i for i in items if not i["passed"])
    assert bad["item_id"] == "rl.mean_episode_reward:incomplete" and bad["extra"]["points"] == 4
    assert bad["extra"]["declared_points"] == 6


def test_declaration_must_match_sources_and_count(tmp_path, capsys):
    argv, logs, out = make_env(tmp_path)
    (tmp_path / "sources.json").write_text((tmp_path / "sources.json").read_text() + " ")
    assert jc.main(argv, logs=logs) == 2 and "test_set_sha256" in capsys.readouterr().out
    (tmp_path / "b").mkdir()
    argv, logs, out = make_env(tmp_path / "b", n_items=99)
    assert jc.main(argv, logs=logs) == 2
    assert not out.exists()


def test_metric_lines_and_paging():
    msgs = ["noise", "METRIC timesteps=1024\nMETRIC episode_reward_mean=-4.5\nMETRIC episode_length_mean=24.0",
            "METRIC timesteps=2048", "METRIC episode_reward_mean=3.9e+01", "METRIC episode_reward_mean=abc",
            "  METRIC timesteps=3072  ", "METRIC episode_reward_mean=nan", f"x METRIC timesteps=9 {HOST}"]
    blocks, scanned = jc.read_metric_blocks(msgs)
    assert [b["timesteps"] for b in blocks] == [1024, 2048, 3072]
    assert blocks[0]["episode_reward_mean"] == -4.5 and blocks[1]["episode_reward_mean"] == 39.0
    got = list(jc.iter_log_messages(FakeLogs([f"m{i}" for i in range(20)], page=6), "g", JOB))
    assert got == [f"m{i}" for i in range(20)]


def test_nonfinite_reward_is_dropped_and_flagged(tmp_path):
    argv, logs, out = make_env(tmp_path, rl_rewards=[1.0, float("nan"), 3.0, 4.0])
    jc.main(argv, logs=logs)
    res, items = result_of(out)
    assert len(res["details"]["rl"]["series"]["mean_episode_reward"]) == 3 and not res["passed"]
    assert "3 points found, 4 declared" in res["details"]["rl"]["series_reasons"]["mean_episode_reward"]


def test_committed_sources_and_example_declaration():
    src = json.loads(jc.SOURCES_PATH.read_text())
    total = jc.expected_records(src)
    ex = json.loads((ROOT / "tools" / "sm" / "examples" / "t1-curves.json").read_text())
    from tools.sm import run as runner

    assert total == ex["n_items"] == 366
    assert set(ex) <= set(runner.DECLARED_KEYS) | set(runner.TEMPLATE_ONLY)
    assert ex["unit"] == jc.UNIT and ex["test_set"] == "t1/curves/sources.json"
    assert ex["target"]["metric"] in evlib.METRICS and ex["synthetic"] is False
    assert src["sft"]["run_id"] == SFT_RUN and src["rl"]["train_job_name"] == JOB
    assert src["downsample"]["max_points"] == 500
