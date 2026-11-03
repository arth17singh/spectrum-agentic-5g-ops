"""t1/rl/job_evaluate.py offline: synthetic inputs, a fake SDK run that writes the SYNTHETIC fixtures, local stores.

No AWS, no twin, no ran_rl needed (one test uses ran_rl's compute_gates when it is installed, to show the fixture
has the SDK's shape).
"""

import copy
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from t1.rl import job_evaluate as je
from t1.rl.mask import Masker
from tools import evlib

FIX = Path(__file__).resolve().parent / "fixtures"
RID = "t1.rl.eval-20261103T060000Z-test"
URL = "https://twin-synthetic.example.internal:8443/api"
TOKEN = "tok-SYNTHETIC-not-a-real-token-0123"
ENV = {"RAN_RL_TWIN_URL": URL, "RAN_RL_ADAPTER_TOKEN": TOKEN}
PROJECT = {
    "config/eval_scenarios.yaml": b"scenario_set_version: SYNTHETIC\n",
    "config/env.yaml": b"use_case: cco\ntwin:\n  backend: http\n",
    "config/reward.yaml": b"reward:\n  function: team.rewards:r\n",
    "config/train.yaml": b"algorithm: ppo\n",
    "team/__init__.py": b"",
    "team/rewards.py": b"def r(ctx):\n    return 0.0\n",
    "team/__pycache__/rewards.cpython-311.pyc": b"\x00junk",
    ".env": b"RAN_RL_TWIN_URL=https://should-never-be-extracted.example\n",
    "requirements.txt": b"x\n",
    "sagemaker_entry.py": b"print('not ours')\n",
}
MODEL_ZIP = b"SYNTHETIC model.zip bytes"


@pytest.fixture(autouse=True)
def _no_aws_env(monkeypatch):
    for k in ("EVIDENCE_S3", "EVIDENCE_DIR", "AWS_PROFILE", "RAN_RL_TWIN_URL", "RAN_RL_ADAPTER_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing-not-a-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing-not-a-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tgz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(body)
            t.addfile(ti, io.BytesIO(body))
    return buf.getvalue()


def declaration(model_sha: str, project_sha: str) -> dict:
    items = {k: f"synthetic requirement for {k}" for k in je.ITEM_IDS}
    return {
        "run_id": RID, "unit": "t1.rl.eval", "track": "track-1",
        "target": {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"}, "n_items": 6,
        "test_set_sha256": sha(PROJECT["config/eval_scenarios.yaml"]), "model_id": "ppo-synthetic",
        "grader": "synthetic", "code_sha": "a" * 40, "region": "us-east-2", "synthetic": True,
        "twin": {"kind": "maveric-hosted", "version": "1.2.9", "scenario": "nominal", "seed": 42, "config": {
            "items": items, "env_config_sha256": sha(PROJECT["config/env.yaml"]),
            "reward_config_sha256": sha(PROJECT["config/reward.yaml"]),
            "team_rewards_sha256": sha(PROJECT["team/rewards.py"]),
            "eval": {"training_run_id": "t1.rl.train-synthetic", "training_job_name": "train-job-synthetic",
                     "model_artifact_uri": "s3://synthetic-bucket/models/model.tar.gz",
                     "model_artifact_sha256": model_sha, "project_sha256": project_sha}}},
    }


class World:
    """Channels, stores and a fake SDK run for one test."""

    def __init__(self, tmp: Path, gates_patch=None, coverage=None, decl_patch=None):
        self.tmp = tmp
        self.model = tgz({"model.zip": MODEL_ZIP, "metrics.json": b"{}", "run_metadata.json": b"{}",
                          "checkpoints/x.zip": b"no"})
        self.project = tgz(PROJECT)
        (tmp / "ch" / "model").mkdir(parents=True)
        (tmp / "ch" / "project").mkdir(parents=True)
        (tmp / "ch" / "model" / "model.tar.gz").write_bytes(self.model)
        (tmp / "ch" / "project" / "sourcedir.tar.gz").write_bytes(self.project)
        self.store = evlib.LocalStore(tmp / "s3-evidence")
        decl = declaration(sha(self.model), sha(self.project))
        if decl_patch:
            decl_patch(decl)
        evlib.write_declared(decl, self.store)
        self.gates_patch, self.coverage = gates_patch, coverage
        self.calls: list[dict] = []

    def run_eval(self, argv: list[str]) -> int:
        a = dict(zip(argv[::1], argv[1:] + [""]))
        self.calls.append({"argv": argv, "cwd": os.getcwd(), "dotenv": Path(".env").exists(),
                           "team_on_path": os.getcwd() in __import__("sys").path})
        ev = json.loads((FIX / "evaluation.SYNTHETIC.json").read_text())
        bl = json.loads((FIX / "baseline.SYNTHETIC.json").read_text())
        ev["model"]["checksum"] = sha(Path(a["--model"]).read_bytes())
        if self.gates_patch:
            self.gates_patch(ev["gates"])
        if self.coverage is not None:
            ev["aggregates"]["overall"]["coverage_mean"] = self.coverage
        out = Path(a["--output"])
        out.write_text(json.dumps(ev))
        Path(a["--baseline-output"]).write_text(json.dumps(bl))
        out.with_suffix(".md").write_text("# Evaluation report (SYNTHETIC)\n")
        # what the SDK's httpx logging would print, and an error carrying the address
        print(f"HTTP Request: POST {URL}/reset \"HTTP/1.1 200 OK\"")
        print(f"bearer {TOKEN} host twin-synthetic.example.internal 10.1.2.3", file=__import__("sys").stderr)
        return 0

    def argv(self) -> list[str]:
        return ["--run-id", RID, "--evidence-dir", str(self.tmp / "local-ev"), "--output-dir", str(self.tmp / "out"),
                "--model-channel", str(self.tmp / "ch" / "model"),
                "--project-channel", str(self.tmp / "ch" / "project"), "--work-dir", str(self.tmp / "work")]

    def run(self, env=ENV) -> int:
        return je.main(self.argv(), store=self.store, environ=dict(env), run_eval=self.run_eval)

    def local(self):
        return evlib.LocalStore(self.tmp / "local-ev")

    def items(self) -> dict[str, dict]:
        return {r["item_id"]: r for r in evlib.read_records(RID, "item", self.local())}

    def result(self):
        return evlib.read_result(RID, self.local())


def no_secret_in(text: str) -> None:
    assert URL not in text and TOKEN not in text
    assert "twin-synthetic.example.internal" not in text and "10.1.2.3" not in text


def test_all_pass(tmp_path, capfd):
    w = World(tmp_path)
    assert w.run() == 0
    items, res = w.items(), w.result()
    assert list(items) == list(je.ITEM_IDS) and all(it["passed"] for it in items.values())
    assert res["passed"] is True and res["value"] == 1.0 and res["n_items"] == 6
    assert res["details"]["training_run_id"] == "t1.rl.train-synthetic"
    # the SDK ran from the project with team/ importable and no .env reachable
    call = w.calls[0]
    assert Path(call["cwd"]).resolve() == (tmp_path / "work" / "project").resolve()
    assert call["team_on_path"] and not call["dotenv"]
    assert call["argv"][:3] == ["--model", str(tmp_path / "work" / "model" / "model.zip"), "--with-baseline"]
    assert not (tmp_path / "work" / "project" / ".env").exists()
    assert not (tmp_path / "work" / "project" / "team" / "__pycache__").exists()
    assert not (tmp_path / "work" / "model" / "checkpoints").exists()
    # artifacts: job output dir, evidence store (outputs/<run_id>/eval/), attested hashes
    out = capfd.readouterr()
    for n in je.ARTIFACTS:
        b = (tmp_path / "out" / "eval" / n).read_bytes()
        assert w.store.get(f"outputs/{RID}/eval/{n}") == b
        assert f'"sha256": "{sha(b)}"' in out.out and res["details"]["artifacts"][n]["sha256"] == sha(b)
    assert out.out.count(je.ATTEST_MARK) == 3
    no_secret_in(out.out + out.err)
    assert "<twin>" in out.out
    # every record validates against the contract (evlib.validate uses team13_contracts when importable)
    for it in items.values():
        evlib.validate(it)
    evlib.validate(res)


def test_uplift_fail_misses_the_target(tmp_path):
    def fail_uplift(g):
        g["UPLIFT"].update({"pass": False, "measured": 10.0})

    w = World(tmp_path, gates_patch=fail_uplift)
    assert w.run() == 0
    items, res = w.items(), w.result()
    assert items["UPLIFT"]["passed"] is False and items["UPLIFT"]["extra"]["counted_as"] == "not passed"
    assert sum(it["passed"] for it in items.values()) == 5
    assert res["passed"] is False and res["value"] == pytest.approx(5 / 6)


def test_repeat_null_counts_as_not_passed_with_the_reason(tmp_path):
    def repeat_null(g):
        g["REPEAT"].update({"pass": None, "status": "not_measured", "measured": "not measured"})

    w = World(tmp_path, gates_patch=repeat_null)
    assert w.run() == 0
    rep = w.items()["REPEAT"]
    assert rep["passed"] is False and rep["score"] == 0.0
    assert rep["extra"]["sdk_pass"] is None and "not measured" in rep["extra"]["reason"]
    assert "NOT passed" in rep["extra"]["reason"]
    assert w.result()["passed"] is False


def test_coverage_below_baseline_fails_coverage_only(tmp_path):
    w = World(tmp_path, coverage=0.85)  # baseline fixture: 0.86
    assert w.run() == 0
    items = w.items()
    assert items["COVERAGE"]["passed"] is False and items["COVERAGE"]["extra"]["threshold"] == 0.86
    assert all(items[g]["passed"] for g in je.SDK_GATES)


@pytest.mark.parametrize("which", ["model", "project"])
def test_sha_mismatch_is_refused_and_nothing_is_written(tmp_path, which):
    key = "model_artifact_sha256" if which == "model" else "project_sha256"

    def wrong(d):
        d["twin"]["config"]["eval"][key] = "f" * 64

    w = World(tmp_path, decl_patch=wrong)
    assert w.run() == je.EXIT_REFUSED
    assert w.calls == []  # the SDK never ran
    assert not (tmp_path / "local-ev").exists() or not any((tmp_path / "local-ev").rglob("*"))
    assert w.store.list("outputs/") == [] and w.store.list("records/") == []


def test_a_changed_config_file_is_refused(tmp_path):
    def wrong(d):
        d["test_set_sha256"] = "e" * 64

    w = World(tmp_path, decl_patch=wrong)
    assert w.run() == je.EXIT_REFUSED and w.calls == []


def test_twin_variables_come_only_from_the_environment(tmp_path, capfd):
    w = World(tmp_path)
    assert w.run(env={}) == je.EXIT_REFUSED and w.calls == []
    assert "RAN_RL_TWIN_URL" in capfd.readouterr().out
    assert w.run(env={**ENV, "RAN_RL_TWIN_URL": '{"url": "https://x.example"}'}) == je.EXIT_REFUSED
    assert "JSON" in capfd.readouterr().out
    opts = {o for act in je.parser()._actions for o in act.option_strings}
    assert not any(("twin" in o or "token" in o or "url" in o) for o in opts)


def test_evaluation_failure_writes_nothing(tmp_path):
    w = World(tmp_path)

    def boom(argv):
        raise RuntimeError(f"connect to {URL} failed")

    rc = je.main(w.argv(), store=w.store, environ=dict(ENV), run_eval=boom)
    assert rc == je.EXIT_EVAL
    assert w.store.list("outputs/") == [] and evlib.read_result(RID, w.local()) is None


def test_unsafe_archive_member_is_refused(tmp_path):
    bad = tmp_path / "bad.tgz"
    bad.write_bytes(tgz({"../escape.txt": b"x"}))
    with pytest.raises(je.Refused):
        je.safe_extract(bad, tmp_path / "d", lambda n: True)


def test_masker_hides_url_host_ip_and_token_but_keeps_file_names():
    m = Masker([URL, TOKEN])
    s = m.mask(f"POST {URL}/reset token={TOKEN} via twin-synthetic.example.internal:8443 and 10.0.0.7:80; "
               "wrote evaluation.json config/env.yaml ran_rl.evaluation.evaluate t1.rl.job_evaluate coverage=0.891")
    no_secret_in(s)
    assert "10.0.0.7" not in s
    for keep in ("evaluation.json", "config/env.yaml", "ran_rl.evaluation.evaluate", "coverage=0.891"):
        assert keep in s


def test_fixture_has_the_sdk_shape():
    ev = json.loads((FIX / "evaluation.SYNTHETIC.json").read_text())
    bl = json.loads((FIX / "baseline.SYNTHETIC.json").read_text())
    assert "SYNTHETIC" in ev["_synthetic"] and "SYNTHETIC" in bl["_synthetic"]
    assert set(ev["gates"]) == set(je.SDK_GATES)
    evaluate = pytest.importorskip("ran_rl.evaluation.evaluate")
    assert evaluate.compute_gates(copy.deepcopy(ev), bl, evaluate.GateThresholds(), "cco") == ev["gates"]
