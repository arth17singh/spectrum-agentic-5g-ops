"""Track 1 RL scoring job: the SDK's evaluation of our trained PPO policy, as six item records and a result.

Run by tools/sm/entry.py in a SageMaker CPU job started by tools/sm/run.py (see t1/rl/eval_declaration.py
for the declaration and the exact launch command):

    -m t1.rl.job_evaluate --run-id {run_id} --evidence-s3 {evidence_s3} --evidence-dir {evidence_dir}
        --output-dir {output_dir} --model-channel {channel:model} --project-channel {channel:project}

In order (nothing is written before step 6; any refusal exits non-zero, so the runner uploads nothing):
  1. read the declaration the launcher stored in S3 before the job (declared/<run_id>.json); its
     twin.config.eval names the model artifact and the packed project with their sha256;
  2. check the twin variables RAN_RL_TWIN_URL and RAN_RL_ADAPTER_TOKEN are set (the runner fetched them
     from Secrets Manager into this process; never from arguments or files, never printed);
  3. sha256 of model.tar.gz and of the packed project (sourcedir.tar.gz) against the declaration;
  4. extract model.zip from model.tar.gz, and only config/ and team/ from the project; their config files'
     sha256 must equal the declared ones (eval_scenarios.yaml is the declared test set); refuse if any
     `.env` is reachable (ran_rl loads the nearest .env on import and it would override the secrets);
  5. run `ran_rl.evaluation.evaluate --model model.zip --with-baseline ...` in this process, with the
     httpx loggers at WARNING and stdout/stderr masked (the SDK would log the twin host per call).
     `--model` is the bare model.zip on purpose: its checksum is sha256(model.zip), which is what the
     SDK registry later requires to equal the package manifest's model_sha256;
  6. write evaluation.json, baseline.json and evaluation.md to {output_dir}/eval/ (SageMaker uploads them
     with the job's output) and to the evidence store at outputs/<run_id>/eval/ (never overwritten), and
     print each one's sha256 as a `team13-evidence` line;
  7. six item records (SAFETY, SERVICE, UPLIFT, REPEAT, TRACE from the SDK's `gates`; COVERAGE ours:
     model aggregates.overall.coverage_mean >= the baseline's, no tolerance) and one result (pass_rate
     against the declared target) into the local evidence store {evidence_dir}; the runner uploads and
     attests them. A gate with `pass: null` (REPEAT not measured) counts as NOT passed, with the reason.

Exit codes: 0 done (whatever the verdict); 7 refused (inputs, declaration or environment); 8 the
evaluation itself failed (twin unavailable, SDK error): no records and no result are written.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "contracts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from t1.rl.mask import Masker, MaskingStream  # noqa: E402
from tools import evlib  # noqa: E402

SCRIPT = "t1/rl/job_evaluate.py"
SDK_GATES = ("SAFETY", "SERVICE", "UPLIFT", "REPEAT", "TRACE")
ITEM_IDS = SDK_GATES + ("COVERAGE",)
TWIN_VARS = ("RAN_RL_TWIN_URL", "RAN_RL_ADAPTER_TOKEN")
MODEL_FILES = ("model.zip", "metrics.json", "run_metadata.json", "effective_config.json", "reward_plugin.py")
PROJECT_DIRS = ("config", "team")
# project file -> the twin.config key holding its declared sha256 (eval_scenarios.yaml: test_set_sha256)
CONFIG_SHAS = {"config/env.yaml": "env_config_sha256", "config/reward.yaml": "reward_config_sha256",
               "team/rewards.py": "team_rewards_sha256"}
ARTIFACTS = ("evaluation.json", "baseline.json", "evaluation.md")
ATTEST_MARK = "team13-evidence"
EXIT_REFUSED, EXIT_EVAL = 7, 8
GATE_GRADER = "ran_rl.evaluation.evaluate.compute_gates (RAN RL SDK {v})"
COVERAGE_GRADER = "t1/rl/job_evaluate.py:coverage_vs_baseline (model >= baseline, no tolerance)"


class Refused(RuntimeError):
    pass


def say(msg: str) -> None:
    print(msg, flush=True)


def canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(obj: Any) -> str:
    return hashlib.sha256(canon(obj)).hexdigest()


# --------------------------------------------------------------------------- the declaration

def eval_block(decl: dict) -> dict:
    """twin.config.eval of the stored declaration, checked for what this job needs."""
    if decl.get("unit") != "t1.rl.eval":
        raise Refused(f"the declaration's unit is {decl.get('unit')!r}, not 't1.rl.eval'")
    t = decl.get("target") or {}
    if t.get("metric") != "pass_rate":
        raise Refused(f"the declared target metric is {t.get('metric')!r}; this job writes pass_rate")
    if decl.get("n_items") != len(ITEM_IDS):
        raise Refused(f"the declaration has n_items={decl.get('n_items')!r}; this job writes {len(ITEM_IDS)} items")
    cfg = ((decl.get("twin") or {}).get("config") or {})
    ev = cfg.get("eval")
    if not isinstance(ev, dict):
        raise Refused("the declaration has no twin.config.eval block (fill it with t1/rl/eval_declaration.py)")
    for k in ("training_run_id", "model_artifact_sha256", "project_sha256"):
        if not ev.get(k):
            raise Refused(f"twin.config.eval.{k} is missing from the declaration")
    items = cfg.get("items") or {}
    if sorted(items) != sorted(ITEM_IDS):
        raise Refused(f"twin.config.items names {sorted(items)}, this job writes {sorted(ITEM_IDS)}")
    return ev


# --------------------------------------------------------------------------- inputs

def channel_file(channel: Path, name: str) -> Path:
    """The named file in an input channel; else the channel's only file."""
    p = channel / name
    if p.is_file():
        return p
    files = [f for f in channel.rglob("*") if f.is_file()] if channel.is_dir() else []
    if len(files) == 1:
        return files[0]
    raise Refused(f"input channel {channel} holds {len(files)} files and no {name}")


def safe_extract(archive: Path, dest: Path, keep: Callable[[str], bool]) -> list[str]:
    """Extract regular files whose normalised path `keep` accepts. Links, devices, absolute paths and '..'
    are refused outright (the archives are inputs, so they are untrusted)."""
    out = []
    with tarfile.open(archive, "r:*") as tar:
        for m in tar.getmembers():
            name = m.name
            while name.startswith("./"):
                name = name[2:]
            pp = PurePosixPath(name)
            if pp.is_absolute() or ".." in pp.parts:
                raise Refused(f"{archive.name}: unsafe member path {m.name!r}")
            if m.issym() or m.islnk() or m.isdev() or m.isfifo():
                if keep(name):
                    raise Refused(f"{archive.name}: {m.name!r} is a link or device")
                continue
            if not m.isfile() or not keep(name):
                continue
            target = dest / pp
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            with src, open(target, "wb") as f:
                f.write(src.read())
            out.append(name)
    return sorted(out)


def keep_model(name: str) -> bool:
    return name in MODEL_FILES


def keep_project(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (len(parts) >= 2 and parts[0] in PROJECT_DIRS and "__pycache__" not in parts
            and not name.endswith(".pyc") and parts[-1] != ".env")


def dotenv_reachable(start: Path) -> Path | None:
    """python-dotenv's find_dotenv(usecwd=True) walks from the cwd up to the root."""
    for d in (start, *start.parents):
        if (d / ".env").exists():
            return d / ".env"
    return None


def twin_env(environ) -> Masker:
    missing = [k for k in TWIN_VARS if not (environ.get(k) or "").strip()]
    if missing:
        raise Refused(f"{', '.join(missing)} not set: launch with --secret NAME=<Secrets Manager ARN> for "
                      f"{' and '.join(TWIN_VARS)} (never --env, never a file)")
    url = environ["RAN_RL_TWIN_URL"].strip()
    if not url.lower().startswith(("http://", "https://")):
        raise Refused("RAN_RL_TWIN_URL is not an http(s) URL (was the secret stored as a JSON object instead of "
                      "the bare URL?); its value is not printed")
    return Masker([environ.get(k) for k in TWIN_VARS])


# --------------------------------------------------------------------------- the SDK

def sdk_evaluate(argv: list[str]) -> int:
    """`python -m ran_rl.evaluation.evaluate <argv>`, in this process (cwd = the project)."""
    import logging

    from ran_rl.evaluation import evaluate  # noqa: PLC0415 - slow (torch); after the checks
    from ran_rl.twin.base import TwinUnavailable  # noqa: PLC0415

    for name in ("httpx", "httpcore"):  # INFO would log every twin call with its URL
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        return evaluate.main(argv)
    except TwinUnavailable as e:
        say(f"evaluation failed: twin unavailable ({e.__class__.__name__}: {e})")
        return EXIT_EVAL


def sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("ran-rl")
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- items and result

def build_items(decl: dict, evaluation: dict, baseline: dict, artifact_shas: dict[str, str],
                backend: str, sdk_v: str) -> list[dict]:
    cfg = decl["twin"]["config"]
    requirements = cfg["items"]
    gates = evaluation.get("gates") or {}
    model_ck = (evaluation.get("model") or {}).get("checksum")
    common = {"evaluation_sha256": artifact_shas.get("evaluation.json"),
              "baseline_sha256": artifact_shas.get("baseline.json"),
              "report_sha256": artifact_shas.get("evaluation.md"),
              "model_checksum": model_ck, "baseline_checksum": (baseline.get("model") or {}).get("checksum"),
              "config_hash": evaluation.get("config_hash"),
              "scenario_set_version": evaluation.get("scenario_set_version"),
              "use_case": evaluation.get("use_case"), "objective_version": evaluation.get("objective_version")}
    out = []
    for item_id in ITEM_IDS:
        inp = {"item_id": item_id, "requirement": requirements[item_id], "test_set_sha256": decl["test_set_sha256"],
               "model_checksum": model_ck, "config_hash": evaluation.get("config_hash")}
        extra: dict[str, Any] = dict(common)
        if item_id == "COVERAGE":
            m = ((evaluation.get("aggregates") or {}).get("overall") or {}).get("coverage_mean")
            b = ((baseline.get("aggregates") or {}).get("overall") or {}).get("coverage_mean")
            ok = isinstance(m, (int, float)) and isinstance(b, (int, float))
            passed = bool(ok and m >= b)
            output = {"measured": m, "threshold": b, "op": ">=", "pass": passed if ok else None}
            extra.update(measured=m, threshold=b, op=">=", source="aggregates.overall.coverage_mean (model vs baseline)",
                         counted_as="passed" if passed else "not passed")
            if not ok:
                extra["reason"] = "coverage_mean missing from evaluation.json or baseline.json: counted as not passed"
            grader = COVERAGE_GRADER
        else:
            g = gates.get(item_id)
            grader = GATE_GRADER.format(v=sdk_v)
            if not isinstance(g, dict):
                passed, output = False, {"gate": None}
                extra.update(gate=None, counted_as="not passed",
                             reason=f"evaluation.json has no {item_id} gate: counted as not passed")
            else:
                passed = g.get("pass") is True
                output = {"gate": g}
                extra.update(gate=g, sdk_pass=g.get("pass"), sdk_status=g.get("status"), measured=g.get("measured"),
                             threshold=g.get("threshold"), counted_as="passed" if passed else "not passed")
                if g.get("pass") is None:
                    extra["reason"] = (f"the SDK reports {item_id} pass: null (status {g.get('status')!r}): not "
                                       "measured, so counted as NOT passed, as the declaration states")
        out.append({"item_id": item_id, "input_sha256": sha(inp), "output_sha256": sha(output),
                    "score": 1.0 if passed else 0.0, "passed": passed, "grader": grader, "backend": backend,
                    "extra": extra, "synthetic": bool(decl.get("synthetic", False))})
    return out


def overall(doc: dict) -> dict:
    o = (doc.get("aggregates") or {}).get("overall") or {}
    return {k: o.get(k) for k in ("episodes", "objective_mean", "reward_mean", "sla_ratio_mean", "coverage_mean",
                                  "energy_score", "spectral_efficiency", "safety_violations") if k in o}


# --------------------------------------------------------------------------- main

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m t1.rl.job_evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--evidence-s3", default=os.environ.get("EVIDENCE_S3"), help="where the declaration is stored")
    ap.add_argument("--evidence-dir", required=True, help="local evidence store; the runner uploads it")
    ap.add_argument("--output-dir", required=True, help="job output dir (SageMaker uploads it)")
    ap.add_argument("--model-channel", required=True)
    ap.add_argument("--project-channel", required=True)
    ap.add_argument("--work-dir", help="scratch (default: a new temporary directory)")
    return ap


def main(argv: list[str] | None = None, *, store=None, environ=None,
         run_eval: Callable[[list[str]], int] = sdk_evaluate) -> int:
    a = parser().parse_args(argv)
    environ = os.environ if environ is None else environ
    try:
        return _main(a, store, environ, run_eval)
    except Refused as e:
        say(f"t1.rl.job_evaluate refused: {e}")
        return EXIT_REFUSED


def _main(a, store, environ, run_eval) -> int:
    rid = a.run_id
    # 1. the declaration, as stored before the job
    if store is None:
        if not a.evidence_s3:
            raise Refused("no --evidence-s3 (or EVIDENCE_S3): the declaration cannot be read")
        store = evlib.S3Store(a.evidence_s3)
    decl = evlib.read_declared(rid, store)
    if decl is None:
        raise Refused(f"no declaration at {store.uri(f'declared/{rid}.json')}")
    ev = eval_block(decl)
    cfg = decl["twin"]["config"]
    # 2. the twin variables (from Secrets Manager via the runner)
    masker = twin_env(environ)
    say("twin variables set: " + ", ".join(TWIN_VARS) + " (values not printed)")
    # 3. inputs against the declaration
    model_tgz = channel_file(Path(a.model_channel), "model.tar.gz")
    project_tgz = channel_file(Path(a.project_channel), "sourcedir.tar.gz")
    for what, path, want in (("model artifact", model_tgz, ev["model_artifact_sha256"]),
                             ("packed project", project_tgz, ev["project_sha256"])):
        got = evlib.sha256_file(path)
        if got != want:
            raise Refused(f"{what} {path.name} sha256 {got} is not the declared {want}; nothing evaluated")
        say(f"{what} {path.name} sha256 {got} (declared)")
    # 4. extract
    work = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="t1-rl-eval-"))
    work.mkdir(parents=True, exist_ok=True)
    mdir, pdir, odir = work / "model", work / "project", work / "out"
    got_model = safe_extract(model_tgz, mdir, keep_model)
    if "model.zip" not in got_model:
        raise Refused(f"model.tar.gz has no model.zip (members kept: {got_model})")
    got_project = safe_extract(project_tgz, pdir, keep_project)
    checks = {"config/eval_scenarios.yaml": decl["test_set_sha256"]}
    checks.update({p: cfg.get(k) for p, k in CONFIG_SHAS.items() if cfg.get(k)})
    for rel, want in checks.items():
        f = pdir / rel
        if not f.is_file():
            raise Refused(f"the packed project has no {rel}")
        got = evlib.sha256_file(f)
        if got != want:
            raise Refused(f"{rel} sha256 {got} is not the declared {want}")
    say(f"project: {len(got_project)} files from config/ and team/; "
        f"{', '.join(sorted(checks))} match the declaration")
    env_file = dotenv_reachable(pdir)
    if env_file is not None:
        raise Refused(f"a .env is reachable from the project dir ({env_file}); ran_rl would load it over the secrets")
    model_zip = mdir / "model.zip"
    model_sha = evlib.sha256_file(model_zip)
    say(f"model.zip sha256 {model_sha}")
    # 5. evaluate
    odir.mkdir(parents=True, exist_ok=True)
    sdk_argv = ["--model", str(model_zip), "--with-baseline", "--scenario-set", "config/eval_scenarios.yaml",
                "--env-config", "config/env.yaml", "--reward-config", "config/reward.yaml",
                "--output", str(odir / "evaluation.json"), "--baseline-output", str(odir / "baseline.json")]
    say("evaluate: python -m ran_rl.evaluation.evaluate " + " ".join(sdk_argv))
    rc = _run_masked(run_eval, sdk_argv, pdir, masker)
    if rc != 0:
        say(f"evaluation failed (exit {rc}); no records and no result written")
        return EXIT_EVAL
    files = {n: odir / n for n in ARTIFACTS}
    missing = [n for n, p in files.items() if not p.is_file()]
    if missing:
        say(f"evaluation wrote no {', '.join(missing)}; no records and no result written")
        return EXIT_EVAL
    raw = {n: p.read_bytes() for n, p in files.items()}
    evaluation, baseline = json.loads(raw["evaluation.json"]), json.loads(raw["baseline.json"])
    ck = (evaluation.get("model") or {}).get("checksum")
    if ck != model_sha:
        say(f"evaluation.json model.checksum {ck} is not sha256(model.zip) {model_sha}; no result written")
        return EXIT_EVAL
    # 6. artifacts: job output, evidence store, attested hashes
    shas = {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()}
    out_eval = Path(a.output_dir) / "eval"
    out_eval.mkdir(parents=True, exist_ok=True)
    uris: dict[str, str | None] = {}
    upload_errors: dict[str, str] = {}
    for n, b in raw.items():
        (out_eval / n).write_bytes(b)
        rel = f"outputs/{rid}/eval/{n}"
        try:
            uris[n] = store.put_new(rel, b)
        except FileExistsError:
            raise
        except Exception as e:  # noqa: BLE001 - the job output still carries the file; say so, do not fail
            code = ((getattr(e, "response", None) or {}).get("Error") or {}).get("Code") or e.__class__.__name__
            uris[n], upload_errors[n] = None, str(code)
            say(f"warning: {rel} not stored in the evidence store ({code}); it is in the job output (eval/{n})")
            continue
        say(f"{ATTEST_MARK} " + json.dumps({"run_id": rid, "path": rel, "sha256": shas[n], "bytes": len(b),
                                            "store": store.uri("")}, sort_keys=True))
    # 7. items and result, into the local evidence store
    backend = f"{decl['twin']['kind']} (http twin), ran-rl {sdk_version()}"
    items = build_items(decl, evaluation, baseline, shas, backend, sdk_version())
    local = evlib.LocalStore(a.evidence_dir)
    evlib.append_records(rid, "item", items, store=local)
    gates = evaluation.get("gates") or {}
    details = {
        "items": {it["item_id"]: it["passed"] for it in items},
        "sdk_gates": {g: (gates.get(g) or {}).get("pass") for g in SDK_GATES},
        "repeat_rule": "a gate with pass: null (not measured) counts as not passed",
        "training_run_id": ev["training_run_id"], "training_job_name": ev.get("training_job_name"),
        "model_artifact_uri": ev.get("model_artifact_uri"), "model_artifact_sha256": ev["model_artifact_sha256"],
        "project_sha256": ev["project_sha256"], "model_zip_sha256": model_sha,
        "artifacts": {n: {"uri": uris[n], "sha256": shas[n], "job_output_member": f"eval/{n}",
                          **({"upload_error": upload_errors[n]} if n in upload_errors else {})} for n in ARTIFACTS},
        "job_output": "output.tar.gz of training job {} (SageMaker OutputDataConfig)".format(
            os.environ.get("TRAINING_JOB_NAME") or "<this job>"),
        "model_overall": overall(evaluation), "baseline_overall": overall(baseline),
        "gate_thresholds": evaluation.get("gate_thresholds"), "config_hash": evaluation.get("config_hash"),
    }
    result = evlib.make_result(decl, items, SCRIPT, evlib.code_sha(), details=details)
    evlib.write_result(result, local)
    for it in items:
        say(f"item {it['item_id']:<8} {'PASS' if it['passed'] else 'FAIL'}"
            + (f"  ({it['extra']['reason']})" if it["extra"].get("reason") else ""))
    say(f"result {result['metric']} = {result['value']:.4f} {result['target']['op']} {result['target']['value']} -> "
        f"{'PASSED' if result['passed'] else 'MISSED'}")
    return 0


def _run_masked(run_eval, argv: list[str], cwd: Path, masker: Masker) -> int:
    """Run the SDK with cwd = the project (and the project first on sys.path, for team.*), stdout and
    stderr masked."""
    old_cwd, old_path = Path.cwd(), list(sys.path)
    out, err = MaskingStream(sys.stdout, masker), MaskingStream(sys.stderr, masker)
    os.chdir(cwd)
    sys.path.insert(0, str(cwd))
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                return int(run_eval(argv) or 0)
            except SystemExit as e:
                return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            except Exception as e:  # noqa: BLE001 - the SDK failed: masked message, non-zero exit
                print(f"evaluation raised {e.__class__.__name__}: {e}", file=sys.stderr)
                return EXIT_EVAL
    finally:
        out.flush()
        err.flush()
        sys.path[:] = old_path
        os.chdir(old_cwd)


if __name__ == "__main__":
    sys.exit(main())
