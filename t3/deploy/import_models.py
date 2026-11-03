"""Bring the SageMaker-trained models into the tools image, at a commit, with their provenance.

The tools Lambda serves the policy (ran___propose) and the response model (ran___predict) from files
baked into the image (t3/tools/image/build.sh: `git archive <sha>`), so both files are committed here,
small, next to a provenance JSON that names the S3 object they came from, its sha256 and the training
and evaluation runs. Nothing here touches AWS: the files are downloaded beforehand (read-only), then

    python -m t3.deploy.import_models policy <ppo_50000_steps.zip> \
        --s3-uri s3://.../ppo_50000_steps.zip --train-run t1.rl.train-... --eval-result <eval result.json>
    python -m t3.deploy.import_models predictor <model.safetensors> \
        --s3-uri s3://.../model.safetensors --train-result <t2 result.json> --train-declared <declared.json>

`policy` copies the SB3 zip as t1/rl/artifacts/ppo_sm_50k.zip with a sidecar meta (obs = "sdk": the SDK
trainer's own observation, see t1/rl/policy.py) and t1/rl/artifacts/ppo_sm_50k.provenance.json.

`predictor` converts W11's MLP (silos/nn_core.py, safetensors fc1/fc2/out) to t2.predict's npz without
changing its function: W0 = fc1.weight^T ... and y_std = 1 / LABEL_SCALE, then CHECKS that the t2 network
output equals silos.nn_core.forward_np on random rows (to 1e-6, float rounding). t2.predict
serves the ANCHORED delta net(state, change) - net(state, no change) (zero change predicts exactly 0); the
provenance records how far that is from W11's raw output on the probe rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
POLICY_ZIP = REPO / "t1/rl/artifacts/ppo_sm_50k.zip"
PREDICTOR_NPZ = REPO / "t2/artifacts/t2_response_central_disjoint_v1.npz"
POLICY_META = {"model_id": "t1-ppo-cco-sm", "version": "50k-a09e", "obs": "sdk"}
PREDICTOR_META = {"model_id": "t2-response-central-disjoint", "version": "v1-041620Z"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _load_json(p):
    return json.loads(Path(p).read_text()) if p else None


def import_policy(src: Path, s3_uri: str, train_run: str, eval_result: dict | None, out: Path = POLICY_ZIP) -> dict:
    digest = sha256_file(src)
    if eval_result and eval_result.get("details", {}).get("checkpoint_sha256") not in (None, digest):
        raise SystemExit(f"sha256 {digest} is not the checkpoint the eval run scored "
                         f"({eval_result['details']['checkpoint_sha256']})")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    meta = dict(POLICY_META, label="SageMaker-trained (official SDK trainer, hosted Maveric twin), 50k steps")
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    d = (eval_result or {}).get("details", {})
    prov = {
        "artifact": str(out.relative_to(REPO)), "artifact_sha256": digest, "source_s3_uri": s3_uri,
        "model_id": meta["model_id"], "version": meta["version"], "model_kind": "ppo",
        "observation": "sdk (RANOptimizationEnv observation, no mask); prev_action = 0 in a stateless tool call",
        "training": {"run_id": train_run, "job_name": d.get("training_job_name"), "algorithm": "ppo",
                     "use_case": "cco", "steps": d.get("checkpoint_steps"),
                     "note": "the 100k run crashed at about 67.6k steps; this is its 50k checkpoint (the last "
                             "complete one). The SDK registry step was not run (no run_metadata.json)."},
        "evaluation": None if not eval_result else {
            "run_id": eval_result.get("run_id"), "unit": eval_result.get("unit"), "passed": eval_result.get("passed"),
            "value": eval_result.get("value"), "gates": d.get("items"),
            "model_overall": d.get("model_overall"), "baseline_overall": d.get("baseline_overall"),
            "honest_summary": ("5 of 6 SDK gates pass (SAFETY, SERVICE, COVERAGE, REPEAT, TRACE). UPLIFT FAILS, and "
                               "not narrowly: the scoring objective is {:.2f} for the policy vs {:.2f} for the "
                               "baseline (threshold {:.2f}). The policy buys coverage ({:.3f} vs {:.3f}) and SLA "
                               "({:.3f} vs {:.3f}) with energy ({:.3f} vs {:.3f}).").format(
                d.get("model_overall", {}).get("objective_mean", float("nan")),
                d.get("baseline_overall", {}).get("objective_mean", float("nan")),
                d.get("baseline_overall", {}).get("objective_mean", float("nan")) * d.get("gate_thresholds", {}).get("objective_uplift", float("nan")),
                d.get("model_overall", {}).get("coverage_mean", float("nan")),
                d.get("baseline_overall", {}).get("coverage_mean", float("nan")),
                d.get("model_overall", {}).get("sla_ratio_mean", float("nan")),
                d.get("baseline_overall", {}).get("sla_ratio_mean", float("nan")),
                d.get("model_overall", {}).get("energy_score", float("nan")),
                d.get("baseline_overall", {}).get("energy_score", float("nan")))},
    }
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return prov


def silos_to_t2_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from silos import nn_core as nn

    from t2 import features as tf

    nn.check_params(params)
    # The two encodings are affine per column but use different ranges for d_power_db (t2 [-8, 8],
    # silos [-6, 8]); fold the map e_silos = a * e_t2 + c into the first layer so the function is unchanged.
    a = (tf._HI - tf._LO) / (nn._HI - nn._LO)
    c = a + 2.0 * (tf._LO - nn._LO) / (nn._HI - nn._LO) - 1.0
    w0 = params["fc1.weight"].T.astype(np.float64)           # (38, 128): x_enc @ W0
    W0 = a[:, None] * w0
    b0 = params["fc1.bias"].astype(np.float64) + c @ w0
    return {"W0": W0, "b0": b0,
            "W1": params["fc2.weight"].T.astype(np.float32), "b1": params["fc2.bias"].astype(np.float32),
            "W2": params["out.weight"].T.astype(np.float32), "b2": params["out.bias"].astype(np.float32),
            "y_std": (1.0 / nn.LABEL_SCALE).astype(np.float32)}


def verify_conversion(params: dict[str, np.ndarray], npz: Path, rows: np.ndarray) -> dict:
    """The t2 network (before anchoring) must equal W11's forward on the same raw rows."""
    from silos import nn_core as nn
    from t2 import features as tf
    from t2.predict import load

    assert list(nn.INPUT_COLUMNS) == list(tf.FEATURE_NAMES), "input layouts differ"
    assert [f"d_{x}" for x in tf.LABELS] == list(nn.LABELS), "label orders differ"
    p = load(npz)
    ours = p.residual_scaled(rows) * p._y_std
    theirs = nn.forward_np(params, rows)
    raw_err = float(np.max(np.abs(ours - theirs)))
    served = p.predict_batch(rows)
    return {"rows": int(len(rows)), "max_abs_diff_unanchored_vs_silos": raw_err,
            "max_abs_anchor_shift_per_label": [float(x) for x in np.max(np.abs(served - theirs), axis=0)],
            "equal_to_1e-6": raw_err <= 1e-6}


def probe_rows(n_random: int = 512, seed: int = 0) -> np.ndarray:
    from t2 import features as tf
    rng = np.random.default_rng(seed)
    lo, hi = tf._LO, tf._HI
    return lo + (hi - lo) * rng.random((n_random, len(lo)))


def no_change_bias(params, scenarios=("nominal", "congestion", "interference", "outage"), seeds=range(1, 6),
                   steps: int = 3) -> dict:
    """W11's raw output for a ZERO change on realistic in-process twin states: exactly what anchoring removes."""
    try:
        from silos import nn_core as nn
        from t2.features import state_change_to_row
        from twin.replay import ReplayTwin
    except Exception as exc:  # noqa: BLE001 - the twin needs the SDK; say so instead of guessing
        return {"measured": False, "why": f"{type(exc).__name__}: {exc}"}
    rows = []
    for sc in scenarios:
        for sd in seeds:
            t = ReplayTwin(sc, int(sd), None)
            s = t.reset()
            for _ in range(steps):
                rows.append(state_change_to_row(s, []))
                s = t.apply([])
    out = nn.forward_np(params, np.asarray(rows))
    return {"measured": True, "rows": len(rows), "labels": list(nn.LABELS),
            "mean_abs": [float(x) for x in np.mean(np.abs(out), axis=0)],
            "max_abs": [float(x) for x in np.max(np.abs(out), axis=0)],
            "score_scale": [float(x) for x in nn.SCORE_SCALE]}


def import_predictor(src: Path, s3_uri: str, train_result: dict | None, train_declared: dict | None,
                     out: Path = PREDICTOR_NPZ) -> dict:
    from silos import nn_core as nn

    digest = sha256_file(src)
    if train_result and train_result.get("details", {}).get("artifact_sha256") not in (None, digest):
        raise SystemExit(f"sha256 {digest} is not the artifact the training run recorded")
    params = nn.safetensors_loads(Path(src).read_bytes())
    t2p = silos_to_t2_params(params)
    meta = dict(PREDICTOR_META, label="SageMaker-trained central reference (pooled simulated silos), MLP 38-128-128-4",
                contract_version="0.2", source_sha256=digest, converted_from="silos/nn_core.py safetensors",
                anchoring="t2.predict serves net(x) - net(x with no change): a zero change predicts exactly 0")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, meta_json=np.array(json.dumps(meta, sort_keys=True)), **t2p)
    check = verify_conversion(params, out, probe_rows())
    if not check["equal_to_1e-6"]:
        out.unlink()
        raise SystemExit(f"conversion changed the function: {check}")
    check["no_change_output_on_twin_states"] = no_change_bias(params)
    d = (train_result or {}).get("details", {})
    prov = {"artifact": str(out.relative_to(REPO)), "artifact_sha256": sha256_file(out), "source_s3_uri": s3_uri,
            "source_sha256": digest, "model_id": meta["model_id"], "version": meta["version"],
            "model_kind": "t2-response", "conversion_check": check,
            "training": {"run_id": (train_result or {}).get("run_id"),
                         "job_name": ((train_declared or {}).get("twin", {}).get("config", {}).get("runner", {})
                                      .get("job_name")),
                         "code_sha": (train_result or {}).get("checker", {}).get("code_sha"),
                         "method": d.get("method"), "split": d.get("split"), "epochs": d.get("epochs"),
                         "population": "three simulated operators on the in-process Maveric twin, one account; "
                                       "pooled (central reference, not private, not federated)"},
            "evaluation": None if not train_result else {
                "shared_test_rows": train_result.get("n_items"), "mean_score": d.get("mean_score"),
                "mean_score_noop": d.get("mean_score_noop"), "mae": d.get("mae"),
                "direction_accuracy": d.get("direction_accuracy"),
                "honest_summary": "a measurement, no pass bar: scaled MAE {:.3f} vs {:.3f} for predicting no effect; "
                                  "weakest on the viavi-like slices".format(d.get("mean_score", float("nan")),
                                                                           d.get("mean_score_noop", float("nan")))}}
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("policy")
    p.add_argument("src", type=Path)
    p.add_argument("--s3-uri", required=True)
    p.add_argument("--train-run", required=True)
    p.add_argument("--eval-result")
    q = sub.add_parser("predictor")
    q.add_argument("src", type=Path)
    q.add_argument("--s3-uri", required=True)
    q.add_argument("--train-result")
    q.add_argument("--train-declared")
    a = ap.parse_args(argv)
    if a.cmd == "policy":
        prov = import_policy(a.src, a.s3_uri, a.train_run, _load_json(a.eval_result))
    else:
        prov = import_predictor(a.src, a.s3_uri, _load_json(a.train_result), _load_json(a.train_declared))
    print(json.dumps(prov, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
