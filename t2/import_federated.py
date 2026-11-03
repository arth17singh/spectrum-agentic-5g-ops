"""Bring the federated aggregate (the FL service's final model) into the tools image, with its provenance.

    python -m t2.import_federated <federated.safetensors> --s3-uri s3://.../federated.safetensors \
        --session-id <id> --rounds 20 --final-iteration 21 --evaluate-result <result.json> \
        --baseline-result local_A=<result.json> ... [--comparison name=<run id> ...]

Reuses the converter of t3/deploy/import_models.py (silos MLP -> t2.predict npz; the d_power_db input-range
difference is folded into the first layer, so the function is unchanged; the check against silos.nn_core.forward_np
runs on 512 random rows; t2.predict serves the anchored delta, a zero change predicts exactly 0).

Two files are written under t2/artifacts: t2_response_federated_v1.npz (served) and its provenance JSON, plus
t2_response_federated_v1.source.npz, the six tensors of the aggregate (the service's safetensors header carries metadata that
names the service and a membership id; only the tensors matter). `weights_canonical_sha256` is the
sha256 of exactly those tensor bytes, so anyone holding the published file (sha256 in `published_file_sha256`) can tie it to the
committed tensors. The service's own file name is withheld for the same reason.

The account id in the S3 URI is replaced by <account> (committed files never carry it). Nothing here touches AWS: the published file and the result JSONs are downloaded beforehand (read-only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from silos import nn_core as nn  # noqa: E402
from t3.deploy.import_models import no_change_bias, probe_rows, silos_to_t2_params, verify_conversion  # noqa: E402

ART = REPO / "t2" / "artifacts"
NPZ = ART / "t2_response_federated_v1.npz"
SOURCE = ART / "t2_response_federated_v1.source.npz"
PROV = ART / "t2_response_federated_v1.provenance.json"
META = {"model_id": "t2-response-federated-disjoint", "version": "v1-5fb29b2d"}
PLACEHOLDER = "<fill after the compare jobs finish>"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def summary(result: dict | None) -> dict | None:
    if not result:
        return None
    d = result.get("details", {})
    return {"run_id": result.get("run_id"), "n_items": result.get("n_items"), "mean_score": d.get("mean_score"),
            "mean_score_noop": d.get("mean_score_noop"), "mae": d.get("mae"), "direction_accuracy": d.get("direction_accuracy"),
            "per_segment": {k: v.get("mean_score") for k, v in (d.get("per_segment") or {}).items()}}


def import_federated(src: Path, s3_uri: str, session_id: str, rounds: int, final_iteration: int, evaluate_result: dict | None,
                     baselines: dict[str, dict], comparisons: dict[str, str], out_dir: Path = ART) -> dict:
    raw = Path(src).read_bytes()
    published_sha = sha256_bytes(raw)
    params = nn.safetensors_loads(raw)
    nn.check_params(params)
    ed = (evaluate_result or {}).get("details", {})
    if ed.get("artifact_sha256") not in (None, published_sha):
        raise SystemExit(f"sha256 {published_sha} is not the file the evaluate run scored ({ed.get('artifact_sha256')})")
    canonical = nn.safetensors_dumps(params)  # no header metadata
    out_dir.mkdir(parents=True, exist_ok=True)
    npz, source, prov_path = out_dir / NPZ.name, out_dir / SOURCE.name, out_dir / PROV.name
    np.savez(source, **{k: np.asarray(v, dtype=np.float32) for k, v in params.items()})  # the same six tensors
    t2p = silos_to_t2_params(params)
    meta = dict(META, label="FL-service federated aggregate of three simulated silos (secure aggregation, no DP), MLP 38-128-128-4",
                contract_version="0.2", source_sha256=sha256_bytes(canonical), converted_from="silos/nn_core.py safetensors",
                anchoring="t2.predict serves net(x) - net(x with no change): a zero change predicts exactly 0")
    np.savez(npz, meta_json=np.array(json.dumps(meta, sort_keys=True)), **t2p)
    check = verify_conversion(params, npz, probe_rows())
    if not check["equal_to_1e-6"]:
        npz.unlink()
        raise SystemExit(f"conversion changed the function: {check}")
    check["no_change_output_on_twin_states"] = no_change_bias(params)
    prov = {
        "artifact": str(npz.relative_to(REPO)), "artifact_sha256": sha256_bytes(npz.read_bytes()),
        "source_artifact": str(source.relative_to(REPO)), "weights_canonical_sha256": sha256_bytes(canonical),
        "weights_canonical_note": "sha256 of nn_core.safetensors_dumps(tensors) with no header metadata; t2_response_federated_v1.source.npz holds the same six float32 tensors",
        "published_file_sha256": published_sha, "published_s3_uri": re.sub(r"\d{12}", "<account>", s3_uri),
        "model_id": meta["model_id"], "version": meta["version"], "model_kind": "t2-response", "conversion_check": check,
        "federation": {
            "session_id": session_id, "rounds": rounds, "final_iteration": final_iteration,
            "iteration_numbering": f"iteration 1 is the initial weights, iterations 2..{final_iteration} are the {rounds} trained rounds",
            "protocol": "NN_HFL (secure aggregation, no differential privacy)", "split": "disjoint",
            "clients": "3 simulated silos (A: large, moving users, 23-30 dBm; B: mid, 30-37 dBm; C: small, stationary, 43-46 dBm), one account",
            "final_file": "iteration-%d aggregate, one of three identical client copies (the service's file name is withheld: it names the service)" % final_iteration,
            "population": "simulated silos on the in-process Maveric twin; not partner data"},
        "evaluation": summary(evaluate_result),
        "baselines_same_shared_test": {k: summary(v) for k, v in baselines.items()},
        "comparisons": {k: (v or PLACEHOLDER) for k, v in comparisons.items()},
        "honest_summary": None,
    }
    if evaluate_result and baselines:
        sc = {k: v["mean_score"] for k, v in prov["baselines_same_shared_test"].items() if v}
        fed = prov["evaluation"]["mean_score"]
        prov["honest_summary"] = ("a measurement, no pass bar (lower scaled MAE is better): federated {:.3f} vs {:.3f} for predicting no effect; "
                                  "{}").format(fed, prov["evaluation"]["mean_score_noop"],
                                               "; ".join(f"{k} {v:.3f}" for k, v in sorted(sc.items())))
    prov_path.write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("--s3-uri", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--final-iteration", type=int, required=True)
    ap.add_argument("--evaluate-result")
    ap.add_argument("--baseline-result", action="append", default=[], help="name=path of a result JSON")
    ap.add_argument("--comparison", action="append", default=[], help="name=run id (omit to leave a placeholder)")
    a = ap.parse_args(argv)
    load = lambda p: json.loads(Path(p).read_text()) if p else None
    baselines = {kv.split("=", 1)[0]: load(kv.split("=", 1)[1]) for kv in a.baseline_result}
    comps = {k: PLACEHOLDER for k in ("federated_vs_B", "federated_vs_C", "distance_to_central", "federated_vs_localfedavg")}
    comps.update({kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in a.comparison})
    prov = import_federated(a.src, a.s3_uri, a.session_id, a.rounds, a.final_iteration, load(a.evaluate_result), baselines, comps)
    print(json.dumps(prov, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
