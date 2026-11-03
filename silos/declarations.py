"""Declaration templates for the Track 2 units, written BEFORE any run.

    python -m silos.declarations            # regenerates silos/declarations/*.template.json

A template is the declaration document with placeholders for what only exists at run time (<RUN_ID>, <CODE_SHA>, the
Region) and for the hash of the AWS-generated shared test set (the committed manifest is the laptop copy; the run
declares the hash of the data tree it actually scores). The job runner fills them and writes declared/<run_id>.json
through tools/evlib before the run. `python -m silos.job_<x> --print-declaration` prints the same document with real values.

The units (all on the ONE shared test set of a split, scored by the per-row scaled MAE in silos/nn_core.py; "simulated
silos": the three silos run in one account, so say so wherever these numbers are shown):

  (a) local-only      t2.local_only.<A|B|C>.<split>   MEASUREMENT: per-silo MAE on the shared test set. No pass bar.
  (b) federated       t2.compare.federated_vs_B.<split> and ...federated_vs_C...:  the federated model's mean score is
                      better than each of the two smaller silos' local-only models, AND the lower bound of a 95 %
                      paired bootstrap interval (clusters resampled) of the mean difference is > 0. This can fail.
  (c) differential    t2.fl_dp.<split>: the DP model scored on the same set, with epsilon stated; t2.compare.dp_cost:
      privacy         MAE relative to (b), a report.
  (d) central         t2.central.<split> scored; t2.compare.distance_to_central: how far the federated model is from the
                      central reference, a report. (The organisers' rule that a central baseline counts only when they
                      publish one is still open; we train it anyway.)

Epsilon: planned 3.0, set by `privacy_budget()` in the model-code class (the service's privacy-budget method: Opacus
per-sample clipping with a Renyi-DP accountant, noise added to each silo's update, i.e. record-level DP-SGD per silo).
Delta is NOT ours to choose: the framework fixes delta = (training-set size)^-2 per silo (A 24,000 rows: 1.7e-9; B 9,000:
1.2e-8; C 3,000: 1.1e-7) and searches for the noise scale that meets epsilon; it fails with an error if none does. A
planning value of delta = 1e-5 is therefore not what the service will report.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import job_baselines as jb
from . import job_compare as jc
from . import job_evaluate as je
from . import job_data
from . import scorerun as sr
from .generate import DEFAULT_CONFIG, load_config

DIR = Path(__file__).resolve().parent / "declarations"
PH = {"run_id": "<RUN_ID>", "code_sha": "<CODE_SHA: 40-hex commit the job was packaged from>", "region": "<AWS_REGION>"}
TEST_PH = "<sha256 of the shared test file in the AWS-generated data tree (manifest.json of silos.job_data)>"


def _fix(d: dict) -> dict:
    d["region"] = PH["region"]
    d["test_set_sha256"] = TEST_PH
    return d


def build_all() -> dict[str, dict]:
    m = sr.committed_manifest()
    cfg = load_config(DEFAULT_CONFIG)
    out: dict[str, dict] = {}
    d = job_data.declaration_for(cfg, PH["run_id"], PH["code_sha"])
    d["region"] = PH["region"]
    out["data"] = d
    for split in sr.SPLITS:
        for s in sr.SILOS:
            out[f"local_only_{s}_{split}"] = _fix(jb.declaration_for("local", s, split, PH["run_id"], PH["code_sha"], m, 20, 0))
        out[f"central_{split}"] = _fix(jb.declaration_for("central", None, split, PH["run_id"], PH["code_sha"], m, 20, 0))
        out[f"fedavg_local_{split}"] = _fix(jb.declaration_for("fedavg", None, split, PH["run_id"], PH["code_sha"], m, 20, 0))
    split = "disjoint"
    for label, eps in (("federated", None), ("dp", 3.0)):
        dd = sr.score_declaration(run_id=PH["run_id"], unit=f"{je.UNIT_OF[label]}.{split}", model_id=f"t2-response-fl-{label}-{split}@v1",
                                  n_rows=sr.test_rows(m, split), test_sha=TEST_PH, code_sha=PH["code_sha"], note=je.NOTE[label],
                                  extra={"split": split, "session_id": "<FL session id>", "final_aggregation_file": "<highest _aggregation_ file name>",
                                         "epsilon": eps, "entry": f"python -m silos.job_evaluate --label {label} --split {split}"})
        dd["region"] = PH["region"]
        if label == "dp":
            dd["privacy"] = {"epsilon": 3.0, "delta": "(training-set size)^-2 per silo, fixed by the FL service's framework",
                             "delta_by_silo": {"A": 1 / 24000**2, "B": 1 / 9000**2, "C": 1 / 3000**2},
                             "source": "privacy_budget() override in the model-code class; Opacus per-sample clipping with a Renyi-DP accountant, noise on each silo's update (record-level DP-SGD per silo)",
                             "planning_value_not_used": "delta = 1e-5 (the service does not take a delta)"}
        out[f"fl_{label}_{split}"] = dd
    fake = {"test_set_sha256": TEST_PH}
    for name, kind, xk, yk in (("federated_vs_B", "beats", "fl_federated", "local_B"), ("federated_vs_C", "beats", "fl_federated", "local_C"),
                               ("dp_cost", "measure", "fl_federated", "fl_dp"), ("distance_to_central", "measure", "central", "fl_federated")):
        ids = {"fl_federated": "t2-response-fl-federated-disjoint@v1", "fl_dp": "t2-response-fl-dp-disjoint@v1", "central": "t2-response-central-disjoint@v1",
               "local_B": "t2-response-local-b-disjoint@v1", "local_C": "t2-response-local-c-disjoint@v1"}
        dd = jc.declaration_for(kind, name, split, PH["run_id"], PH["code_sha"], {**fake, "model_id": ids[xk]}, {**fake, "model_id": ids[yk]},
                                sr.test_rows(m, split), f"<run id of {xk}>", f"<run id of {yk}>")
        dd["region"] = PH["region"]
        out[f"compare_{name}_{split}"] = dd
    return out


def write_templates() -> list[str]:
    DIR.mkdir(exist_ok=True)
    names = []
    for name, doc in build_all().items():
        (DIR / f"{name}.template.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        names.append(name)
    return names


if __name__ == "__main__":
    print("wrote", len(write_templates()), "templates to", DIR)
