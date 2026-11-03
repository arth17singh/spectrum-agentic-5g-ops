# t3/trust: trust gate and pilot ladder

The trust gate keeps a trust score for each Track 2 predictor version, `(model_id, version)`. After every applied change it compares the predicted KPI delta with the measured one. When a version stops matching the twin, the gate pauses it: it adds a Cedar forbid on `ran___predict` and writes one `control` record.

Settings live in `thresholds.json`. Only `python -m t3.trust.calibrate --apply` changes that file, and its sha256 is recorded in `calibration.json`.

## Synchronous call from `ran___apply_change`

`ran___apply_change` calls the `team13-trust-gate` Lambda with `InvocationType=RequestResponse` before it writes the change record. It then writes that record once, with `trust_before` and `trust_after` filled from the response.

Request (the change record apply is about to write, trust fields empty):

```json
{"record": {
  "kind": "change", "run_id": "...", "case_id": "case-...", "change_id": "chg-...", "item_id": "step-3",
  "change": [{"cell": 1, "d_power_db": 2.0, "d_tilt_deg": -1.0}], "change_text": "cell=1;d_power_db=+2.0;d_tilt_deg=-1.0",
  "twin": {"kind": "maveric-inprocess", "version": "...", "scenario": "nominal", "seed": 0, "step": 4},
  "baseline_kpis": {...}, "measured_kpis": {...}, "measured_delta": {...},
  "predicted_delta": {...}, "predictor": {"model_id": "...", "version": "..."},
  "scored": true, "probe": false,
  "schema_version": "1", "team": "team13", "created_at": "...", "aws_time": null, "aws_time_source": null, "synthetic": false
}}
```

Response (about 10 ms of work, well inside 2 s):

```json
{"trust_before": 0.71, "trust_after": 0.64, "e": 0.42, "score": 0.79, "action": null,
 "direction_agree": true, "status": "active", "band": "trusted", "n": 7,
 "thresholds_sha256": "...", "elapsed_ms": 3.1}
```

- **`action: "pause"`**: returned only on the call that paused the version. That response also carries `policy_name` and `control_record_key`, the evlib object key of the control record the gate wrote. On the local `jsonl` layout the key is just `control`.
- **`skipped`**: present, with trust values null, for probes, `scored: false` (clipped) changes and records without a predictor. It is also present for a retried `change_id`, which is never scored twice; the response then repeats the earlier trust values.
- **`trust_before: null`**: the version's first scored change.

The gate keeps the policy-write rights. On a pause it calls `create_policy` and does not wait for the policy to become ACTIVE, so it can answer quickly; the forbid takes effect a few seconds later. Apply should treat `action: "pause"` as "do not call `ran___predict` for this version again".

Environment:

| Variable | Meaning |
|---|---|
| `T3_TRUST_STORE` | `ddb:<table>` in production. Rows use `pk = t3-trust#<model>@<version>` and `gate_status`, never `status`, so the kit's `approve` never lists them. `file:<path>` or `memory` for local runs. |
| `T3_TRUST_LIVE=1` | Create real policies. Requires `T3_GATEWAY_ARN` and `T3_POLICY_ENGINE_ID`. Otherwise every call is a dry run. |
| `EVIDENCE_S3` / `EVIDENCE_DIR` | Where control records go, through `tools.evlib`. |

The role template is `iam_trust_gate_policy.json`.

## Rule (A6)

```
e = mean over k in {coverage_ratio, sla_ratio, sinr_p10_db} of |pred_delta_k - meas_delta_k| / scale_k
s = 0.5 * (1 - min(1, e)) + 0.5 * [direction right]      (s = 1 - min(1, e) when nothing moved materially)
trust = bias-corrected EWMA of s (alpha)
```

- **Pause trigger:** after at least 3 scored changes, pause when 2 of the last 3 have `e > tau`, or when trust falls below `pause_below`.
- **Resume:** only a person can resume (`gate.resume(..., person=...)`). Resuming starts a new warm-up.

## Ladder

`python -m t3.trust.ladder` runs the same scripted changes on two rungs and writes one `result` per model (`unit: "t3.ladder"`):

| Rung | Label |
|---|---|
| Maveric, official preset | `MAVERIC` |
| Maveric, viavi-like preset | `REHEARSAL (Viavi-like Maveric)` |

Retention is gain(VIAVI) ÷ gain(REHEARSAL), so it is null until a VIAVI rung exists. In that case `details.retention_null_reason` says why.

## Latest runs (real `twin.replay` and `t2.predict`, dry-run policy)

The outputs are in `runs/` and `calibration.json`.

- **Calibration** (288 changes per model, seeds 20000–20011):
  - The healthy model was paused in 1 of 12 episodes. That is within the declared budget of 1.
  - `tilt_reversed` was paused in 12 of 12 episodes, each time at change 3.
- **Demo** (24 scripted changes; held out: seed 0 plus seeds 1000–1009):
  - `tilt_reversed` was paused in 11 of 11 runs, at change 3 or 4.
  - The healthy model was paused in 4 of 11 runs, always at change 15. On scripted steps 12 and 14 its p10-SINR delta has the wrong sign (−0.42 vs +0.84 dB, and +0.49 vs −1.08 dB), so 2 of the last 3 changes have e > τ.
  - Those pauses are real model errors. The thresholds were not tuned after the fact.
- **Ladder** (20 cases × 5 steps):
  - Healthy gain is +0.0154 [0.0098, 0.0213] on MAVERIC and +0.0037 [0.0003, 0.0076] on REHEARSAL.
  - Retention is null because there is no VIAVI rung.

## Thresholds v2 (declared, evaluated once, FAILED its criterion)

v2 was chosen on calibration data only: twin seeds 20000–20039, 960 changes. It was declared before any evaluation in commit ed7d3c4, 2026-11-02T13:26:06-08:00, as `thresholds_v2.json`, sha256 `395ed282e042b4c365f6ad2d92f5b8e560511df1cfc6b572d3aed369a7fef852`.

Settings: residual-p90 scales 0.0333 / 0.0391 / 0.8046 dB, α 0.10, pause line 0.50, τ 1.10. The A6 rule itself is unchanged. The procedure is in the `calibrate_v2.py` docstring.

- **Forecast from calibration:** healthy false-pause rate 14% per run, degraded paused by change 6 in 93%, P(pass) 0.27. The same estimator gave v1 38%, matching v1's observed held-out rate of 4 in 11.
- **Criterion** (seeds 2000–2019, 24-change script): healthy false pauses ≤ 2 of 20, and the degraded model paused by change 6 in ≥ 19 of 20 runs.

One-shot result (`runs/v2_eval.json`):

| Thresholds | Healthy false pauses | Degraded paused by change 6 | Criterion |
|---|---|---|---|
| v2 (395ed282) | 6/20 (5 at change 3, 1 at 20) | 20/20, all at change 3 | **failed** |
| v1 (d051c1e0), same seeds, for reference | 8/20 (at 3, 3, 4, 15, 15, 19, 19, 24) | 20/20, at 3–4 | failed |

v1's original results (seeds 0 and 1000–1009) stay as recorded above.

**The deployed default stays v1, as pre-declared.** Switching to v2 after seeing the 2000–2019 outcome would mean choosing thresholds on the test set. v2 also failed its own declared bar. A future v3 needs its own calibration-only declaration and fresh evaluation seeds.

Five of v2's six false pauses fire on the EWMA line at change 3, the first change where a decision is allowed. They come on the script's opening two-cell changes; calibration showed that change type has 3.5× the exceedance rate. The handler still defaults to v1 (`T3_TRUST_THRESHOLDS` selects the version).

## Commands

```
python -m pytest -q t3/trust/tests
python -m t3.trust.calibrate --out t3/trust/calibration.json --apply
python -m t3.trust.demo --out <evidence dir> --seeds 10
python -m t3.trust.ladder --out <evidence dir>
```
