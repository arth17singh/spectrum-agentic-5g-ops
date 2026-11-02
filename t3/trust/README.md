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

## Commands

```
python -m pytest -q t3/trust/tests
python -m t3.trust.calibrate --out t3/trust/calibration.json --apply
python -m t3.trust.demo --out <evidence dir> --seeds 10
python -m t3.trust.ladder --out <evidence dir>
```
