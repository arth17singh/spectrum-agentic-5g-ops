# remediation-v1: a fault-aware policy behind `ran___propose`

ppo_sm_50k is a coverage-capacity optimiser. It proposes nearly the same change on every incident, about +7 to +8 dB on cell 2, whichever cell is faulted. remediation-v1 is a drop-in replacement:

- **Same format:** an SB3 PPO zip plus a `.meta.json` sidecar, loaded by `t1.rl.policy.load`.
- **Same architecture:** the same spaces and actor as ppo_sm_50k.
- **Same conventions:** the same observation and action conventions.

**How it is trained:** it learns to put the network back to its commissioned configuration.

- Training is behaviour cloning, plus 2 DAgger rounds, on twin-labelled incidents (`data.py`, `bc.py`). No LLM labels anything.
- An incident is used only if the twin verifies that the fault's exact inverse recovers it.
- A 1 dB / 1° deadzone in the meta makes "no change" an exact no-op. The policy wrapper reads it; it is absent for the older policies, so they behave as before.

**Seeds:** 2400 fresh dev-split seeds, 1000001 to 1004759.

- The brief asked for 200000 and up. That is inside the silo seed range (100000–499999) and the Track 1 SFT hash range (0–899999), so this range starts above every known range.
- `data-meta.json` records the exclusion checks: no seed collision, no spec-hash overlap with the test, dev, candidate or confirmation files, and a clean literal scan of the repo.

Targets and secondary measures are declared in `targets.json` before any run.

## 1. Data and training (SageMaker, CPU)

```
python -m t1.rl.remediation.job_train --print-declaration > /tmp/rem-train-decl.json
python -m tools.sm.run --run-id t1.rl.remediation-train-<UTC> --declaration /tmp/rem-train-decl.json \
  --instance-type ml.m5.xlarge --max-run 5400 \
  --requirement "stable-baselines3==2.9.2" --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \
  --entry "-m t1.rl.remediation.job_train --run-id {run_id} --evidence-dir {evidence_dir} --output-dir {output_dir} --model-dir {model_dir} --workers 4"
```

- **Cost:** ml.m5.xlarge is $0.230/h, so at most $0.35 for the 90-minute bound. It is expected to finish in about 20–30 minutes.
- **Records:** one item per candidate (2400) and a result with n_items == 2400.
- **The model:** SageMaker writes `/opt/ml/model` to `s3://<bucket>/jobs/<run_id>/output/<job>/output/model.tar.gz`. It holds `remediation_v1.zip`, `.meta.json` and `.provenance.json`.

Commit the three files into `t1/rl/artifacts/`. The image build and the evaluation ship them from git.

## 1b. The large held-out set (SageMaker, x86; frozen before any policy is evaluated on it)

`heldout.py` builds 1,008 masked faults with the t3 test split's generator (`incidents.build` v4 keep rule),
stratified: 7 templates x 4 base cells x 2 presets = 56 strata, 18 each. Candidate k is stratum k % 56,
attempt k // 56, seed 2,000,001 + k; per stratum the first 18 kept candidates in attempt order, so the set
does not depend on workers or timing. The seed block and its exclusion check (known ranges, every suite,
the training candidates, spec hashes, a literal scan of the repo) are in `heldout-reservation.json`.

```
python -m t1.rl.remediation.heldout --print-declaration > /tmp/rem-heldout-decl.json
python -m tools.sm.run --run-id t1.rl.remediation-heldout-<UTC> --declaration /tmp/rem-heldout-decl.json \
  --instance-type ml.m5.xlarge --max-run 3600 --requirement "stable-baselines3==2.9.2" \
  --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \
  --entry "-m t1.rl.remediation.heldout --run-id {run_id} --evidence-dir {evidence_dir} --output-dir {output_dir} --workers 4"
```

Then commit `heldout-v2.jsonl`, `heldout-v2.sha256` and `heldout-v2-meta.json` from the job's outputs into
`t1/rl/remediation/suite/` (about 2 MB) and check the sidecar: `heldout.load_heldout()` refuses a changed file.

The job pins every process to one torch thread, records its host in the meta, and replays every chosen
incident through the evaluation's own check before it writes anything (`self_replay`; nothing is written and
the job fails otherwise). `heldout-v1` (frozen 08:21Z) stays committed as history: the first evaluation could
not replay it on ml.m5.4xlarge, before any result. Run the evaluation and the learning curve on the instance
type the set was frozen on (the meta's `host`).

## 2. Evaluation (SageMaker, x86, the platform the held-out set and the test split were frozen on)

Run this at the commit that holds the artifact and the frozen held-out set:

```
python -m t1.rl.remediation.evaluate --print-declaration > /tmp/rem-eval-decl.json
python -m tools.sm.run --run-id t3.evals.remediation-policy-<UTC> --declaration /tmp/rem-eval-decl.json \
  --instance-type ml.m5.xlarge --max-run 5400 \
  --requirement "stable-baselines3==2.9.2" --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \
  --entry "-m t1.rl.remediation.evaluate --run-id {run_id} --evidence-dir {evidence_dir} --output-dir {output_dir} --workers 4 --instance ml.m5.xlarge"
```

- **Fail-fast:** before any pool starts, the first 3 incidents of every suite are replayed through run_incident's own 1e-6 check; a mismatch stops the job in its first minutes, before any item or result.
- **Items:** 1,159, one per incident: 1,008 heldout (primary), 18 test-masked (continuity), 15 test-visible, 18 confirmation and 100 dev-holdout (replication).
- **Operators:** v1, ppo_sm_50k, config_revert, rule_baseline and `oracle_inverse` (`oracle.py`: the exact inverse of the injected fault, the per-cell ceiling; reported on every set, never pass/fail; `details.flags` carries `ORACLE_CEILING_LOW` if its per-cell rate on a set is below 0.90).
- **Item score:** percell_recovered(v1) − percell_recovered(ppo_sm_50k), from `t3/evals/percell.py` (percell-v1: every cell green against its own healthy state at the grade step). `extra.diff` holds all three component differences.
- **Target (joint, `targets.json` version 3):** on the 1,008 heldout items, each 95% paired-bootstrap lower bound must clear its threshold:
  1. per-cell recovered, v1 − ppo_sm_50k: > 0;
  2. network-level recovery, v1 − ppo_sm_50k: > −0.05 (non-inferior);
  3. diagnosis hit, v1 − ppo_sm_50k: > 0.

  PASS only if all three pass. The unit's value is the smallest margin (lower bound minus threshold); `tools/check_unit.py` part e replays every component (`JOINT_METRIC`). The same components are reported on test-masked (continuity) and on test-visible, confirm and dev-holdout (replication), outside pass/fail.
- **Why the target changed twice before any real evaluation** (`targets.json` records both, with what was seen):
  - version 2: network-level recovery is capped at 1.0 and rewards overshoot (Live Fix run `t3.livefix-20261104T073206Z-9695`: network recovery 1.00 while cells 0 and 3 stayed degraded);
  - version 3: n = 18 gives a ±0.25 interval on a difference; the 1,008-incident held-out set about ±0.04.
- **Secondary measures,** reported per suite and operator with intervals vs each baseline: per-cell recovered, degraded and collateral cells, diagnosis hit, cells touched, total |d_power_db|, total |d_tilt_deg|, headroom used, power above healthy, and false changes on the healthy network.
- **Confirmation suite:** it was frozen on Darwin-arm64, so the job rebuilds it from its frozen specs, and records that it did.

## 2b. Learning curve (SageMaker, x86; descriptive, decision rule declared first)

```
python -m t1.rl.remediation.learning_curve --print-declaration > /tmp/rem-lc-decl.json
python -m tools.sm.run --run-id t1.rl.remediation-learning-curve-<UTC> --declaration /tmp/rem-lc-decl.json \
  --instance-type ml.m5.xlarge --max-run 5400 \
  --requirement "stable-baselines3==2.9.2" --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \
  --inputs train=s3://bb26-p-team13-data-483598626742-use2/jobs/t1.rl.remediation-train-20261104T073837Z/output/t1-rl-remediation-train-20261104T073837Z-664093/output/output.tar.gz \
  --entry "-m t1.rl.remediation.learning_curve --run-id {run_id} --evidence-dir {evidence_dir} --output-dir {output_dir} --train-output {channel:train} --workers 4 --instance ml.m5.xlarge"
```

Behaviour-cloning round 0 on 10/25/50/100% of v1's train incidents (nested), each played on all 1,008 heldout incidents. Target `percell_gain_100_vs_50_ci_low > 0`: PASS = data-limited (next generation: more data); MISS = next generation: better inputs. DAgger is left out and declared (see the module docstring); the 100% point is checked against the training job's round 0.

## 3. Serving it (after the evaluation; each step needs the user's go)

1. **Build the image** from the commit that holds both artifacts: `t3/tools/image/build.sh <commit>`. It produces `team13-ran-tools:<sha12>`, arm64, about 2.07 GB, with both policies inside. The image also needs the new `t1/rl/policy.py`, which the deadzone requires.
2. **Push it to ECR:** `docker tag` the image to `<account>.dkr.ecr.us-east-2.amazonaws.com/team13-ran-tools:<sha12>`, log in to ECR, then `docker push`.
3. **Switch the policy.** Record the current `Code.ImageUri` of `team13-ran-tools` first, for rollback. There are two ways:
   - **Narrow (recommended):**
     - Run `aws lambda update-function-code --function-name team13-ran-tools --image-uri <new>` and wait for it.
     - Then change only `T1_POLICY_PATH` to `/var/task/t1/rl/artifacts/remediation_v1.zip`.
     - To do that, read the current environment, change that one key, and write it back with `update-function-configuration`. Never print the values, and keep any temp file 0600 and outside the repo.
   - **`t3/deploy/register.py --apply`:**
     - It rewrites the whole environment from its own flags, re-applies the Gateway targets and the Cedar rules, and moves the trust gate to the new image.
     - So every flag must match the current deployment: `--kit-dir`, `--evidence-bucket`, `--predictor-path /var/task/t2/artifacts/t2_response_federated_v1.npz`, `--nlm-endpoint team13-otel-nlm`, `--standin-model`, and `--trust-live` if the gate is live.
     - A dry run with placeholders prints `T1_POLICY_PATH` = remediation_v1 and the rest of the plan.
4. **Name the served policy in records.** For Live Fix and swarm runs, set `T3_POLICY_ID=t1-remediation-bc`, `T3_POLICY_VERSION=<meta version>` and `T3_POLICY_ARTIFACT=t1/rl/artifacts/remediation_v1.zip` (`t3/evals/live_swarm.py`).
5. **Smoke test:** one `ran___propose` on a seeded case. Its `policy` should read `t1-remediation-bc`, and the change should land on the faulted cell. A healthy case should get `no_change`.
6. **Rollback:** set `T1_POLICY_PATH=/var/task/t1/rl/artifacts/ppo_sm_50k.zip`, which is still in the new image, or `update-function-code` back to the recorded image.

## Local smoke (at most 50 incidents, labelled synthetic)

```
python -m t1.rl.remediation.job_train --run-id smoke-x --evidence-dir E --output-dir O --model-dir M --limit 32 --smoke --declare
python -m t1.rl.remediation.evaluate --run-id smoke-y --evidence-dir E --output-dir O2 --policy M/remediation_v1.zip \
  --smoke --limit-test 5 --limit-confirm 5 --holdout 5 --declare
```
