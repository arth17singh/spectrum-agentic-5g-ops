# Track 1 evaluation (t1/evals)

Frozen item sets, one prompt file, one parser and deterministic graders. Every model, base or tuned, gets the same items, the same prompt and the same parser. No model judges anything.

## Rules for anyone building training (SFT) data

1. **Seeds of 900000 and above are reserved for evaluation.** The execution items use twin seeds from `SEED_BASE = 900000` upward (`gen_execution.py`). SFT or best-of-K data generated on the twin must use seeds below 900000. If it doesn't, the tuned model has seen the test states.
2. **SFT data must use `prompt.txt`'s sections and the canonical change string.**
   - Build each example with `prompting.render(item)`: the `### SYSTEM` section plus the matching `### TASK <template>` section.
   - Write the target answer in the same JSON shape the parser reads. For execution and safety that is `{"decision", "change", "reason"}`, where `change` is the A2 string from `team13_contracts.change.canon`.
   - Turn each example into model text with the model's own chat template: `tokenizer.apply_chat_template([{"role": "system", ...}, {"role": "user", ...}], add_generation_prompt=True, enable_thinking=False)`. That is exactly what `local_hf.LocalHFClient.render` does. Thinking is OFF, so the target is the bare JSON answer, with no `<think>` text.
   - A tuned model trained on any other format is measured on a format it never saw.
3. **Use only the dev split for prompt or format development.** The test split is frozen; its hash goes into every declaration.

## Units and items

| unit | file | items | grader |
|---|---|---|---|
| interface | `items/interface.jsonl` | 320 | exact match of `{intent, cells, parameter, direction, amount, kpi}`; cells are 0-based; direction is the operator's word (up/down) |
| execution | `items/execution.jsonl` | 80 | replays the item on W2's `twin.replay.ReplayTwin` and scores the model's change against no change, the SDK rule baseline and a best-of-K reference |
| safety | `items/safety.jsonl` | 170 | per-item rules, checked with the SDK `ActionSafetyFilter` |
| knowledge | `items/knowledge.jsonl` | 600 | exact letter, label, root-cause C-number or number (1%); GSMA ot-lite at a pinned revision; groups core, telelogs and telemath, reported separately |
| price_latency | `items/price_latency.jsonl` | 50 | no correctness score; records latency, tokens and cost per call |

The split is 20% dev and 80% test, by sha256 of the unit and item id. `items/MANIFEST.json` holds:
- each file's sha256;
- the sha256 of each split, and of each knowledge group;
- the prompt hash, and the parser and grader versions.

### How the execution unit is scored

- **KPI and gain.** The KPI is `coverage_ratio + sla_ratio`. Gain is that KPI after the change minus the KPI after no change, both measured from the same state and random stream, using W2's `counterfactual`.
- **Score.** `clip(gain / best-of-K gain, -1, 1)`.
- **Clipped changes.** A change the safety filter clips is applied as clipped. A hard violation (a change that would cross an absolute power or tilt bound) never scores above 0.
- **Replay check.** If the replay does not reproduce an item's frozen no-change KPI or rule gain, the grader raises `TwinDrift` and the run stops.
- **Fixed-answer filter.** Items where the fixed answer `cell=2;+8 dB` earns more than half of the best-of-K gain were dropped.

## Required modules (no stand-ins)

- `team13_contracts` (W1): the canonical change string and record validation. If it is not installed, `t1/evals/__init__.py` puts the in-repo `contracts/` on `sys.path`.
- `twin.replay` (W2): the in-process Maveric twin.
- `tools.evlib` (W6): the evidence writer.

`""` means "no change" inside the eval (a refusal or a no-op step). It is never sent to a tool, because A2 has no empty change.

## Commands (run from the repository root)

```
python -m pytest -q -p no:cacheprovider t1/evals/tests
python -m t1.evals.freeze --verify
python -m t1.evals.freeze --source-cache <dir>      # regenerate everything (ot-lite cached in <dir>)
python -m t1.evals.run --unit interface --client fake --evidence-dir /tmp/ev   # dry run, synthetic
```

For a real comparison, run the base model first. Then run the tuned model with the same flags plus `--baseline-run-id <base run id>`.

**The gain over the baseline.** It is computed as `paired-bootstrap-v1`, the one method `tools/check_unit.py` part e replays from the item records:
- **Pairs**: sorted `item_id` pairs; `passed` counts as 1/0.
- **Resampling**: numpy `default_rng(20261102).integers`, 10,000 resamples. The statistic is the mean difference, and the interval is the linear-quantile 95% interval.
- **What it is computed on**: execution gains are on `score`; interface, safety and knowledge gains are on `passed` (pass rate). `--gain-metric` overrides this.
- **Where it is recorded**: the full spec goes in `result.details.gain`. Knowledge records one gain per group in `details.groups.<g>.gain`, each with `subset {field: extra.group}`.
- **Claiming the interval**: when the claim is about the interval itself (for example "CI lower bound > 0"), declare the bound as the target: `--target-metric gain_ci_low_vs_base --target-op '>' --target-value 0`. The result's value is then `ci_low`, and check_unit replays it.
- **Latency**: p95-latency gains are a difference of quantiles, which v1 does not cover. They appear only as `details.latency_gain_note` and are never a target.

**Default targets and secondary checks with a baseline** (plan v5.1, lead decision 3 Nov). With `--baseline-run-id` and no `--target-*` flags, the target is `gain_ci_low_vs_base`:
- `> 0` for execution and safety, where the claim is an improvement;
- `> -0.02` for interface and `> -0.03` for knowledge, where the claim is no regression.

price_latency keeps `p95_latency_ms <= 3000`.

Plan v5.1's other bars are point-estimate secondary checks. They are declared in `decoding.secondary_checks` (`--print-declaration` emits them, so they are fixed before the run) and evaluated into `result.details.secondary_checks` as PASS / FAIL / NOT MEASURED:
- **execution**: mean ≥ 0.149; gain ≥ +0.10.
- **interface**: pass rate ≥ 0.90; ≥ base − 0.02.
- **safety**: pass rate ≥ 0.95; ≥ base − 0.02; benign over-refusal ≤ 0.10.
- **knowledge**: ≥ base − 0.03.
- **price_latency**: p95 ≤ base × 1.1; ≤ $1 per 1,000 calls. A base run checks only the cost bar.

`gain_value` is the v1 gain's point value: this run minus the baseline on paired items. The harness refuses a baseline that used other items, another prompt or another parser. Prices must be passed in: `--price-in-per-1m` and `--price-out-per-1m`, or `--instance-price-per-hour`. The default targets are placeholders, so declare the real ones with `--target-*`.

## Running as a SageMaker job

Every real evaluation runs inside a SageMaker GPU job (ml.g5.2xlarge for both base and tuned), never on a laptop. The runner in `tools/sm` launches the jobs; this harness is the entry point it calls. One job runs one unit for one model. The model is loaded in-process with `--client local-hf`, so base and tuned go through the same code path, the same prompt and the same parser.

```
python -m t1.evals.run --unit <interface|execution|safety|knowledge|price_latency> \
  --client local-hf --model-dir $SM_CHANNEL_MODEL [--adapter-dir $SM_CHANNEL_ADAPTER] \
  --model-id <label, e.g. otel-llm-1.7b-it or otel-llm-1.7b-it+lora-v1> \
  --instance-price-per-hour <on-demand USD/h of the job instance> --price-source "<price list, date>" \
  --target-metric <m> --target-op <op> --target-value <v> --target-unit <u> \
  [--baseline-run-id <the base model's run id for this unit>] [--run-id <id>]
```

**Inputs**
- Channel `model`: the base model directory (config, weights, tokenizer).
- Optional channel `adapter`: a LoRA adapter. It is merged into the base before generation unless `--no-merge-adapter` is passed. A merged model directory also works: put it in `model` and give no adapter.

**Environment** (the runner sets these; nothing needs passing by hand)
- `CODE_SHA`: the commit the source was packaged from. A job has no `.git`, and the harness asks git only when `CODE_SHA` and `--code-sha` are both absent.
- `EVIDENCE_S3`. Under the runner, always pass `--evidence-dir {evidence_dir}`. The launcher has already written `declared/<run_id>.json` to S3, so the module writes its declaration, items and result to the local store. `tools/sm/entry.py` then compares the declarations, uploads one object per record and attests them.
  - Raw model outputs go straight to `EVIDENCE_S3` under `outputs/<run_id>/`, because the runner uploads only records and results.
  - A baseline's declaration and items are read from S3.
- `AWS_REGION`.

**Image**
- transformers ≥ 4.51 (Qwen3), torch and peft. The versions used are recorded in `declared.decoding.versions`.
- pydantic and numpy.
- For `--unit execution` only: ran-rl with the maveric extra, because the grader replays the twin in the job.

**Prompt format (binding)**
- The `prompt.txt` SYSTEM section is the system message; the filled TASK section is the user message.
- The model's own chat template is applied with `add_generation_prompt=True` and `enable_thinking=False`.
- Decoding is greedy, batch size 1, with per-unit `max_new_tokens`.
- `declared.decoding` records the chat template's sha256 and whether the template actually honours the thinking flag. A tuned run whose template or decoding differs from its baseline's is refused.

**Latency and price**
- Each item's `latency_ms` is the wall-clock time of the whole call on the job's GPU: chat template, tokenise, generate and decode, with CUDA synchronised before and after.
- Warm-up calls (`--warmup`, default 3) run after the declaration is written and are excluded from the items; they are listed in the result.
- `cost_usd` = instance $/h × latency / 3600. This is the cost when calls are served one at a time.
- `result.details.latency_and_price` gives the p50, p95 (nearest rank) and mean latency, $ per 1,000 calls, the price, its source and the instance type.
- The instance type is read from `/opt/ml/input/config/resourceconfig.json` unless `--instance-type` is given.

**Knowledge**
- One run per model covers all groups (`--group all`, the default). The headline value is over all 475 test items, which is what `check_unit` recomputes.
- `result.details.groups` gives core, telelogs and telemath separately, each with its own paired gain when a baseline is given.

**Run ids per model**
- One run per unit: interface, execution, safety, knowledge and price_latency. That is 5 run ids per model, 10 for base plus tuned.
- Run the base model first. Its declaration must be in the same evidence store, because the tuned run checks it before starting.

**Weights provenance**
- `outputs/<run_id>/model_files.json` holds the sha256 of every file in the model and adapter directories. Their digests are in `declared.decoding`.

### Launching through tools/sm

The launcher stores a declaration before the job. The module's own declaration must match it on every judged field (unit, target, n_items, test_set_sha256, model_id, grader, subset, baseline_run_id, synthetic and twin); otherwise `entry.py` refuses the upload. So print it with the same flags first:

```
python -m t1.evals.run <the same flags as the entry, without --run-id/--evidence-dir> --print-declaration > /tmp/decl.json
python -m tools.sm.run --run-id <id> --declaration /tmp/decl.json --instance-type <type> [--gpu --model-s3 s3://…] \
  [--no-sdk] --entry "-m t1.evals.run <the same flags> --run-id {run_id} --evidence-dir {evidence_dir}" [--dry-run]
```

- `--no-sdk` is for interface, knowledge and price_latency only. execution replays the twin, and safety grades with the SDK `ActionSafetyFilter`, so both need the SDK. Always add `--requirement "pydantic>=2.7,<3"` (team13_contracts needs it; the SDK brings it, the bare image may not), plus `transformers==4.57.6` and `peft==0.21.2` for `--client local-hf`.
- With the SDK, the generated requirements.txt (ran-rl[maveric], gpytorch 1.15.2, linear-operator 0.6.1, plus `--requirement` lines such as `transformers==4.57.6` and `peft==0.21.2`) resolves with torch pinned at the DLC's 2.9.0. I checked this with `uv pip compile … -c torch==2.9.0` for x86_64 manylinux and py3.12. pip leaves a satisfied torch alone, so the image's CUDA torch survives.
- The image's torch (2.9.0) differs from the Mac venv's (2.15.0). That is one more reason for the platform check below.

### Execution items on the job platform (replay check, re-freeze)

The execution items were frozen on a Mac (Darwin arm64, torch 2.15.0). The GP twin computes in float32 torch, so an x86 job may replay slightly different numbers, and the grader stops with `TwinDrift` beyond 1e-9. Before the first real execution run, replay the frozen items on the job platform. This is a CPU job with no model, a few cents:

```
python -m t1.evals.replay_job check --print-declaration > /tmp/t1-replay-decl.json
python -m tools.sm.run --run-id t1.execution.replay-check-x86-1 --declaration /tmp/t1-replay-decl.json \
  --instance-type ml.m5.xlarge --max-run 1800 \
  --entry "-m t1.evals.replay_job check --run-id {run_id} --evidence-dir {evidence_dir}" [--dry-run]
```

- There is one item record per execution item, with the frozen and replayed `noop_kpi`, `rule_gain`, `oracle_gain` and `constant_gain` and each |difference|. The result's `pass_rate == 1.0` means no drift, and `details.max_abs_diff` and `details.platform` are reported.
- **Only if it drifts**, re-freeze on the platform the execution evals run on (the GPU image, ml.g5.2xlarge):
  - Run `replay_job refreeze` the same way, with `--gpu --instance-type ml.g5.2xlarge` and `--output-dir {output_dir}`. Print its declaration with `refreeze --print-declaration`.
  - It rebuilds every item from its own spec (same ids, seeds, prefixes and candidates) and writes `execution.jsonl`, a regenerated `price_latency.jsonl` and `refreeze-summary.json` to `{output_dir}`, which SageMaker uploads as `jobs/<run_id>/output/<job>/output/output.tar.gz`. The summary has the new hashes and how many items changed values, state text, best-of-K or rule choice, or would now fail the item filter.
  - Adopt it with `python -m t1.evals.adopt_refreeze <extracted dir> --run-id <run id>`. It refuses output from other items or with changed identities. It records the old hashes in `items/superseded.json` and the job platform in `items/execution.platform.json`, copies the files and rewrites MANIFEST. Then commit.
  - Declarations then carry the new `test_set_sha256`. Run `replay_job check` once more on the GPU image to confirm `pass_rate == 1.0` there.
- On the Mac, `refreeze` reproduces the committed files byte for byte, so the re-freeze path is a fixed point on the platform that froze them.
- The tests that need exact twin values (`FROZEN_HERE`) run only on the platform recorded in `items/execution.platform.json`. Elsewhere they skip, and `replay_job check` on that platform is the check.
- **First x86 check** (`t1.execution.replay-check-x86-1`, CPU DLC, torch 2.9.0+cpu): 75 of 80 items reproduced exactly.
  - In 5 items, one value moved by one user's worth (1/200 to 1/300 of a ratio): ex-outage-006 `constant_gain` 5.0e-3, ex-power_drop-009 `rule_gain` 1.67e-2, ex-tilt_misconfig-001 `oracle_gain` 3.3e-3, ex-tilt_misconfig-009 and -013 `constant_gain` 5.0e-3.
  - `noop_kpi` never moved.
- **Re-freeze** (`t1.execution.refreeze-x86-1`, GPU DLC on ml.g5.2xlarge, torch 2.9.0+cu130), adopted:
  - 45 of 80 items changed.
  - In 43 of them, the 2-decimal state text in the prompt changed: float32 RSRP/SINR round differently in the last digit.
  - No best-of-K or rule choice changed, and every item still passes the item filter.
  - Reference values moved in 6 items by one user's worth. ex-tilt_misconfig-002, -004 and -005 moved on the GPU image although they matched on the CPU image.
  - So **the CPU and GPU DLCs do not replay identically either**. The execution items now replay exactly only on the GPU image (`items/execution.platform.json`). Run every execution eval, and its confirming replay check, on the GPU image.

### Which units depend on platform numerics

Only execution does.
- **interface and knowledge**: generated by hashing and graded by string, JSON and `math.isclose` comparisons.
- **safety**: its grader runs the SDK `ActionSafetyFilter`, which is float64 adds, clips and comparisons on small deltas. IEEE results there are identical on every platform.
- **price_latency**: copies prompt text and scores nothing.

None of these four calls the twin, torch or a transcendental function.

### Bedrock baselines (jobs with a Bedrock-enabled role)

The organisers' job role, `bb26-SageMakerExecution`, has no Bedrock actions, so `--client bedrock` cannot run in a job under it. The Bedrock runs use `tools.sm.run --role bb26/team13-sm-bedrock-eval`, which the user creates in CloudShell. They run on ml.m5.xlarge, except **execution**, whose grader replays the twin: it must run on the GPU image the items were frozen on (`--gpu`, ml.g6.xlarge is the cheapest GPU type), or the grader stops with TwinDrift.

- **Prices**: `python -m t1.evals.bedrock_prices --model-id <id>` reads the AWS Price List (read-only) and prints the price flags. It fails closed if there is not exactly one standard on-demand input price and one output price.
  - Read 2026-11-02, effective 2026-10-01, us-east-2, per 1M tokens: gpt-oss-20b costs 0.07 in and 0.30 out; Qwen3-32B costs 0.15 in and 0.60 out.
- **Thinking**:
  - gpt-oss cannot turn reasoning off. We send `reasoning_effort: low`, the lowest setting.
  - Qwen3 on Bedrock has no documented switch, and `/no_think` would change the shared prompt, so it is measured as served.
  - In both cases the reasoning comes back as `reasoningContent`, is excluded from the parsed answer, and is billed in `outputTokens`.
  - What was sent is recorded in `declared.decoding`.
  - Use `--max-tokens 2048`, because reasoning counts against the output limit.
- **Fail fast**: if the first 3 calls all fail (for example, the role lacks Bedrock), the run stops instead of recording a run of errors.

## Superseded hashes

`items/superseded.json` lists every replaced freeze, and `MANIFEST.json` repeats it under `superseded`. Never declare a run against these hashes. None of them was ever used by a model run.

| replaced by | unit | superseded file sha256 | superseded test-split sha256 |
|---|---|---|---|
| commit 2959c9b → W2 twin, W1 canon | execution | `4686121964eaf4811fec665ac73927af8ee8f32ffe5375e3c4074f87dc80b72c` | `8ac4c54536f8affdfdc75bccbed02f4d0ece78ad81e6ef0d93e44e858cdc0feb` |
| run t1.execution.refreeze-x86-1 (x86 GPU image) | execution | `63d04053892d118721eed5c40460c9ae0c878043d464db7b9edc10e2223efd74` | `5dbc837713851902e455d750534f7aee49c9b7a1b7836439c280f2606615fc22` |
| run t1.execution.refreeze-x86-1 (x86 GPU image) | price_latency | `4d55adddb52171878f30000e7d7fa969d77bb41cd6e0f33c041927cde185b408` | `802e3621d2bc13d5202caa33af7df7b2585d89d6fc7c27c72216d8a4533f6c44` |

**First replacement.**
- Why: the execution items were built on a local stand-in twin before W2's `ReplayTwin` was merged, and change strings came from a local canon that dropped zero-change cells.
- What changed: the same items were rebuilt, with the reference values now stored unrounded and `meta.twin` naming the W2 replay.

**Second replacement.** The Mac-frozen values did not replay exactly on x86. The items were re-frozen on the GPU image the evals run on: same ids, splits, scenarios, seeds and prefixes. price_latency changed with them, because it copies execution prompt text.

**Current hashes** (file / test split):
- execution: `e7d58194bab6b276c44a8ea73f70b273413e425eb5d951a0c9c2e67897200df2` / `5bf3773ffe1b7e776486b79197e9518e2bd7b634c018ffd17f85b00365e39383`
- price_latency: `c459173aa269a80276d07983b0adb0c0c97aa09bdd4c9976a133479274b56f12` / `17d7d80720e56671af96d6646cb6e1c76c470fc4e3b58f17e444b75b890f5ec2`
