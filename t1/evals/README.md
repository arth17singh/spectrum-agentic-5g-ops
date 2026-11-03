# Track 1 evaluation (t1/evals)

Frozen item sets, one prompt file, one parser and deterministic graders. Every model, base or tuned, gets the same items, the same prompt and the same parser. No model judges anything.

## Rules for anyone building training (SFT) data

1. **Seeds of 900000 and above are reserved for evaluation.** The execution items use twin seeds from `SEED_BASE = 900000` upward (`gen_execution.py`). SFT or best-of-K data generated on the twin must use seeds below 900000. If it doesn't, the tuned model has seen the test states.
2. **SFT data must use `prompt.txt`'s sections and the canonical change string.**
   - Build each example with `prompting.render(item)`: the `### SYSTEM` section plus the matching `### TASK <template>` section.
   - Write the target answer in the same JSON shape the parser reads. For execution and safety that is `{"decision", "change", "reason"}`, where `change` is the A2 string from `team13_contracts.change.canon`.
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

For a real comparison, run the base model first. Then run the tuned model with the same flags plus `--baseline-run-id <base run id>`. The harness refuses a baseline that used other items, another prompt or another parser. Prices must be passed in: `--price-in-per-1m` and `--price-out-per-1m`, or `--instance-price-per-hour`. The default targets are placeholders, so declare the real ones with `--target-*`.

## Superseded hashes

These were frozen at commit `2959c9b` and replaced before any model ran on them.

| unit | superseded file sha256 | superseded test-split sha256 |
|---|---|---|
| execution | `4686121964eaf4811fec665ac73927af8ee8f32ffe5375e3c4074f87dc80b72c` | `8ac4c54536f8affdfdc75bccbed02f4d0ece78ad81e6ef0d93e44e858cdc0feb` |

Why they were replaced:
- the execution items were built on a local stand-in twin, before W2's `ReplayTwin` was merged;
- change strings came from a local canon that dropped zero-change cells.

Both now come from W2 and W1. Regenerating on W2's twin reproduced the same 80 item ids, specs, states and prompt text, and the same change strings. What changed:
- the reference values (`noop_kpi` and the rule, best-of-K and fixed-answer gains) are now stored unrounded. They used to be rounded to 6 decimals, so they differed from the replay by up to 3.3e-7, and the drift check is now exact to 1e-9;
- `meta.twin` names the W2 replay.

The other four files are byte-identical; price_latency keeps its hash because the prompt text it copies is unchanged. Never declare a run against the hashes above. The same list is in `MANIFEST.json` under `superseded`.
