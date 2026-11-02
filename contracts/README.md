# team13-contracts

The single source of truth for every Team13 evidence record (`schema_version` "1"). Field lists follow `BUILD_CONTRACT.md` ("Record contracts"). The pydantic models in `team13_contracts/records.py` are the source. `gen.py` derives the JSON Schema and the TypeScript types from them.

## Kinds and where they are stored

Paths are relative to the evidence root: the local evidence dir, or the S3 prefix `evidence/`.

| kind | what it is | stored at |
|---|---|---|
| `declared` | what a run will be judged on (target, test-set hash, n_items, twin), written **before** the run | `declared/<run_id>.json` |
| `item` | one graded test item | `records/<run_id>/item.jsonl` |
| `decision` | one checkpoint (Cedar) ALLOW/DENY. **v0:** unknown keys allowed until a real AgentCore span is captured | `records/<run_id>/decision.jsonl` |
| `change` | one applied change: predicted vs measured KPIs, error, trust before/after | `records/<run_id>/change.jsonl` |
| `control` | a pause/resume of a model's tool, by `person:<pseudonym>` or `trust-gate` | `records/<run_id>/control.jsonl` |
| `job` | one AWS job (training, processing, endpoint, lambda) and its cost | `records/<run_id>/job.jsonl` |
| `model` | a model version and its artifact hash and status | `models/<model_id>/<version>.json` |
| `result` | a unit's checked outcome against its declared target | `results/<run_id>.json` |

Other files that sit next to these but have no contract here: `outputs/<run_id>/…` and `logs/<run_id>/…`.

## Rules every record follows

The envelope is `schema_version, kind, run_id, team, created_at, aws_time, aws_time_source, synthetic`.

- **Timestamps** are ISO-8601 UTC with a trailing `Z`, for example `2026-11-02T21:30:00.123Z`. `+00:00` is rejected, so each instant has one spelling. Use `team13_contracts.utc_iso(dt)` to write them.
- **Numbers** are JSON numbers. Bools, numeric strings, NaN and Infinity are rejected. Ints stay ints.
- **Identifiers that become paths** (`run_id`, `model_id`, `version`) match `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`, so they contain no slashes and no leading dot.
- **Unknown keys are rejected**, with two exceptions:
  - `decision` (v0);
  - the free-form dicts: `item.extra`, `model.metrics`, `result.details`, `declared.decoding` and `declared.twin.config`.
- **Python-only cross-field checks.** JSON Schema cannot express these, so they run only in pydantic:
  - `aws_time` requires `aws_time_source`;
  - in `change`:
    - each `error.<kpi>` equals `|predicted − measured|` (to 1e-6);
    - cells are unique;
  - in `result`:
    - `metric == target.metric`;
    - `passed == (value <op> target.value)`;
  - in `job`, `aws_end ≥ aws_start`.

## Use from other streams (Python)

Nothing needs installing. Put `contracts/` on the path:

```bash
PYTHONPATH=contracts python your_script.py
# or, in your own venv:  pip install -e contracts
```

```python
from team13_contracts import parse_record, dump_record, canonical_json, sha256_of, utc_iso, ChangeRecord

rec = parse_record(raw_dict)        # dispatches on raw_dict["kind"]; raises ValidationError / UnknownKind
line = canonical_json(dump_record(rec))   # sorted keys, no whitespace, UTF-8: one JSONL line
digest = sha256_of(rec)                   # hex SHA-256 of that text
```

The canonical form of a stored record is `dump_record(parse_record(raw))`. It contains every field, with nulls kept. `canonical_json` is the Python reference. It is **not** the dashboard ledger's `canonical.ts`, which turns floats into strings.

## Use from the frontend

Import the types from `dashboard/frontend/src/contracts/records.gen.ts`:

- `AnyRecord` and `RecordByKind`;
- `RecordKind` and `RECORD_KINDS`;
- one interface per kind, for example `ChangeRecord`.

That file is generated, so never edit it.

## Regenerate and check

```bash
python contracts/gen.py            # writes contracts/schema/*.schema.json and records.gen.ts
python contracts/gen.py --check    # exit 1 if anything is stale (or: git diff --exit-code)
python -m pytest contracts/tests -q
```

The fixtures are in `fixtures/<kind>/`:

- `valid-*.json` files are all `synthetic: true`;
- each `invalid-*.json` file is wrong in exactly one way, and that way is documented in `fixtures/invalid-reasons.json`.

To change a field, edit `records.py`, regenerate, update the fixtures and tell the other streams. Bump `schema_version` for any change that breaks existing records.
