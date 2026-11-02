# team13-contracts

The single source of truth for every Team13 evidence record (`schema_version` "1"). Field lists follow `BUILD_CONTRACT.md` ("Record contracts") as changed by `BUILD_CONTRACT_AMENDMENT_1.md` (A2, A5, A11, A14, A15). The pydantic models in `team13_contracts/records.py` are the source. `gen.py` derives the JSON Schema and the TypeScript types from them.

## Kinds and where they are stored

Paths are relative to the evidence root: the local evidence dir, or the S3 prefix `evidence/`.

| kind | what it is | stored at |
|---|---|---|
| `declared` | what a run will be judged on (target, test-set hash, n_items, twin), written **before** the run | `declared/<run_id>.json` |
| `item` | one graded test item | `records/<run_id>/item…` (see below) |
| `decision` | one checkpoint (Cedar) ALLOW/DENY. **v0:** unknown keys allowed until a real AgentCore span is captured | `records/<run_id>/decision…` |
| `change` | one applied change: measured KPI delta against the no-change replay, the predicted delta, the error, and trust before and after | `records/<run_id>/change…` |
| `control` | a pause/resume of a model's tool, by `person:<pseudonym>` or `trust-gate` | `records/<run_id>/control…` |
| `job` | one AWS job (training, processing, endpoint, lambda) and its cost | `records/<run_id>/job…` |
| `model` | a model version and its artifact hash and status | `models/<model_id>/<version>.json` |
| `result` | a unit's checked outcome against its declared target | `results/<run_id>.json` |

Other files that sit next to these but have no contract here: `outputs/<run_id>/…` and `logs/<run_id>/…`.

**Per-run records have two layouts (A11).** S3 cannot append, so both are valid:

- **One object per record**, for Lambdas and tools: `records/<run_id>/<kind>/<record_key>.json`. The `record_key` must be a safe path segment and unique within the run. Use `change_id` for `change`, and an id unique to that record for the other kinds.
- **JSON Lines**, for batch writers: `records/<run_id>/<kind>.jsonl`, one `canonical_json(dump_record(rec))` per line.

Every reader (`tools/evlib.py`, the dashboard's `RecordsReader`) merges both layouts for a run.

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
  - in `change` (amendment 1):
    - `change_text == canon(change)` and `change == parse(change_text)`. Store the rounded list, not the raw proposal;
    - `measured_delta == measured_kpis − baseline_kpis`, to 1e-6;
    - each `error.<kpi>` equals `|predicted_delta − measured_delta|`, to 1e-6. Compute it from unrounded values. `error` needs `predicted_delta`;
    - when `scored` is false, `unscored_reason` is required, for example `"clipped"` (A5) or `"probe"`. When `scored` is true, it must be absent;
    - `predictor`, `predicted_delta`, `error` and `direction_agree` may be omitted only when `probe` is true and `scored` is false (A14);
    - cells are unique;
  - in `result`:
    - `metric == target.metric`;
    - `passed == (value <op> target.value)`;
  - in `job`, `aws_end ≥ aws_start`.

## The canonical change string (A2)

A change goes to the tools as one string. That keeps the kit's approval key, `"chg-" + sha256(json.dumps([case_id, item_id, change]))[:12]`, stable:

```
cell=<int>;d_power_db=<+x.x|-x.x>;d_tilt_deg=<+x.x|-x.x>
```

- Cells are 0-based.
- Each value has one decimal and an explicit sign. Zero, including -0.0, is written `+0.0`.
- Several cells are joined with `|` and sorted by cell.

```python
from team13_contracts.change import canon, parse, approval_key
text = canon([{"cell": 3, "d_power_db": -1, "d_tilt_deg": 0.5}, {"cell": 1, "d_power_db": 2, "d_tilt_deg": 0}])
# 'cell=1;d_power_db=+2.0;d_tilt_deg=+0.0|cell=3;d_power_db=-1.0;d_tilt_deg=+0.5'
parse(text)        # [{'cell': 1, 'd_power_db': 2.0, 'd_tilt_deg': 0.0}, {'cell': 3, ...}]
canon(text)        # returns text unchanged, or raises ValueError if it is not canonical
approval_key(case_id, item_id, text)   # the kit's change_key for that change
```

How the functions behave:

- `canon(list)` rounds with Python's `format`, which rounds half to even.
- `parse` accepts **only** canonical strings. It raises `ValueError` on:
  - a missing sign or decimal;
  - `-0.0`;
  - unsorted or duplicate cells;
  - whitespace or a wrong separator.

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

The canonical form of a stored record is `dump_record(parse_record(raw))`. It contains every field, with nulls kept.

`canonical_json` keeps floats as JSON numbers, and that is intended. The dashboard's demo ledger, `core.canonical` and `canonical.ts`, turns floats into `.6g` strings so it can hash records in the browser. The two serve different purposes and are not interchangeable. Evidence records use `canonical_json`.

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
python -m pytest contracts/tests -q   # also passes with --import-mode=importlib
```

The committed schema and TypeScript files were generated with pydantic 2.14.2, the same version pinned in `dashboard/backend/uv.lock`. Other pydantic 2.x versions may write the JSON Schema slightly differently. If yours does, regenerate with 2.14.2 before trusting a `git diff`.

The fixtures are in `fixtures/<kind>/`:

- `valid-*.json` files are all `synthetic: true`;
- each `invalid-*.json` file is wrong in exactly one way, and that way is documented in `fixtures/invalid-reasons.json`.

To change a field, edit `records.py`, regenerate, update the fixtures and tell the other streams. Bump `schema_version` for any change that breaks existing records.
