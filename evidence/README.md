# evidence/

The real evidence for judged runs, and the signed ledger that records it. Nothing here is simulated.
The SPECTRUM backend scans this folder every 30 s (set `SPECTRUM_EVIDENCE_POLL_S` to change it, or
press **Scan now** on the Evidence screen) and appends one signed ledger record per new file.

## Layout

| Path | What | Written by |
|---|---|---|
| `declared/<run_id>.json` | The declaration, written **before** the run (schema below) | you, before the run |
| `outputs/<run_id>/...` | Per-item outputs | the run |
| `records/<run_id>/...` | Per-run records | the run |
| `logs/<run_id>/...` | Logs | the run |
| `judging.json` | The run ids handed in for judging, at most 20: `{"run_ids": [...]}` | you |
| `ledger.jsonl` | The evidence ledger: one signed record per line, append-only | the backend |
| `anchors.jsonl` | A Merkle root over every 32 ledger records | the backend |
| `public_keys.json` | The Ed25519 public key that verifies `ledger.jsonl` | the backend, on first run |

Anything else is ignored. One `run_id` links the declaration, outputs, records and logs.

Declaration schema (every field required; `test_set_sha256` is 64 lowercase hex characters; the file
name must equal `run_id`). A file that does not match is still recorded, as an **invalid** record.

```json
{ "run_id": "2026-11-02T14-03-knowledge-01", "unit": "knowledge",
  "target": {"metric": "exact_match", "value": 0.70},
  "test_set_sha256": "<64 hex>", "model_id": "otel-llm-1.7b-it", "grader": "rule-based-parser-v2" }
```

## Ledger rules

- Each ledger record holds the run id, artifact kind, path, the file's SHA-256, its size and its source
  time (file modification time, or S3 `LastModified` and `VersionId` when the S3 source is used).
- Re-scanning adds nothing for unchanged files. A file whose content changes after it was recorded
  gets an `evidence.changed` record; the original record stays. A deleted file gets `evidence.missing`.
- On startup the backend re-verifies the whole chain with `public_keys.json` only. If anything fails
  it reports **BROKEN at #n** in the API and on the Evidence screen, stops appending, and changes nothing.

## Keys

The signing secret is `dashboard/backend/.evidence-key` (mode 0600, gitignored, created on first run;
`SPECTRUM_EVIDENCE_KEY` moves it). Never commit or share it. **Do commit `public_keys.json`** once
it exists: it is what lets anyone verify the ledger. If the secret is lost, the backend reports a key
mismatch and refuses to append until a person decides what to do.

## Rebuild (offline, no AWS)

```sh
cd dashboard && make evidence-rebuild          # or:
cd dashboard/backend && uv run python -m spectrum.evidence rebuild [--dir ../../evidence]
```

It re-verifies the chain, anchors and signatures, re-hashes every recorded file, and reprints the
per-run summary. Exit 0 = clean; 1 = a file changed or is missing (or the ledger recorded a change);
2 = the ledger, an anchor or the public key file does not verify. With the S3 source, sync the bucket
prefix into this folder first: rebuild only reads local files.

## S3 source (optional, read-only)

`SPECTRUM_EVIDENCE_S3=s3://<bucket>/<prefix>/` makes the backend read artifacts from S3 instead of this
folder (ListObjectsV2 and GetObject only; credentials come from the environment). The ledger itself
still lives here. Copying the ledger to S3 is not implemented yet.

## Known limits

- Deleting lines from the **end** of `ledger.jsonl` after the last anchor cannot be detected from the
  file alone; an external copy of the head hash would be needed.
- Local source times are file modification times, which anyone with access can change.
  "Times agree with AWS" and "independent grader" are not checked yet.
