"""Live Fix evidence through tools/evlib (the same validator and layout as every judged unit).

Each record is validated and serialised ONCE, then written with put_new (never overwrite) to the team's S3
evidence prefix (EVIDENCE_S3, else TEAM13_DATA_BUCKET, else the team bucket named from the account: the same
rule as evidence/live_verify.py) and, byte-identical, to this laptop's evidence folder (SPECTRUM_EVIDENCE_DIR) so
the Evidence screen shows the run after its next scan. Layout: declared/<run>.json, records/<run>/item/<key>.json,
results/<run>.json. The declaration is written before the first Gateway call of the run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from .suite import REPO, repo_import


class Writer(Protocol):
    def declared(self, rec: dict) -> list[str]: ...
    def item(self, rec: dict) -> list[str]: ...
    def result(self, rec: dict) -> list[str]: ...


class EvlibWriter:
    def __init__(self, stores: list[Any]):
        self.ev = repo_import("tools.evlib")
        self.stores = stores

    @classmethod
    def from_env(cls, account: str | None) -> "EvlibWriter":
        ev = repo_import("tools.evlib")
        uri = os.environ.get("EVIDENCE_S3") or (
            f"s3://{os.environ['TEAM13_DATA_BUCKET']}/evidence/" if os.environ.get("TEAM13_DATA_BUCKET")
            else f"s3://bb26-p-team13-data-{account}-use2/evidence/")
        stores: list[Any] = [ev.S3Store(uri, region=os.environ.get("AWS_REGION") or "us-east-2")]
        local = os.environ.get("SPECTRUM_EVIDENCE_DIR")
        if local and Path(local).is_dir():
            stores.append(ev.LocalStore(local))
        return cls(stores)

    def _put(self, rel: str, data: bytes) -> list[str]:
        return [s.put_new(rel, data) for s in self.stores]

    def declared(self, rec: dict) -> list[str]:
        ev = self.ev
        r = ev._envelope(rec, "declared", rec["run_id"])
        r.setdefault("declared_at", ev.utc_now())
        ev.validate(r)
        return self._put(f"declared/{r['run_id']}.json", ev.dumps_doc(r))

    def item(self, rec: dict) -> list[str]:
        ev = self.ev
        r = ev._envelope(rec, "item", rec["run_id"])
        ev.validate(r)
        line = ev.dumps_line(r)
        return self._put(f"records/{r['run_id']}/item/{ev.record_key(line)}.json", line)

    def result(self, rec: dict) -> list[str]:
        ev = self.ev
        r = ev._envelope(rec, "result", rec["run_id"])
        r.setdefault("checked_at", ev.utc_now())
        ev.validate(r)
        return self._put(f"results/{r['run_id']}.json", ev.dumps_doc(r))


def code_sha() -> str:
    return repo_import("tools.evlib").code_sha(REPO)
