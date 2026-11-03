"""The panel's own memory: cases and turns this laptop started, kept so a restart neither forgets nor resends.

Default file: ~/.spectrum/asknoc/state.json (mode 0600, outside every repo; ASKNOC_STATE_DIR moves it).
Writes are atomic (temp file in the same folder, fsync, os.replace). A turn is saved as `sending` BEFORE
its request leaves the laptop, so a crash during a send is recovered as `outcome_unknown`, never resent.

Finished turns are also appended to logs/turns-<YYYYMMDD>.jsonl under the same folder: ids, times, states
and SHA-256 hashes only (no operator text, no reply text). That log is local operator history, not judged
evidence: it is never written into the evidence root.
When `memory=True` (offline runs: tests, the merge gate) nothing touches the disk.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


def default_dir() -> Path:
    return Path(os.environ.get("ASKNOC_STATE_DIR") or Path.home() / ".spectrum" / "asknoc").expanduser()


class Store:
    def __init__(self, directory: Path | None = None, memory: bool = False):
        self.memory = memory
        self.dir = None if memory else Path(directory or default_dir())
        self.cases: dict[str, dict] = {}
        self.turns: dict[str, dict] = {}
        if not memory:
            self._load()

    @property
    def path(self) -> Path | None:
        return None if self.dir is None else self.dir / "state.json"

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError, OSError):
            return
        if isinstance(data, dict):
            self.cases = {k: v for k, v in (data.get("cases") or {}).items() if isinstance(v, dict)}
            self.turns = {k: v for k, v in (data.get("turns") or {}).items() if isinstance(v, dict)}

    def save(self) -> None:
        if self.memory:
            return
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        body = json.dumps({"cases": self.cases, "turns": self.turns}, separators=(",", ":"), default=str)
        fd, tmp = tempfile.mkstemp(prefix=".state-", dir=self.dir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def log_turn(self, record: dict) -> None:
        if self.memory:
            return
        logs = self.dir / "logs"
        logs.mkdir(parents=True, exist_ok=True, mode=0o700)
        p = logs / f"turns-{time.strftime('%Y%m%d', time.gmtime())}.jsonl"
        with open(p, "a") as f:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        os.chmod(p, 0o600)
