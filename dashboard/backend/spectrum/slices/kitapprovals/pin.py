"""The approver PIN (review F4, F13). Required on EVERY decision, loopback included: a process on this
laptop (a script, an AI agent with a shell, another slice) can reach 127.0.0.1, so the PIN is the human
factor. Stored only as scrypt(salt) in a 0600 file outside git, written by the user:

    cd dashboard/backend && uv run python -m spectrum.slices.kitapprovals.pin

Five wrong PINs lock decisions for 15 minutes (in memory; a restart clears it).
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import os
import re
import secrets
import stat
import sys
import threading
import time
from pathlib import Path

N, R, P = 2**14, 8, 1
MAX_WRONG, LOCK_S = 5, 15 * 60
PIN_RE = re.compile(r"^\d{6,12}$")


def state_dir() -> Path:
    return Path(os.environ.get("APPROVALS_STATE_DIR") or Path.home() / ".spectrum" / "approvals")


def pin_path() -> Path:
    return state_dir() / "pin"


def hash_pin(pin: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, n=N, r=R, p=P)
    return f"scrypt${base64.b64encode(salt).decode()}${digest.hex()}"


def write_pin(pin: str, path: Path | None = None) -> Path:
    if not PIN_RE.fullmatch(pin):
        raise ValueError("the PIN must be 6 to 12 digits")
    path = path or pin_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(hash_pin(pin) + "\n")
    os.chmod(path, 0o600)
    return path


class PinCheck:
    def __init__(self, path: Path | None = None, clock=time.monotonic):
        self.path = path
        self.clock = clock
        self.wrong: list[float] = []
        self.locked_until = 0.0
        self.lock = threading.Lock()

    def _file(self) -> Path:
        return self.path or pin_path()

    def stored(self) -> str | None:
        """The stored hash, or None if the file is missing, unreadable, not 0600, or malformed."""
        p = self._file()
        try:
            st = p.stat()
            if stat.S_IMODE(st.st_mode) & 0o077:
                return None
            text = p.read_text().strip()
        except OSError:
            return None
        return text if re.fullmatch(r"scrypt\$[A-Za-z0-9+/=]+\$[0-9a-f]{128}", text) else None

    def state(self) -> dict:
        left = max(0, int(self.locked_until - self.clock()))
        return {"set": self.stored() is not None, "locked_for_s": left}

    def verify(self, pin: str) -> str:
        """'ok' | 'pin_not_set' | 'pin_locked' | 'pin_wrong'."""
        with self.lock:
            now = self.clock()
            if now < self.locked_until:
                return "pin_locked"
            stored = self.stored()
            if stored is None:
                return "pin_not_set"
            _, salt_b64, want = stored.split("$")
            ok = isinstance(pin, str) and PIN_RE.fullmatch(pin) is not None and secrets.compare_digest(
                hash_pin(pin, base64.b64decode(salt_b64)).split("$")[2], want)
            if ok:
                self.wrong.clear()
                return "ok"
            self.wrong = [t for t in self.wrong if now - t < LOCK_S] + [now]
            if len(self.wrong) >= MAX_WRONG:
                self.locked_until = now + LOCK_S
                self.wrong.clear()
                return "pin_locked"
            return "pin_wrong"


def main() -> int:
    a = getpass.getpass("New approver PIN (6-12 digits): ")
    if getpass.getpass("Repeat it: ") != a:
        print("The two PINs differ; nothing written.", file=sys.stderr)
        return 1
    try:
        path = write_pin(a)
    except ValueError as e:
        print(f"{e}; nothing written.", file=sys.stderr)
        return 1
    print(f"Wrote the PIN hash (scrypt, mode 0600) to {path}. Restart is not needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
