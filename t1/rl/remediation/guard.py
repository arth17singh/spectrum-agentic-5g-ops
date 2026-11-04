"""The held-out gate stays unread while a policy is developed.

``install()`` adds a process-wide audit hook (PEP 578) that refuses to open any file in
t1/rl/remediation/suite/ whose name starts with ``heldout-`` (the frozen gate sets heldout-v1/v2 and their
sidecars). It is called first thing by the v2 training and tuning entry points (job_train_v2.py), so nothing
in those processes (data generation, behaviour cloning, DAgger, the deadzone tuning on the dev validation set)
can read the gate, by accident or otherwise. Audit hooks cannot be removed: the guard lasts for the process.

The evaluation (evaluate.py) never installs it: reading the gate is its job. The freeze-time exclusion
checks for the v2 candidates do not read it either: they test disjointness from the held-out seed block
(t3/evals/lessons.py HELDOUT_SEED_BLOCK) by seed, which a spec-hash overlap needs anyway (a spec holds its seed).
"""

from __future__ import annotations

import os
import sys
from typing import Any

SUITE_PART = "/t1/rl/remediation/suite/"
PREFIX = "heldout-"
_installed: dict[str, Any] = {}


class HeldoutReadRefused(PermissionError):
    """Something in a development process tried to open the held-out gate."""


def is_heldout(path: Any) -> bool:
    try:
        p = os.fspath(path)
    except TypeError:
        return False
    if isinstance(p, bytes):
        p = p.decode("utf-8", "replace")
    p = os.path.abspath(p).replace("\\", "/")
    head, name = os.path.split(p)
    return (head + "/").endswith(SUITE_PART) and name.startswith(PREFIX)


def _hook(event: str, args: tuple) -> None:
    if event == "open" and args and is_heldout(args[0]):
        raise HeldoutReadRefused(f"refused: {args[0]} is the remediation held-out gate; this process "
                                 f"({_installed.get('reason')}) develops a policy and must not read it "
                                 "(t1/rl/remediation/guard.py)")


def install(reason: str) -> None:
    """Install the guard for the rest of this process (idempotent)."""
    if _installed:
        return
    _installed["reason"] = reason
    sys.addaudithook(_hook)


def installed() -> bool:
    return bool(_installed)
