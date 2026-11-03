"""Import W6's evlib robustly.

``from tools import evlib`` is tried first (repo root on sys.path, or the Lambda/job package layout).
When another top-level ``tools`` package shadows it (W3's ``t3/tools`` when ``t3/`` is on sys.path, as
pytest arranges), the repo's ``tools/evlib.py`` is loaded by path instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CACHE = {}


def get_evlib():
    try:
        from tools import evlib  # type: ignore

        if hasattr(evlib, "append_records"):
            return evlib
    except ImportError:
        pass
    if "evlib" in _CACHE:
        return _CACHE["evlib"]
    path = Path(__file__).resolve().parents[2] / "tools" / "evlib.py"
    if not path.is_file():
        raise ImportError("tools/evlib.py (W6) is not available")
    spec = importlib.util.spec_from_file_location("team13_tools_evlib", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["team13_tools_evlib"] = mod
    spec.loader.exec_module(mod)
    _CACHE["evlib"] = mod
    return mod
