"""Load, validate and hash the frozen Track 3 call suite.

The hash is sha256 over the canonical suite: each item re-serialised as JSON with sorted keys and no
whitespace, in file order, joined by "\\n". Re-formatting the file does not change it; changing any
field, adding, removing or reordering an item does.

    python -m t3.suite.suite_hash [path]        # prints the hash and the item count
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

DEFAULT_SUITE = Path(__file__).with_name("call_suite.jsonl")

CALLERS = {"noc_lead", "ran_monitor", "ran_optimizer", "ran_explainer", "team_app"}
RULE_CLASSES = {"allow", "default_deny", "require_case_id", "require_human_approval", "pause",
                "guardrail_injection"}
REQUIRED = ("id", "caller", "tool", "args", "expect", "rule_class", "adversarial", "why")
APPROVAL_STEPS = {"request", "after_approval", "replay"}
PAUSE_WINDOWS = {"inside", "after_resume"}
# amendment A2: `change` is one canonical string, cells sorted, one decimal with an explicit sign
_ONE = r"cell=\d+;d_power_db=[+-]\d+\.\d;d_tilt_deg=[+-]\d+\.\d"
CHANGE_RE = re.compile(rf"{_ONE}(\|{_ONE})*")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_suite(path: str | Path = DEFAULT_SUITE) -> list[dict]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                items.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: not JSON ({exc})") from exc
    return items


def validate(items: list[dict]) -> list[str]:
    """Returns a list of problems; empty means the suite is well formed."""
    problems, seen = [], set()
    for it in items:
        where = it.get("id", "?")
        for key in REQUIRED:
            if key not in it:
                problems.append(f"{where}: missing {key}")
        if it.get("id") in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(it.get("id"))
        if it.get("caller") not in CALLERS:
            problems.append(f"{where}: unknown caller {it.get('caller')}")
        if "___" not in str(it.get("tool", "")):
            problems.append(f"{where}: tool must be <target>___<tool>")
        if it.get("expect") not in ("ALLOW", "DENY"):
            problems.append(f"{where}: expect must be ALLOW or DENY")
        if it.get("rule_class") not in RULE_CLASSES:
            problems.append(f"{where}: unknown rule_class {it.get('rule_class')}")
        if not isinstance(it.get("args"), dict):
            problems.append(f"{where}: args must be an object")
        if not isinstance(it.get("adversarial"), bool):
            problems.append(f"{where}: adversarial must be a bool")
        if "approval_step" in it and it["approval_step"] not in APPROVAL_STEPS:
            problems.append(f"{where}: bad approval_step")
        if "pause_window" in it and it["pause_window"] not in PAUSE_WINDOWS:
            problems.append(f"{where}: bad pause_window")
        if "case_slot" in it and not (isinstance(it["case_slot"], int) and 0 < it["case_slot"] < 100):
            problems.append(f"{where}: case_slot must be an int 1-99")
        change = (it.get("args") or {}).get("change") if isinstance(it.get("args"), dict) else None
        if change is not None and not (isinstance(change, str) and CHANGE_RE.fullmatch(change)):
            problems.append(f"{where}: change must be the canonical string (amendment A2)")
    return problems


def suite_sha256(items_or_path=DEFAULT_SUITE) -> str:
    items = items_or_path if isinstance(items_or_path, list) else load_suite(items_or_path)
    return hashlib.sha256("\n".join(canonical(i) for i in items).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else DEFAULT_SUITE
    items = load_suite(path)
    problems = validate(items)
    for p in problems:
        print("INVALID", p)
    print(f"{suite_sha256(items)}  {len(items)} items  {path}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
