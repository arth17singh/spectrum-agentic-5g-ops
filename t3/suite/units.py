"""The two Track 3 units graded from the one frozen suite, each with its own declaration, records and result.

Deny  = the items expected DENY, without the approval chain.
Allow = the items expected ALLOW, plus the whole approval chain (t3-033 refused, t3-034 allowed after a
        person approves, t3-035 refused as already applied), since the chain proves an allowed change.
Both declarations carry test_set_sha256 = the hash of the FULL frozen suite, plus `subset` and their own n_items.
"""
from __future__ import annotations

UNITS = {
    "deny": {"unit": "t3.deny", "subset": "expect=DENY;exclude=approval_step"},
    "allow": {"unit": "t3.allow", "subset": "expect=ALLOW;include=approval_step"},
}
BY_NAME = {v["unit"]: k for k, v in UNITS.items()}


def in_unit(it: dict, key: str) -> bool:
    if it.get("approval_step"):
        return key == "allow"
    return it["expect"] == ("DENY" if key == "deny" else "ALLOW")


def select(items: list[dict], key: str) -> list[dict]:
    if key not in UNITS:
        raise ValueError(f"unknown unit {key!r}: use deny or allow")
    return [it for it in items if in_unit(it, key)]


def run_ids(run_id: str, unit: str) -> list[tuple[str, str]]:
    """--unit all runs both units under two run ids, <run_id>-deny and <run_id>-allow."""
    if unit == "all":
        return [(f"{run_id}-deny", "deny"), (f"{run_id}-allow", "allow")]
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}")
    return [(run_id, unit)]
