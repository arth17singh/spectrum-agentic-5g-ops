"""For the PERSON on duty: approve the changes a Track 3 run is waiting for, one by one, with the kit.

It never approves anything on its own: it lists what the organisers' kit says is waiting, shows each change,
asks you, and only on your "y" runs the kit's own command (`checkpoint_kit.py approve chg-... "why"`), which
records the decision with YOUR AWS identity. Run it from a terminal signed in as you (AWS_PROFILE=fyuz).

    python -m t3.suite.approve --kit-dir ~/team13-kit --run-id 1102a-t3-allow [--watch]

--run-id limits the list to that run's cases (case-<first 8 hex of sha256(run id)>-NN, and the run's front-door
case). --watch keeps polling every 5 s until you press Ctrl-C (an allow run waits for one approval; a trace
run waits for the lead's change).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

LINE = re.compile(r"^(chg-[0-9a-f]{12})\s+(\S*)\s+(\S*)\s+(.*)$")


def pending(kit_dir: Path, python: str = sys.executable) -> list[dict]:
    out = subprocess.run([python, "checkpoint_kit.py", "approve"], cwd=kit_dir, capture_output=True, text=True,
                         check=True).stdout
    rows = []
    for line in out.splitlines():
        m = LINE.match(line.strip())
        if m:
            rows.append({"approval_id": m.group(1), "case_id": m.group(2), "agent": m.group(3), "change": m.group(4)})
    return rows


def run_prefix(run_id: str) -> str:
    import hashlib
    return hashlib.sha256(run_id.encode()).hexdigest()[:8]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit-dir", type=Path, required=True, help="the folder with checkpoint_kit.py and kit.json")
    ap.add_argument("--run-id", help="only this run's cases")
    ap.add_argument("--case", action="append", default=[], help="also this case id (e.g. a trace run's case)")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--python", default=sys.executable, help="the python of the kit's venv")
    a = ap.parse_args(argv)
    prefix = f"case-{run_prefix(a.run_id)}-" if a.run_id else None
    seen = set()
    while True:
        rows = [r for r in pending(a.kit_dir, a.python) if r["approval_id"] not in seen
                and (prefix is None or r["case_id"].startswith(prefix) or r["case_id"] in a.case)]
        for r in rows:
            seen.add(r["approval_id"])
            print(f"\n{r['approval_id']}  case {r['case_id']}  by {r['agent']}\n  change: {r['change']}")
            answer = input("Approve this change? [y/N/r=reject] ").strip().lower()
            if answer in ("y", "yes", "r", "reject"):
                verb = "approve" if answer.startswith("y") else "reject"
                why = input("Why (one line, recorded): ").strip() or f"{verb}d by the person on duty"
                subprocess.run([a.python, "checkpoint_kit.py", verb, r["approval_id"], why], cwd=a.kit_dir, check=True)
            else:
                print("left waiting (the run will time out if nobody approves)")
        if not a.watch:
            if not rows:
                print("nothing waiting" + (f" for run {a.run_id}" if a.run_id else ""))
            return 0
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
