"""Runner self-test (not a scored unit): proves the evidence chain end to end on a real job, with a check
whose answer does not depend on the CPU it runs on.

    python -m tools.sm.selftest --test-set <repo path> --run-id {run_id} --evidence-dir {evidence_dir}

It reads the declared test set from the job's copy of the commit, refuses if its sha256 differs from the
declaration's test_set_sha256, then writes one item per non-empty line: passed when the line is canonical
JSON (sorted keys, compact, UTF-8: re-serialising the parsed line gives the same bytes), score 1.0 / 0.0.
The result is the declared target's metric over those items. Evidence goes to a local evlib store at
--evidence-dir; tools/sm/entry.py uploads and attests it. Run it with tools/sm/examples/runner-selftest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from tools import evlib

GRADER = "tools/sm/selftest.py:v1 (line is canonical JSON)"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def grade(line: str) -> tuple[bool, str]:
    try:
        again = canonical(json.loads(line))
    except ValueError:
        return False, ""
    return again == line, again


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test-set", required=True, help="repo path, relative to the code root")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--evidence-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    s3 = evlib.S3Store(os.environ["EVIDENCE_S3"]) if os.environ.get("EVIDENCE_S3") else None
    decl = evlib.read_declared(a.run_id, s3) if s3 else None
    if decl is None:
        print("selftest: no declaration for this run in EVIDENCE_S3; the runner writes it first", file=sys.stderr)
        return 3
    data = Path(a.test_set).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != decl["test_set_sha256"]:
        print(f"selftest: {a.test_set} has sha256 {digest}, the declaration says {decl['test_set_sha256']}",
              file=sys.stderr)
        return 4
    items = []
    for n, line in enumerate(x for x in data.decode("utf-8").splitlines() if x.strip()):
        ok, again = grade(line)
        try:
            item_id = str(json.loads(line).get("incident_id") or f"line-{n:04d}")
        except (ValueError, AttributeError):
            item_id = f"line-{n:04d}"
        items.append({"item_id": item_id, "input_sha256": hashlib.sha256(line.encode()).hexdigest(),
                      "output_sha256": hashlib.sha256(again.encode()).hexdigest(), "passed": ok,
                      "score": 1.0 if ok else 0.0, "grader": GRADER, "backend": "python-json",
                      "extra": {"line": n}})
    local = evlib.LocalStore(a.evidence_dir)
    evlib.append_records(a.run_id, "item", items, store=local)
    res = evlib.make_result(decl, items, "tools/sm/selftest.py", decl["code_sha"],
                            details={"test_set": a.test_set, "lines": len(items)})
    evlib.write_result(res, store=local)
    print(json.dumps({k: res[k] for k in ("metric", "value", "passed", "n_items")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
