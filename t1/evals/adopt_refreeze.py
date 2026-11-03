"""Adopt the output of a `replay_job refreeze` job into t1/evals/items/.

    aws s3 cp s3://<bucket>/jobs/<run_id>/output/<job>/output/output.tar.gz /tmp/rf/ && tar -xzf /tmp/rf/output.tar.gz -C /tmp/rf
    python -m t1.evals.adopt_refreeze /tmp/rf --run-id <refreeze run id>

Refuses unless:
  - refreeze-summary.json's previous hashes are the files committed now (the job ran on these items),
  - its new hashes are the files in the directory (nothing changed in transit),
  - every new item keeps its item_id, split, scenario, seed and prefix (only twin-derived values move).
Then it records the old hashes in items/superseded.json, the job's platform in
items/execution.platform.json, copies execution.jsonl and price_latency.jsonl into items/ and rewrites
MANIFEST.json. Commit the result; declarations then carry the new test_set_sha256.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from t1.evals import freeze
from t1.evals.common import lines_sha256

UNITS = ("execution", "price_latency")


def _lines(p: Path) -> list[str]:
    return [x for x in p.read_text(encoding="utf-8").split("\n") if x != ""]


def adopt(src: Path, run_id: str) -> dict:
    summary = json.loads((src / "refreeze-summary.json").read_text())
    if summary.get("run_id") != run_id:
        raise SystemExit(f"refreeze-summary.json is for run {summary.get('run_id')!r}, not {run_id!r}")
    old: dict[str, dict] = {}
    for u in UNITS:
        h = summary["hashes"][u]
        cur = freeze.read_lines(u)
        if lines_sha256(cur) != h["previous_sha256"]:
            raise SystemExit(f"{u}: the job started from {h['previous_sha256']}, the committed file is "
                             f"{lines_sha256(cur)}; re-run the refreeze on the current items")
        new = _lines(src / f"{u}.jsonl")
        if lines_sha256(new) != h["sha256"] or len(new) != h["n_items"]:
            raise SystemExit(f"{u}.jsonl in {src} does not match the job's summary hash")
        a, b = [json.loads(x) for x in cur], [json.loads(x) for x in new]
        for x, y in zip(a, b, strict=True):
            keep = ("item_id", "split")
            if any(x[k] != y[k] for k in keep) or any(
                    x["meta"].get(k) != y["meta"].get(k) for k in ("scenario", "seed", "prefix", "source_item_id")):
                raise SystemExit(f"{u}: {x['item_id']} changed identity, not only twin values")
        if new != cur:
            old[u] = {"sha256": lines_sha256(cur), "test_sha256": freeze.select(u, "test")[1]}
    if not old:
        return {"changed": [], "note": "the job reproduced the committed items exactly; nothing to adopt"}
    sup = freeze.load_superseded()
    sup.append({"run_id": run_id, "reason": f"re-frozen on the job platform ({summary['platform']['system']} "
                f"{summary['platform']['machine']}, torch {summary['platform']['versions'].get('torch')}) because the "
                "frozen twin values did not replay there exactly (t1/evals/replay_job.py check); same item ids, "
                "splits, scenarios, seeds and prefixes", "units": old})
    freeze.SUPERSEDED_FILE.write_text(json.dumps(sup, indent=2, sort_keys=True) + "\n")
    plat = freeze.frozen_on() or {}
    history = plat.get("previous", []) + ([{k: v for k, v in plat.items() if k != "previous"}] if plat else [])
    freeze.PLATFORM_FILE.write_text(json.dumps({
        "frozen_on": summary["platform"], "how": f"python -m t1.evals.replay_job refreeze, run {run_id}",
        "changes": {k: summary.get(k) for k in ("n_changed", "n_state_text_changed", "n_changes_changed",
                                                "n_not_kept_by_filter_here")},
        "previous": history}, indent=2, sort_keys=True) + "\n")
    for u in old:
        shutil.copyfile(src / f"{u}.jsonl", freeze.items_path(u))
    m = freeze.write_manifest()
    return {"changed": sorted(old), "new": {u: {"sha256": m["units"][u]["sha256"],
                                                "test_sha256": m["units"][u]["splits"]["test"]["sha256"]} for u in old},
            "superseded": old}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="directory with execution.jsonl, price_latency.jsonl, refreeze-summary.json")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(adopt(a.src, a.run_id), indent=2, sort_keys=True))
    errs = freeze.verify()
    print("OK: every hash matches MANIFEST.json" if not errs else f"MISMATCH {errs}")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
