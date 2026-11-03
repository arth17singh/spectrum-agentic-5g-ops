"""Generate, freeze and verify the item sets.

    python -m t1.evals.freeze                  # regenerate every unit, write items/*.jsonl + MANIFEST.json
    python -m t1.evals.freeze --units interface safety
    python -m t1.evals.freeze --verify         # recompute every hash and compare with MANIFEST.json

Each line of items/<unit>.jsonl is one item in canonical JSON (sorted keys, compact). The MANIFEST
records each file's sha256 and, per split (and per knowledge group), the sha256 of exactly those
lines in file order: that is the `test_set_sha256` a declaration carries. Execution needs ran-rl
(the in-process Maveric twin); knowledge needs the pinned ot-lite files (downloaded once into
--source-cache and checked against their pinned sha256).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from t1.evals import common
from t1.evals.common import ITEMS_DIR, dumps, lines_sha256, sha256_text
from t1.evals.graders import GRADER_VERSION
from t1.evals.parse import PARSER_VERSION
from t1.evals.prompting import prompt_sha256

MANIFEST = ITEMS_DIR / "MANIFEST.json"
DEFAULT_CACHE = Path.home() / ".cache" / "t1-evals" / "ot-lite"
GENERATORS = {
    "interface": "t1/evals/gen_interface.py",
    "execution": "t1/evals/gen_execution.py",
    "safety": "t1/evals/gen_safety.py",
    "knowledge": "t1/evals/gen_knowledge.py",
    "price_latency": "t1/evals/gen_price_latency.py",
}


def items_path(unit: str) -> Path:
    return ITEMS_DIR / f"{unit}.jsonl"


def read_lines(unit: str) -> list[str]:
    text = items_path(unit).read_text(encoding="utf-8")
    return [line for line in text.split("\n") if line != ""]


def load_items(unit: str) -> list[dict]:
    return [json.loads(line) for line in read_lines(unit)]


def select(unit: str, split: str = "test", group: str | None = None) -> tuple[list[dict], str]:
    """(items, sha256 of exactly their lines in file order) for a split ('dev', 'test' or 'all')
    and, for knowledge, a group ('core', 'telelogs', 'telemath' or None for all)."""
    lines = read_lines(unit)
    chosen = []
    for line in lines:
        it = json.loads(line)
        if split != "all" and it["split"] != split:
            continue
        if group is not None and it["meta"].get("group") != group:
            continue
        chosen.append((line, it))
    return [it for _, it in chosen], lines_sha256([line for line, _ in chosen])


def _counts(items: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        v = it["meta"].get(key)
        if v is not None:
            out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def unit_manifest(unit: str) -> dict:
    lines = read_lines(unit)
    items = [json.loads(line) for line in lines]
    entry = {
        "file": f"t1/evals/items/{unit}.jsonl",
        "sha256": lines_sha256(lines),
        "n_items": len(items),
        "generator": GENERATORS[unit],
        "splits": {},
        "families": _counts(items, "family") or _counts(items, "config"),
    }
    for split in ("dev", "test"):
        sel, sha = select(unit, split)
        entry["splits"][split] = {"n": len(sel), "sha256": sha}
    if unit == "knowledge":
        entry["groups"] = {}
        for group in ("core", "telelogs", "telemath"):
            entry["groups"][group] = {}
            for split in ("dev", "test", "all"):
                sel, sha = select(unit, split, group)
                entry["groups"][group][split] = {"n": len(sel), "sha256": sha}
        from t1.evals import gen_knowledge as K

        entry["source"] = {"dataset": K.DATASET, "revision": K.REVISION, "license": K.LICENSE,
                           "files": {c: {"file": f, "sha256": s} for c, (f, s) in K.FILES.items()},
                           "take": K.TAKE}
    if unit == "execution":
        from t1.evals import gen_execution as E

        entry["twin"] = {"kind": "maveric-inprocess", "config": "official", "kpi": "+".join(E.KPI_KEYS),
                         "seed_base_reserved": E.SEED_BASE, "min_oracle_gain": E.MIN_ORACLE_GAIN,
                         "sdk_version": items[0]["meta"]["twin"]["sdk_version"] if items else None}
    return entry


def write_unit(unit: str, items: list[dict]) -> None:
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{unit}: duplicate item ids")
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    items_path(unit).write_text("".join(dumps(it) + "\n" for it in items), encoding="utf-8")


# Hashes frozen at commit 2959c9b and replaced before any model ran on them. Why: the execution
# items were built on a local stand-in twin (W2's twin.replay was not merged yet), and change
# strings followed a local canon that dropped zero-change cells; both now come from W2's ReplayTwin
# and W1's team13_contracts.change. Never declare a run against these.
SUPERSEDED = {
    "commit": "2959c9b",
    "reason": "built on twin_standin.py and a local canonical-change fallback; re-frozen on W2 "
              "twin.replay.ReplayTwin and W1 team13_contracts.change before any model was run. Same "
              "80 item ids, specs and prompt text; the reference values (noop_kpi, rule/oracle/"
              "constant gain) are now stored unrounded (they were rounded to 6 decimals, so they "
              "differed from the replay by up to 3.3e-7) and meta.twin names the W2 replay",
    "units": {
        "execution": {"sha256": "4686121964eaf4811fec665ac73927af8ee8f32ffe5375e3c4074f87dc80b72c",
                      "test_sha256": "8ac4c54536f8affdfdc75bccbed02f4d0ece78ad81e6ef0d93e44e858cdc0feb"},
    },
}


def build_manifest() -> dict:
    return {
        "manifest_version": 2,
        "superseded": SUPERSEDED,
        "split": {"salt": common.SPLIT_SALT, "dev_fraction": common.DEV_FRACTION,
                  "rule": "dev if sha256(unit, item_id, salt)[:8] % 10000 < dev_fraction * 10000"},
        "prompt": {"file": "t1/evals/prompt.txt", "sha256": prompt_sha256()},
        "parser": PARSER_VERSION,
        "graders": GRADER_VERSION,
        "units": {u: unit_manifest(u) for u in common.UNITS if items_path(u).is_file()},
    }


def write_manifest() -> dict:
    m = build_manifest()
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return m


def verify() -> list[str]:
    """Differences between MANIFEST.json and the files on disk (empty list = all hashes match)."""
    want = json.loads(MANIFEST.read_text(encoding="utf-8"))
    got = build_manifest()
    errs = []
    for key in ("prompt", "parser", "graders", "split"):
        if want.get(key) != got.get(key):
            errs.append(f"{key}: manifest {want.get(key)} != now {got.get(key)}")
    for u in sorted(set(want["units"]) | set(got["units"])):
        if want["units"].get(u) != got["units"].get(u):
            errs.append(f"unit {u}: manifest and files differ")
    return errs


def verify_unit(unit: str) -> list[str]:
    """Differences for one unit: its file hash and split hashes, plus the prompt/parser/graders."""
    want = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errs = []
    if want.get("prompt", {}).get("sha256") != prompt_sha256():
        errs.append("prompt.txt sha256 differs from MANIFEST")
    if want.get("parser") != PARSER_VERSION or want.get("graders") != GRADER_VERSION:
        errs.append("parser/grader version differs from MANIFEST")
    w = want["units"].get(unit)
    if w is None:
        return errs + [f"{unit} not in MANIFEST"]
    lines = read_lines(unit)
    if lines_sha256(lines) != w["sha256"] or len(lines) != w["n_items"]:
        errs.append(f"{unit}.jsonl sha256 differs from MANIFEST")
    return errs


def generate(unit: str, cache: Path, existing: dict[str, list[dict]]) -> list[dict]:
    if unit == "interface":
        from t1.evals import gen_interface

        return gen_interface.generate()
    if unit == "safety":
        from t1.evals import gen_safety

        return gen_safety.generate()
    if unit == "execution":
        from t1.evals import gen_execution

        return gen_execution.generate()
    if unit == "knowledge":
        from t1.evals import gen_knowledge

        return gen_knowledge.generate(gen_knowledge.fetch(cache))
    if unit == "price_latency":
        from t1.evals import gen_price_latency

        src = {u: existing.get(u) or load_items(u) for u in gen_price_latency.MIX}
        return gen_price_latency.generate(src)
    raise ValueError(unit)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="*", default=list(common.UNITS))
    ap.add_argument("--source-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        errs = verify()
        for e in errs:
            print("MISMATCH", e)
        print("OK: every hash matches MANIFEST.json" if not errs else f"{len(errs)} mismatches")
        return 0 if not errs else 1
    done: dict[str, list[dict]] = {}
    for unit in common.UNITS:  # fixed order: price_latency last, it draws from the others
        if unit not in a.units:
            continue
        items = generate(unit, a.source_cache, done)
        write_unit(unit, items)
        done[unit] = items
        print(f"{unit}: {len(items)} items -> {items_path(unit)}")
    m = write_manifest()
    for u, e in m["units"].items():
        print(f"{u:14s} n={e['n_items']:4d} dev={e['splits']['dev']['n']:4d} test={e['splits']['test']['n']:4d} "
              f"sha256={e['sha256']}")
    print("prompt sha256", m["prompt"]["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
