"""Offline rebuild: re-read evidence/, re-verify the persisted chain and every artifact hash, and
reprint the per-run summary. No network, no AWS, no signing secret: only the public key file.

Exit codes: 0 clean; 1 an artifact changed, is missing, or the ledger recorded a change; 2 the
chain, an anchor or the public key file is broken or unreadable.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from .service import build_index, summarise_runs
from .sources import LocalSource
from .store import LEDGER_FILE, PUBKEYS_FILE, PublicKeyRing, evidence_dir, load_disk, verify_chain

OK, CHANGED, BROKEN = 0, 1, 2


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rebuild(directory: Path, out=sys.stdout) -> int:
    say = lambda s="": print(s, file=out)  # noqa: E731
    directory = directory.resolve()
    say(f"evidence rebuild: {directory}")
    if not directory.is_dir():
        say(f"FAIL: {directory} is not a directory")
        return BROKEN
    records, anchors, bad_line = load_disk(directory)
    if not records and bad_line is None:
        say(f"no {LEDGER_FILE} records yet: nothing was ingested, nothing to rebuild")
        return OK
    try:
        keys = PublicKeyRing.from_file(directory / PUBKEYS_FILE)
    except (OSError, ValueError, KeyError, TypeError) as e:
        say(f"FAIL: cannot read {PUBKEYS_FILE}: {e}")
        return BROKEN

    code = OK
    v = verify_chain(records, anchors, keys, bad_line)
    if v["broken"]:
        say(f"FAIL: chain broken at #{v['broken']['seq']}: {v['broken']['reason']}")
        code = BROKEN
    elif not v["anchors_ok"]:
        say(f"FAIL: {v['anchor_problem']}")
        code = BROKEN
    else:
        say(f"chain OK: {v['total']} records, {v['anchors_checked']} Merkle anchors, signatures verified "
            f"with {PUBKEYS_FILE}")

    problems: list[str] = []
    idx = build_index(records)
    for path, e in sorted(idx.items()):
        first = e["first"]
        for ev in e["events"]:
            if ev["type"] in ("evidence.changed", "evidence.missing"):
                problems.append(f"{path}: ledger #{ev['seq']} recorded it {ev['type'].split('.')[1]} after ingest "
                                f"at #{first['seq']}")
        f = directory / path
        if not f.is_file():
            problems.append(f"{path}: MISSING (ingested at #{first['seq']})")
            continue
        got = _sha(f)
        appends = [ev for ev in e["events"] if ev["type"] == "evidence.appended"]
        if appends and not any(ev["type"] == "evidence.changed" for ev in e["events"]):
            # an append-only records file: every recorded (sha256, size) must still be the file's first bytes
            data = f.read_bytes()
            for rec in [first, *appends]:
                p = rec["payload"]
                if not isinstance(p.get("size"), int) or hashlib.sha256(data[: p["size"]]).hexdigest() != p["sha256"]:
                    problems.append(f"{path}: CHANGED: the content recorded at #{rec['seq']} is no longer the start "
                                    "of the file")
                    break
            else:
                if got != e["last_sha"]:
                    problems.append(f"{path}: grew after its last ledger record #{appends[-1]['seq']} (not ingested yet)")
            continue
        want = first["payload"]["sha256"]
        if got != want:
            problems.append(f"{path}: CHANGED since ingest at #{first['seq']} (sha256 {want[:12]}… is now {got[:12]}…)")
    for p in problems:
        say(f"FAIL: {p}")
    if problems and code == OK:
        code = CHANGED

    on_disk = {a.path for a in LocalSource(directory).list_artifacts()}
    for p in sorted(on_disk - set(idx)):
        say(f"note: {p} is not in the ledger yet (not ingested); it is not counted")

    runs, judged = summarise_runs(records, chain_ok=v["ok"])
    say()
    say(f"{'run_id':<40} {'unit':<12} {'target':<22} {'model':<22} {'artifacts':>9}  status")
    for s in runs:
        d = s["declaration"] or {}
        tgt = f"{d['target']['metric']} {d['target']['value']}" if d else "-"
        bad = [k for k, c in s["checks"].items() if c["state"] == "fail"]
        if not v["ok"]:
            status = "UNVERIFIED (chain broken)"
        else:
            status = "OK" if not bad else "FAIL " + ",".join(bad)
        if s["judged"]:
            status += " (judged)"
        say(f"{s['run_id']:<40} {d.get('unit', '-'):<12} {tgt:<22} {d.get('model_id', '-'):<22} "
            f"{s['artifacts']:>9}  {status}")
    if judged:
        say(f"judging.json lists {len(judged)} run(s): {', '.join(judged)}")
    say()
    say({OK: "rebuild OK: every artifact matches its ledger hash",
         CHANGED: "rebuild FAILED: evidence changed after it was recorded",
         BROKEN: "rebuild FAILED: the ledger itself does not verify"}[code])
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m spectrum.evidence")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rb = sub.add_parser("rebuild", help="re-verify evidence/ offline and reprint per-run results")
    rb.add_argument("--dir", type=Path, default=None, help="evidence directory (default: $SPECTRUM_EVIDENCE_DIR or <repo>/evidence)")
    args = ap.parse_args(argv)
    return rebuild(args.dir or evidence_dir())
