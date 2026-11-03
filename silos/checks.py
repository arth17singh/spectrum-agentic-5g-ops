"""Data checks that must pass before any training on the silo datasets.

    python -m silos.checks --data <dir> [--manifest silos/manifest.json]    exit 0 = all pass, 1 = any fail

Checks: manifest sha256 and row counts; exact schema; no forbidden fields; fixed value ranges;
no-op/zero-change consistency; seeds (train vs test classes by hash, no overlap between any train
file and any test file, no seed shared between silos); per-silo row counts proportional to cluster
counts; operating points actually enforced (start power in each silo's region, Viavi-like start
exactly 40 dBm and 6 deg); constants per silo; shared_test equals Viavi-like slices plus the silo
test slices.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from silos.features import (
    COLUMNS,
    CONTRACT_VERSION,
    FORBIDDEN_ANYWHERE,
    FORBIDDEN_INPUTS,
    INPUT_COLUMNS,
    LABEL_RANGES,
    LABELS,
    N_CELLS,
    input_range,
)
from silos.generate import DEFAULT_CONFIG, DEFAULT_MANIFEST, config_sha256, file_sha256, load_config, seed_hash, split_of

TOL = 1e-9
CHANGE_COLS = [f"{f}[{i}]" for f in ("d_power_db", "d_tilt_deg") for i in range(N_CELLS)]


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, bool(ok), detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.results)


def _read(data: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(data / rel)


def run_checks(data: Path, manifest_path: Path = DEFAULT_MANIFEST, config_path: Path | None = DEFAULT_CONFIG) -> Report:
    rep = Report()
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except Exception as e:  # noqa: BLE001
        rep.add("manifest readable", False, repr(e))
        return rep
    cfg = manifest["config"]
    files = manifest["files"]
    rep.add("manifest config hash", config_sha256(cfg) == manifest["config_sha256"], manifest["config_sha256"][:12])
    if config_path is not None and Path(config_path).exists():
        rep.add("manifest config equals silos/config.yaml", config_sha256(load_config(config_path)) == manifest["config_sha256"],
                "data is stale if the config changed after generation")
    rep.add("manifest columns equal the code's schema", manifest["columns"] == COLUMNS)

    # 1. files, hashes, rows ------------------------------------------------------------------
    frames: dict[str, pd.DataFrame] = {}
    bad: list[str] = []
    for f in files:
        p = data / f["path"]
        if not p.exists():
            bad.append(f'{f["path"]}: missing')
            continue
        if file_sha256(p) != f["sha256"]:
            bad.append(f'{f["path"]}: sha256 differs from the manifest')
            continue
        df = pd.read_csv(p)
        if len(df) != f["rows"]:
            bad.append(f'{f["path"]}: {len(df)} rows, manifest says {f["rows"]}')
            continue
        frames[f["path"]] = df
    rep.add("every file exists, sha256 and rows match the manifest", not bad, "; ".join(bad) or f"{len(files)} files")
    by_path = {f["path"]: f for f in files}
    ok_files = [p for p in by_path if p in frames]

    def per_file(name: str, fn: Callable[[str, pd.DataFrame, dict[str, Any]], str | None]) -> None:
        problems = []
        for p in ok_files:
            msg = fn(p, frames[p], by_path[p])
            if msg:
                problems.append(f"{p}: {msg}")
        rep.add(name, not problems, "; ".join(problems[:6]) or f"{len(ok_files)} files")

    # 2. schema, forbidden fields -------------------------------------------------------------
    per_file("schema: header equals the contract columns exactly",
             lambda p, df, f: None if list(df.columns) == COLUMNS else f"header differs: {sorted(set(df.columns) ^ set(COLUMNS))[:6]}")

    def forbidden(p: str, df: pd.DataFrame, f: dict[str, Any]) -> str | None:
        cols = {c.split("[")[0] for c in df.columns}
        anywhere = cols & FORBIDDEN_ANYWHERE
        inputs = {c.split("[")[0] for c in df.columns[: len(INPUT_COLUMNS)]} & FORBIDDEN_INPUTS
        extra_inputs = set(df.columns[: len(INPUT_COLUMNS)]) ^ set(INPUT_COLUMNS)
        bad_ = sorted(anywhere | inputs | extra_inputs)
        return f"forbidden or unexpected input fields {bad_}" if bad_ else None

    per_file("no forbidden fields (hour, phase, alarm, asleep, extra, counts, seed, silo, ... never inputs)", forbidden)

    # 3. ranges --------------------------------------------------------------------------------
    def ranges(p: str, df: pd.DataFrame, f: dict[str, Any]) -> str | None:
        bad_ = []
        for c in INPUT_COLUMNS:
            lo, hi = input_range(c)
            v = df[c].to_numpy(dtype=float)
            if not np.isfinite(v).all() or v.min() < lo - TOL or v.max() > hi + TOL:
                bad_.append(c)
        for c in LABELS:
            lo, hi = LABEL_RANGES[c]
            v = df[c].to_numpy(dtype=float)
            if not np.isfinite(v).all() or v.min() < lo - TOL or v.max() > hi + TOL:
                bad_.append(c)
        sh = df[[f"user_share[{i}]" for i in range(N_CELLS)]].to_numpy(dtype=float).sum(axis=1)
        if not np.allclose(sh, 1.0, atol=1e-6):
            bad_.append("user_share rows do not sum to 1")
        if (df["contract_version"].astype(str) != CONTRACT_VERSION).any():
            bad_.append("contract_version")
        return f"out of range or not finite: {bad_[:6]}" if bad_ else None

    per_file("fixed ranges (inputs, labels, user_share sums to 1, contract version)", ranges)

    def noop(p: str, df: pd.DataFrame, f: dict[str, Any]) -> str | None:
        z = df["change_kind"] == "zero"
        ch = df[CHANGE_COLS].to_numpy(dtype=float)
        lab = df[list(LABELS)].to_numpy(dtype=float)
        if z.sum() == 0 and len(df) >= 200:
            return "no zero-change rows"
        if (np.abs(ch[z.to_numpy()]) > 0).any() or (np.abs(lab[z.to_numpy()]) > 0).any():
            return "zero-change rows with a change or a non-zero delta label"
        nz = ch[~z.to_numpy()]
        if len(nz) and not (np.abs(nz).sum(axis=1) > 0).all():
            return "a change row with no change"
        if (df["clipped"] != 0).any():
            return "a generated change was clipped by the safety filter"
        return None

    per_file("zero-change rows have zero deltas; every other row has a change; none clipped", noop)

    # 4. seeds ----------------------------------------------------------------------------------
    seeds_ok, seed_msgs = True, []
    test_seeds: set[int] = set()
    for p in ok_files:
        f, df = by_path[p], frames[p]
        s_df = sorted(int(s) for s in df["seed"].unique())
        s_mf = [e["seed"] for e in f["seeds"]]
        if s_df != s_mf:
            seeds_ok = False
            seed_msgs.append(f"{p}: seeds in the file differ from the manifest")
        for e in f["seeds"]:
            if e["sha256"] != seed_hash(e["seed"]) or e["class"] != split_of(e["seed"]):
                seeds_ok = False
                seed_msgs.append(f'{p}: manifest hash/class wrong for seed {e["seed"]}')
        if f["role"] == "train":
            for s in s_df:
                if split_of(s) != "train":
                    seeds_ok = False
                    seed_msgs.append(f"{p}: seed {s} is a test seed by hash")
        else:
            for s in s_df:
                test_seeds.add(s)
                if split_of(s) != "test":
                    seeds_ok = False
                    seed_msgs.append(f"{p}: seed {s} is a train seed by hash")
    rep.add("seeds: train files hold only train-hash seeds, test files only test-hash seeds", seeds_ok, "; ".join(seed_msgs[:5]))
    leak = []
    for p in ok_files:
        if by_path[p]["role"] == "train":
            both = set(int(s) for s in frames[p]["seed"].unique()) & test_seeds
            if both:
                leak.append(f"{p}: {sorted(both)[:3]}")
    rep.add("no seed overlap between any train file and any test file", not leak, "; ".join(leak))
    dup = []
    for split in ("disjoint", "overlap"):
        seen: dict[int, str] = {}
        for p in ok_files:
            f = by_path[p]
            if f["split"] == split and f["role"] == "train":
                for s in frames[p]["seed"].unique():
                    if int(s) in seen:
                        dup.append(f"{split}: seed {int(s)} in {seen[int(s)]} and {f['silo']}")
                    seen[int(s)] = f["silo"]
    rep.add("no seed shared between two silos' training data", not dup, "; ".join(dup))

    # 5. proportional row counts ---------------------------------------------------------------
    prop_bad = []
    for split in ("disjoint", "overlap"):
        tr = {by_path[p]["silo"]: by_path[p] for p in ok_files if by_path[p]["split"] == split and by_path[p]["role"] == "train"}
        for silo, f in tr.items():
            want_clusters = cfg["silos"][silo]["clusters"]
            if f["clusters"] != want_clusters or f["rows"] != want_clusters * cfg["steps_per_cluster"]:
                prop_bad.append(f'{split}/{silo}: {f["rows"]} rows, {f["clusters"]} clusters, want {want_clusters} x {cfg["steps_per_cluster"]}')
        if len(tr) == len(cfg["silos"]):
            per = {round(f["rows"] / f["clusters"], 6) for f in tr.values()}
            if len(per) != 1:
                prop_bad.append(f"{split}: rows per cluster differ between silos {sorted(per)}")
    rep.add("per-silo training rows are proportional to cluster counts", not prop_bad and bool(ok_files), "; ".join(prop_bad))

    # 6. operating points ----------------------------------------------------------------------
    op_bad = []
    for p in ok_files:
        f, df = by_path[p], frames[p]
        pw = df[[f"power_dbm[{i}]" for i in range(N_CELLS)]].to_numpy(dtype=float)
        tl = df[[f"tilt_deg[{i}]" for i in range(N_CELLS)]].to_numpy(dtype=float)
        if f["silo"] in cfg["silos"]:
            s = cfg["silos"][f["silo"]]
            lo, hi = s["power_start_dbm"] if f["split"] == "disjoint" else cfg["overlap_power_start_dbm"]
            m = cfg["drift_margin_db"]
            if df["start_power_dbm"].min() < lo - TOL or df["start_power_dbm"].max() > hi + TOL:
                op_bad.append(f"{p}: start power outside {lo}-{hi}")
            if pw.min() < max(10.0, lo - m) - TOL or pw.max() > min(46.0, hi + m) + TOL:
                op_bad.append(f"{p}: power drifted outside {lo}-{m}..{hi}+{m}")
            if set(df["n_ues"]) != {s["n_ues"]} or set(df["noise_floor_dbm"]) != {s["noise_floor_dbm"]} or set(df["moving"]) != {int(s["mobility"] == "moving")}:
                op_bad.append(f"{p}: users or noise floor differ from the silo's fixed settings")
        elif f["silo"] == "shared":
            v = cfg["test"]["viavi_like"]
            for seg, d in df.groupby("segment"):
                if seg not in v["slices"]:
                    continue
                first = d[d["step"] % v["episode_len"] == 0]
                p0 = first[[f"power_dbm[{i}]" for i in range(N_CELLS)]].to_numpy(dtype=float)
                t0 = first[[f"tilt_deg[{i}]" for i in range(N_CELLS)]].to_numpy(dtype=float)
                if not (np.allclose(p0, v["power_dbm"]) and np.allclose(t0, v["tilt_deg"])):
                    op_bad.append(f"{p}:{seg}: episode starts are not exactly {v['power_dbm']} dBm / {v['tilt_deg']} deg")
                if set(d["n_ues"]) != {v["n_ues"]} or set(d["moving"]) != {0}:
                    op_bad.append(f"{p}:{seg}: not 60 stationary users")
                if set(d["noise_floor_dbm"]) != {v["slices"][seg]["noise_floor_dbm"]}:
                    op_bad.append(f"{p}:{seg}: wrong noise floor")
    rep.add("operating points enforced (start power in region, bounded drift, fixed users and noise, Viavi-like start 40 dBm / 6 deg)", not op_bad, "; ".join(op_bad[:5]))
    d_rng = [tuple(cfg["silos"][s]["power_start_dbm"]) for s in sorted(cfg["silos"])]
    rep.add("disjoint split: the silos' start ranges are ordered and do not overlap (they may touch)",
            all(a[1] <= b[0] for a, b in zip(sorted(d_rng), sorted(d_rng)[1:])), str(d_rng))

    # 7. shared_test ----------------------------------------------------------------------------
    sh_bad = []
    for split in ("disjoint", "overlap"):
        p = f"{split}/shared_test/test.csv"
        if p not in frames:
            sh_bad.append(f"{p} unreadable")
            continue
        sh = frames[p]
        want = set(cfg["test"]["viavi_like"]["slices"]) | {f"region_{s}" for s in cfg["silos"]}
        if set(sh["segment"]) != want:
            sh_bad.append(f"{p}: segments {sorted(set(sh['segment']))} != {sorted(want)}")
        for silo in cfg["silos"]:
            q = f"{split}/{silo}/test.csv"
            if q in frames:
                a = sh[sh["segment"] == f"region_{silo}"].reset_index(drop=True)
                if not a.equals(frames[q].reset_index(drop=True)):
                    sh_bad.append(f"{p}: region_{silo} rows differ from {q}")
    rep.add("shared_test = Viavi-like slices + each silo's test slice", not sh_bad, "; ".join(sh_bad))
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    a = ap.parse_args(argv)
    rep = run_checks(a.data, a.manifest, a.config)
    for name, ok, detail in rep.results:
        print(f'{"PASS" if ok else "FAIL"}  {name}' + (f"  [{detail}]" if detail else ""))
    print("ALL CHECKS PASSED" if rep.ok else "CHECKS FAILED")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
