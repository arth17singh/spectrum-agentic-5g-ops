"""Parity: the organisers' adapter vs ReplayTwin, same scenario / seed / actions, every state field.

    python -m twin.parity --twinhost <twin-host checkout> [--python <venv python>] [--port 8796]

(run from the repo root with ``-m``: a file named twin/http.py would shadow the stdlib ``http``
module if twin/ itself were on sys.path.)

It starts ``python -m ran_rl_host.adapter.serve --backend maveric`` on 127.0.0.1 with PYTHONPATH set
to <twinhost>/src, a throwaway random token that exists only in that process's environment, and the
model and topology paths the twin-host README uses (<twinhost>/models/gp/sim_tilt_sweep_4cell and
<twinhost>/data/topologies/sim_tilt_sweep_4cell/topology.csv). For each case it drives the adapter
through ``AdapterClient`` and an in-process ``ReplayTwin`` (shipped assets of the installed SDK) with the
same changes, and compares reset and every post-step state recursively. Numbers must agree to 1e-6
relative; booleans, integers and strings exactly. It also checks that the two action logs agree
(filtered vectors, rule ids, KPIs) and that ``ReplayTwin.from_log`` of the log lands on a state equal
to the live one. The adapter is killed afterwards, also on error.

The report goes to twin/reports/parity-<UTC timestamp>.json. It contains no token, URL or path.
Exit code 0 only if every case passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from twin.http import AdapterClient
from twin.replay import ReplayTwin, sdk_version

RTOL = 1e-6
ATOL = 1e-12
REPORTS = Path(__file__).resolve().parent / "reports"

CASES = [
    {"scenario": "nominal", "seed": 7, "steps": 12},
    {"scenario": "congestion", "seed": 11, "steps": 12},
    {"scenario": "outage_cell_2", "seed": 3, "steps": 12},  # cell 0 alarmed from step 6: freeze rule
]


def make_changes(seed: int, steps: int, n_cells: int = 4) -> list[list[dict[str, Any]]]:
    """A deterministic action sequence with in-range, out-of-range and empty steps."""
    rng = np.random.default_rng(1000 + seed)
    out: list[list[dict[str, Any]]] = []
    for i in range(steps):
        if i == 3:
            out.append([])  # a no-op step
            continue
        cells = rng.choice(n_cells, size=int(rng.integers(1, 4)), replace=False)
        change = [
            {
                "cell": int(c),
                "d_power_db": round(float(rng.uniform(-10, 12)), 3),  # filter bounds -6..8
                "d_tilt_deg": round(float(rng.uniform(-7, 7)), 3),  # filter bounds +-5
            }
            for c in cells
        ]
        if i == 5 or i == 8:
            change.append({"cell": 0, "d_power_db": 2.0, "d_tilt_deg": 1.0})  # alarm_freeze target
        out.append(change)
    return out


def compare(a: Any, b: Any, path: str, stats: dict[str, Any]) -> None:
    """Recursive compare; updates stats (fields, max_rel, worst, mismatches)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                stats["mismatches"].append(f"{path}.{key}: present on one side only")
            else:
                compare(a[key], b[key], f"{path}.{key}", stats)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            stats["mismatches"].append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{path}[{i}]", stats)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        stats["fields"] += 1
        if a != b:
            stats["mismatches"].append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        stats["fields"] += 1
        diff = abs(a - b)
        scale = max(abs(a), abs(b))
        rel = 0.0 if diff == 0 else diff / scale if scale > 0 else math.inf
        if rel > stats["max_rel"]:
            stats["max_rel"], stats["worst"] = rel, path
        if diff > RTOL * scale + ATOL:
            stats["mismatches"].append(f"{path}: {a!r} vs {b!r} (rel {rel:.3g})")
    else:
        stats["mismatches"].append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")


def run_case(client: AdapterClient, case: dict[str, Any]) -> dict[str, Any]:
    scenario, seed, steps = case["scenario"], case["seed"], case["steps"]
    changes = make_changes(seed, steps)
    stats: dict[str, Any] = {"fields": 0, "max_rel": 0.0, "worst": None, "mismatches": []}
    local = ReplayTwin(scenario, seed)
    compare(client.reset(scenario, seed), local.reset(), "reset", stats)
    for i, change in enumerate(changes):
        compare(client.apply(change), local.apply(change), f"step{i + 1}", stats)
    logs_stats: dict[str, Any] = {"fields": 0, "max_rel": 0.0, "worst": None, "mismatches": []}
    compare(client.action_log, local.action_log, "action_log", logs_stats)
    replayed = ReplayTwin.from_log(scenario, seed, None, json.loads(json.dumps(local.action_log)))
    from_log_equal = replayed.state == local.state
    clamped = sum(1 for e in local.action_log if e["clamped"])
    frozen = sum(1 for e in local.action_log if any(r.startswith("alarm_freeze") for r in e["rules"]))
    return {
        "scenario": scenario,
        "seed": seed,
        "steps": steps,
        "state_fields_compared": stats["fields"],
        "state_max_rel_diff": stats["max_rel"],
        "state_worst_field": stats["worst"],
        "state_mismatches": stats["mismatches"][:20],
        "action_log_fields_compared": logs_stats["fields"],
        "action_log_max_rel_diff": logs_stats["max_rel"],
        "action_log_mismatches": logs_stats["mismatches"][:20],
        "from_log_state_equal": from_log_equal,
        "steps_with_clamping": clamped,
        "steps_with_alarm_freeze": frozen,
        "passed": not stats["mismatches"] and not logs_stats["mismatches"] and from_log_equal,
    }


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def start_adapter(twinhost: Path, python: str, port: int, token: str, log_path: Path) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")}
    env.update(
        PYTHONPATH=str(twinhost / "src"),
        RAN_RL_ADAPTER_TOKEN=token,
        RAN_RL_MAVERIC_MODELS_DIR=str(twinhost / "models" / "gp" / "sim_tilt_sweep_4cell"),
        RAN_RL_MAVERIC_TOPOLOGY=str(
            twinhost / "data" / "topologies" / "sim_tilt_sweep_4cell" / "topology.csv"
        ),
    )
    log = open(log_path, "wb")
    return subprocess.Popen(
        [python, "-m", "ran_rl_host.adapter.serve", "--backend", "maveric",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def wait_healthy(url: str, proc: subprocess.Popen, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"adapter exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.3)
    raise TimeoutError("adapter did not become healthy")


def stop_adapter(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def twinhost_version(twinhost: Path) -> str | None:
    try:
        text = (twinhost / "pyproject.toml").read_text()
    except OSError:
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twinhost", default=os.environ.get("TWINHOST_DIR"), help="twin-host checkout (or TWINHOST_DIR)")
    ap.add_argument("--python", default=sys.executable, help="python for the adapter process")
    ap.add_argument("--port", type=int, default=8796)
    ap.add_argument("--out-dir", default=str(REPORTS))
    args = ap.parse_args(argv)
    if not args.twinhost:
        ap.error("--twinhost (or TWINHOST_DIR) is required")
    twinhost = Path(args.twinhost).resolve()
    if not _port_free(args.port):
        print(f"port {args.port} is in use", file=sys.stderr)
        return 2

    token = secrets.token_urlsafe(24)
    url = f"http://127.0.0.1:{args.port}"
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "adapter.log"
        proc = start_adapter(twinhost, args.python, args.port, token, log_path)
        try:
            wait_healthy(url, proc)
            with AdapterClient(url, token) as client:
                for case in CASES:
                    results.append(run_case(client, case))
        except Exception as exc:
            tail = log_path.read_text(errors="replace")[-1500:].replace(token, "<token>")
            print(f"parity run failed: {exc!r}\n--- adapter log tail ---\n{tail}", file=sys.stderr)
            return 3
        finally:
            stop_adapter(proc)

    passed = all(r["passed"] for r in results)
    report = {
        "kind": "twin-parity",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tolerance": {"relative": RTOL, "absolute": ATOL},
        "compared": "ReplayTwin (in-process, installed SDK assets) vs organisers' adapter (local, maveric backend)",
        "ran_rl_installed": sdk_version(),
        "ran_rl_adapter_source": twinhost_version(twinhost),
        "preset": "official",
        "cases": results,
        "state_fields_compared": sum(r["state_fields_compared"] for r in results),
        "max_rel_diff": max(max(r["state_max_rel_diff"], r["action_log_max_rel_diff"]) for r in results),
        "passed": passed,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"parity-{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print(f"report: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
