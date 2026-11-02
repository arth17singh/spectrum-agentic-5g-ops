"""Timing of the replay path (Lambda tools replay on every call).

    python -m twin.bench [--out-dir twin/reports]

Measures, on this machine (a laptop, NOT a Lambda: scale by your Lambda's CPU share):
  build_cold_ms       first ReplayTwin(...) in the process: imports ran_rl/torch, loads the GP models (once per Lambda container)
  build_warm_ms       later ReplayTwin(...) with reuse_models=True (models shared)
  build_no_reuse_ms   ReplayTwin(..., reuse_models=False)
  reset_ms            reset()
  step_ms             one apply() (filter + advance + observe + KPIs), mean / p50 / p95 over many steps
  replay_{5,10,20}_ms from_log for a logged episode of that many steps, warm and with verification,
                      i.e. what a tool call pays: build + reset + N steps. Median of 20 repeats.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / "reports"


def _ms(f, repeat=1):
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def measure() -> dict:
    from twin.replay import ReplayTwin, sdk_version

    change = lambda i: [{"cell": i % 4, "d_power_db": (i % 5) - 2.0, "d_tilt_deg": 1.0 - (i % 3)}]

    cold = _ms(lambda: ReplayTwin("nominal", 1))[0]
    warm = _ms(lambda: ReplayTwin("nominal", 1), 20)
    noreuse = _ms(lambda: ReplayTwin("nominal", 1, reuse_models=False), 5)
    t = ReplayTwin("nominal", 1)
    resets = _ms(t.reset, 20)
    t.reset()
    steps = _ms(lambda: t.apply(change(t.step)), 40)

    replay = {}
    for n in (5, 10, 20):
        live = ReplayTwin("nominal", 3)
        live.reset()
        for i in range(n):
            live.apply(change(i))
        log = live.action_log
        for verify in (True, False):
            times = _ms(lambda: ReplayTwin.from_log("nominal", 3, None, log, verify=verify), 20)
            replay[f"replay_{n}_steps_{'verified' if verify else 'unverified'}_ms"] = round(statistics.median(times), 1)
        times = _ms(lambda: ReplayTwin.from_log("nominal", 3, None, log, reuse_models=False), 5)
        replay[f"replay_{n}_steps_no_model_reuse_ms"] = round(statistics.median(times), 1)

    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    return {
        "machine": f"{platform.system()} {platform.machine()}, python {platform.python_version()}",
        "ran_rl": sdk_version(),
        "load_average_1m": round(os.getloadavg()[0], 1),  # a loaded machine inflates every number
        "build_cold_ms": round(cold, 1),
        "build_warm_ms": round(statistics.median(warm), 2),
        "build_no_reuse_ms": round(statistics.median(noreuse), 1),
        "reset_ms": round(statistics.median(resets), 1),
        "step_ms": {
            "mean": round(statistics.mean(steps), 2),
            "p50": round(statistics.median(steps), 2),
            "p95": round(q(steps, 0.95), 2),
            "n": len(steps),
        },
        **replay,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(REPORTS))
    args = ap.parse_args(argv)
    report = {
        "kind": "twin-timing",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **measure(),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"timing-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
