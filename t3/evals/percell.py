"""Per-cell success, percell-v1: every cell back near its own healthy state, and no neighbour hurt by the fix.

The single definition of per-cell success that the Live Fix grader and the remediation-policy evaluation
share. The network-level grader (``graders.grade_kpis``) looks at network totals only, so a broad power
boost can lift those totals above the healthy network while neighbours end worse than their own healthy
state. Live Fix run ``t3.livefix-20261104T073206Z-9695`` is the case that prompted this: network recovery
1.00, cells 0 and 3 degraded. percell-v1 judges each cell against itself.

Every threshold comes from ``faults.cell_health`` and ``faults.HEALTH_THRESHOLDS``; no number is restated
here.

- **health:** each cell after the fix, against the same cell in the healthy counterfactual at the same
  step: green, amber or red (``faults.cell_health``).
- **recovered:** every cell green.
- **degraded:** the cells that are not green.
- **worse_than_before** (needs ``cells_before``, the state right before the fix): the cell is not green
  after the fix, and either its health level against healthy got worse, or one of coverage share, SLA
  share or p10 SINR dropped from before the fix to after it by at least the amber line
  (``faults.cell_health(after, reference=before)``), or it newly serves no UEs, or it newly went into
  alarm or asleep. A cell that ends green is back at its own healthy state and is never counted: during
  a fault a neighbour can look better than healthy on a KPI (it briefly gained the faulted cell's
  users), and returning from that is not harm.
- **collateral:** the cells worse than before, excluding the faulted cells (``fault_cells``; when it is
  not given, no cell is excluded).
- **overshoot:** power above healthy (the sum over cells of positive after-minus-healthy power, dB); power
  and tilt away from healthy (sums of absolute differences); and network coverage after minus healthy.
  That last one comes from network headlines when they are passed, else it is the UE-weighted mean of the
  cells' coverage shares, else None.

    v = percell_verdict(cells_after, cells_healthy, cells_before, fault_cells=[2])
    v = grade_incident(incident, ["cell=2;d_power_db=+6.0;d_tilt_deg=+0.0", "", ""])
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from t3.evals import faults
from t3.evals import incidents as inc
from t3.evals.changes import nonzero
from t3.evals.graders import load_targets

VERSION = "percell-v1"
_ORDER = {"green": 0, "amber": 1, "red": 2}
_KPI_KEYS = ("coverage_share", "sla_share", "sinr_p10_db")


class _KpisView:
    """Lets ``faults._cell_rows`` read an already computed ``kpis()`` result."""

    def __init__(self, k: Mapping[str, Any]) -> None:
        self._k = k

    def kpis(self) -> Mapping[str, Any]:
        return self._k


def cells_from_twin(twin_or_state: Any) -> list[dict[str, Any]]:
    """Per-cell rows in the shape ``faults.make_incident`` gives ``cells_before`` and ``cells_after``.

    Accepts a twin with ``.kpis()`` (``ReplayTwin``), a ``kpis()`` result (it has ``per_cell``), or a raw
    twin state, to which ``twin.kpis.kpis`` is applied. A raw state has no per-cell coverage or SLA
    shares (the twin computes them from its UEs), so those keys are None and only SINR, UEs and alarms
    are judged."""
    if hasattr(twin_or_state, "kpis") and callable(twin_or_state.kpis):
        return faults._cell_rows(twin_or_state)
    if isinstance(twin_or_state, Mapping):
        if "per_cell" in twin_or_state:
            return faults._cell_rows(_KpisView(twin_or_state))
        from twin.kpis import kpis as twin_kpis
        return faults._cell_rows(_KpisView(twin_kpis(dict(twin_or_state))))
    raise TypeError(f"cells_from_twin needs a twin, a kpis() result or a twin state, not {type(twin_or_state)!r}")


def _by_cell(rows: Iterable[Mapping[str, Any]], name: str) -> dict[int, Mapping[str, Any]]:
    out = {int(r["cell"]): r for r in rows}
    if not out:
        raise ValueError(f"{name} is empty")
    return out


def _f(v: Any) -> float | None:
    return None if v is None else float(v)


def _ue_weighted_coverage(rows: Iterable[Mapping[str, Any]]) -> float | None:
    num = den = 0.0
    for r in rows:
        share, n = r.get("coverage_share"), r.get("ue_count")
        if share is None or not n:
            continue
        num += float(share) * float(n)
        den += float(n)
    return num / den if den else None


def percell_verdict(cells_after: list[Mapping[str, Any]], cells_healthy: list[Mapping[str, Any]],
                    cells_before: list[Mapping[str, Any]] | None = None, *,
                    thresholds: Mapping[str, Any] | None = None, fault_cells: Iterable[int] | None = None,
                    network_after: Mapping[str, Any] | None = None,
                    network_healthy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The percell-v1 verdict (see the module docstring). ``thresholds`` defaults to
    ``faults.HEALTH_THRESHOLDS`` (same shape: ``{"red": {...}, "amber": {...}}``)."""
    after = _by_cell(cells_after, "cells_after")
    healthy = _by_cell(cells_healthy, "cells_healthy")
    if set(after) != set(healthy):
        raise ValueError(f"cells_after {sorted(after)} and cells_healthy {sorted(healthy)} name different cells")
    before = _by_cell(cells_before, "cells_before") if cells_before is not None else None
    if before is not None and set(before) != set(after):
        raise ValueError(f"cells_before {sorted(before)} and cells_after {sorted(after)} name different cells")
    th = thresholds or faults.HEALTH_THRESHOLDS
    faulted = {int(c) for c in fault_cells} if fault_cells is not None else set()
    cells = sorted(after)

    vs_healthy = {h["cell"]: h for h in faults.cell_health([after[c] for c in cells],
                                                           [healthy[c] for c in cells], thresholds=th)}
    was = ({h["cell"]: h for h in faults.cell_health([before[c] for c in cells],
                                                     [healthy[c] for c in cells], thresholds=th)}
           if before is not None else None)
    vs_before = ({h["cell"]: h for h in faults.cell_health([after[c] for c in cells],
                                                           [before[c] for c in cells], thresholds=th)}
                 if before is not None else None)

    out_cells: list[dict[str, Any]] = []
    for c in cells:
        h = vs_healthy[c]
        reasons = [f"vs healthy: {w}" for w in h["why"]]
        worse: bool | None = None
        if before is not None:
            assert was is not None and vs_before is not None
            drops: list[str] = []
            if _ORDER[h["health"]] > _ORDER[was[c]["health"]]:
                drops.append(f"health {was[c]['health']} before the fix, {h['health']} after")
            for w in vs_before[c]["why"]:
                newly = (w == "alarm" and not before[c].get("alarm")) or \
                        (w == "asleep" and not before[c].get("asleep"))
                if w.split(" ")[0] in _KPI_KEYS or w == "serves no UEs" or newly:
                    drops.append(f"vs before the fix: {w}")
            # A cell that ends green is back at its own healthy state: a KPI that falls from a
            # fault-time high (a neighbour that briefly gained the faulted cell's users) is not harm.
            worse = bool(drops) and h["health"] != "green"
            reasons += drops if worse else [f"{d} (back to healthy: not counted)" for d in drops]
        out_cells.append({"cell": c, "health": h["health"], "worse_than_before": worse,
                          "fault_cell": (c in faulted) if fault_cells is not None else None,
                          "reasons": reasons})

    pw = [(_f(after[c].get("power_dbm")), _f(healthy[c].get("power_dbm"))) for c in cells]
    tl = [(_f(after[c].get("tilt_deg")), _f(healthy[c].get("tilt_deg"))) for c in cells]
    pw = [(a, b) for a, b in pw if a is not None and b is not None]
    tl = [(a, b) for a, b in tl if a is not None and b is not None]
    if network_after is not None and network_healthy is not None \
            and network_after.get("coverage_ratio") is not None \
            and network_healthy.get("coverage_ratio") is not None:
        cov = float(network_after["coverage_ratio"]) - float(network_healthy["coverage_ratio"])
        cov_src = "network"
    else:
        ca = _ue_weighted_coverage(after[c] for c in cells)
        ch = _ue_weighted_coverage(healthy[c] for c in cells)
        cov, cov_src = ((ca - ch), "ue-weighted-cells") if ca is not None and ch is not None else (None, None)
    return {
        "version": VERSION,
        "recovered": all(oc["health"] == "green" for oc in out_cells),
        "cells": out_cells,
        "degraded": [oc["cell"] for oc in out_cells if oc["health"] != "green"],
        "collateral": [oc["cell"] for oc in out_cells if oc["worse_than_before"] and oc["cell"] not in faulted],
        "overshoot": {
            "power_above_healthy_db": round(sum(max(0.0, a - b) for a, b in pw), 6),
            "power_off_healthy_db": round(sum(abs(a - b) for a, b in pw), 6),
            "tilt_off_healthy_deg": round(sum(abs(a - b) for a, b in tl), 6),
            "coverage_above_healthy": None if cov is None else round(cov, 6),
            "coverage_source": cov_src,
        },
        "thresholds": "faults.HEALTH_THRESHOLDS" if thresholds is None else "custom",
    }


def grade_incident(incident: Mapping[str, Any], proposals: Iterable[Any], targets: dict[str, Any] | None = None,
                   *, twin_cls: Any = None) -> dict[str, Any]:
    """percell-v1 for one incident and the changes applied at its operator steps.

    Replays the incident twice on this machine, the same way ``lessons.verify`` does: the healthy
    counterfactual (no fault, no-op operator steps) and the faulted run with ``proposals`` (one change per
    operator step; "", None or [] is a no-op). It grades at the step after the last operator step (the
    grade step). ``cells_before`` is the faulted state right before the first operator step."""
    targets = targets or load_targets()
    ep = targets["episode"]
    if twin_cls is None:
        from t3.evals.twin_backend import ReplayTwin as twin_cls
    props = [nonzero(p) for p in proposals][: ep["operator_steps"]]
    props += [[]] * (ep["operator_steps"] - len(props))

    def run(with_fault: bool, changes: list[list[dict[str, Any]]]):
        t = twin_cls(incident["scenario"], incident["seed"], incident["preset"])
        t.reset()
        for ch in inc.script(dict(incident), targets, with_fault):
            t.apply(ch)
        rows_before = cells_from_twin(t)
        for ch in changes:
            t.apply(ch)
        return rows_before, cells_from_twin(t), t.kpis(), t.step

    _, healthy, k_healthy, s_healthy = run(False, [[]] * len(props))
    before, after, k_after, s_after = run(True, props)
    if s_healthy != s_after:
        raise RuntimeError(f"healthy and fixed runs ended at different steps ({s_healthy} vs {s_after})")
    fault_cells = incident.get("fault_cells")
    if fault_cells is None:
        fault_cells = sorted({int(f["cell"]) for f in incident.get("fault", [])})
    v = percell_verdict(after, healthy, before, fault_cells=fault_cells,
                        network_after=k_after, network_healthy=k_healthy)
    v["grade_step"] = s_after
    v["cells_healthy"], v["cells_before"], v["cells_after"] = healthy, before, after
    return v
