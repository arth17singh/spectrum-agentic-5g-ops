"""Team13's governed RAN tools: one Lambda behind the AgentCore Gateway.

    handler(event, context)

The Gateway passes the tool arguments as `event` and the tool's full name as
`context.client_context.custom["bedrockAgentCoreToolName"]` (`<target>___<tool>`). Local runs
without a client context may put the name in `event["__tool__"]`.

Tools (see catalog.py for the schemas): ran___observe, ran___propose, ran___predict,
ran___preflight, ran___apply_change and the NLM tool (default nlm13___ask). Every tool needs
`case_id`. Case state is stored per case and the twin is replayed deterministically from
(scenario, seed, config, action_log) on every call: the Lambda holds no state.

Collaborators are injected through `Deps` (tests use fakes; `Deps.from_env()` builds the real
ones lazily): the twin (W2 `twin.replay`), the predictor (W9 `t2.predict`), the RL policy
(W9 `t1.rl.policy`), the trust gate (W5 `t3.trust`) and the evidence writer (W6 `tools.evlib`).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import time
from typing import Any, Callable

from . import catalog
from .store import SAFE_ID, ConflictError

KPI_KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
DIRECTION_KEYS = ("coverage_ratio", "sla_ratio")   # direction_agree is judged on these
DEADBAND = {"coverage_ratio": 0.002, "sla_ratio": 0.002, "sinr_p10_db": 0.1, "sinr_mean_db": 0.1}
MAX_PREDICTIONS = 50
RUN_ID_LINE = re.compile(r"(?m)^\s*run_id\s*:\s*(\S+)\s*$")


class ToolError(Exception):
    """A refusal the agent should read (returned as {"error": ...}, never raised to the Gateway)."""


# ------------------------------------------------------------------ helpers
def audit(**fields) -> None:
    """One JSON line per event, like the kit's records, so `trace case-...` finds them."""
    print(json.dumps({"ts": time.time(), "source": "team13-ran-tools", **fields}, default=str))


def change_key(args: dict) -> str:
    """The approval id. Byte-for-byte the kit's key (checkpoint_kit.py 128-131): json.dumps of
    [case_id, item_id, change] with default separators, no key sorting, missing -> null."""
    fields = [args.get(k) for k in ("case_id", "item_id", "change")]
    return "chg-" + hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:12]


def change_hash(cells: list[dict]) -> str:
    """Order-independent identity of a change, to match an apply with its earlier prediction."""
    return hashlib.sha256(json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _number(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ToolError(f"{what} must be a finite number")
    return float(v)


def parse_change(change: Any) -> list[dict]:
    """`{"cells": [{"cell", "d_power_db", "d_tilt_deg"}, ...]}` (or the same as JSON text) -> cells."""
    if isinstance(change, str):
        try:
            change = json.loads(change)
        except ValueError as exc:
            raise ToolError("change is not valid JSON") from exc
    if not isinstance(change, dict) or not isinstance(change.get("cells"), list) or not change["cells"]:
        raise ToolError('change must look like {"cells": [{"cell": 2, "d_power_db": 0, "d_tilt_deg": -2}]}')
    cells = []
    for i, c in enumerate(change["cells"]):
        if not isinstance(c, dict):
            raise ToolError(f"change.cells[{i}] must be an object")
        cell = c.get("cell")
        if isinstance(cell, float) and cell.is_integer():
            cell = int(cell)
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0:
            raise ToolError(f"change.cells[{i}].cell must be a cell index (integer >= 0)")
        cells.append({"cell": cell, "d_power_db": _number(c.get("d_power_db", 0), f"change.cells[{i}].d_power_db"),
                      "d_tilt_deg": _number(c.get("d_tilt_deg", 0), f"change.cells[{i}].d_tilt_deg")})
    if len({c["cell"] for c in cells}) != len(cells):
        raise ToolError("change lists a cell twice")
    return cells


def jsonable(v: Any) -> Any:
    """numpy scalars/arrays -> plain JSON; NaN/inf -> None."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if hasattr(v, "item"):
        return jsonable(v.item())
    return v


def headline(kpis: dict) -> dict:
    return {k: (None if kpis.get(k) is None else float(kpis[k])) for k in KPI_KEYS}


def _sign(d: float, band: float) -> int:
    return 0 if abs(d) <= band else (1 if d > 0 else -1)


def compare(before: dict, predicted: dict, measured: dict) -> dict:
    """Predicted vs measured: per-key absolute error, a summary and direction agreement."""
    error = {k: abs(predicted[k] - measured[k]) for k in KPI_KEYS}
    error["summary"] = sum(error[k] for k in DIRECTION_KEYS) / len(DIRECTION_KEYS)
    per_key = {k: _sign(predicted[k] - before[k], DEADBAND[k]) == _sign(measured[k] - before[k], DEADBAND[k])
               for k in KPI_KEYS}
    return {"error": error, "direction_per_key": per_key,
            "direction_agree": all(per_key[k] for k in DIRECTION_KEYS)}


def require_case_id(args: dict) -> str:
    case_id = args.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ToolError("case_id is required: pass the case_id from open_case to every tool")
    if not SAFE_ID.fullmatch(case_id):
        raise ToolError("case_id has unexpected characters")
    return case_id


def run_id_from(args: dict) -> str | None:
    """A run_id the agent copied in: the `run_id` argument, or a `run_id:` line in any text argument."""
    value = args.get("run_id")
    if isinstance(value, str) and value.strip():
        m = RUN_ID_LINE.search(value)
        return (m.group(1) if m else value.strip())
    for v in args.values():
        if isinstance(v, str):
            m = RUN_ID_LINE.search(v)
            if m:
                return m.group(1)
    return None


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _cached(loader: Callable[[], Any]) -> Callable[[], Any]:
    box: dict = {}

    def get():
        if "v" not in box:
            box["v"] = loader()
        return box["v"]
    return get


# ------------------------------------------------------------------ dependencies
@dataclasses.dataclass
class Deps:
    cases: Any                                   # store.CaseStore
    approvals: Any                               # store.ApprovalStore
    twin_factory: Callable[[str, int, Any], Any]  # (scenario, seed, config) -> ReplayTwin-like
    kpis: Callable[[Any], dict]                  # twin.replay.kpis
    predictor: Callable[[], Any] = lambda: None   # -> t2.predict Predictor | None
    policy: Callable[[], Any] = lambda: None      # -> t1.rl.policy Policy | None
    trust: Callable[[], Any] = lambda: None       # -> t3.trust TrustGate | None
    writer: Callable[[], Any] = lambda: None      # -> evidence writer (append_records)
    nlm: Any = None                              # nlm.NLM
    emulate_interceptor: bool = False            # local runs: create PENDING approvals like the kit
    defaults: dict = dataclasses.field(default_factory=lambda: {
        "scenario": "nominal", "seed": 0, "config": "official", "twin_kind": "maveric-inprocess",
        "twin_version": "unknown"})
    synthetic: bool = False
    now: Callable[[], str] = now_iso

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Deps":
        env = dict(os.environ if env is None else env)
        from . import evidence, nlm, store

        cases, approvals, emulate = store.from_env(env)

        def twin_module():
            import twin.replay as tr  # W2
            return tr

        def twin_factory(scenario, seed, config):
            tr = twin_module()
            presets = getattr(tr, "PRESETS", None) or getattr(tr, "CONFIGS", None) or {}
            if isinstance(config, str) and config in presets:
                config = presets[config]
            return tr.ReplayTwin(scenario, seed, config)

        def load_predictor():
            path = env.get("T2_PREDICTOR_PATH")
            if not path:
                return None
            from t2.predict import load  # W9
            return load(path)

        def load_policy():
            path = env.get("T1_POLICY_PATH")
            if not path:
                return None
            from t1.rl.policy import load  # W9
            return load(path)

        def load_trust():
            if env.get("T3_TRUST", "on") == "off":
                return None
            import t3.trust as trust  # W5
            if hasattr(trust, "from_env"):
                return trust.from_env()
            return None

        version = env.get("T3_TWIN_VERSION")
        if not version:
            try:
                from importlib.metadata import version as pkg_version
                version = "ran-rl " + pkg_version("ran-rl")
            except Exception:  # noqa: BLE001
                version = "unknown"
        return cls(
            cases=cases, approvals=approvals, emulate_interceptor=emulate,
            twin_factory=twin_factory, kpis=lambda s: twin_module().kpis(s),
            predictor=_cached(load_predictor), policy=_cached(load_policy), trust=_cached(load_trust),
            writer=_cached(lambda: evidence.from_env(env)), nlm=nlm.NLM(env),
            defaults={"scenario": env.get("T3_SCENARIO", "nominal"), "seed": int(env.get("T3_SEED", "0")),
                      "config": env.get("T3_TWIN_CONFIG", "official"),
                      "twin_kind": env.get("T3_TWIN_KIND", "maveric-inprocess"), "twin_version": version},
            synthetic=env.get("T3_SYNTHETIC", "0") == "1")


# ------------------------------------------------------------------ the tools
class Tools:
    def __init__(self, deps: Deps):
        self.d = deps

    # ---- case state
    def load_case(self, args: dict) -> dict:
        case_id = require_case_id(args)
        case = self.d.cases.get(case_id)
        if case is not None and all(case.get(k) is not None for k in ("scenario", "seed", "config", "twin_kind")):
            case.setdefault("action_log", [])
            case.setdefault("predictions", [])
            return case
        given = run_id_from(args)
        if given and not SAFE_ID.fullmatch(given):
            raise ToolError("run_id has unexpected characters")
        base = case or {}
        new = {
            "case_id": case_id,
            "run_id": base.get("run_id") or given or f"adhoc-{case_id}",
            "scenario": base.get("scenario") or self.d.defaults["scenario"],
            "seed": int(base["seed"]) if base.get("seed") is not None else int(self.d.defaults["seed"]),
            "config": base.get("config") or self.d.defaults["config"],
            "action_log": base.get("action_log") or [],
            "twin_kind": base.get("twin_kind") or self.d.defaults["twin_kind"],
            "predictions": base.get("predictions") or [],
            "created_at": base.get("created_at") or self.d.now(),
            "version": base.get("version") if case is not None else None,
        }
        self.d.cases.put(new, new["version"])
        audit(record="ran_case_created", case_id=case_id, run_id=new["run_id"], scenario=new["scenario"],
              seed=new["seed"], prewritten=case is not None)
        return self.d.cases.get(case_id)

    def save(self, case: dict) -> None:
        case["predictions"] = case.get("predictions", [])[-MAX_PREDICTIONS:]
        self.d.cases.put(case, case.get("version"))

    def replay(self, case: dict, extra: list[list[dict]] = ()) -> tuple[Any, Any]:
        twin = self.d.twin_factory(case["scenario"], int(case["seed"]), case["config"])
        state = twin.reset()
        for cells in list(case.get("action_log") or []) + list(extra):
            state = twin.apply(cells)
        return twin, state

    def kpis(self, state) -> dict:
        return jsonable(self.d.kpis(state))

    def twin_info(self, case: dict) -> dict:
        return {"kind": case["twin_kind"], "version": self.d.defaults["twin_version"],
                "scenario": case["scenario"], "seed": int(case["seed"]), "step": len(case["action_log"])}

    # ---- tools
    def observe(self, args: dict) -> dict:
        case = self.load_case(args)
        _, state = self.replay(case)
        return {"case_id": case["case_id"], "run_id": case["run_id"], "twin": self.twin_info(case),
                "kpis": self.kpis(state)}

    def propose(self, args: dict) -> dict:
        case = self.load_case(args)
        policy = self.d.policy()
        if policy is None:
            raise ToolError("no RL policy is configured (T1_POLICY_PATH)")
        _, state = self.replay(case)
        cells = parse_change({"cells": jsonable(policy.propose(state))})
        meta = {"model_id": str(policy.model_id), "version": str(policy.version)}
        self._remember(case, {"kind": "propose", "change_hash": change_hash(cells), "cells": cells,
                              "policy": meta, "step": len(case["action_log"]), "at": self.d.now()})
        return {"case_id": case["case_id"], "run_id": case["run_id"], "change": {"cells": cells}, "policy": meta,
                "note": "Pass this change unchanged to ran___predict, ran___preflight and ran___apply_change."}

    def predict(self, args: dict) -> dict:
        case = self.load_case(args)
        cells = parse_change(args.get("change"))
        predictor = self.d.predictor()
        if predictor is None:
            raise ToolError("no response model is configured (T2_PREDICTOR_PATH)")
        _, state = self.replay(case)
        before = headline(self.kpis(state))
        raw = jsonable(predictor.predict(state, cells))
        try:
            predicted = {k: float(raw[k]) for k in KPI_KEYS}
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError(f"the response model returned no {exc}") from exc
        meta = {"model_id": str(predictor.model_id), "version": str(predictor.version)}
        step = len(case["action_log"])
        self._remember(case, {"kind": "predict", "change_hash": change_hash(cells), "cells": cells,
                              "predicted": predicted, "before": before, "predictor": meta, "step": step,
                              "at": self.d.now()})
        return {"case_id": case["case_id"], "run_id": case["run_id"], "step": step, "predictor": meta,
                "before": before, "predicted": predicted,
                "delta": {k: predicted[k] - before[k] for k in KPI_KEYS if before[k] is not None}}

    def preflight(self, args: dict) -> dict:
        case = self.load_case(args)
        cells = parse_change(args.get("change"))
        _, state = self.replay(case)
        before = self.kpis(state)
        twin, _ = self.replay(case)
        after = self.kpis(twin.apply(cells))
        hb, ha = headline(before), headline(after)
        return {"case_id": case["case_id"], "run_id": case["run_id"], "step": len(case["action_log"]),
                "before": hb, "after": ha, "after_per_cell": after.get("per_cell"),
                "delta": {k: ha[k] - hb[k] for k in KPI_KEYS if None not in (ha[k], hb[k])},
                "note": "Preflight runs the same simulator on a copy: nothing was changed, and it is not "
                        "a learned prediction, so the trust gate does not score it."}

    def apply_change(self, args: dict) -> dict:
        case_id = require_case_id(args)
        if not isinstance(args.get("item_id"), str) or not args["item_id"]:
            raise ToolError("item_id is required")
        cells = parse_change(args.get("change"))   # refuse a malformed change before using the approval
        key = change_key(args)
        if self.d.emulate_interceptor:
            status = self.d.approvals.ensure_pending(key, {
                "case_id": case_id, "agent": "local",
                "change": json.dumps({k: args.get(k) for k in ("item_id", "change", "reason")})})
            if status != "APPROVED":
                return self._approval_refusal(key, status)
        applied, seen = self.d.approvals.mark_applied(key)
        if not applied:
            return self._approval_refusal(key, seen)

        for attempt in range(4):
            case = self.load_case(args)
            twin, state = self.replay(case)
            before = self.kpis(state)
            after = self.kpis(twin.apply(cells))
            case["action_log"] = list(case["action_log"]) + [cells]
            try:
                self.save(case)
                break
            except ConflictError:
                if attempt == 3:
                    raise
        twin_info = {**self.twin_info(case), "step": len(case["action_log"])}
        hb, measured = headline(before), headline(after)
        h = change_hash(cells)
        prediction = next((p for p in reversed(case.get("predictions", []))
                           if p.get("kind") == "predict" and p.get("change_hash") == h), None)
        proposal = next((p for p in reversed(case.get("predictions", []))
                         if p.get("kind") == "propose" and p.get("change_hash") == h), None)
        record = {
            "schema_version": "1", "kind": "change", "run_id": case["run_id"], "team": "team13",
            "created_at": self.d.now(), "aws_time": None, "aws_time_source": None, "synthetic": self.d.synthetic,
            "case_id": case["case_id"], "change_id": key, "twin": twin_info, "change": cells,
            "predictor": prediction["predictor"] if prediction else None,
            "policy": proposal["policy"] if proposal else None,
            "predicted": prediction["predicted"] if prediction else None, "measured": measured,
            "error": None, "direction_agree": None, "trust_before": None, "trust_after": None,
        }
        result: dict = {"applied": True, "approval_id": key, "case_id": case["case_id"], "run_id": case["run_id"],
                        "item_id": args["item_id"], "change": args.get("change"), "twin": twin_info,
                        "before": hb, "measured": measured, "measured_per_cell": after.get("per_cell")}
        if prediction and None not in measured.values() and None not in hb.values():
            cmp = compare(hb, prediction["predicted"], measured)
            record.update(error=cmp["error"], direction_agree=cmp["direction_agree"])
            result.update(predicted=prediction["predicted"], error=cmp["error"],
                          direction_agree=cmp["direction_agree"], direction_per_key=cmp["direction_per_key"],
                          prediction_step=prediction["step"],
                          stale_prediction=prediction["step"] != twin_info["step"] - 1)
            result["trust"] = self._trust(record)
        else:
            result["note"] = "No ran___predict output for this change in this case: nothing to score."
        try:
            result["evidence"] = self.d.writer().append_records(case["run_id"], "change", [record])
        except Exception as exc:  # noqa: BLE001 — the change is applied; say the record failed
            result["evidence_error"] = str(exc)
            audit(record="evidence_error", case_id=case["case_id"], run_id=case["run_id"], error=str(exc))
        audit(record="change_applied", case_id=case["case_id"], run_id=case["run_id"], approval_id=key,
              item_id=args["item_id"], change=args.get("change"), step=twin_info["step"],
              predicted=record["predicted"], measured=measured, direction_agree=record["direction_agree"])
        return result

    def ask(self, args: dict) -> dict:
        case = self.load_case(args)
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolError("question is required")
        out = self.d.nlm.ask(question)
        audit(record="nlm_answer", case_id=case["case_id"], run_id=case["run_id"],
              answered_by=out.get("answered_by"), error=out.get("error"))
        return {"case_id": case["case_id"], "run_id": case["run_id"], **out}

    # ---- internals
    def _remember(self, case: dict, entry: dict) -> None:
        for attempt in range(4):
            case["predictions"] = list(case.get("predictions", [])) + [entry]
            try:
                self.save(case)
                return
            except ConflictError:
                if attempt == 3:
                    raise
                case = self.d.cases.get(case["case_id"])

    def _trust(self, record: dict) -> dict:
        try:
            gate = self.d.trust()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"trust gate unavailable: {exc}"}
        if gate is None:
            return {"note": "trust gate not configured"}
        try:
            out = jsonable(gate.update(record)) or {}
        except Exception as exc:  # noqa: BLE001 — the change stands; report the gate's failure
            audit(record="trust_error", case_id=record["case_id"], run_id=record["run_id"], error=str(exc))
            return {"error": f"trust gate failed: {exc}"}
        record["trust_after"] = out.get("trust_after")
        record["trust_before"] = out.get("trust_before")
        return out

    @staticmethod
    def _approval_refusal(key: str, status: str | None) -> dict:
        if status in ("APPLIED", "REJECTED"):
            msg = (f"Refused: approval {key} was already {status.lower()}. "
                   f"A new change needs a new approval: tell the person.")
        else:
            msg = (f"Refused: this change needs a person's approval. Approval id {key}, status "
                   f"{status or 'MISSING'}. Tell the person this approval id. When they say it is approved, "
                   f"call apply_change again with exactly the same values.")
        audit(record="apply_refused", approval_id=key, approval=status)
        return {"error": msg, "applied": False, "approval_id": key, "approval": status or "MISSING"}


# ------------------------------------------------------------------ dispatch
def _dispatch_table(tools: Tools) -> dict[str, Callable[[dict], dict]]:
    return {catalog.ran(catalog.OBSERVE): tools.observe, catalog.ran(catalog.PROPOSE): tools.propose,
            catalog.ran(catalog.PREDICT): tools.predict, catalog.ran(catalog.PREFLIGHT): tools.preflight,
            catalog.ran(catalog.APPLY): tools.apply_change, catalog.nlm_tool(): tools.ask}


def tool_name(event: dict, context: Any) -> str | None:
    try:
        return context.client_context.custom["bedrockAgentCoreToolName"]
    except (AttributeError, KeyError, TypeError):
        pass
    name = event.get("__tool__") if isinstance(event, dict) else None
    return name if isinstance(name, str) else None


_DEPS: Deps | None = None


def set_deps(deps: Deps | None) -> None:
    global _DEPS
    _DEPS = deps


def get_deps() -> Deps:
    global _DEPS
    if _DEPS is None:
        _DEPS = Deps.from_env()
    return _DEPS


def handler(event, context):
    event = event if isinstance(event, dict) else {}
    name = tool_name(event, context)
    args = {k: v for k, v in event.items() if k != "__tool__"}
    case_id = args.get("case_id")
    started = time.time()
    try:
        fn = _dispatch_table(Tools(get_deps())).get(name or "")
        if fn is None:
            raise ToolError(f"unknown tool {name!r}")
        out = fn(args)
        ok = "error" not in out
    except ToolError as exc:
        out, ok = {"error": str(exc)}, False
    except Exception as exc:  # noqa: BLE001 — the agent sees the error; the log keeps it
        audit(record="tool_error", tool=name, case_id=case_id, error=f"{type(exc).__name__}: {exc}")
        out, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
    audit(record="ran_tool", tool=name, case_id=case_id, run_id=out.get("run_id"), ok=ok,
          ms=round((time.time() - started) * 1000))
    return jsonable(out)
