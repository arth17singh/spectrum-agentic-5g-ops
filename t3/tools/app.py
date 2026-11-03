"""Team13's governed RAN tools: one Lambda behind the AgentCore Gateway.

    handler(event, context)          Lambda: 3008 MB, 120 s (contract amendment 1, A16)

The Gateway passes the tool arguments as `event` and the tool's full name as
`context.client_context.custom["bedrockAgentCoreToolName"]` (`<target>___<tool>`). Local runs
without a client context set `T3_TOOL_NAME_FALLBACK=1` and put the name in `event["_tool"]`
(never set that in AWS).

Tools (schemas in catalog.py): ran___observe, ran___propose, ran___predict, ran___preflight,
ran___apply_change, nlm___ask (and the kit's nlm___ask_nlm, served by the same code because
register.py takes over the kit's `nlm` target). Every tool needs `case_id`.

The Lambda holds no case state: each call replays the case's twin from (scenario, seed, config,
action_log), all stored in the case row. The twin object itself is cached per (scenario, seed,
config) and reset on every use (A16).

Measurements are deltas against the no-change counterfactual at the same step (A5): the twin is
replayed twice, once advancing with no change (baseline) and once with the change (measured).
Only ran___predict runs the predictor; it stores its predicted delta in the case row and
ran___apply_change compares against that stored prediction (A4). Before the change record is
written, the separate trust-gate Lambda is invoked synchronously (about 5 s); its trust_before and
trust_after go into the record, which is written once. The tools never write policies (A7), and a
gate error or timeout never fails an apply (the trust fields stay null).

Collaborators are injected through `Deps` (tests use fakes; `Deps.from_env()` builds the real
ones lazily): the twin (W2 `twin.replay`), the predictor (W9 `t2.predict`), the RL policy
(W9 `t1.rl.policy`), the trust-gate invoker, the evidence writer and the NLM client.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from . import catalog, changes
from .store import SAFE_ID, ConflictError

KPI_KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
SCORED_KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db")      # error.summary is over these (A6)
DIRECTION_KEYS = ("coverage_ratio", "sla_ratio")                  # direction_agree is judged on these
DEADBAND = {"coverage_ratio": 0.002, "sla_ratio": 0.002, "sinr_p10_db": 0.1, "sinr_mean_db": 0.1}
MAX_PREDICTIONS = 50
THRESHOLDS_FILE = Path(__file__).resolve().parents[1] / "trust" / "thresholds.json"


class ToolError(Exception):
    """A refusal the agent should read (returned as {"error": ...}, never raised to the Gateway)."""


class ReplayMismatch(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def audit(**fields) -> None:
    """One JSON line per event in the function's log group (/aws/lambda/team13-ran-tools)."""
    print(json.dumps({"ts": time.time(), "source": "team13-ran-tools", **fields}, default=str))


def change_key(args: dict) -> str:
    """The approval id. Byte-for-byte the kit's key (checkpoint_kit.py 128-131): json.dumps of
    [case_id, item_id, change] with default separators, no key sorting, missing -> null. It hashes
    the change exactly as sent, which is why `change` is one canonical string (A2)."""
    fields = [args.get(k) for k in ("case_id", "item_id", "change")]
    return "chg-" + hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:12]


def parse_change(raw: Any) -> tuple[list[dict], str]:
    """-> (cells, canonical text). Refuses anything that is not a change."""
    try:
        cells = changes.parse(raw)
        return cells, changes.canon(cells)
    except (ValueError, TypeError) as exc:
        raise ToolError(f"bad change: {exc}. Use the canonical form, e.g. "
                        "cell=2;d_power_db=+0.0;d_tilt_deg=-2.0") from exc


def is_no_change(raw: Any) -> bool:
    """True for an empty proposal ([] / "" / None) or one whose every cell is zero after the
    canonical one-decimal rounding. Anything else (including malformed input) is not a no-op."""
    if raw is None or raw == [] or (isinstance(raw, str) and not raw.strip()):
        return True
    if isinstance(raw, str):
        try:
            raw = changes.parse(raw)
        except (ValueError, TypeError):
            return False
    if not isinstance(raw, (list, tuple)) or not all(isinstance(c, dict) for c in raw):
        return False
    try:
        return all(round(float(c.get(k, 0.0)), 1) == 0.0 for c in raw for k in ("d_power_db", "d_tilt_deg"))
    except (TypeError, ValueError):
        return False


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
    out = {}
    for k in KPI_KEYS:
        v = kpis.get(k)
        out[k] = None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)
    return out


def delta(after: dict, before: dict) -> dict:
    return {k: (None if after[k] is None or before[k] is None else after[k] - before[k]) for k in KPI_KEYS}


def _close(a: Any, b: Any) -> bool:
    """Same structure, floats equal to 1e-6 relative (W2's replay tolerance across platforms)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    return a == b


def _sign(d: float, band: float) -> int:
    return 0 if abs(d) <= band else (1 if d > 0 else -1)


def compare(predicted: dict, measured: dict, scales: dict) -> dict:
    """Predicted vs measured deltas, exact (never rounded): per-KPI absolute error, the summary the
    trust gate scores (mean over SCORED_KEYS of |error| / scale) and direction agreement."""
    keys = [k for k in KPI_KEYS if predicted.get(k) is not None and measured.get(k) is not None]
    error = {k: abs(predicted[k] - measured[k]) for k in keys}
    error["summary"] = sum(error[k] / float(scales.get(k, 1.0)) for k in SCORED_KEYS) / len(SCORED_KEYS)
    direction = all(_sign(predicted[k], DEADBAND[k]) == _sign(measured[k], DEADBAND[k]) for k in DIRECTION_KEYS)
    return {"error": error, "direction_agree": direction}


def require_case_id(args: dict) -> str:
    case_id = args.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ToolError("case_id is required: pass the case_id from open_case to every tool")
    if not SAFE_ID.fullmatch(case_id):
        raise ToolError("case_id has unexpected characters")
    return case_id


def require_item_id(args: dict) -> str:
    item = args.get("item_id")
    if not isinstance(item, str) or not item.strip():
        raise ToolError("item_id is required: copy it from ran___propose (e.g. step-0)")
    return item


def check_model_version(args: dict, serving: str, what: str) -> None:
    """A pause forbids a (tool, version) unless the caller names another version (A8). So a named
    version must be the one really served, or a paused model could be reached by naming another."""
    wanted = args.get("model_version")
    if wanted is not None and str(wanted) != str(serving):
        raise ToolError(f"{what} version {wanted!r} is not being served (serving {serving!r})")


CELL_KEYS = ("cell", "power_dbm", "tilt_deg", "ue_count", "coverage_share", "sla_share", "sinr_mean_db",
             "sinr_p10_db", "rsrp_mean_dbm", "rsrp_p10_dbm", "alarm", "asleep")
INT_CELL_KEYS = ("cell", "ue_count")
SHARE_KEYS = ("coverage_share", "sla_share")   # W2 2977a7e, ReplayTwin only


def snapshot(step: int, kpis: dict) -> dict:
    """A TwinSnapshot (W1): {step, coverage_ratio, sla_ratio, per_cell: [CellState]}. Only the
    contract's cell keys are kept (W2's kpis also carry bounds and method notes)."""
    cells = []
    for c in kpis.get("per_cell") or []:
        # Non-finite twin values (e.g. -inf RSRP of a dark cell) are written as null (W1 cc92085).
        cell = {k: jsonable(c[k]) for k in CELL_KEYS if k in c}
        for k in SHARE_KEYS:  # always present: None where the twin cannot tell (hosted adapters, 0 UEs)
            cell[k] = jsonable(c.get(k))
        for k in INT_CELL_KEYS:
            if cell.get(k) is not None:
                cell[k] = int(cell[k])
        for k in ("alarm", "asleep"):
            if cell.get(k) is not None:
                cell[k] = bool(cell[k])
        cells.append(cell)
    h = headline(kpis)
    return {"step": int(step), "coverage_ratio": h["coverage_ratio"], "sla_ratio": h["sla_ratio"], "per_cell": cells}


_DECLARED: dict = {}


def declared(model: str, field: str) -> bool:
    """Does W1's contract declare this optional field? Its models forbid unknown keys, so a field
    still on its way (twin_state, twin.preset) is written only once declared. Without
    team13_contracts nothing can validate, so the full shape is written."""
    key = (model, field)
    if key not in _DECLARED:
        try:
            from team13_contracts import records
            _DECLARED[key] = field in getattr(records, model).model_fields
        except ImportError:
            _DECLARED[key] = True
        except Exception:  # noqa: BLE001
            _DECLARED[key] = False
    return _DECLARED[key]


AWS_TIME_SOURCE = "lambda:clock"   # W1 cc92085: what a Lambda stamping its own clock writes


def lambda_clock(env: dict) -> Callable[[], tuple[str | None, str | None]]:
    """aws_time from the Lambda's own clock (inside AWS Lambda only; local runs stay null)."""
    fn = env.get("AWS_LAMBDA_FUNCTION_NAME")

    def clock():
        if not fn:
            return None, None
        return now_iso(), AWS_TIME_SOURCE
    return clock


def utc_from_epoch(value: Any) -> str | None:
    """Epoch seconds (the kit's decided_at) -> ISO-8601 UTC ending in Z, as the contract wants."""
    if value is None or isinstance(value, bool):
        return None
    try:
        ts = dt.datetime.fromtimestamp(float(value), dt.timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _cached(loader: Callable[[], Any]) -> Callable[[], Any]:
    box: dict = {}

    def get():
        if "v" not in box:
            box["v"] = loader()
        return box["v"]
    return get


def load_scales(env: dict) -> tuple[dict, str]:
    if env.get("T3_ERROR_SCALES"):
        return json.loads(env["T3_ERROR_SCALES"]), "env:T3_ERROR_SCALES"
    try:
        data = json.loads(THRESHOLDS_FILE.read_text())
        scales = data.get("scales") or data.get("error_scales")
        if isinstance(scales, dict):
            return {k: float(v) for k, v in scales.items()}, "t3/trust/thresholds.json"
    except (OSError, ValueError):
        pass
    return {k: 1.0 for k in SCORED_KEYS}, "unit (no declared scales found)"


# ------------------------------------------------------------------ dependencies
@dataclasses.dataclass
class Deps:
    store: Any                                    # store.Store
    twin_factory: Callable[[str, int, Any], Any]   # (scenario, seed, config) -> ReplayTwin-like
    kpis: Callable[[Any], dict]                   # twin.kpis
    predictor: Callable[[], Any] = lambda: None    # -> Predictor | None
    policy: Callable[[], Any] = lambda: None       # -> Policy | None
    trust_invoker: Callable[[], Any] = lambda: None  # -> callable(record) | None
    writer: Callable[[], Any] = lambda: None       # -> evidence.Writer
    nlm: Any = None                               # nlm.NLM
    defaults: dict = dataclasses.field(default_factory=lambda: {
        "scenario": "nominal", "seed": 0, "config": "official", "twin_kind": "maveric-inprocess"})
    twin_version: str = "unknown"
    scales: dict = dataclasses.field(default_factory=lambda: {k: 1.0 for k in SCORED_KEYS})
    scales_source: str = "unit"
    synthetic: bool = False
    tool_name_fallback: bool = False
    # Grounded mode (T3_TRUST_LIVE=1): apply refuses a change it cannot ground, i.e. with no
    # prediction for the item, a paused response model, a clipped change, or no trust-gate answer.
    # Lenient mode (0, the default): those changes apply and are recorded unscored or with null trust.
    strict_grounding: bool = False
    # When set, a call must carry the Gateway's context marker bedrockAgentCoreGatewayId equal to it.
    expected_gateway_id: str | None = None
    # () -> (aws_time, aws_time_source): the Lambda's own clock; (None, None) outside Lambda.
    aws_clock: Callable[[], tuple] = lambda: (None, None)
    now: Callable[[], str] = now_iso
    twin_cache: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Deps":
        env = dict(os.environ if env is None else env)
        from . import evidence, nlm, store

        def twin_factory(scenario, seed, config):
            from twin.replay import ReplayTwin  # W2
            return ReplayTwin(scenario, seed, config)

        def kpis(state):
            from twin.kpis import kpis as twin_kpis  # W2
            return twin_kpis(state)

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

        def load_invoker():
            if env.get("T3_TRUST", "on") == "off":
                return None
            fn = env.get("T3_TRUST_FUNCTION", "team13-trust-gate")
            timeout = float(env.get("T3_TRUST_TIMEOUT_S", "5"))
            import boto3
            from botocore.config import Config
            client = boto3.client("lambda", config=Config(connect_timeout=2, read_timeout=timeout,
                                                          retries={"max_attempts": 0}))

            def invoke(payload: dict) -> dict:
                resp = client.invoke(FunctionName=fn, InvocationType="RequestResponse",
                                     Payload=json.dumps(payload, allow_nan=False).encode())
                body = resp["Payload"].read()
                if resp.get("FunctionError"):
                    raise RuntimeError(f"{fn} {resp['FunctionError']}: {body[:300]!r}")
                return json.loads(body or b"{}")
            return invoke

        version = env.get("T3_TWIN_VERSION")
        if not version:
            try:
                from importlib.metadata import version as pkg_version
                version = "ran-rl " + pkg_version("ran-rl")
            except Exception:  # noqa: BLE001
                version = "unknown"
        scales, scales_source = load_scales(env)
        return cls(
            store=store.from_env(env), twin_factory=twin_factory, kpis=kpis,
            predictor=_cached(load_predictor), policy=_cached(load_policy), trust_invoker=_cached(load_invoker),
            writer=_cached(lambda: evidence.from_env(env)), nlm=nlm.NLM(env),
            defaults={"scenario": env.get("T3_SCENARIO", "nominal"), "seed": int(env.get("T3_SEED", "0")),
                      "config": env.get("T3_TWIN_CONFIG", "official"),
                      "twin_kind": env.get("T3_TWIN_KIND", "maveric-inprocess")},
            twin_version=version, scales=scales, scales_source=scales_source,
            synthetic=env.get("T3_SYNTHETIC", "0") == "1",
            tool_name_fallback=env.get("T3_TOOL_NAME_FALLBACK", "0") == "1",
            strict_grounding=env.get("T3_TRUST_LIVE", "0") == "1",
            expected_gateway_id=env.get("T3_GATEWAY_ID") or None,
            aws_clock=lambda_clock(env))


# ------------------------------------------------------------------ the tools
class Tools:
    def __init__(self, deps: Deps):
        self.d = deps

    # ---- case state and the twin
    def twin(self, scenario: str, seed: int, config: Any) -> Any:
        def key(cfg):
            return (scenario, int(seed), json.dumps(jsonable(cfg), sort_keys=True, default=str))
        if key(config) not in self.d.twin_cache:
            twin = self.d.twin_factory(scenario, int(seed), config)
            self.d.twin_cache[key(config)] = twin
            resolved = getattr(twin, "config", None)
            if isinstance(resolved, dict):  # cases store the resolved config: same twin
                self.d.twin_cache.setdefault(key(resolved), twin)
        return self.d.twin_cache[key(config)]

    def load_case(self, args: dict) -> dict:
        case_id = require_case_id(args)
        case = self.d.store.get_case(case_id)
        if case is not None:
            return case
        dflt = self.d.defaults
        twin = self.twin(dflt["scenario"], dflt["seed"], dflt["config"])
        resolved = getattr(twin, "config", None)
        new = {"case_id": case_id, "run_id": self.d.store.active_run_id() or f"adhoc-{case_id}",
               "scenario": dflt["scenario"], "seed": int(dflt["seed"]),
               "config": jsonable(resolved) if isinstance(resolved, dict) else dflt["config"],
               "action_log": [], "twin_kind": dflt["twin_kind"], "predictions": [], "created_at": self.d.now()}
        try:
            self.d.store.put_case(new, None)
        except ConflictError:
            pass  # created by a concurrent call: read theirs
        audit(record="ran_case_created", case_id=case_id, run_id=new["run_id"])
        return self.d.store.get_case(case_id)

    def replay(self, case: dict) -> tuple[Any, Any]:
        """The case's twin at its current step (reset, then every logged change again)."""
        twin = self.twin(case["scenario"], case["seed"], case["config"])
        state = twin.reset()
        for i, logged in enumerate(case.get("action_log") or []):
            state = twin.apply(logged["change"])
            new = twin.action_log[-1]
            if "filtered" in logged and not _close(jsonable(new.get("filtered")), logged["filtered"]):
                raise ReplayMismatch(f"case {case['case_id']} step {i}: the safety filter now passes "
                                     f"{new.get('filtered')} where the log has {logged['filtered']}")
        return twin, state

    def counterfactual(self, case: dict, cells: list[dict]) -> dict:
        """Baseline (advance with no change) and measured (advance with the change) at the same step.

        Uses W2's `ReplayTwin.counterfactual(change)` when the twin has it (both futures from one
        snapshot, live twin untouched), then applies the change for real to extend the action log.
        Otherwise replays twice: once with an empty change (W2: a no-op step), once with the change.
        """
        twin, state = self.replay(case)
        before = self.state_kpis(twin, state)       # this step, exactly as ran___observe shows it
        cf_fn = getattr(twin, "counterfactual", None)
        cf = jsonable(cf_fn(cells)) if callable(cf_fn) else None   # W2: does not move the live twin
        if cf is not None and cf.get("after_kpis") and cf.get("baseline_kpis"):
            # W2 (2977a7e): full kpis() of both futures, per-cell shares included. One replay.
            after_full, baseline_full = cf["after_kpis"], cf["baseline_kpis"]
            twin.apply(cells)                                        # the change, applied for real
        else:
            baseline_full = jsonable(self.d.kpis(twin.apply([])))   # no-change step
            twin, _ = self.replay(case)
            after_full = jsonable(self.d.kpis(twin.apply(cells)))   # the change, applied for real
        if cf is not None:
            hb, hm = headline(cf["baseline"]), headline(cf["after"])
            d = {k: (None if cf["delta"].get(k) is None else float(cf["delta"][k])) for k in KPI_KEYS}
            clipped, rules, source = bool(cf.get("clamped")), cf.get("rules") or [], "twin.counterfactual"
        else:
            entry = twin.action_log[-1] if twin.action_log else {}
            hb, hm = headline(baseline_full), headline(after_full)
            d = delta(hm, hb)
            clipped, rules, source = bool(entry.get("clamped")), entry.get("rules") or [], "two replays"
        return {"baseline_kpis": hb, "measured_kpis": hm, "measured_delta": d,
                "measured_per_cell": after_full.get("per_cell"), "clipped": clipped, "safety_rules": rules,
                "counterfactual_source": source, "action_log": jsonable(list(twin.action_log)),
                "before_full": before, "baseline_full": baseline_full, "measured_full": after_full}

    def state_kpis(self, twin: Any, state: Any) -> dict:
        """KPIs of the twin's current state: W2's ReplayTwin.kpis() (per-cell shares included) when
        it has it, else twin.kpis(state). Hosted adapters have no per-UE data, so no shares there."""
        fn = getattr(twin, "kpis", None)
        if callable(fn):
            try:
                return jsonable(fn())
            except RuntimeError:  # shares rely on ran-rl 1.2.9 internals
                pass
        return jsonable(self.d.kpis(state))

    def twin_state(self, cf: dict, step: int) -> dict:
        """Per-cell twin state for the record: before (this step), after (the change plus one step)
        and baseline (no change plus one step). 4 cells x 3 snapshots; no address or token."""
        return {"before": snapshot(step, cf["before_full"]), "after": snapshot(step + 1, cf["measured_full"]),
                "baseline": snapshot(step + 1, cf["baseline_full"])}

    def twin_info(self, case: dict, step: int) -> dict:
        return {"kind": case["twin_kind"], "version": self.d.twin_version, "scenario": case["scenario"],
                "seed": int(case["seed"]), "step": step}

    def twin_preset(self, case: dict) -> str | None:
        """The preset the case's twin runs ("official", "viavi-like"): W2's resolved config names it."""
        cfg = case.get("config")
        if isinstance(cfg, dict):
            return cfg.get("preset") or None
        return cfg if isinstance(cfg, str) and cfg else None

    def _remember(self, case: dict, entry: dict) -> None:
        for attempt in range(4):
            case["predictions"] = (list(case.get("predictions") or []) + [entry])[-MAX_PREDICTIONS:]
            try:
                self.d.store.put_case(case, case.get("version"))
                return
            except ConflictError:
                if attempt == 3:
                    raise
                case = self.d.store.get_case(case["case_id"])

    # ---- tools
    def observe(self, args: dict) -> dict:
        case = self.load_case(args)
        twin, state = self.replay(case)
        return {"case_id": case["case_id"], "run_id": case["run_id"],
                "twin": self.twin_info(case, len(case["action_log"])), "kpis": self.state_kpis(twin, state)}

    def propose(self, args: dict) -> dict:
        case = self.load_case(args)
        policy = self.d.policy()
        if policy is None:
            raise ToolError("no RL policy is configured (T1_POLICY_PATH)")
        check_model_version(args, policy.version, "policy")
        _, state = self.replay(case)
        raw = jsonable(policy.propose(state))
        step = len(case["action_log"])
        meta = {"model_id": str(policy.model_id), "version": str(policy.version)}
        if is_no_change(raw):  # W1's canon refuses an empty change; a no-op is an answer, not an error
            audit(record="ran_no_change", case_id=case["case_id"], run_id=case["run_id"], step=step, policy=meta)
            return {"case_id": case["case_id"], "run_id": case["run_id"], "no_change": True, "item_id": None,
                    "change": None, "policy": meta, "step": step, "reason": "policy proposes no change"}
        try:
            text = changes.canon(changes.round_items(raw))
        except (ValueError, TypeError) as exc:
            raise ToolError(f"the RL policy proposed an unusable change: {exc}") from exc
        _, text = parse_change(text)
        item_id = f"step-{step}"
        self._remember(case, {"kind": "propose", "item_id": item_id, "change_text": text, "policy": meta,
                              "step": step, "at": self.d.now()})
        return {"case_id": case["case_id"], "run_id": case["run_id"], "item_id": item_id, "change": text,
                "policy": meta, "note": "Pass item_id and change verbatim to ran___predict and ran___apply_change."}

    def predict(self, args: dict) -> dict:
        case = self.load_case(args)
        cells, text = parse_change(args.get("change"))
        predictor = self.d.predictor()
        if predictor is None:
            raise ToolError("no response model is configured (T2_PREDICTOR_PATH)")
        check_model_version(args, predictor.version, "response model")
        _, state = self.replay(case)
        raw = jsonable(predictor.predict(state, cells)) or {}
        try:  # the record's KpiDelta needs all four (W1); the trust gate scores three of them
            predicted = {k: float(raw[k]) for k in KPI_KEYS}
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError(f"the response model returned no {exc}") from exc
        if not all(math.isfinite(v) for v in predicted.values()):
            raise ToolError("the response model returned a non-finite delta")
        step = len(case["action_log"])
        item_id = args.get("item_id") if isinstance(args.get("item_id"), str) and args["item_id"] else f"step-{step}"
        meta = {"model_id": str(predictor.model_id), "version": str(predictor.version)}
        self._remember(case, {"kind": "predict", "item_id": item_id, "change_text": text,
                              "predicted_delta": predicted, "predictor": meta, "step": step, "at": self.d.now()})
        return {"case_id": case["case_id"], "run_id": case["run_id"], "item_id": item_id, "change": text,
                "step": step, "predictor": meta, "predicted_delta": predicted,
                "note": "Delta against doing nothing at this step. Stored; ran___apply_change compares it "
                        "with what the twin measures."}

    def preflight(self, args: dict) -> dict:
        case = self.load_case(args)
        cells, text = parse_change(args.get("change"))
        cf = self.counterfactual(case, cells)
        return {"case_id": case["case_id"], "run_id": case["run_id"], "change": text,
                "step": len(case["action_log"]), "baseline_kpis": cf["baseline_kpis"],
                "measured_kpis": cf["measured_kpis"], "measured_delta": cf["measured_delta"],
                "per_cell": cf["measured_per_cell"], "clipped": cf["clipped"], "safety_rules": cf["safety_rules"],
                "note": "Preflight ran the same simulator on a copy: nothing was changed, and it is not "
                        "a learned prediction, so the trust gate does not score it."}

    def apply_change(self, args: dict) -> dict:
        require_case_id(args)
        item_id = require_item_id(args)
        cells, text = parse_change(args.get("change"))  # a bad change is refused before any approval is used
        if is_no_change(cells):
            raise ToolError("nothing to apply: every cell of this change is zero (use ran___propose's no_change "
                            "answer instead of applying)")
        key = change_key(args)
        store = self.d.store
        if store.emulate_interceptor:  # local runs only: no Gateway in front of us
            status = store.ensure_pending(key, {"case_id": args["case_id"], "agent": "local",
                                                "change": json.dumps({k: args.get(k) for k in
                                                                      ("item_id", "change", "reason")})})
        else:
            status = store.approval_status(key)
        if status != "APPROVED":  # the table decides; an `approval` argument is never trusted here
            return self._approval_refusal(key, status)
        # Who approved: the kit's `approve` stores decided_by (the approver's AWS identity) and
        # decided_at (epoch s) on the row. Read once, before the gate sees the record.
        approval = store.approval_item(key) or {}
        approved_by = approval.get("decided_by") or None
        approved_at = utc_from_epoch(approval.get("decided_at"))
        strict = self.d.strict_grounding
        grounding = "strict" if strict else "lenient"

        for attempt in range(4):
            case = self.load_case(args)
            step = len(case["action_log"])
            prediction, proposal = self._find_prediction(case, item_id, text, strict)
            if strict:  # every refusal here leaves the approval APPROVED and unused
                self._require_grounding(prediction, item_id, step)
            cf = self.counterfactual(case, cells)
            if strict and cf["clipped"]:
                raise ToolError(f"refused (grounded mode): the safety filter would change this request "
                                f"({', '.join(cf['safety_rules']) or 'clipped'}), so its effect could not be "
                                f"compared with the prediction. The approval {key} is unchanged; propose a "
                                f"smaller change.")
            record = self._change_record(case, key, item_id, cells, text, step, cf, prediction, proposal)
            record["approved_by"], record["approved_at"] = approved_by, approved_at  # W1 ChangeRecord (2bf3325)
            if strict and not record["scored"]:
                raise ToolError(f"refused (grounded mode): this change cannot be scored "
                                f"({record['unscored_reason']}). The approval {key} is unchanged.")
            trust = None
            if record["scored"]:  # the gate decides before anything is committed
                trust = self._ask_trust(record, prediction["predictor"]["version"])
                if strict and (trust.get("note") or trust.get("error") or trust.get("trust_after") is None):
                    raise ToolError(f"refused (grounded mode): the trust gate did not answer cleanly "
                                    f"({trust.get('note') or trust.get('error') or 'no trust_after'}). Nothing was "
                                    f"applied and the approval {key} is unchanged; try again later.")
                record["trust_before"], record["trust_after"] = trust.get("trust_before"), trust.get("trust_after")
            updated = {**case, "action_log": cf["action_log"]}
            try:
                applied, seen = store.apply_and_save(key, updated, case.get("version"))
            except ConflictError:
                if attempt == 3:
                    raise
                continue
            if not applied:
                if trust is not None:  # the gate scored a change that another call won the race for
                    audit(record="trust_scored_unapplied", case_id=case["case_id"], approval_id=key, approval=seen)
                return self._approval_refusal(key, seen)
            break

        result = {"applied": True, "approval_id": key, "case_id": case["case_id"], "run_id": case["run_id"],
                  "item_id": item_id, "change": text, "step": step, "baseline_kpis": cf["baseline_kpis"],
                  "measured_kpis": cf["measured_kpis"], "measured_delta": cf["measured_delta"],
                  # not "error": that key means a refusal to agents and to the handler's ok flag
                  "predicted_delta": record["predicted_delta"], "prediction_error": record["error"],
                  "direction_agree": record["direction_agree"], "scored": record["scored"],
                  "unscored_reason": record["unscored_reason"], "clipped": cf["clipped"],
                  "safety_rules": cf["safety_rules"], "grounding": grounding, "approved_by": approved_by}
        if trust is not None:
            result["trust"] = trust
        try:
            result["evidence"] = self.d.writer().put_record(case["run_id"], "change", key, record)
        except Exception as exc:  # noqa: BLE001 — the change is applied; say the record failed
            result["evidence_error"] = f"{type(exc).__name__}: {exc}"
            audit(record="evidence_error", case_id=case["case_id"], run_id=case["run_id"], error=str(exc))
        audit(record="change_applied", case_id=case["case_id"], run_id=case["run_id"], approval_id=key,
              change_text=text, item_id=item_id, step=step, scored=record["scored"],
              unscored_reason=record["unscored_reason"], evidence=result.get("evidence"),
              trust_after=record["trust_after"], grounding=grounding, approved_by=approved_by,
              approved_at=approved_at)
        return result

    def _find_prediction(self, case: dict, item_id: str, text: str, strict: bool) -> tuple[dict | None, dict | None]:
        """The latest ran___predict output for this change (grounded mode: for this exact item too),
        and the latest proposal of the same change."""
        preds = [p for p in case.get("predictions") or [] if p.get("change_text") == text]
        same_item = next((p for p in reversed(preds)
                          if p.get("kind") == "predict" and p.get("item_id") == item_id), None)
        prediction = same_item if strict else (
            same_item or next((p for p in reversed(preds) if p.get("kind") == "predict"), None))
        proposal = next((p for p in reversed(preds) if p.get("kind") == "propose"), None)
        return prediction, proposal

    def _require_grounding(self, prediction: dict | None, item_id: str, step: int) -> None:
        if prediction is None:
            raise ToolError(f"refused (grounded mode, T3_TRUST_LIVE=1): there is no ran___predict output for "
                            f"item {item_id} with this change, so its effect could not be checked. Ask "
                            f"ran_optimizer to call ran___predict first. If ran___predict is refused, the response "
                            f"model is paused and no change can be applied until a person resumes it. The approval "
                            f"is unchanged.")
        if prediction.get("step") != step:
            raise ToolError(f"refused (grounded mode): the prediction for item {item_id} was made at step "
                            f"{prediction.get('step')} but the twin is at step {step}; predict again. "
                            f"The approval is unchanged.")
        meta = prediction.get("predictor") or {}
        if self.d.store.model_paused(str(meta.get("model_id")), str(meta.get("version"))):
            raise ToolError(f"refused (grounded mode): the response model {meta.get('model_id')}@"
                            f"{meta.get('version')} is paused, so no prediction can ground this change. "
                            f"A person must resume it first. The approval is unchanged.")

    def _change_record(self, case, key, item_id, cells, text, step, cf, prediction, proposal) -> dict:
        record = {
            "schema_version": "1", "kind": "change", "run_id": case["run_id"], "team": "team13",
            "created_at": self.d.now(), "aws_time": None, "aws_time_source": None, "synthetic": self.d.synthetic,
            "case_id": case["case_id"], "change_id": key, "item_id": item_id, "change": cells, "change_text": text,
            "twin": self.twin_info(case, step), "baseline_kpis": cf["baseline_kpis"],
            "measured_kpis": cf["measured_kpis"], "measured_delta": cf["measured_delta"],
            "predicted_delta": prediction["predicted_delta"] if prediction else None,
            "predictor": prediction["predictor"] if prediction else None,
            "policy": proposal["policy"] if proposal else None,
            "error": None, "direction_agree": None, "scored": False, "unscored_reason": None, "probe": False,
            "trust_before": None, "trust_after": None,
        }
        aws_time, aws_source = self.d.aws_clock()
        if aws_time:  # inside Lambda: the record carries an AWS-side time and says where it came from
            record["aws_time"], record["aws_time_source"] = aws_time, aws_source
        if declared("ChangeTwin", "preset"):
            record["twin"]["preset"] = self.twin_preset(case)
        if declared("ChangeRecord", "twin_state"):
            record["twin_state"] = self.twin_state(cf, step)
        if cf["clipped"]:
            record["unscored_reason"] = "clipped"
        elif prediction is None:
            record["unscored_reason"] = "no_prediction"
        elif None in (cf["measured_delta"][k] for k in SCORED_KEYS):
            record["unscored_reason"] = "measurement_missing"
        else:
            cmp = compare(prediction["predicted_delta"], cf["measured_delta"], self.d.scales)
            record.update(error=cmp["error"], direction_agree=cmp["direction_agree"], scored=True)
        return record

    def ask(self, args: dict) -> dict:
        case = self.load_case(args)
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolError("question is required")
        if self.d.nlm is None:
            raise ToolError("no NLM client")
        check_model_version(args, self.d.nlm.describe().get("version"), "NLM")
        out = self.d.nlm.ask(question)
        audit(record="nlm_answer", case_id=case["case_id"], run_id=case["run_id"],
              answered_by=out.get("answered_by"), error=out.get("error"))
        return {"case_id": case["case_id"], "run_id": case["run_id"], **out}

    # ---- internals
    def _ask_trust(self, record: dict, model_version: str) -> dict:
        """Ask the separate trust-gate Lambda synchronously (about 5 s timeout) before the record is
        written. It answers {trust_before, trust_after, e, score, action, control_record_key?} and
        keeps the policy-write rights (A7). Apply never fails because of the gate: on an error or a
        timeout the trust fields stay null and the result carries a note."""
        try:
            invoke = self.d.trust_invoker()
            if invoke is None:
                return {"trust_before": None, "trust_after": None,
                        "note": "trust gate not configured (T3_TRUST=off); trust fields left null"}
            out = jsonable(invoke({"record": record, "model_version": model_version})) or {}
            if not isinstance(out, dict):
                raise ValueError(f"unexpected gate answer {str(out)[:100]}")
            audit(record="trust_decided", case_id=record["case_id"], run_id=record["run_id"],
                  change_id=record["change_id"], model_version=model_version,
                  **{k: out.get(k) for k in ("trust_before", "trust_after", "e", "score", "action",
                                             "control_record_key")})
            return out
        except Exception as exc:  # noqa: BLE001 — the change stands; report the failure
            audit(record="trust_error", case_id=record["case_id"], run_id=record["run_id"],
                  error=f"{type(exc).__name__}: {exc}")
            return {"trust_before": None, "trust_after": None,
                    "note": f"trust gate unavailable ({type(exc).__name__}: {exc}); trust fields left null"}

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
            catalog.ran(catalog.APPLY): tools.apply_change, catalog.NLM_ASK: tools.ask,
            catalog.KIT_NLM_ASK: tools.ask}


def tool_name(event: dict, context: Any, fallback: bool) -> str | None:
    try:
        return context.client_context.custom["bedrockAgentCoreToolName"]
    except (AttributeError, KeyError, TypeError):
        pass
    if fallback:
        name = event.get("_tool")
        return name if isinstance(name, str) else None
    return None


def gateway_marker(context: Any) -> dict:
    try:
        custom = context.client_context.custom
        return custom if isinstance(custom, dict) else {}
    except AttributeError:
        return {}


def check_gateway(context: Any, expected_gateway_id: str | None) -> None:
    """Defence in depth against a direct Lambda invoke that skips the checkpoint. The Gateway puts
    bedrockAgentCoreGatewayId (and the tool name) in the invocation's client context. Anyone allowed
    to call lambda:InvokeFunction can forge a client context, so this is not proof of origin: IAM
    (only the Gateway's role may invoke this function) is the real control."""
    if not expected_gateway_id:
        return
    custom = gateway_marker(context)
    if custom.get("bedrockAgentCoreGatewayId") != expected_gateway_id or not custom.get("bedrockAgentCoreToolName"):
        raise ToolError("refused: this tool only answers calls that come through the team13 checkpoint")


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
    args = {k: v for k, v in event.items() if k != "_tool"}
    case_id = args.get("case_id")
    started = time.time()
    name = None
    try:
        deps = get_deps()
        check_gateway(context, deps.expected_gateway_id)
        name = tool_name(event, context, deps.tool_name_fallback)
        fn = _dispatch_table(Tools(deps)).get(name or "")
        if fn is None:
            raise ToolError(f"unknown tool {name!r}")
        out = fn(args)
        ok = not out.get("error")
    except ToolError as exc:
        out, ok = {"error": str(exc)}, False
    except Exception as exc:  # noqa: BLE001 — the agent sees the error; the log keeps it
        audit(record="tool_error", tool=name, case_id=case_id, error=f"{type(exc).__name__}: {exc}")
        out, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
    audit(record="ran_tool", tool=name, case_id=case_id, run_id=out.get("run_id"), ok=ok,
          ms=round((time.time() - started) * 1000))
    return jsonable(out)
