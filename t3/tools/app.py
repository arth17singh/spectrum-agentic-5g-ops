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
ran___apply_change compares against that stored prediction (A4). After the change record is
written, the separate trust-gate Lambda is invoked asynchronously (A7); the tools never write policies.

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
            import boto3
            client = boto3.client("lambda")

            def invoke(record: dict) -> dict:
                resp = client.invoke(FunctionName=fn, InvocationType="Event",
                                     Payload=json.dumps(record, allow_nan=False).encode())
                return {"invoked": fn, "status": resp.get("StatusCode")}
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
            tool_name_fallback=env.get("T3_TOOL_NAME_FALLBACK", "0") == "1")


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
        twin, _ = self.replay(case)
        cf_fn = getattr(twin, "counterfactual", None)
        if callable(cf_fn):
            cf = jsonable(cf_fn(cells))
            measured = jsonable(self.d.kpis(twin.apply(cells)))
            hb, hm = headline(cf["baseline"]), headline(cf["after"])
            d = {k: (None if cf["delta"].get(k) is None else float(cf["delta"][k])) for k in KPI_KEYS}
            clipped, rules, source = bool(cf.get("clamped")), cf.get("rules") or [], "twin.counterfactual"
        else:
            baseline = jsonable(self.d.kpis(twin.apply([])))
            twin, _ = self.replay(case)
            measured = jsonable(self.d.kpis(twin.apply(cells)))
            entry = twin.action_log[-1] if twin.action_log else {}
            hb, hm = headline(baseline), headline(measured)
            d = delta(hm, hb)
            clipped, rules, source = bool(entry.get("clamped")), entry.get("rules") or [], "two replays"
        return {"baseline_kpis": hb, "measured_kpis": hm, "measured_delta": d,
                "measured_per_cell": measured.get("per_cell"), "clipped": clipped, "safety_rules": rules,
                "counterfactual_source": source, "action_log": jsonable(list(twin.action_log))}

    def twin_info(self, case: dict, step: int) -> dict:
        return {"kind": case["twin_kind"], "version": self.d.twin_version, "scenario": case["scenario"],
                "seed": int(case["seed"]), "step": step}

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
        _, state = self.replay(case)
        return {"case_id": case["case_id"], "run_id": case["run_id"],
                "twin": self.twin_info(case, len(case["action_log"])), "kpis": jsonable(self.d.kpis(state))}

    def propose(self, args: dict) -> dict:
        case = self.load_case(args)
        policy = self.d.policy()
        if policy is None:
            raise ToolError("no RL policy is configured (T1_POLICY_PATH)")
        check_model_version(args, policy.version, "policy")
        _, state = self.replay(case)
        _, text = parse_change(changes.canon(changes.round_items(jsonable(policy.propose(state)))))
        step = len(case["action_log"])
        meta = {"model_id": str(policy.model_id), "version": str(policy.version)}
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
        key = change_key(args)
        store = self.d.store
        if store.emulate_interceptor:  # local runs only: no Gateway in front of us
            status = store.ensure_pending(key, {"case_id": args["case_id"], "agent": "local",
                                                "change": json.dumps({k: args.get(k) for k in
                                                                      ("item_id", "change", "reason")})})
        else:
            status = store.approval_status(key)
        if status != "APPROVED":
            return self._approval_refusal(key, status)

        for attempt in range(4):
            case = self.load_case(args)
            step = len(case["action_log"])
            cf = self.counterfactual(case, cells)
            updated = {**case, "action_log": cf["action_log"]}
            try:
                applied, seen = store.apply_and_save(key, updated, case.get("version"))
            except ConflictError:
                if attempt == 3:
                    raise
                continue
            if not applied:
                return self._approval_refusal(key, seen)
            break

        preds = [p for p in case.get("predictions") or [] if p.get("change_text") == text]
        prediction = next((p for p in reversed(preds) if p.get("kind") == "predict" and p.get("item_id") == item_id),
                          next((p for p in reversed(preds) if p.get("kind") == "predict"), None))
        proposal = next((p for p in reversed(preds) if p.get("kind") == "propose"), None)
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
        if cf["clipped"]:
            record["unscored_reason"] = "clipped"
        elif prediction is None:
            record["unscored_reason"] = "no_prediction"
        elif None in (cf["measured_delta"][k] for k in SCORED_KEYS):
            record["unscored_reason"] = "measurement_missing"
        else:
            cmp = compare(prediction["predicted_delta"], cf["measured_delta"], self.d.scales)
            record.update(error=cmp["error"], direction_agree=cmp["direction_agree"], scored=True)

        result = {"applied": True, "approval_id": key, "case_id": case["case_id"], "run_id": case["run_id"],
                  "item_id": item_id, "change": text, "step": step, "baseline_kpis": cf["baseline_kpis"],
                  "measured_kpis": cf["measured_kpis"], "measured_delta": cf["measured_delta"],
                  "predicted_delta": record["predicted_delta"], "error": record["error"],
                  "direction_agree": record["direction_agree"], "scored": record["scored"],
                  "unscored_reason": record["unscored_reason"], "clipped": cf["clipped"],
                  "safety_rules": cf["safety_rules"]}
        try:
            result["evidence"] = self.d.writer().put_record(case["run_id"], "change", key, record)
        except Exception as exc:  # noqa: BLE001 — the change is applied; say the record failed
            result["evidence_error"] = f"{type(exc).__name__}: {exc}"
            audit(record="evidence_error", case_id=case["case_id"], run_id=case["run_id"], error=str(exc))
        audit(record="change_applied", case_id=case["case_id"], run_id=case["run_id"], approval_id=key,
              change_text=text, item_id=item_id, step=step, scored=record["scored"],
              unscored_reason=record["unscored_reason"], evidence=result.get("evidence"))
        if record["scored"]:
            result["trust"] = self._notify_trust(record)
        return result

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
    def _notify_trust(self, record: dict) -> dict:
        """Hand the record to the separate trust-gate Lambda, asynchronously (A7)."""
        try:
            invoke = self.d.trust_invoker()
            if invoke is None:
                return {"note": "trust gate not configured (T3_TRUST=off)"}
            out = jsonable(invoke(record)) or {}
            audit(record="trust_notified", case_id=record["case_id"], run_id=record["run_id"],
                  change_id=record["change_id"], **{k: v for k, v in out.items() if k != "record"})
            return out
        except Exception as exc:  # noqa: BLE001 — the change stands; report the failure
            audit(record="trust_error", case_id=record["case_id"], run_id=record["run_id"], error=str(exc))
            return {"error": f"trust gate not notified: {exc}"}

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
