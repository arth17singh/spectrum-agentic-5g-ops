"""/api/kitapprovals: the REAL approvals of the kit's pending changes (table team13-approvals).

Laptop only. Four routes, one of them a write:
  GET  /status                  always 200: mode, approver pseudonym, PIN state, audit outbox
  GET  /pending                 always 200: chg- rows (PENDING first), or an empty list with the mode
  GET  /{approval_id}           one row with its case, prediction and trust (missing data shown as missing)
  POST /{approval_id}/decide    approve | reject, with the PIN and a reason, exactly the kit's update

The decision is the kit's own (KIT approve(), L982-1003): UpdateItem SET status, decided_by, decided_at,
decision_reason on condition status = PENDING, plus the stricter `AND #c = :change` (the person decides the
change they saw). decided_by is the approver PSEUDONYM (person:team13-op1), never the ARN. The control line
in /team13/checkpoint carries the caller ARN in `by`, exactly as the kit writes it; the ARN never reaches a
response. Approving does not apply: noc_lead applies an APPROVED change on its next apply_change call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Path as PathParam, Request
from fastapi.responses import JSONResponse

from spectrum.access import is_loopback

from . import audit, logic
from . import aws as awsmod
from .pin import PinCheck, state_dir

log = logging.getLogger("spectrum.slices.kitapprovals")
router = APIRouter(prefix="/api/kitapprovals")

DEFAULT_PSEUDONYM = "person:team13-op1"
CLIENT_HEADER = "kitapprovals"
MAX_BODY = 4096
SAFE_CASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
NO_STORE = {"Cache-Control": "no-store"}
NOT_APPLIED_NOTE = ("Approving does not apply the change. noc_lead applies it on its next apply_change call: "
                    "say \"Approved.\" in Ask the NOC (or use the kit's `ask --case`).")


class Service:
    """Everything the routes share. Tests replace it through app.dependency_overrides[get_service]."""

    def __init__(self, aws: awsmod.Aws | None = None, offline: bool | None = None, pin: PinCheck | None = None,
                 clock=time.time):
        self.fixture = os.environ.get("KITAPPROVALS_FIXTURE")
        self._aws = aws
        self.offline = (os.environ.get("SPECTRUM_OFFLINE") == "1") if offline is None else offline
        if self._aws is None and self.fixture:
            from .fake import fixture

            d, lg, st = fixture(self.fixture)
            self._aws = awsmod.Aws(d, lg, st, "FIXTURE")
            self.offline = False
        self.pin = pin or PinCheck()
        self.clock = clock
        self.decide_lock = asyncio.Lock()
        self._sts_cache: tuple[float, str] | None = None
        self._key: bytes | None = None
        self._last_drain = 0.0

    @property
    def label(self) -> str:
        if self.offline:
            return "OFFLINE"
        return self.aws.label

    @property
    def aws(self) -> awsmod.Aws:
        if self._aws is None:
            self._aws = awsmod.live()
        return self._aws

    def approvers(self) -> dict[str, str]:
        p = os.environ.get("APPROVERS")
        if not p:
            return {}
        try:
            data = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)
                and logic.PSEUDONYM_RE.fullmatch(v)} if isinstance(data, dict) else {}

    def key(self) -> bytes:
        """HMAC key for display pseudonyms of non-pseudonymous decided_by values (stable across restarts)."""
        if self._key is None:
            p = state_dir() / "display.key"
            try:
                self._key = p.read_bytes()
            except OSError:
                k = secrets.token_bytes(32)
                try:
                    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(k)
                except OSError:
                    pass
                self._key = k
        return self._key

    def pseudonym_for(self, arn: str) -> str:
        """Who is written into decided_by: APPROVERS[session name] if mapped, else the fixed pseudonym."""
        name = logic.session_name(arn)
        mapped = self.approvers().get(name or "")
        if mapped:
            return mapped
        p = os.environ.get("APPROVALS_PSEUDONYM", DEFAULT_PSEUDONYM)
        return p if logic.PSEUDONYM_RE.fullmatch(p) else DEFAULT_PSEUDONYM

    def caller_arn(self, fresh: bool) -> str:
        now = self.clock()
        if not fresh and self._sts_cache and now - self._sts_cache[0] < 60:
            return self._sts_cache[1]
        arn = self.aws.sts.get_caller_identity()["Arn"]
        self._sts_cache = (now, arn)
        return arn

    def view(self, item: dict) -> dict:
        return logic.row_view(logic.plain(item), self.clock(), self.approvers(), self.key())


_service: Service | None = None


def get_service() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service


def _err(http: int, error: str, message: str, /, **extra) -> JSONResponse:
    return JSONResponse({"error": error, "message": message, **extra}, status_code=http, headers=NO_STORE)


def _host_ok(request: Request) -> bool:
    host = (request.headers.get("host") or "").lower()
    name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
    return name in ("localhost", "127.0.0.1", "[::1]")


def _laptop(request: Request) -> JSONResponse | None:
    if not is_loopback(tuple(request.client) if request.client else None) or not _host_ok(request):
        return _err(403, "laptop_only", "Real approvals work only on the presenter laptop (http://localhost:8787).")
    return None


def _post_guard(request: Request) -> JSONResponse | None:
    refused = _laptop(request)
    if refused:
        return refused
    h = request.headers
    if h.get("origin") != f"http://{h.get('host')}":
        return _err(403, "bad_origin", "The request must come from this console's own page.")
    if h.get("sec-fetch-site") not in (None, "same-origin", "none"):
        return _err(403, "bad_origin", "Cross-site requests are refused.")
    if h.get("x-spectrum-client") != CLIENT_HEADER:
        return _err(403, "bad_client", "Missing the X-Spectrum-Client header.")
    if not (h.get("content-type") or "").startswith("application/json"):
        return _err(415, "bad_content_type", "JSON only.")
    try:
        n = int(h.get("content-length") or "")
    except ValueError:
        return _err(411, "length_required", "Content-Length is required.")
    if n > MAX_BODY:
        return _err(413, "too_large", "The request is too large.")
    return None


async def _call(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)


def _aws_failure(exc: BaseException) -> JSONResponse:
    mode = awsmod.classify(exc)
    log.warning("kitapprovals: AWS call failed (%s)", awsmod.error_code(exc))
    return _err(503, mode, awsmod.MESSAGES[mode])


# ------------------------------------------------------------------------------------------ reads
@router.get("/status")
async def status(request: Request, svc: Service = Depends(get_service)):
    if _laptop(request):
        return JSONResponse({"mode": "laptop_only", "label": "LAPTOP ONLY", "message":
                             "Real approvals work only on the presenter laptop."}, headers=NO_STORE)
    out: dict[str, Any] = {"label": svc.label, "approver": None, "pin": svc.pin.state(),
                           "audit_outbox": audit.pending_count(), "note": NOT_APPLIED_NOTE}
    if svc.offline:
        return JSONResponse({**out, "mode": "offline", "message": awsmod.MESSAGES["offline"]}, headers=NO_STORE)
    try:
        arn = await _call(svc.caller_arn, False)
    except Exception as exc:  # noqa: BLE001
        mode = awsmod.classify(exc)
        return JSONResponse({**out, "mode": mode, "message": awsmod.MESSAGES[mode]}, headers=NO_STORE)
    out["approver"] = {"pseudonym": svc.pseudonym_for(arn), "aws_session": "AWS SSO session OK"}
    if out["audit_outbox"] and svc.clock() - svc._last_drain > 30:
        svc._last_drain = svc.clock()
        try:
            out["audit_outbox"] = await _call(audit.drain, svc.aws.logs)
        except Exception:  # noqa: BLE001
            pass
    return JSONResponse({**out, "mode": "ok"}, headers=NO_STORE)


def _scan_all(svc: Service) -> list[dict]:
    kw: dict[str, Any] = dict(
        TableName=awsmod.TABLE, ConsistentRead=True,
        FilterExpression="begins_with(pk, :chg) AND attribute_exists(#s)",
        ProjectionExpression="pk, #s, #ag, case_id, #c, created_at, decided_by, decided_at, decision_reason, applied_at",
        ExpressionAttributeNames={"#s": "status", "#ag": "agent", "#c": "change"},
        ExpressionAttributeValues={":chg": {"S": "chg-"}},
    )
    items: list[dict] = []
    for _ in range(1000):
        resp = svc.aws.ddb.scan(**kw)
        items += [awsmod.to_item(i) for i in resp.get("Items", [])]
        if not resp.get("LastEvaluatedKey"):
            return items
        kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    raise RuntimeError("Scan did not finish")


@router.get("/pending")
async def pending(request: Request, svc: Service = Depends(get_service)):
    refused = _laptop(request)
    if refused:
        return refused
    read_at = time.strftime("%H:%M:%S")
    if svc.offline:
        return JSONResponse({"mode": "offline", "label": "OFFLINE", "rows": [], "read_at": read_at,
                             "message": awsmod.MESSAGES["offline"]}, headers=NO_STORE)
    try:
        items = await _call(_scan_all, svc)
    except Exception as exc:  # noqa: BLE001 - shown as a state, never as an empty "nothing waiting"
        mode = awsmod.classify(exc)
        log.warning("kitapprovals: scan failed (%s)", awsmod.error_code(exc))
        return JSONResponse({"mode": mode, "label": svc.label, "rows": [], "read_at": read_at,
                             "message": awsmod.MESSAGES[mode]}, headers=NO_STORE)
    rows = logic.order_rows([svc.view(i) for i in items if str(i.get("pk", "")).startswith("chg-") and "status" in i])
    return JSONResponse({"mode": "ok", "label": svc.label, "rows": rows, "read_at": read_at}, headers=NO_STORE)


def _get(svc: Service, pk: str) -> dict | None:
    resp = svc.aws.ddb.get_item(TableName=awsmod.TABLE, Key={"pk": {"S": pk}}, ConsistentRead=True)
    return awsmod.to_item(resp.get("Item"))


def _context(svc: Service, row: dict) -> dict:
    """Case, prediction and trust for one row. Every part may be missing; nothing is filled in."""
    out: dict[str, Any] = {"case": None, "case_note": None, "prediction": None, "prediction_note": None,
                           "trust": None, "warnings": []}
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not SAFE_CASE.fullmatch(case_id):
        out["case_note"] = "No case id on this row."
        return out
    case_row = _get(svc, "t3-" + case_id)
    doc = None
    if case_row is None:
        out["case_note"] = "No twin case row (t3-<case>) for this case: not a RAN-tools case, or not started."
    elif not isinstance(case_row.get("doc"), str):
        out["case_note"] = "The case row has no state yet (no doc)."
    else:
        try:
            doc = json.loads(case_row["doc"])
        except ValueError:
            out["case_note"] = "The case row's state could not be read."
    if not isinstance(doc, dict):
        return out
    step_now = len(doc.get("action_log") or [])
    out["case"] = {"run_id": doc.get("run_id"), "step_now": step_now,
                   "twin_kind": (doc.get("twin") or {}).get("kind") if isinstance(doc.get("twin"), dict) else None}
    p = logic.find_prediction(doc, row.get("item_id"), row.get("canonical_text"))
    if p is None:
        out["prediction_note"] = ("No prediction is stored for this change (ran___predict wasn't called). "
                                  "You'd be approving without a forecast; in grounded mode the apply will be refused.")
        return out
    pred = p.get("predictor") if isinstance(p.get("predictor"), dict) else {}
    stale = p.get("step") != step_now
    out["prediction"] = {"predicted_delta": logic.plain(p.get("predicted_delta")), "step": p.get("step"),
                         "stale": stale, "predictor": {"model_id": pred.get("model_id"), "version": pred.get("version")},
                         "source": "case row t3-<case> doc.predictions (ran___predict)"}
    if stale:
        out["warnings"].append(f"Predicted at step {p.get('step')}; the twin is now at step {step_now}. "
                               "In grounded mode the apply will be refused.")
    mid, ver = pred.get("model_id"), pred.get("version")
    if isinstance(mid, str) and isinstance(ver, str):
        t = _get(svc, f"t3-trust#{mid}@{ver}")
        if t:
            out["trust"] = {"trust": logic.plain(t.get("trust")), "gate_status": t.get("gate_status", "active"),
                            "n": logic.plain(t.get("n")), "source": f"t3-trust#{mid}@{ver}"}
            if t.get("gate_status") == "paused":
                out["warnings"].append("The response model is paused. In grounded mode the apply will be refused "
                                       "until a person resumes it.")
    return out


@router.get("/{approval_id}")
async def detail(request: Request, approval_id: str = PathParam(pattern=logic.APPROVAL_ID_RE),
                 svc: Service = Depends(get_service)):
    refused = _laptop(request)
    if refused:
        return refused
    if svc.offline:
        return _err(503, "offline", awsmod.MESSAGES["offline"])
    try:
        item = await _call(_get, svc, approval_id)
        if item is None or "status" not in item:
            return _err(404, "not_found", f"{approval_id} is not in the approvals table.")
        row = svc.view(item)
        ctx = await _call(_context, svc, row)
    except Exception as exc:  # noqa: BLE001
        return _aws_failure(exc)
    body = logic.parse_change_attr(item.get("change"))
    row["agent_reason"] = logic.redact(body.get("reason"))
    return JSONResponse({**row, **ctx, "label": svc.label, "note": NOT_APPLIED_NOTE,
                         "links": {"ask_noc": f"/ask-noc?case={row['case_id']}", "trace": f"/trace?case={row['case_id']}",
                                   "twin_view": f"/twin-view?case={row['case_id']}"}}, headers=NO_STORE)


# ------------------------------------------------------------------------------------------ the decision
def _body(raw: Any) -> tuple[dict | None, JSONResponse | None]:
    """Manual validation, so a refused body is never echoed back (it holds the PIN)."""
    if not isinstance(raw, dict) or set(raw) != {"decision", "reason", "change_sha256", "pin"}:
        return None, _err(422, "bad_body", "Send exactly: decision, reason, change_sha256, pin.")
    if raw["decision"] not in ("approve", "reject"):
        return None, _err(422, "bad_body", "decision must be approve or reject.")
    if not isinstance(raw["change_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", raw["change_sha256"]):
        return None, _err(422, "bad_body", "change_sha256 must be the 64-hex hash shown with the change.")
    if not isinstance(raw["pin"], str) or not isinstance(raw["reason"], str):
        return None, _err(422, "bad_body", "pin and reason must be strings.")
    try:
        reason = logic.clean_reason(raw["reason"])
    except ValueError as e:
        return None, _err(422, "bad_reason", str(e).capitalize() + ".")
    return {**raw, "reason": reason}, None


def _who(svc: Service, item: dict) -> dict:
    return {"status": item.get("status"), "by": logic.display_who(item.get("decided_by"), svc.approvers(), svc.key()),
            "decided_at": logic.num(logic.plain(item.get("decided_at")))}


def _decide(svc: Service, approval_id: str, body: dict) -> JSONResponse:
    """Blocking part of the decision (runs in a thread)."""
    from botocore.exceptions import ClientError

    if audit.pending_count():
        if audit.drain(svc.aws.logs):
            return _err(503, "audit_pending", "An earlier decision is not audited yet (CloudWatch control line). "
                        "No new decision until it is written; press Refresh to retry.")
    arn = svc.caller_arn(fresh=True)
    pseudonym = svc.pseudonym_for(arn)
    item = _get(svc, approval_id)
    if item is None or "status" not in item:
        return _err(404, "not_found", f"{approval_id} is not in the approvals table.")
    if item["status"] != "PENDING":
        w = _who(svc, item)
        return _err(409, "not_pending", f"Already {w['status']}. Nothing changed.", **w)
    raw_change = item.get("change")
    if not isinstance(raw_change, str) or logic.sha256_text(raw_change) != body["change_sha256"]:
        return _err(409, "change_mismatch", "The change on record differs from what you reviewed. Review it again.")
    status = {"approve": "APPROVED", "reject": "REJECTED"}[body["decision"]]
    t = svc.clock()
    decided_at = int(t)
    try:
        resp = svc.aws.ddb.update_item(
            TableName=awsmod.TABLE, Key={"pk": {"S": approval_id}}, ReturnValues="ALL_NEW",
            UpdateExpression="SET #s = :s, decided_by = :by, decided_at = :t, decision_reason = :why",
            ConditionExpression="#s = :pending AND #c = :change",
            ExpressionAttributeNames={"#s": "status", "#c": "change"},
            ExpressionAttributeValues={":s": {"S": status}, ":by": {"S": pseudonym}, ":t": {"N": str(decided_at)},
                                       ":why": {"S": body["reason"]}, ":pending": {"S": "PENDING"},
                                       ":change": {"S": raw_change}},
            ReturnValuesOnConditionCheckFailure="ALL_OLD")
    except ClientError as exc:
        if awsmod.error_code(exc) != "ConditionalCheckFailedException":
            raise
        old = awsmod.to_item(exc.response.get("Item")) or {}
        if old.get("status") == "PENDING":
            return _err(409, "change_mismatch", "The change on record differs from what you reviewed. Review it again.")
        w = _who(svc, old)
        return _err(409, "not_pending", f"Someone decided first: {w['status'] or 'gone'}. Nothing changed.", **w)
    new = awsmod.to_item(resp.get("Attributes")) or {}
    message = audit.control_message(arn, t, status, approval_id, new.get("case_id", item.get("case_id")),
                                    new.get("change", raw_change), body["reason"])
    audited = audit.write(svc.aws.logs, message)
    log.info("kitapprovals: %s %s by %s (audit %s)", approval_id, status, pseudonym, audited)
    return JSONResponse({"approval_id": approval_id, "status": status, "decided_at": decided_at, "by": pseudonym,
                         "audit": audited, "note": NOT_APPLIED_NOTE if status == "APPROVED" else None,
                         "audit_message": None if audited == "written" else
                         "Decided but NOT audited: the CloudWatch control line could not be written. It is kept "
                         "locally with its original time and retried; no new decision until it is written."},
                        headers=NO_STORE)


@router.post("/{approval_id}/decide")
async def decide(request: Request, approval_id: str = PathParam(pattern=logic.APPROVAL_ID_RE),
                 svc: Service = Depends(get_service)):
    refused = _post_guard(request)
    if refused:
        return refused
    try:
        raw = json.loads(await request.body())
    except ValueError:
        return _err(422, "bad_body", "The body is not JSON.")
    body, bad = _body(raw)
    if bad:
        return bad
    if svc.offline:
        return _err(503, "offline", awsmod.MESSAGES["offline"])
    if svc.decide_lock.locked():
        return _err(429, "busy", "Another decision is in flight.")
    async with svc.decide_lock:
        verdict = await _call(svc.pin.verify, body["pin"])
        if verdict != "ok":
            msg = {"pin_wrong": "Wrong PIN. Nothing was decided.",
                   "pin_locked": "Too many wrong PINs: decisions are locked for 15 minutes.",
                   "pin_not_set": "No approver PIN is set on this laptop. Set one first (see the page)."}[verdict]
            return _err(403, verdict, msg, pin=svc.pin.state())
        try:
            return await _call(_decide, svc, approval_id, body)
        except Exception as exc:  # noqa: BLE001
            if awsmod.maybe_landed(exc):  # a timeout: the update may have happened
                log.warning("kitapprovals: decision outcome unknown (%s)", awsmod.error_code(exc))
                return _err(503, "outcome_unknown", "AWS did not answer in time: the decision may or may not "
                            "have been recorded. Press Refresh and look at the row before trying again.")
            return _aws_failure(exc)
