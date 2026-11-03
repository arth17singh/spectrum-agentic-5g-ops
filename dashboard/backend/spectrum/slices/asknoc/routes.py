"""HTTP routes of Ask the NOC (/api/asknoc). Handlers are `async def` on the loop thread (main.py:64-66);
every blocking call (urllib, boto3) runs in asyncio.to_thread inside the service. Every response is
no-store. Errors are {"detail": {"error": code, "message": operator copy}}.
"""

from __future__ import annotations

import hashlib
import os
import re

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from .aws import AwsReads
from .config import KitConfig
from .guard import guard, refuse
from .kit import KitClient
from .service import AskNoc, RefusedRequest
from .store import Store
from .validate import CASE_ID, validate_text

router = APIRouter(prefix="/api/asknoc", dependencies=[Depends(guard)])

KEY = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
TURN = re.compile(r"^t-[0-9a-f]{12}$")
_service: AskNoc | None = None


def build_service(offline: bool | None = None) -> AskNoc:
    off = os.environ.get("SPECTRUM_OFFLINE") == "1" if offline is None else offline
    config = KitConfig()
    return AskNoc(config, KitClient(config), Store(memory=off), AwsReads(), offline=off)


def get_service() -> AskNoc:
    global _service
    if _service is None:
        _service = build_service()
    return _service


def set_service(svc: AskNoc | None) -> None:
    global _service
    _service = svc


class TextBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=4000)


class EmptyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _http(e: RefusedRequest):
    headers = {"Retry-After": str(int(e.extra["retry_after_s"]) + 1)} if "retry_after_s" in e.extra else None
    exc = refuse(e.status, e.code, e.message, headers)
    exc.detail.update({k: v for k, v in e.extra.items()})
    return exc


def _limit(svc: AskNoc, bucket: str, n: int, per_s: float) -> None:
    wait = svc.limited(bucket, n, per_s)
    if wait is not None:
        raise _http(RefusedRequest(429, "rate_limited", f"Too many requests; try again in {int(wait) + 1} s.",
                                   retry_after_s=wait))


def _key(value: str | None) -> str:
    if not value or not KEY.match(value):
        raise refuse(400, "idempotency_key_required", "Each send needs an Idempotency-Key (a UUID).")
    return value.lower()


def _text(svc: AskNoc, raw: str, case_id: str | None = None) -> str:
    text, reasons = validate_text(raw, case_id, svc.redactor() if svc.config.load().configured else None)
    if reasons:
        exc = refuse(400, "invalid_text", "The message can't be sent as it is.")
        exc.detail["reasons"] = reasons
        raise exc
    return text


@router.get("/health")
async def health(request: Request, response: Response):
    _no_store(response)
    svc = get_service()
    _limit(svc, "health", 20, 10)
    return await svc.health(bool(getattr(request.state, "loopback", False)))


@router.get("/cases")
async def cases(response: Response):
    _no_store(response)
    svc = get_service()
    _limit(svc, "get", 40, 1)
    return {"cases": svc.cases(), "pending": svc.pending_turns(), "offline": svc.offline}


@router.post("/cases", status_code=202)
async def start(body: TextBody, response: Response, idempotency_key: str | None = Header(default=None)):
    _no_store(response)
    svc = get_service()
    key = _key(idempotency_key)
    digest = hashlib.sha256(("start\0" + body.text).encode()).hexdigest()
    try:  # no await between the key check, the limits and the in-flight reservation
        replay = svc.reserve_key(key, digest)
        if replay is not None:
            return replay
        text = _text(svc, body.text)
        _limit(svc, "start", 6, 600)
        out = svc.start(text)
    except RefusedRequest as e:
        if e.code not in ("idempotency_conflict", "duplicate_in_progress"):
            svc.release_key(key)
        raise _http(e) from None
    except BaseException:
        svc.release_key(key)
        raise
    svc.remember_key(key, digest, out)
    return out


@router.get("/cases/{case_id}")
async def case(case_id: str, response: Response):
    _no_store(response)
    svc = get_service()
    _limit(svc, "get", 40, 1)
    if not CASE_ID.fullmatch(case_id):
        raise refuse(404, "unknown_case", "No such case.")
    try:
        return svc.case_view(case_id)
    except RefusedRequest as e:
        raise _http(e) from None


@router.post("/cases/{case_id}/turns", status_code=202)
async def follow_up(case_id: str, body: TextBody, response: Response,
                    idempotency_key: str | None = Header(default=None)):
    _no_store(response)
    svc = get_service()
    key = _key(idempotency_key)
    if not CASE_ID.fullmatch(case_id):
        raise refuse(404, "unknown_case", "No such case.")
    digest = hashlib.sha256(f"continue\0{case_id}\0{body.text}".encode()).hexdigest()
    try:
        replay = svc.reserve_key(key, digest)
        if replay is not None:
            return replay
        text = _text(svc, body.text, case_id)
        _limit(svc, "continue", 20, 600)
        out = await svc.follow_up(case_id, text)
    except RefusedRequest as e:
        if e.code not in ("idempotency_conflict", "duplicate_in_progress"):
            svc.release_key(key)
        raise _http(e) from None
    except BaseException:
        svc.release_key(key)
        raise
    svc.remember_key(key, digest, out)
    return out


@router.get("/cases/{case_id}/activity")
async def activity(case_id: str, response: Response):
    _no_store(response)
    svc = get_service()
    _limit(svc, "get", 40, 1)
    if not CASE_ID.fullmatch(case_id):
        raise refuse(404, "unknown_case", "No such case.")
    try:
        return await svc.activity(case_id)
    except RefusedRequest as e:
        raise _http(e) from None


@router.get("/turns/{turn_id}")
async def turn(turn_id: str, response: Response):
    _no_store(response)
    svc = get_service()
    _limit(svc, "get", 40, 1)
    t = svc.store.turns.get(turn_id) if TURN.match(turn_id) else None
    if t is None:
        raise refuse(404, "unknown_turn", "No such message.")
    return svc.turn_view(t)


@router.post("/turns/{turn_id}/stop")
async def stop(turn_id: str, body: EmptyBody, response: Response):
    _no_store(response)
    svc = get_service()
    _limit(svc, "stop", 30, 60)
    if not TURN.match(turn_id):
        raise refuse(404, "unknown_turn", "No such message.")
    try:
        return svc.stop_following(turn_id)
    except RefusedRequest as e:
        raise _http(e) from None
