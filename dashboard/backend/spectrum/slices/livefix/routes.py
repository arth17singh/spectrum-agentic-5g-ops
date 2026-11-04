"""HTTP routes of Live Fix (/api/livefix). Laptop only, with Ask the NOC's request guard rules (loopback client,
local Host, same-origin POST with a custom header). Every response is no-store. No route approves anything: the
decision is the person's, on the Approvals path (PIN) or the kit CLI.
"""

from __future__ import annotations

import asyncio
import os
import re

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from spectrum.access import is_loopback

from ..asknoc.guard import LOCAL, MAX_BODY, hostname, origin_ok, refuse
from .service import LiveFix

CLIENT_HEADER = "livefix"
RUN_ID = re.compile(r"^(t3\.livefix|recorded)-[0-9TZ]+(-[0-9a-f]{4})?$")


async def guard(request: Request) -> None:
    if not is_loopback(request.client):
        raise refuse(403, "loopback_only", "Live Fix works only on the presenter laptop.")
    if hostname(request.headers.get("host", "")) not in LOCAL:
        raise refuse(403, "bad_host", "Open the console on the presenter laptop (its loopback address).")
    site = (request.headers.get("sec-fetch-site") or "").lower()
    if request.method in ("GET", "HEAD"):
        if site == "cross-site":
            raise refuse(403, "cross_site", "Cross-site requests are refused.")
        return
    origin = request.headers.get("origin")
    if not origin or not origin_ok(origin):
        raise refuse(403, "bad_origin", "Requests must come from this console's own page.")
    if site and site not in ("same-origin", "none"):
        raise refuse(403, "cross_site", "Cross-site requests are refused.")
    if request.headers.get("x-spectrum-client") != CLIENT_HEADER:
        raise refuse(403, "missing_client_header", "Requests must come from the Live Fix page.")
    if (request.headers.get("content-type") or "").split(";")[0].strip().lower() != "application/json":
        raise refuse(415, "bad_content_type", "Send JSON.")
    try:
        n = int(request.headers.get("content-length") or "")
    except ValueError:
        raise refuse(411, "length_required", "Content-Length is required.") from None
    if n > MAX_BODY:
        raise refuse(413, "too_large", "The request is too large.")


router = APIRouter(prefix="/api/livefix", dependencies=[Depends(guard)])
_service: LiveFix | None = None


def build_service(offline: bool | None = None) -> LiveFix:
    off = os.environ.get("SPECTRUM_OFFLINE") == "1" if offline is None else offline
    if off:
        return LiveFix()
    from ..asknoc.config import KitConfig
    from .gateway import LiveFixKit

    def ddb():
        import boto3
        from botocore.config import Config

        return boto3.session.Session(region_name=os.environ.get("ASKNOC_REGION") or "us-east-2").client(
            "dynamodb", config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 3}))

    def writer():
        import boto3

        from .evidence import EvlibWriter

        account = boto3.client("sts").get_caller_identity()["Account"]
        return EvlibWriter.from_env(account)

    kit_cfg = KitConfig()
    kit = LiveFixKit(kit_cfg) if kit_cfg.load().configured else None
    try:
        client = ddb()
    except Exception:  # noqa: BLE001 - no AWS session: live mode answers 503, recorded mode still works
        client = None
    from .twinside import TwinSide

    def attack_gateway():
        """The suite's Gateway client with the kit's agent tokens (tokens/<agent>.txt): the rogue-agent attack calls
        as ran_monitor. The URL and tokens never leave this process."""
        from .suite import repo_import

        common = repo_import("t3.suite.common")
        view = kit_cfg.load()
        return common.Gateway(view.gateway_url.reveal(), common.Tokens(tokens_dir=str(kit_cfg.dir / "tokens")))

    svc = LiveFix(kit=kit, ddb=client, writer_factory=writer, twinside=TwinSide(),
                  attack_gateway=attack_gateway if kit is not None else None,
                  table=os.environ.get("ASKNOC_APPROVALS_TABLE") or "team13-approvals")
    if kit is not None:
        from ..asknoc.redact import Redactor

        svc.redact = lambda s: Redactor(kit_cfg.load().redact_values).text(s)
    return svc


def get_service() -> LiveFix:
    global _service
    if _service is None:
        _service = build_service()
    return _service


def set_service(svc: LiveFix | None) -> None:
    global _service
    _service = svc


class StartBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    incident_id: str | None = Field(default=None, max_length=80, pattern=r"^inc-[A-Za-z0-9:+|._-]{1,72}$")
    fault_key: str | None = Field(default=None, pattern=r"^(tilt_drift|pa_fault|bad_config|hardware|cascade)$")
    cell: int | None = Field(default=None, ge=0, le=3)
    preset: str = Field(default="official", pattern=r"^(official|viavi-like)$")
    mode: str = Field(pattern=r"^(live|recorded)$")


class AttackBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(pattern=r"^(poisoned_alarm|rogue_agent)$")
    variant: str = Field(default="judged", pattern=r"^(judged|alarm)$")


class EmptyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _ns(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _run(svc: LiveFix, run_id: str):
    if not RUN_ID.fullmatch(run_id) or run_id not in svc.runs:
        raise refuse(404, "unknown_run", "No such Live Fix run on this console.")
    return svc.runs[run_id]


@router.get("/incidents")
async def incidents(response: Response):
    _ns(response)
    svc = get_service()
    out = await asyncio.to_thread(svc.incidents)
    return {**out, "live_ready": svc.kit is not None and svc.ddb is not None,
            "faults_ready": svc.twinside.available, "attacks_ready": svc.attack_gateway is not None}


@router.get("/runs")
async def runs(response: Response):
    _ns(response)
    svc = get_service()
    return {"runs": [svc.view(r) for r in list(svc.runs.values())[-10:]]}


@router.post("/runs", status_code=202)
async def start(body: StartBody, response: Response):
    _ns(response)
    svc = get_service()
    try:
        if body.fault_key is None and body.incident_id is None:
            raise refuse(422, "bad_body", "Send a fault_key and cell, or an incident_id.")
        run = await asyncio.to_thread(svc.start, body.incident_id, body.mode, fault_key=body.fault_key,
                                      cell=body.cell, preset=body.preset)
    except KeyError:
        raise refuse(404, "unknown_incident", "That incident is not in the list.") from None
    except RuntimeError as e:
        raise refuse(503, "live_unavailable", str(e)) from None
    except ValueError as e:
        raise refuse(409, "busy", str(e)) from None
    return svc.view(run)


@router.get("/runs/{run_id}")
async def run_view(run_id: str, response: Response):
    _ns(response)
    svc = get_service()
    return svc.view(_run(svc, run_id))


@router.post("/runs/{run_id}/inject", status_code=202)
async def inject(run_id: str, body: EmptyBody, response: Response):
    _ns(response)
    svc = get_service()
    _run(svc, run_id)
    try:
        return svc.view(svc.inject(run_id))
    except ValueError as e:
        raise refuse(409, "not_ready", str(e)) from None


@router.post("/runs/{run_id}/attack", status_code=202)
async def attack(run_id: str, body: AttackBody, response: Response):
    """One attack call through the Gateway on this run's case (expected: refused). The `alarm` variant is not a
    judged case and is labelled so."""
    _ns(response)
    svc = get_service()
    _run(svc, run_id)
    try:
        return svc.view(svc.attack(run_id, body.key, body.variant))
    except RuntimeError as e:
        raise refuse(503, "attack_unavailable", str(e)) from None
    except ValueError as e:
        raise refuse(409, "not_ready", str(e)) from None


@router.post("/runs/{run_id}/continue", status_code=202)
async def recorded_continue(run_id: str, body: EmptyBody, response: Response):
    """Recorded runs only: move past the (recorded) gate. A live run never moves on without a person's
    decision in the approvals table."""
    _ns(response)
    svc = get_service()
    _run(svc, run_id)
    try:
        return svc.view(svc.recorded_continue(run_id))
    except ValueError as e:
        raise refuse(409, "not_recorded", str(e)) from None


@router.get("/phone")
async def phone(request: Request, response: Response):
    """Can a phone on the Wi-Fi reach this server? A TCP connect to our own LAN address and port: a server bound to
    127.0.0.1 refuses it. Also whether the approvals slice accepts paired phones (KITAPPROVALS_PHONE=1). Never the
    token: the page takes it from /api/config, which hands it to this laptop only."""
    _ns(response)
    import socket

    from spectrum.main import _lan_ip as lan_ip

    ip, port = lan_ip(), request.url.port or 80

    def probe() -> bool:
        if ip.startswith("127."):
            return False
        try:
            with socket.create_connection((ip, port), timeout=0.5):
                return True
        except OSError:
            return False

    reachable = await asyncio.to_thread(probe)
    enabled = os.environ.get("KITAPPROVALS_PHONE") == "1"
    note = None if reachable and enabled else (
        "Phone approval needs `make demo` on the same Wi-Fi: this server listens on this laptop only."
        if not reachable else "Phone approval is off on this server (start it with `make demo`).")
    return {"lan_ip": ip, "port": port, "reachable": reachable, "phone_approvals": enabled,
            "ready": reachable and enabled, "note": note}


@router.get("/runs/{run_id}/activity")
async def activity(run_id: str, response: Response):
    """What the agents did in the faulted case: the kit's audit log lines in CloudWatch (Ask the NOC's parser)."""
    _ns(response)
    svc = get_service()
    run = _run(svc, run_id)
    case_id = run.cases.get("faulted")
    if run.mode != "live" or not case_id:
        return {"available": False, "events": []}
    from ..asknoc.activity import parse_events
    from ..asknoc.aws import AwsReads, AwsUnavailable
    from ..asknoc.config import KitConfig
    from ..asknoc.redact import Redactor

    try:
        events = await asyncio.to_thread(AwsReads().filter_events, case_id, int((run.created - 300) * 1000))
    except AwsUnavailable as e:
        return {"available": False, "message": str(e), "events": []}
    redactor = Redactor(KitConfig().load().redact_values)
    return {"available": True, "source": "the kit's audit log in CloudWatch, filtered by case id",
            "lag_note": "Log lines can arrive 5 to 30 s after the call.",
            "events": parse_events(events, case_id, redactor)}
