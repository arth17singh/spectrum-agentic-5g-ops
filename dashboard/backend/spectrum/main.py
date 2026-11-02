"""FastAPI app: REST commands + snapshot, one WebSocket stream (batched every 100 ms).

Run:  uv run uvicorn spectrum.main:app --host 127.0.0.1 --port 8787   (this laptop only, as `make dev`)
      uv run uvicorn spectrum.main:app --host 0.0.0.0 --port 8787     (phones too, as `make demo`)

Devices other than this laptop must send the access token (see access.py): the X-Spectrum-Token
header, or ?t= on the WebSocket. It changes on every start unless SPECTRUM_TOKEN pins it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .access import TOKEN, AccessGate, is_loopback, log_phone_link
from .adapters import load_adapters
from .engine import Engine
from .evidence.service import EvidenceService
from .evidence.store import utc_iso

MODE = os.environ.get("SPECTRUM_MODE", "mock")
SEED = int(os.environ.get("SPECTRUM_SEED", "7"))
FLAGS_FILE = Path(__file__).parent / "flags.json"

engine = Engine(seed=SEED)
adapters = load_adapters(MODE)
# The REAL evidence ledger lives beside the engine, never inside it: reset() rebuilds the engine (and
# its bus) and must not touch it. Events resolve engine.bus at emit time for the same reason.
def _emit_evidence(*a, **k) -> dict:
    """Emit on the engine's current bus, stamped with real UTC time: evidence never uses the sim clock."""
    ev = engine.bus.emit(*a, **k)
    ev["ts"] = utc_iso()  # same dict the queues and history hold; the socket serialises it later
    ev["clock"] = "wall"
    return ev


evidence = EvidenceService(emit=_emit_evidence)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_phone_link(_lan_ip())
    task = asyncio.create_task(engine.run())
    ev_task = asyncio.create_task(evidence.run())
    yield
    task.cancel()
    ev_task.cancel()


# Every handler is `async def` on purpose: it then runs on the event-loop thread, serialised with
# the simulation loop. Plain `def` handlers run in a thread pool and race the loop (interleaved
# ledger appends corrupt the hash chain).
app = FastAPI(title="SPECTRUM console API", version="0.1.0", lifespan=lifespan)
# No CORS: the UI is same-origin under `make demo`, and Vite proxies /api and /ws under `make dev`.
app.add_middleware(AccessGate, token=TOKEN)


def ok(**kw):
    return {"ok": True, **kw}


# --------------------------------------------------------------------------- config & snapshot

@app.get("/api/config")
async def config(request: Request):
    cfg = {"mode": MODE, "seed": SEED, "version": "0.1.0", "warm_ms": engine.warm_ms,
           "adapters": adapters.describe(), "region": "us-west-2"}
    # the Director puts the token in its phone link; only this laptop is ever handed it
    return {**cfg, "access_token": TOKEN} if is_loopback(request.client) else cfg


@app.get("/api/config/flags")
async def remote_flags():
    return json.loads(FLAGS_FILE.read_text()) if FLAGS_FILE.exists() else {}


@app.get("/api/snapshot")
async def snapshot():
    return {**engine.snapshot(), "evidence": evidence.snapshot()}


# --------------------------------------------------------------------------- simulation

class SimBody(BaseModel):
    paused: bool | None = None
    speed: float | None = None


@app.post("/api/sim")
async def sim(body: SimBody):
    if body.paused is not None:
        engine.clock.paused = body.paused
    if body.speed is not None:
        engine.clock.speed = max(0.25, min(10.0, body.speed))
    engine.bus.emit("sim.clock", "system", "simulation " + ("paused" if engine.clock.paused else
                                                            f"running at {engine.clock.speed}×"),
                    engine.clock_view(), "notice", "director")
    return ok(clock=engine.clock_view())


# --------------------------------------------------------------------------- federation

class FlConfig(BaseModel):
    changes: dict


@app.post("/api/fl/{action}")
async def fl_action(action: str):
    fl = engine.fl
    fn = {"start": fl.start, "pause": fl.pause, "resume": fl.start, "stop": fl.stop, "next_round": fl.next_round,
          "checkpoint": lambda: fl.save_checkpoint(manual=True)}.get(action)
    if not fn:
        raise HTTPException(404, f"unknown action {action}")
    res = fn()
    return ok(result=res, run=fl.view_run())


@app.post("/api/fl-config")
async def fl_config(body: FlConfig):
    return ok(config=engine.fl.set_config(body.changes))


@app.get("/api/fl/projected_eps")
async def projected_eps(sigma: float, rounds: int | None = None):
    return engine.fl.projected_eps(sigma, rounds)


@app.post("/api/fl/checkpoint/{ck_id}/{action}")
async def checkpoint_action(ck_id: str, action: str):
    if action == "restore":
        return ok(checkpoint=engine.fl.restore_checkpoint(ck_id))
    if action == "promote":
        return ok(checkpoint=engine.fl.promote(ck_id))
    raise HTTPException(404)


@app.post("/api/fl/client/{op}/{action}")
async def fl_client(op: str, action: str):
    engine.fl.client(op, action)
    return ok()


@app.post("/api/fl/reveal")
async def fl_reveal(on: bool = True):
    engine.fl.reveal = on
    return ok()


# --------------------------------------------------------------------------- proving & release

class EvalBody(BaseModel):
    model: str | None = None


@app.post("/api/eval/run")
async def eval_run(body: EvalBody):
    if body.model is None and engine.proving.pending_candidate is None and engine.fl.checkpoints:
        engine.fl.promote(engine.fl.checkpoints[-1]["id"])
    return ok(run=engine.proving.run(body.model))


@app.get("/api/eval/aibom/{model}/verify")
async def aibom_verify(model: str):
    return engine.proving.verify_aibom(model)


@app.post("/api/release/{action}")
async def release_action(action: str):
    rl = engine.release
    if action == "advance":
        rl.advance()
    elif action == "rollback":
        rl.rollback("manual rollback by A. Rivera")
    elif action == "pause":
        rl.paused = not rl.paused
    elif action == "regression":
        rl.regression = True
    elif action == "shadow" and engine.proving.pending_candidate:
        rl.start_shadow(engine.proving.pending_candidate["id"])
    else:
        raise HTTPException(404)
    return ok(release=rl.view())


# --------------------------------------------------------------------------- twin & swarm

class InjectBody(BaseModel):
    kind: str
    target: str | None = None


@app.post("/api/twin/inject")
async def twin_inject(body: InjectBody):
    if body.kind == "incident":
        return ok(incident=engine.swarm.start_incident())
    if body.kind == "telemetry_injection":
        engine.swarm.telemetry_injection(body.target or "site-17")
        return ok()
    return ok(alarm=engine.twin.inject(body.kind, body.target))


@app.post("/api/swarm/{action}")
async def swarm_action(action: str):
    sw = engine.swarm
    if action == "freeze":
        sw.freeze(True)
    elif action == "resume":
        sw.freeze(False)
    elif action == "pause":
        sw.paused = not sw.paused
    else:
        raise HTTPException(404)
    return ok(frozen=sw.frozen, paused=sw.paused)


class AutonomyBody(BaseModel):
    cls: str
    level: int
    reason: str = "changed by operator"


@app.post("/api/autonomy")
async def autonomy(body: AutonomyBody):
    engine.swarm.set_autonomy(body.cls, body.level, body.reason)
    engine.bus.emit("autonomy.change", "prove", f"{body.cls} autonomy set to L{body.level}: {body.reason}",
                    {"class": body.cls, "to": body.level}, "notice", "human:a.rivera")
    return ok(autonomy=engine.swarm.autonomy)


@app.post("/api/agents/{aid}/{action}")
async def agent_action(aid: str, action: str):
    if aid not in engine.swarm.agents:
        raise HTTPException(404)
    if action == "quarantine":
        engine.swarm.quarantine(aid, "quarantined by operator", actor="human:a.rivera")
    elif action == "release":
        engine.swarm.release(aid)
    else:
        raise HTTPException(404)
    return ok(agent=engine.swarm.view_agent(engine.swarm.agents[aid]))


# --------------------------------------------------------------------------- oversight

class DecisionBody(BaseModel):
    by: str  # required: the decision is signed as this approver, so a missing one must not default to anybody
    device: str = "console"
    reason: str = ""
    changes: dict | None = None


@app.post("/api/approvals/{aid}/{decision}")
async def approval(aid: str, decision: str, body: DecisionBody):
    from .engines.oversight import HUMANS
    if aid not in engine.oversight.items:
        raise HTTPException(404)
    mapping = {"approve": "approved", "approve_with_changes": "approved_with_changes", "reject": "rejected",
               "escalate": "escalated"}
    if decision not in mapping:
        raise HTTPException(400, f"unknown decision {decision!r}; allowed: {', '.join(mapping)}")
    if body.by not in HUMANS:
        raise HTTPException(400, f"unknown approver {body.by!r}; known approvers: {', '.join(HUMANS)}")
    item = engine.oversight.decide(aid, mapping[decision], body.by, body.device, body.reason, body.changes)
    if item["status"] == "approved" and item["plan"]["id"].startswith("restore-"):
        engine.swarm.release(item["plan"]["id"][len("restore-"):])
    return ok(item=engine.oversight.view(item))


# --------------------------------------------------------------------------- trust: policy test bench

class PolicyTestBody(BaseModel):
    principal: str
    action: str
    resource: dict
    context: dict = {}
    status: str | None = None


@app.post("/api/trust/test")
async def trust_test(body: PolicyTestBody):
    """Evaluate a hypothetical request against the live policy set (no side effects, not ledgered)."""
    agent = engine.swarm.agents.get(body.principal)
    if not agent:
        raise HTTPException(404, f"unknown principal {body.principal}")
    principal = engine.swarm._principal(agent)
    if body.status:
        principal["status"] = body.status
    t0 = time.perf_counter()
    d = engine.trust.authorize(principal, body.action, body.resource, body.context, emit=False, record=False)
    return {**d, "eval_us": round((time.perf_counter() - t0) * 1e6, 1), "principal_view": principal}


# --------------------------------------------------------------------------- ledger & provenance

@app.post("/api/ledger/verify")
async def ledger_verify():
    return engine.ledger.verify()


@app.post("/api/ledger/tamper")
async def ledger_tamper(seq: int | None = None):
    return engine.ledger.tamper(seq)


@app.post("/api/ledger/restore")
async def ledger_restore():
    engine.ledger.restore()
    engine.bus.emit("ledger.restore", "prove", "tampered records restored from originals", {}, "notice", "director")
    return ok()


@app.get("/api/ledger/{seq}")
async def ledger_record(seq: int):
    if not 1 <= seq <= len(engine.ledger.records):
        raise HTTPException(404)
    rec = engine.ledger.records[seq - 1]
    prev = engine.ledger.records[seq - 2]["hash"] if seq > 1 else "0" * 64
    return {"record": rec, "verification": engine.ledger.verify_record(rec, prev),
            "public_key": engine.ledger.public_key(rec["key_id"])}


@app.get("/api/ledger/proof/{seq}")
async def ledger_proof(seq: int):
    return engine.ledger.proof(seq)


@app.get("/api/keys/{key_id:path}")
async def key_info(key_id: str):
    return engine.ledger.public_key(key_id)


@app.get("/api/provenance/latest")
async def provenance_latest():
    from .provenance import build
    return build(engine)


@app.get("/api/dossier/{incident}")
async def dossier(incident: str):
    from .provenance import dossier as build_dossier
    return build_dossier(engine, incident)


# --------------------------------------------------------------------------- evidence (REAL, persisted)

@app.get("/api/evidence")
async def evidence_snapshot():
    evidence.check_disk(force=True)  # re-read and re-verify from disk on every read, never trust a cache
    return evidence.snapshot()


@app.get("/api/evidence/chain")
async def evidence_chain():
    evidence.check_disk(force=True)  # serve the records exactly as they are on disk now, for the browser to verify
    return evidence.chain()


@app.get("/api/evidence/runs/{run_id}")
async def evidence_run(run_id: str):
    d = evidence.run_detail(run_id)
    if d is None:
        raise HTTPException(404, f"no evidence for run {run_id!r}")
    return d


@app.post("/api/evidence/scan")
async def evidence_scan():
    return ok(scan=await evidence.scan(), evidence=evidence.snapshot())


@app.post("/api/evidence/verify")
async def evidence_verify():
    return ok(verify=evidence.verify_now(), evidence=evidence.snapshot())


# --------------------------------------------------------------------------- director

@app.post("/api/director/beat/{n}")
async def director_beat(n: int):
    if not 1 <= n <= 14:
        raise HTTPException(404)
    return ok(beat=engine.beat(n))


@app.post("/api/director/reset")
async def director_reset():
    engine.reset()
    return ok(seq=engine.bus.seq)


@app.get("/api/director/preflight")
async def director_preflight():
    # reuse the last full verification while the chain hasn't grown (tamper/restore clear the cache);
    # a full verify takes ~0.5-0.9 s and would stall the event loop on every poll
    lv = engine.ledger.last_verify
    v = lv if lv and lv["total"] == len(engine.ledger.records) else engine.ledger.verify(emit=False)
    return {"backend": True, "lan_ip": _lan_ip(), "ledger_clean": v["ok"], "ledger_height": v["total"],
            "ws_clients": len(engine.bus.subscribers), "fl_state": engine.fl.state, "round": engine.fl.round,
            "clock": engine.clock_view(), "beat": engine.director["beat"]}


def _lan_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


# --------------------------------------------------------------------------- websocket

@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    q = engine.bus.subscribe()
    try:
        await socket.send_text(json.dumps({"kind": "hello", "seq": engine.bus.seq, "mode": MODE}))
        while True:
            await asyncio.sleep(0.1)
            batch = []
            while not q.empty() and len(batch) < 800:
                batch.append(q.get_nowait())
            if batch:
                await socket.send_text(json.dumps({"kind": "events", "events": batch}))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        engine.bus.unsubscribe(q)


# --------------------------------------------------------------------------- built frontend (demo day: one process)

def serve_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built UI: /assets as static files, any other GET as the file it names in `dist`,
    else index.html so client-side routes such as /federation load the app."""
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    root = dist.resolve()
    index = root / "index.html"

    @app.get("/{path:path}")
    def spa(path: str):  # sync on purpose: the filesystem calls run in the thread pool, off the loop
        # `path` comes from the URL: `..`, an absolute path (//etc/hosts) or a symlink in dist could
        # otherwise reach any file the server can read. Serve only regular files that resolve inside.
        try:
            f = (root / path).resolve()
            if f.is_relative_to(root) and f.is_file():
                return FileResponse(f)
        except (OSError, ValueError, RuntimeError):  # name too long, NUL byte, symlink loop
            pass
        return FileResponse(index)


DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.exists():
    serve_frontend(app, DIST)
