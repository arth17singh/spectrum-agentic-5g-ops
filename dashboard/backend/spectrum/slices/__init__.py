"""Slice discovery: every package under spectrum/slices/ is one dashboard slice.

A slice is `spectrum/slices/<name>/__init__.py` exporting:
  router: fastapi.APIRouter      with prefix "/api/<name>" (required)
  async def run(ctx) -> None     optional background task, started in the app lifespan

main.py includes every slice router BEFORE the SPA catch-all (a route added after it is never
reached: the catch-all answers first). Packages whose name starts with "_" (e.g. _shared) are
helpers, not slices. A slice that is malformed raises at import, so a typo fails pytest instead of
silently falling through to index.html.

Anything in a slice that calls boto3 must run via `asyncio.to_thread` (handlers and run() share the
event loop with the simulation).
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from fastapi import APIRouter, FastAPI

log = logging.getLogger("spectrum.slices")


class SliceError(RuntimeError):
    pass


@dataclass
class SliceContext:
    """What a slice's run(ctx) receives. Read-only use is expected; slices own their own state.

    engine:   the demo Engine (spectrum.engine.Engine); its bus is engine.bus (replaced on reset)
    evidence: the EvidenceService (the real, persisted ledger)
    emit:     emit(type, plane, summary, data, severity, actor) on the CURRENT engine bus, wall-clock stamped
    mode:     SPECTRUM_MODE ("mock" | "aws")
    offline:  True when SPECTRUM_OFFLINE=1 (tests): never call AWS
    """
    engine: Any
    evidence: Any
    emit: Any
    mode: str = "mock"
    offline: bool = False
    extra: dict = field(default_factory=dict)


def discover(package: ModuleType | None = None) -> list[ModuleType]:
    """Import every slice package (sorted by name) and check its contract."""
    if package is None:
        package = importlib.import_module(__name__)
    mods = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if info.name.startswith("_") or not info.ispkg:
            continue
        mod = importlib.import_module(f"{package.__name__}.{info.name}")
        router = getattr(mod, "router", None)
        if not isinstance(router, APIRouter):
            raise SliceError(f"slice {info.name!r} must export `router: fastapi.APIRouter`")
        if router.prefix != f"/api/{info.name}":
            raise SliceError(f"slice {info.name!r}: router prefix is {router.prefix!r}, expected '/api/{info.name}'")
        run = getattr(mod, "run", None)
        if run is not None and not inspect.iscoroutinefunction(run):
            raise SliceError(f"slice {info.name!r}: run must be `async def run(ctx)`")
        mods.append(mod)
    return mods


def include_slices(app: FastAPI, package: ModuleType | None = None) -> list[str]:
    """Include every slice router in `app`; remember the slices on app.state for the lifespan."""
    mods = discover(package)
    for mod in mods:
        app.include_router(mod.router)
    app.state.slices = mods
    return [m.__name__.rsplit(".", 1)[-1] for m in mods]


def start_slices(app: FastAPI, ctx: SliceContext) -> list[asyncio.Task]:
    """Start each slice's run(ctx) as its own task. One slice crashing is logged and never stops another."""
    tasks = []
    for mod in getattr(app.state, "slices", []):
        run = getattr(mod, "run", None)
        if run is None:
            continue
        name = mod.__name__.rsplit(".", 1)[-1]

        async def guarded(run=run, name=name):
            try:
                await run(ctx)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a slice's background task must not take the app down
                log.exception("slice %s: run() failed", name)

        tasks.append(asyncio.create_task(guarded(), name=f"slice:{name}"))
    return tasks
