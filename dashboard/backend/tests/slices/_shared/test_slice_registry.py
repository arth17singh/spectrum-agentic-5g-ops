"""A slice package registers its router BEFORE the SPA catch-all, so /api/<x>/... answers JSON even
when frontend/dist is built; unknown /api, /ws and /docs paths are JSON 404s, never index.html.

The throwaway slice is a real package written to tmp_path and put on spectrum.slices.__path__, so
discovery finds it exactly as it finds a committed slice."""

import asyncio
import importlib
import sys
import textwrap

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import spectrum.slices as slices_pkg
from spectrum.slices import SliceContext, SliceError, discover, include_slices, start_slices

INDEX = "<!doctype html><title>SPECTRUM</title><div id=\"root\"></div>"
PING = '''
from fastapi import APIRouter

router = APIRouter(prefix="/api/{name}")
ran = []


@router.get("/ping")
async def ping():
    return {{"slice": "{name}", "pong": True}}


async def run(ctx):
    ran.append(ctx.mode)
'''


@pytest.fixture
def make_slice(tmp_path):
    made = []

    def make(name, body=PING):
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(textwrap.dedent(body.format(name=name)))
        made.append(name)
        return name

    slices_pkg.__path__.append(str(tmp_path))
    importlib.invalidate_caches()
    yield make
    slices_pkg.__path__.remove(str(tmp_path))
    for name in made:
        sys.modules.pop(f"spectrum.slices.{name}", None)


@pytest.fixture
def dist(tmp_path):
    d = tmp_path / "frontend-dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(INDEX)
    (d / "assets" / "app.js").write_text("console.log(1)")
    return d


def build(dist):
    """The same order main.py uses: core routes, then slices, then the SPA catch-all."""
    from spectrum.main import serve_frontend
    app = FastAPI()

    @app.get("/api/config")
    async def config():
        return {"mode": "mock"}

    names = include_slices(app)
    serve_frontend(app, dist)
    return app, names


def test_throwaway_slice_answers_json_with_the_spa_catch_all_in_place(make_slice, dist):
    make_slice("zzping")
    app, names = build(dist)
    assert "zzping" in names
    c = TestClient(app)
    r = c.get("/api/zzping/ping")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.json() == {"slice": "zzping", "pong": True}
    for path in ("/api/nope", "/api/zzping/nope", "/api", "/ws", "/ws/x", "/docs/x"):
        r = c.get(path)
        assert r.status_code == 404 and r.json() == {"detail": "Not Found"}, path
    assert c.get("/api/config").json() == {"mode": "mock"}
    for path in ("/", "/federation", "/trace", "/m/approve", "/apis", "/wsx"):  # client routes still get the app
        r = c.get(path)
        assert r.status_code == 200 and r.text == INDEX, path
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert app.routes[-1].path == "/{path:path}"  # the catch-all stays last


def test_slice_run_tasks_start_with_the_context(make_slice, dist):
    make_slice("zzrun")
    app, _ = build(dist)
    mod = sys.modules["spectrum.slices.zzrun"]

    async def go():
        tasks = start_slices(app, SliceContext(engine=None, evidence=None, emit=None, mode="mock", offline=True))
        await asyncio.gather(*tasks)

    asyncio.run(go())
    assert mod.ran == ["mock"]


@pytest.mark.parametrize("body, match", [
    ("x = 1\n", "must export"),
    ('from fastapi import APIRouter\nrouter = APIRouter(prefix="/api/other")\n', "expected '/api/"),
    ('from fastapi import APIRouter\nrouter = APIRouter(prefix="/api/{name}")\ndef run(ctx): pass\n', "async def run"),
])
def test_a_malformed_slice_fails_loudly(make_slice, body, match):
    make_slice("zzbad", body)
    with pytest.raises(SliceError, match=match):
        discover()


def test_helper_packages_are_not_slices():
    assert "_shared" not in [m.__name__.rsplit(".", 1)[-1] for m in discover()]


def test_the_real_app_keeps_its_routes_and_puts_slices_before_the_catch_all():
    from spectrum.main import DIST, app
    c = TestClient(app, client=("127.0.0.1", 50000))  # loopback: the access gate needs no token
    assert c.get("/api/config").status_code == 200
    assert c.get("/api/evidence").status_code == 200
    if DIST.exists():
        assert app.routes[-1].path == "/{path:path}"
        r = c.get("/api/nope")
        assert r.status_code == 404 and r.json() == {"detail": "Not Found"}
        r = c.get("/federation")
        assert r.status_code == 200 and '<div id="root">' in r.text
