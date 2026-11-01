"""The demo-day frontend routes (main.serve_frontend) must only ever serve files inside frontend/dist.

Requests are raw ASGI scopes, not httpx/TestClient calls, which may normalise `..` away before the
app sees it. Targets are written as sent on the wire; `get` percent-decodes them into scope["path"]
as uvicorn does, because the decoded path is what Starlette routes on."""

import asyncio
from urllib.parse import unquote

import pytest
from fastapi import FastAPI

from spectrum.main import serve_frontend

INDEX = b"<!doctype html><title>SPECTRUM</title>"
SVG = b"<svg xmlns='http://www.w3.org/2000/svg'/>"
JS = b"console.log('spectrum')"
SECRET = b"secret: must never leave the server"
UP = "../" * 40  # more levels than any temp dir is deep, so these really climb to /


async def asgi_get(app, path, raw_path=None, client=("127.0.0.1", 50000), headers=()):
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": path,
             "raw_path": path.encode() if raw_path is None else raw_path, "query_string": b"",
             "root_path": "", "headers": [(b"host", b"127.0.0.1:8787"), *headers],
             "client": client, "server": ("127.0.0.1", 8787)}
    out, done, first = [], asyncio.Event(), [True]

    async def receive():
        # one request message, then wait for the response: answering http.request forever makes
        # FileResponse's disconnect listener spin
        if first[0]:
            first[0] = False
            return {"type": "http.request", "body": b"", "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(m):
        out.append(m)
        if m["type"] == "http.response.body" and not m.get("more_body", False):
            done.set()

    await asyncio.wait_for(app(scope, receive, send), timeout=20)
    status = next(m["status"] for m in out if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in out if m["type"] == "http.response.body")
    return status, body


def get(app, target):
    return asyncio.run(asgi_get(app, unquote(target), raw_path=target.encode()))


@pytest.fixture
def site(tmp_path):
    """tmp/frontend/dist (the built UI) beside tmp/backend/pyproject.toml, as in the repo."""
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_bytes(INDEX)
    (dist / "favicon.svg").write_bytes(SVG)
    (dist / "assets" / "app.js").write_bytes(JS)
    secret = tmp_path / "backend" / "pyproject.toml"
    secret.parent.mkdir()
    secret.write_bytes(SECRET)
    (dist / "alias.svg").symlink_to("favicon.svg")  # stays inside dist
    (dist / "leak.txt").symlink_to(secret)  # file outside dist
    (dist / "backend").symlink_to(secret.parent, target_is_directory=True)  # directory outside dist
    (dist / "assets" / "leak.js").symlink_to(secret)
    (dist / "loop").symlink_to("loop")
    app = FastAPI()
    serve_frontend(app, dist)
    return app, str(tmp_path).lstrip("/")


@pytest.mark.parametrize("target", [
    pytest.param("/" + UP + "etc/hosts", id="dotdot-to-etc-hosts"),
    pytest.param("/../../backend/pyproject.toml", id="dotdot-to-repo-file"),
    pytest.param("//etc/hosts", id="absolute-etc-hosts"),
    pytest.param("//{tmp}/backend/pyproject.toml", id="absolute-repo-file"),
    pytest.param("/%2e%2e/%2e%2e/backend/pyproject.toml", id="encoded-dots"),
    pytest.param("/..%2f..%2fbackend%2fpyproject.toml", id="encoded-slashes"),
    pytest.param("/leak.txt", id="symlinked-file"),
    pytest.param("/backend/pyproject.toml", id="symlinked-dir"),
])
def test_files_outside_dist_get_index_html(site, target):
    app, tmp = site
    assert get(app, target.format(tmp=tmp)) == (200, INDEX)


@pytest.mark.parametrize("target", [
    pytest.param("/%00", id="nul"),
    pytest.param("/index.html%00.js", id="nul-after-name"),
    pytest.param("/%2e%2e/%00/etc/hosts", id="nul-in-traversal"),
    pytest.param("/loop", id="symlink-loop"),
    pytest.param("/" + "x" * 300, id="name-too-long"),
])
def test_paths_that_cannot_be_resolved_get_index_html_not_500(site, target):
    app, _ = site
    assert get(app, target) == (200, INDEX)


@pytest.mark.parametrize("target", [
    pytest.param("/assets/../../../backend/pyproject.toml", id="dotdot"),
    pytest.param("/assets/" + UP + "{tmp}/backend/pyproject.toml", id="dotdot-to-absolute"),
    pytest.param("/assets/%2e%2e/%2e%2e/%2e%2e/backend/pyproject.toml", id="encoded-dots"),
    pytest.param("/assets/leak.js", id="symlinked-file"),
])
def test_assets_paths_do_not_escape_either(site, target):
    app, tmp = site
    _, body = get(app, target.format(tmp=tmp))
    assert SECRET not in body


@pytest.mark.parametrize("target, expected", [
    ("/", INDEX),
    ("/index.html", INDEX),
    ("/favicon.svg", SVG),
    ("/alias.svg", SVG),
    ("/assets/app.js", JS),
    ("/federation", INDEX),
    ("/m/approve", INDEX),
])
def test_built_ui_and_client_routes_are_still_served(site, target, expected):
    app, _ = site
    assert get(app, target) == (200, expected)
