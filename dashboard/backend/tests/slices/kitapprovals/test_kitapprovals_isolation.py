"""Separation (review F4): no code path from the chat slice, agents or the simulator to the decision.

A process on this laptop CAN reach 127.0.0.1, so the PIN is what stops it; these tests pin both halves:
no import or self-call between slices, and the real app refuses a well-formed local request without the PIN.
"""

import ast
import json
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.routing import Match

import spectrum.main as main
from spectrum.slices.kitapprovals.routes import get_service
from slices.kitapprovals.kitapprovals_helpers import PIN, make, pending_row, post_headers, sha

SLICES = Path(main.__file__).parent / "slices"
KIT = SLICES / "kitapprovals"


def _imports(folder: Path) -> set[str]:
    out = set()
    for f in folder.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                out |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                out.add(("." * node.level) + (node.module or ""))
    return out


def test_kitapprovals_imports_no_other_slice_and_no_t3():
    bad = [m for m in _imports(KIT) if m.startswith(("spectrum.slices.", "t3", "..")) and "kitapprovals" not in m]
    assert bad == []


def test_no_other_slice_reaches_kitapprovals():
    for d in SLICES.iterdir():
        if not d.is_dir() or d.name in ("kitapprovals", "__pycache__"):
            continue
        assert not [m for m in _imports(d) if "kitapprovals" in m], d.name
        for f in d.rglob("*.py"):
            text = f.read_text()
            for s in ("kitapprovals", "localhost:8787", "127.0.0.1:8787"):
                assert s not in text, f"{f.relative_to(SLICES)} mentions {s}"


def _resolve(method: str, path: str):
    scope = {"type": "http", "method": method, "path": path, "root_path": "", "headers": [], "query_string": b""}
    for r in main.app.router.routes:
        m, _ = r.matches(scope)
        if m == Match.FULL:
            return r
    return None


def test_routes_are_exactly_four_and_do_not_shadow_the_simulator():
    from spectrum.slices.kitapprovals import router
    mine = sorted((sorted(r.methods), r.path) for r in router.routes)
    assert mine == [(["GET"], "/api/kitapprovals/pending"), (["GET"], "/api/kitapprovals/status"),
                    (["GET"], "/api/kitapprovals/{approval_id}"), (["POST"], "/api/kitapprovals/{approval_id}/decide")]
    sim = _resolve("POST", "/api/approvals/chg-0123456789ab/approve")
    assert sim is not None and "kitapprovals" not in sim.endpoint.__module__
    real = _resolve("POST", "/api/kitapprovals/chg-0123456789ab/decide")
    assert real is not None and getattr(real, "original_router", None) is router


def test_a_local_process_cannot_decide_without_the_pin(tmp_path, monkeypatch):
    """The real app (AccessGate and all) from 127.0.0.1, with every browser header forged, as curl can."""
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    main.app.dependency_overrides[get_service] = lambda: svc
    try:
        c = TestClient(main.app, base_url="http://localhost:8787", client=("127.0.0.1", 50001))
        url = f"/api/kitapprovals/{row['pk']}/decide"
        base = {"decision": "approve", "reason": "agent says ok", "change_sha256": sha(row["change"])}
        for pin in ("", "123456", "000000"):
            r = c.post(url, content=json.dumps({**base, "pin": pin}), headers=post_headers())
            assert r.status_code == 403 and r.json()["error"] == "pin_wrong"
        r = c.post(url, content=json.dumps(base), headers=post_headers())  # no pin at all
        assert r.status_code == 422
        assert not [op for op, _ in ddb.calls if op == "UpdateItem"]
        assert ddb.items[row["pk"]]["status"] == "PENDING"
        # the simulator's approve route never touches the kit table
        c.post("/api/approvals/" + row["pk"] + "/approve", json={"by": "A. Rivera"})
        assert ddb.items[row["pk"]]["status"] == "PENDING"
        # with the PIN a person typed, it works
        r = c.post(url, content=json.dumps({**base, "pin": PIN}), headers=post_headers())
        assert r.status_code == 200
    finally:
        main.app.dependency_overrides.pop(get_service, None)

