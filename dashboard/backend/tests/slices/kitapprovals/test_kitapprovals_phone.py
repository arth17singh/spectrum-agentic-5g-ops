"""Paired-phone approvals (KITAPPROVALS_PHONE=1, `make demo`): off by default; on, only with the access token and the
laptop's LAN address as Host; the decision is the same PIN-checked kit update."""

from spectrum.access import TOKEN
from spectrum.slices.kitapprovals import routes
from slices.kitapprovals.kitapprovals_helpers import client, decide, make, pending_row, post_headers

LAN = "192.168.7.20"
PHONE = ("192.168.7.55", 40000)


def _setup(tmp_path, monkeypatch, on: bool):
    monkeypatch.setattr(routes, "lan_ip", lambda: LAN)
    if on:
        monkeypatch.setenv("KITAPPROVALS_PHONE", "1")
    else:
        monkeypatch.delenv("KITAPPROVALS_PHONE", raising=False)
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, items=[row])
    return svc, ddb, row


def test_phone_refused_by_default(tmp_path, monkeypatch):
    svc, ddb, row = _setup(tmp_path, monkeypatch, on=False)
    c = client(svc, host=f"{LAN}:8787", peer=PHONE)
    h = {**post_headers(f"{LAN}:8787"), "X-Spectrum-Token": TOKEN}
    assert decide(c, row, headers=h).status_code == 403
    assert c.get(f"/api/kitapprovals/{row['pk']}", headers={"X-Spectrum-Token": TOKEN}).status_code == 403


def test_phone_needs_token_and_lan_host(tmp_path, monkeypatch):
    svc, ddb, row = _setup(tmp_path, monkeypatch, on=True)
    c = client(svc, host=f"{LAN}:8787", peer=PHONE)
    assert decide(c, row, headers=post_headers(f"{LAN}:8787")).status_code == 403  # no token
    evil = client(svc, host="evil.example:8787", peer=PHONE)
    h = {**post_headers("evil.example:8787"), "X-Spectrum-Token": TOKEN}
    assert decide(evil, row, headers=h).status_code == 403  # rebinding host name
    h = {**post_headers(f"{LAN}:8787"), "X-Spectrum-Token": TOKEN}
    assert decide(c, row, pin="000000", headers=h).status_code == 403  # wrong PIN: nothing decided
    r = decide(c, row, headers=h)
    assert r.status_code == 200 and r.json()["status"] == "APPROVED"
