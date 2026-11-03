"""Static checks of the Ask the NOC page (the frontend has no unit-test runner; the gate's route smoke loads it)."""

import re
from pathlib import Path

FEATURE = Path(__file__).resolve().parents[4] / "frontend" / "src" / "features" / "asknoc"
SOURCES = sorted(p for p in FEATURE.glob("*.ts*"))


def test_feature_exists():
    assert {p.name for p in SOURCES} >= {"manifest.ts", "AskNocPage.tsx", "api.ts", "states.ts", "parts.tsx"}


def test_text_is_never_rendered_as_html_or_code():
    for p in SOURCES:
        src = p.read_text()
        for bad in ("dangerouslySetInnerHTML", "innerHTML", "eval(", "new Function", "outerHTML"):
            assert bad not in src, f"{p.name} uses {bad}"


def test_no_absolute_urls_or_hosts():
    for p in SOURCES:
        assert not re.search(r"https?://(?!localhost)", p.read_text()), p.name


def test_manifest_route_and_hotkey():
    src = (FEATURE / "manifest.ts").read_text()
    assert "route: '/ask-noc'" in src and "hotkey: 'n'" in src and "data: 'real'" in src


def test_writes_carry_the_guard_headers():
    src = (FEATURE / "api.ts").read_text()
    assert "'X-Spectrum-Client': 'asknoc'" in src and "Idempotency-Key" in src


def test_no_approve_or_tool_call_anywhere():
    for p in SOURCES:
        src = p.read_text()
        assert not re.search(r"/api/approvals|ran___|swarm___|apply_change\(|/approve\b", src), p.name
