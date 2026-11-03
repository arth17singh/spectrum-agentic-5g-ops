"""Static checks on the kitapprovals UI (the live check is in the screenshot script)."""

from pathlib import Path

import spectrum.main as main


def test_confirm_sheet_stops_global_hotkeys():
    """Review F8: hotkeys.ts listens on window; the sheet must stop key events from bubbling there."""
    fe = Path(main.__file__).parents[2] / "frontend" / "src" / "features" / "kitapprovals"
    src = (fe / "ConfirmSheet.tsx").read_text()
    assert "onKeyDown" in src and "stopPropagation()" in src
    for f in fe.glob("*.tsx"):
        assert "dangerouslySetInnerHTML" not in f.read_text()
