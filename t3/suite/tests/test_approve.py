"""The approval helper lists the kit's waiting changes and runs the kit's approve only on the person's yes."""
from __future__ import annotations

import json

from t3.suite import approve

FAKE_KIT = '''import sys
if len(sys.argv) == 2:
    print("chg-aaaaaaaaaaaa case-%(pre)s-33 noc_lead " + %(chg)r)
    print("chg-bbbbbbbbbbbb case-other-01 noc_lead {}")
    print("2 waiting. Approve one with: approve chg-... \\"why\\"")
else:
    open("decided.txt", "a").write(" ".join(sys.argv[1:]) + "\\n")
'''


def test_approve_helper_only_runs_the_kit_on_the_persons_yes(tmp_path, monkeypatch):
    pre = approve.run_prefix("1102a-t3-allow")
    (tmp_path / "checkpoint_kit.py").write_text(FAKE_KIT % {"pre": pre, "chg": json.dumps({"item_id": "step-0"})})
    rows = approve.pending(tmp_path)
    assert [r["approval_id"] for r in rows] == ["chg-aaaaaaaaaaaa", "chg-bbbbbbbbbbbb"]
    answers = iter(["y", "checked by me"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert approve.main(["--kit-dir", str(tmp_path), "--run-id", "1102a-t3-allow"]) == 0
    assert (tmp_path / "decided.txt").read_text().split() == ["approve", "chg-aaaaaaaaaaaa", "checked", "by", "me"]
    (tmp_path / "decided.txt").unlink()
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")   # no: nothing is decided
    assert approve.main(["--kit-dir", str(tmp_path), "--run-id", "1102a-t3-allow"]) == 0
    assert not (tmp_path / "decided.txt").exists()
