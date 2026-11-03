"""smoke_gateway.py: the plan, the approval-key formula, and PASS/FAIL logic with a fake Gateway."""
from __future__ import annotations

from t3.deploy import smoke_gateway as sg
from t3.tools import app


def test_dry_run_lists_every_refusal(tmp_path):
    lines = []
    assert sg.main(["--kit-dir", str(tmp_path)], out=lines.append) == 0
    text = "\n".join(lines)
    assert text.startswith("DRY RUN") and text.count("ran___apply_change") >= 4
    assert "work___apply_change" in text and "nlm___ask_nlm" in text


def test_approval_key_matches_the_tools():
    args = {"case_id": "case-smk00001", "item_id": "step-0", "change": sg.CHANGE}
    assert sg.approval_key(args["case_id"], args["item_id"], args["change"]) == app.change_key(args)


def test_all_refused_and_pending_passes():
    lines = []
    assert sg.run("case-smk00001", lambda agent, tool, args: ("REFUSED", "refused"), lambda k: "PENDING",
                  lines.append) == 0
    assert all(line.startswith("PASS") for line in lines if line[:4] in ("PASS", "FAIL"))


def test_a_spoof_that_gets_through_fails():
    def call(agent, tool, args):
        return ("ALLOW", "applied") if args.get("approval") == "APPROVED" and agent == "noc_lead" \
            and tool == "ran___apply_change" else ("REFUSED", "x")
    assert sg.run("case-smk00001", call, lambda k: "PENDING", lambda *_: None) == 1


def test_a_consumed_approval_fails_even_if_refused():
    assert sg.run("case-smk00001", lambda *a: ("REFUSED", "x"), lambda k: "APPLIED", lambda *_: None) == 1
