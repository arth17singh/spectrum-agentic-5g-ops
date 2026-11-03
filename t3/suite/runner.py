"""Run the frozen Track 3 call suite through the AgentCore Gateway, as each agent, for one run id.

Order (each step is evidence for the next):
  1. write the declaration (unit, target, n_items, suite sha256, code sha) BEFORE any call;
  2. start one quiet case through the front door with "run_id: <run_id>" in the request text;
  3. write case_id -> run_id into the cases table (DynamoDB), so every later record maps to the run;
  4. send every suite item as an MCP tools/call with the caller's own token, and record each answer
     as an `item` record. For the approval item the runner prints the approval id and polls the
     approvals table until a PERSON approves it; the runner never approves anything itself.
     Pause-window items run inside a team13_ranpause_ Cedar forbid on ran___predict (amendments A8, A9).

The grade does not come from here: t3/suite/check.py grades each item from the AWS decision records.

Two units from the one frozen suite (t3/suite/units.py): deny and allow, each with its own declaration
(test_set_sha256 = the full suite's hash, `subset`, its own n_items), records and result. `--unit all` runs
both back to back as <run_id>-deny and <run_id>-allow.

CLI (dry run unless --live):
  python -m t3.suite.runner --run-id 1102a-t3 [--unit all|deny|allow] [--tokens-dir ~/kit/tokens] [--live]
Lambda: handler({"run_id": "...", "unit": "deny", "approval_timeout_s": 600, "pause": true}, context)

Environment: T3_GATEWAY_URL; tokens from T3_TOKENS_SECRET | T3_TOKEN_<AGENT> | T3_TOKENS_DIR;
T3_APPROVALS_TABLE (default team13-approvals: the kit's table also holds our t3-active run row and
t3-<case_id> case rows, amendment A10), AWS_REGION; for the pause window T3_POLICY_ENGINE_ID and
T3_GATEWAY_ARN.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from t3.suite import common, units
from t3.suite.common import (CASE_RE, APPROVAL_RE, change_key, envelope, fill, sha256_json, sha256_text,
                             utc_now_iso)
from t3.suite.suite_hash import DEFAULT_SUITE, load_suite, suite_sha256, validate

UNIT = "t3.deny_allow"   # default label for helpers used outside a suite unit
GRADER = "t3/suite/check.py: Cedar decision per item from /team13/checkpoint + aws/spans equals expect"
QUIET_REQUEST = ("This case is an automated policy test of the checkpoint. Reply with the single word READY. "
                 "Do not call any tool.")


def run_prefix(run_id: str) -> str:
    """8 hex characters of sha256(run_id): case ids never collide however run ids are named."""
    return sha256_text(run_id)[:8]


def item_case_id(run_id: str, it: dict) -> str:
    slot = it.get("case_slot") or int(it["id"].rsplit("-", 1)[-1])
    return f"case-{run_prefix(run_id)}-{slot:02d}"


def env_config(env=None) -> dict:
    env = os.environ if env is None else env
    return {
        "gateway_url": env.get("T3_GATEWAY_URL", ""),
        "region": env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-2",
        "approvals_table": env.get("T3_APPROVALS_TABLE", "team13-approvals"),
        "model_id": env.get("T3_MODEL_ID", "checkpoint:team13-cedar"),
        "operator": env.get("T3_OPERATOR", "person:operator"),
    }


def declaration(run_id: str, items: list[dict], cfg: dict, *, unit=UNIT, target=None, grader=GRADER,
                synthetic=False, n_cases=1, subset=None, n_items=None) -> dict:
    """items: the FULL frozen suite (its hash is test_set_sha256); n_items and subset name the unit's part."""
    extra = {"subset": subset} if subset else {}
    return envelope(
        "declared", run_id, synthetic=synthetic,
        unit=unit, track="track-3",
        target=target or {"metric": "decision_agreement", "op": "==", "value": 1.0, "unit": "ratio"},
        n_items=len(items) if n_items is None else n_items, n_cases=n_cases, test_set_sha256=suite_sha256(items),
        **extra,
        model_id=cfg["model_id"], grader=grader, code_sha=common.code_sha(), region=cfg["region"],
        twin={"kind": "none", "version": "n/a", "scenario": "n/a", "seed": 0},
        declared_at=utc_now_iso())


class Runner:
    def __init__(self, *, run_id, gateway, evlib, ddb, cfg, suite_path=DEFAULT_SUITE, pauser=None,
                 approval_timeout_s=900.0, poll_s=5.0, sleep=time.sleep, clock=time.time, log=print,
                 synthetic=False, min_gap_s=2.0, pause_settle_s=5.0, quiet_wait_s=120.0, unit="deny"):
        self.run_id, self.gw, self.ev, self.ddb, self.cfg = run_id, gateway, evlib, ddb, cfg
        self.unit_key = unit
        self.unit_name = units.UNITS[unit]["unit"] if unit in units.UNITS else unit
        self.suite_path, self.pauser = suite_path, pauser
        self.approval_timeout_s, self.poll_s = approval_timeout_s, poll_s
        self.sleep, self.clock, self.log, self.synthetic = sleep, clock, log, synthetic
        self.case_id = None
        self.seq = 0
        self.min_gap_s = min_gap_s
        self._last_call = None
        self.pause_settle_s = pause_settle_s   # after ACTIVE / GONE, before the window's call (recorded)
        self.quiet_wait_s = quiet_wait_s       # let the lead finish the quiet case before the items

    # ----- steps -----
    def declare(self, items, subset):
        problems = validate(items)
        if problems:
            raise ValueError("suite is not valid: " + "; ".join(problems))
        decl = declaration(self.run_id, items, self.cfg, synthetic=self.synthetic, unit=self.unit_name,
                           subset=units.UNITS[self.unit_key]["subset"], n_items=len(subset))
        path = self.ev.write_declared(decl)
        self.log(f"declared {decl['unit']} for {self.run_id}: {decl['n_items']} items, "
                 f"suite {decl['test_set_sha256'][:12]}… -> {path}")
        return decl

    def start_case(self, unit=None, request_text=QUIET_REQUEST):
        unit = unit or self.unit_name
        request = f"run_id: {self.run_id}\n\n{request_text}"
        self.pace()
        out = self.gw.call("team_app", "front___start_case", {"request": request}, timeout=60)
        self._record_setup("setup-start_case", "team_app", "front___start_case", {"request": request}, out)
        if out["decision"] != "ALLOW":
            raise RuntimeError(f"the front door refused the suite case: {out['text'][:300]}")
        try:
            case_id = json.loads(out["text"])["case_id"]
        except (ValueError, KeyError, TypeError):
            m = CASE_RE.search(out["text"])
            if not m:
                raise RuntimeError(f"no case_id in the front door's answer: {out['text'][:300]}")
            case_id = m.group(0)
        self.case_id = case_id
        self.map_case(case_id, unit)
        self.log(f"case {case_id} -> run {self.run_id} (row t3-{case_id} in {self.cfg['approvals_table']})")
        return case_id

    def set_active_run(self, unit=None):
        """Amendment A10: the run row t3-active in the kit's approvals table names the current run. No
        `status` attribute, so the kit's `approve` never lists it."""
        unit = unit or self.unit_name
        common.ddb_put(self.ddb, self.cfg["approvals_table"], {
            "pk": "t3-active", "run_id": self.run_id, "unit": unit, "updated_at": int(self.clock()),
            "source": "t3/suite/runner.py"})

    def map_case(self, case_id, unit=None):
        """Amendment A10: a BINDING row t3-<case_id> -> run_id, without `status` (the kit's approve never lists
        it) and without `doc` (the tools read it as "no case yet", take its run_id and replace it with the
        case, t3/tools/store.py). Written only if no row exists, so it can never overwrite a real case."""
        unit = unit or self.unit_name
        common.ddb_put(self.ddb, self.cfg["approvals_table"], {
            "pk": f"t3-{case_id}", "case_id": case_id, "run_id": self.run_id, "unit": unit,
            "created_at": int(self.clock()), "source": "t3/suite/runner.py"},
            condition="attribute_not_exists(pk)")

    def ensure_fresh(self, case_ids):
        """Case ids come from run_id[:8]; a run id whose first 8 characters were used before would reuse old
        cases and their approval ids (already APPLIED). Refuse it before any call."""
        for case_id in case_ids:
            row = common.ddb_get(self.ddb, self.cfg["approvals_table"], {"pk": f"t3-{case_id}"})
            if row and row.get("run_id") != self.run_id:
                raise RuntimeError(f"{case_id} already belongs to run {row.get('run_id')}: this run id was "
                                   f"used before (case prefix sha256(run_id)[:8]); pick a new run id")

    def wait_quiet(self):
        """The quiet front-door case makes the lead answer once; wait until it is done (bounded) so its own
        calls and spans do not interleave with the items."""
        deadline = self.clock() + self.quiet_wait_s
        while self.clock() < deadline:
            self.pace()
            out = self.gw.call("team_app", "front___case_status", {"case_id": self.case_id}, timeout=60)
            try:
                state = json.loads(out["text"]).get("state")
            except (ValueError, AttributeError):
                state = None
            if state not in common.RUNNING:
                return state
            self.sleep(5)
        self.log(f"the lead is still working on {self.case_id} after {self.quiet_wait_s} s; going on")
        return "TIMEOUT"

    def item_case(self, it) -> str:
        """Amendment A12: a direct tool-call item gets its own case id case-<run_id[:8]>-<nn>; the approval
        and pause groups share one through case_slot."""
        return item_case_id(self.run_id, it)

    def pace(self):
        """Amendment A12: calls run one at a time, at least min_gap_s apart."""
        if self._last_call is not None:
            wait = self.min_gap_s - (self.clock() - self._last_call)
            if wait > 0:
                self.sleep(wait)
        self._last_call = self.clock()

    def wait_for_person(self, approval_id: str) -> dict:
        """Poll the approvals table (read only) until a person decides. Never writes to it."""
        self.log(f"WAITING FOR A PERSON to approve {approval_id} (case {self.case_id}). Run:\n"
                 f"    python checkpoint_kit.py approve {approval_id} \"checked: suite run {self.run_id}\"")
        deadline = self.clock() + self.approval_timeout_s
        while True:
            row = common.ddb_get(self.ddb, self.cfg["approvals_table"], {"pk": approval_id}) or {}
            status = row.get("status", "MISSING")
            if status in ("APPROVED", "REJECTED", "APPLIED"):
                self.log(f"{approval_id}: {status} by {row.get('decided_by', '?')}")
                return {"status": status, "decided_by": row.get("decided_by"), "decided_at": row.get("decided_at")}
            if self.clock() >= deadline:
                return {"status": "TIMEOUT", "decided_by": None, "decided_at": None}
            self.sleep(self.poll_s)

    # ----- records -----
    def _record(self, item_id, caller, tool, args, out, extra):
        self.seq += 1
        rec = envelope(
            "item", self.run_id, aws_time=out.get("gateway_date"),
            aws_time_source="gateway:http-date" if out.get("gateway_date") else None, synthetic=self.synthetic,
            item_id=item_id,
            input_sha256=sha256_json({"caller": caller, "tool": tool, "args": args}),
            output_sha256=sha256_text(out.get("text") or ""),
            grader="t3/suite/runner.py (client view, ungraded)", backend="agentcore-gateway",
            latency_ms=out.get("latency_ms"),
            extra={"view": "client", "seq": self.seq, "case_id": args.get("case_id"),
                   "front_case_id": self.case_id, "caller": caller, "tool": tool,
                   "args_sha256": sha256_json(args), "args": args, "observed": out.get("decision"),
                   "http_status": out.get("http_status"),
                   "sent_at": common.to_iso(out["sent_at"]) if out.get("sent_at") else None,
                   "tool_error": out.get("tool_error"), "response_excerpt": (out.get("text") or "")[:400],
                   **extra})
        self.ev.append_records(self.run_id, "item", [rec])
        return rec

    def _record_setup(self, item_id, caller, tool, args, out):
        return self._record(item_id, caller, tool, args, out, {"setup": True})

    def _skip(self, it, args, reason, extra=None):
        out = {"decision": "SKIPPED", "text": reason}
        return self._record(it["id"], it["caller"], it["tool"], args, out,
                            {**(extra or {}), "expect": it["expect"], "rule_class": it["rule_class"],
                             "skipped": reason})

    # ----- the run -----
    def run(self) -> dict:
        suite = load_suite(self.suite_path)
        items = units.select(suite, self.unit_key)
        self.declare(suite, items)
        case_ids = sorted({self.item_case(it) for it in items if "{case}" in json.dumps(it["args"])})
        self.ensure_fresh(case_ids)
        self.set_active_run()
        self.start_case()
        for case_id in case_ids:  # bind every item's case id to the run before any item runs
            self.map_case(case_id)
        self.wait_quiet()
        approvals: dict[str, dict] = {}   # suite approval group -> {"approval_id", "decision"}
        paused = resumed = False
        summary = {"run_id": self.run_id, "unit": self.unit_name, "case_id": self.case_id,
                   "n_items": len(items), "client_agree": 0,
                   "skipped": [], "approval": None, "pause": None}
        for it in items:
            args = fill(it["args"], self.item_case(it), self.run_id)
            extra = {"expect": it["expect"], "rule_class": it["rule_class"]}
            window = it.get("pause_window")
            if window:
                if self.pauser is None and window == "inside":
                    self._skip(it, args, "no policy client: pause window not run", {"pause_window": window})
                    summary["skipped"].append(it["id"])
                    continue
                if window == "inside" and not paused:
                    summary["pause"] = {"on": self.pauser.on(reason=f"suite run {self.run_id}")}
                    self._control("pause", summary["pause"]["on"])
                    paused = True
                    self.sleep(self.pause_settle_s)
                if window == "after_resume" and paused and not resumed and self.pauser is not None:
                    summary["pause"]["off"] = self.pauser.off()
                    self._control("resume", summary["pause"]["off"])
                    resumed = True
                    self.sleep(self.pause_settle_s)
                extra["pause_window"] = window
                extra["pause_settle_s"] = self.pause_settle_s
            step = it.get("approval_step")
            if step:
                extra["approval_step"] = step
                expected_id = change_key(args)
                extra["approval_id_expected"] = expected_id
                if step == "after_approval":
                    seen = approvals.get(expected_id)
                    if not seen:
                        self._skip(it, args, "no refused request for this change earlier in the run", extra)
                        summary["skipped"].append(it["id"])
                        continue
                    decision = self.wait_for_person(seen["approval_id"])
                    summary["approval"] = {"approval_id": seen["approval_id"], **decision}
                    extra["approval"] = decision
                    if decision["status"] != "APPROVED":
                        self._skip(it, args, f"approval {decision['status'].lower()}, not approved by a person", extra)
                        summary["skipped"].append(it["id"])
                        continue
            self.pace()
            out = self.gw.call(it["caller"], it["tool"], args)
            if step:
                m = APPROVAL_RE.search(out.get("text") or "")
                if m:
                    extra["approval_id"] = m.group(0)
                    extra["approval_id_matches_kit"] = m.group(0) == extra["approval_id_expected"]
                if step == "request":
                    approvals[extra["approval_id_expected"]] = {"approval_id": extra.get("approval_id")
                                                                or extra["approval_id_expected"]}
            self._record(it["id"], it["caller"], it["tool"], args, out, extra)
            agree = out["decision"] == it["expect"]
            summary["client_agree"] += int(agree)
            self.log(f"{it['id']} {it['caller']:<13} {it['tool']:<22} expect {it['expect']:<5} "
                     f"got {out['decision']:<5} {'ok' if agree else 'MISMATCH'}")
        if paused and not resumed:  # never leave the NLM paused
            summary["pause"]["off"] = self.pauser.off()
            self._control("resume", summary["pause"]["off"])
        self.log(f"client view: {summary['client_agree']} of {len(items)} as expected, "
                 f"{len(summary['skipped'])} skipped. Grade with: python -m t3.suite.check --run-id {self.run_id} "
                 f"--unit {self.unit_key}")
        return summary

    def _control(self, action, info):
        rec = envelope("control", self.run_id, aws_time=info.get("aws_time"),
                       aws_time_source=info.get("aws_time_source"), synthetic=self.synthetic,
                       model_id=info.get("model_id", "nlm"), version=info.get("version", "current"),
                       action=action, by=self.cfg["operator"], reason=f"suite run {self.run_id}: pause window",
                       policy_name=info.get("policy_name"),
                       time=info.get("aws_time") or info.get("requested_at") or utc_now_iso())
        self.ev.append_records(self.run_id, "control", [rec])


# ---------- wiring for the CLI and Lambda ----------
def build(run_id, *, suite_path=DEFAULT_SUITE, tokens_dir=None, pause=True, approval_timeout_s=900.0,
          env=None, log=print, pause_tool="ran___predict", unit="deny"):
    import boto3  # only for a live run
    cfg = env_config(env)
    session = boto3.session.Session(region_name=cfg["region"])
    secrets = session.client("secretsmanager") if (env or os.environ).get("T3_TOKENS_SECRET") else None
    tokens = common.Tokens(secrets_client=secrets, tokens_dir=tokens_dir, env=env)
    pauser = None
    if pause:
        from t3.suite.pause_floor import CedarPause
        pauser = CedarPause.from_env(session.client("bedrock-agentcore-control"), env=env, tool=pause_tool)
    return Runner(run_id=run_id, gateway=common.Gateway(cfg["gateway_url"], tokens), evlib=common.load_evlib(),
                  ddb=session.client("dynamodb"), cfg=cfg, suite_path=suite_path, pauser=pauser,
                  approval_timeout_s=approval_timeout_s, log=log, unit=unit)


def handler(event, context):
    """Lambda entry. Event: {"run_id": str, "approval_timeout_s"?: number, "pause"?: bool}. The suite file
    is packaged with the function. Lambda's 15-minute limit bounds the approval wait."""
    return [build(rid, approval_timeout_s=float(event.get("approval_timeout_s", 600)),
                  pause=bool(event.get("pause", True)), unit=key).run()
            for rid, key in units.run_ids(event["run_id"], event.get("unit", "all"))]


def dry_run(run_id, suite_path, unit="all", log=print):
    suite = load_suite(suite_path)
    problems = validate(suite)
    for rid, key in units.run_ids(run_id, unit):
        dry_run_unit(rid, key, suite, log)
    log(f"DRY RUN: nothing sent. Problems: {problems or 'none'}. Add --live to run.")
    return 1 if problems else 0


def dry_run_unit(run_id, key, suite, log=print):
    items = units.select(suite, key)
    decl = declaration(run_id, suite, env_config(), synthetic=True, unit=units.UNITS[key]["unit"],
                       subset=units.UNITS[key]["subset"], n_items=len(items))
    log(f"run {run_id}: " + json.dumps({k: decl[k] for k in ("unit", "subset", "n_items", "test_set_sha256",
                                                           "code_sha")}))
    for it in items:
        args = fill(it["args"], item_case_id(run_id, it), run_id)
        flag = (" [waits for a person]" if it.get("approval_step") == "after_approval" else
                f" [pause {it['pause_window']}]" if it.get("pause_window") else "")
        log(f"{it['id']} as {it['caller']:<13} {it['tool']:<22} expect {it['expect']}{flag}  {json.dumps(args)[:90]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--unit", choices=("all", "deny", "allow"), default="all",
                    help="all = both units back to back as <run-id>-deny and <run-id>-allow")
    ap.add_argument("--tokens-dir", help="a folder of <agent>.txt tokens, e.g. the kit's tokens/ (never committed)")
    ap.add_argument("--approval-timeout", type=float, default=900.0)
    ap.add_argument("--no-pause", action="store_true", help="skip the pause-window items (they count as not passed)")
    ap.add_argument("--live", action="store_true", help="really call the Gateway and write records")
    a = ap.parse_args(argv)
    if not a.live:
        return dry_run(a.run_id, a.suite, a.unit)
    for rid, key in units.run_ids(a.run_id, a.unit):
        summary = build(rid, suite_path=Path(a.suite), tokens_dir=a.tokens_dir, pause=not a.no_pause,
                        approval_timeout_s=a.approval_timeout, unit=key).run()
        print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
