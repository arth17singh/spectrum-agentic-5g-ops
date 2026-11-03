"""The live phase of the swarm incident eval: the governed swarm proposes a fix for every masked incident.

Scoring needs x86 numerics (the suite was frozen on x86), and the SageMaker job role reaches neither DynamoDB
nor AgentCore, so the eval is split in two (README: "Scoring the swarm"):

1. LIVE (this module, from a terminal signed in with the team profile): for each of the 18 masked test
   incidents and each of the 3 operator steps, open one case at the harness's state and let the swarm work it
   through the checkpoint's front door, as in run.py's proposal mode:
     - map the case to this run (the runner's binding row) and seed it with ran___seed_case (the app's
       token; Cedar refuses it to every agent): scenario, seed, preset, and the steps so far = commissioning
       and fault (incidents.script) + the swarm's earlier proposals for this incident;
     - send the incident's ticket (swarm.task_text, symptoms only) to noc_lead with front___continue_case;
     - wait for the lead; the swarm stops at the checkpoint's approval refusal on ran___apply_change. The
       PROPOSAL is the waiting change of that case (front___case_status); none means "no change".
   Nobody approves: proposal-only (the declared approval policy). Every case is an `item` record; the
   proposals are written to outputs/<run_id>/swarm-actions.jsonl.
2. SCORE (t3/evals/run.py in a SageMaker CPU job, x86): --operator swarm --mode proposal --subset masked
   --swarm-client t3.evals.recorded_client:factory replays exactly these proposals through the harness's own
   twin, scores recovery, and computes the declared paired gain over config_revert (paired-bootstrap-v1).

    python -m t3.evals.live_swarm --run-id t3.swarm-live-<UTC> [--workers 3] [--limit N] [--live]

Environment: as t3/suite/runner.py (T3_GATEWAY_URL, T3_TOKENS_DIR | T3_TOKEN_TEAM_APP, T3_APPROVALS_TABLE,
EVIDENCE_S3, AWS_PROFILE), plus T3_SWARM_MODEL_ID (what the agents think with, recorded as model_id).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from t3.evals import incidents as inc_mod
from t3.evals.changes import canon, nonzero
from t3.evals.swarm import task_text
from t3.suite import common
from t3.suite.common import RUNNING, envelope, sha256_json, sha256_text, utc_now_iso

UNIT = "t3.evals.swarm-proposals"
SCRIPT = "t3/evals/live_swarm.py"
TARGET = {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"}
DEFAULT_MODEL = "noc_lead=openai.gpt-oss-120b-1:0;students=openai.gpt-oss-20b-1:0"
APPROVAL_POLICY = "proposal-only: no change is approved; the proposal is read from the approval refusal"


def masked_incidents(path: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    incidents, digest = inc_mod.load_test(path or inc_mod.TEST_PATH)
    return [i for i in incidents if i.get("tier", "visible") == "masked"], digest


def load_targets() -> dict[str, Any]:
    return json.loads((Path(inc_mod.__file__).with_name("targets.json")).read_text())


def run_prefix(run_id: str) -> str:
    return hashlib.sha256(run_id.encode()).hexdigest()[:8]


def case_id_for(run_id: str, idx: int, step: int) -> str:
    return f"case-{run_prefix(run_id)}-i{idx:02d}s{step}"


def start_log(incident: dict[str, Any], targets: dict[str, Any], proposals: list[str]) -> list[str]:
    """The steps applied so far, as the harness has them: commissioning and fault, then earlier proposals."""
    steps = [canon(nonzero(s)) if nonzero(s) else "" for s in inc_mod.script(incident, targets, with_fault=True)]
    return steps + [p or "" for p in proposals]


def proposal_of(status: dict[str, Any], case_id: str) -> tuple[str, str | None]:
    """The change the swarm got refused for approval in this case (the kit's waiting row), else no change."""
    for w in status.get("waiting_approvals") or []:
        raw = w.get("change")
        try:
            change = (json.loads(raw) if isinstance(raw, str) else raw or {}).get("change")
        except (ValueError, AttributeError):
            change = None
        if change:
            cells = nonzero(change)
            return (canon(cells) if cells else ""), w.get("approval_id")
    return "", None


class LiveSwarm:
    def __init__(self, *, run_id, gateway, ddb, evlib, cfg, targets, model_id=DEFAULT_MODEL, timeout_s=300.0,
                 poll_s=5.0, sleep=time.sleep, clock=time.time, log=print, synthetic=False):
        self.run_id, self.gw, self.ddb, self.ev, self.cfg = run_id, gateway, ddb, evlib, cfg
        self.targets, self.model_id, self.timeout_s, self.poll_s = targets, model_id, timeout_s, poll_s
        self.sleep, self.clock, self.log, self.synthetic = sleep, clock, log, synthetic
        self.rows: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ----- evidence -----
    def declaration(self, incidents, digest, steps) -> dict[str, Any]:
        n = len(incidents) * steps
        return envelope(
            "declared", self.run_id, synthetic=self.synthetic, unit=UNIT, track="track-3", target=dict(TARGET),
            n_items=n, n_cases=n, test_set_sha256=digest, subset="tier=masked", model_id=self.model_id,
            grader=(f"{SCRIPT}: one item per (incident, step) case; passed = the case ran to the lead's answer and "
                    "the proposal was read from the checkpoint (a timed-out or refused case fails). Scored later "
                    "by t3/evals/run.py on x86 with t3/evals/recorded_client.py."),
            code_sha=common.code_sha(), region=self.cfg["region"],
            twin={"kind": "maveric-inprocess", "version": "ran-rl-1.2.9", "scenario": "incident-suite",
                  "seed": min(i["seed"] for i in incidents),
                  "config": {"suite": "t3/evals/suite/incidents-test.jsonl", "suite_version": inc_mod.SUITE_VERSION,
                             "tier": "masked", "operator_steps": steps, "approval_policy": APPROVAL_POLICY,
                             "twin_host": "team13-ran-tools Lambda (arm64 in-process replay); the score is "
                                          "computed on x86 by the scoring run"}},
            declared_at=utc_now_iso())

    def _item(self, row: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
        return envelope(
            "item", self.run_id, aws_time=out.get("gateway_date"),
            aws_time_source="gateway:http-date" if out.get("gateway_date") else None, synthetic=self.synthetic,
            item_id=f"{row['incident_id']}#s{row['step']}",
            input_sha256=sha256_json({"incident_id": row["incident_id"], "step": row["step"],
                                      "start_action_log": row["start_action_log"], "task_text": row["task_text"]}),
            output_sha256=sha256_text(row["change_text"]), score=1.0 if row["ok"] else 0.0, passed=row["ok"],
            grader=SCRIPT, backend="agentcore-gateway", latency_ms=row["wall_ms"],
            extra={k: row[k] for k in ("incident_id", "step", "case_id", "change_text", "approval_id", "refused",
                                       "state", "problem", "reply_excerpt")})

    # ----- one case -----
    def run_case(self, incident: dict[str, Any], idx: int, step: int, proposals: list[str]) -> dict[str, Any]:
        case_id = case_id_for(self.run_id, idx, step)
        log = start_log(incident, self.targets, proposals)
        text = task_text(incident, self.run_id)
        row = {"run_id": self.run_id, "model_id": self.model_id, "incident_id": incident["incident_id"], "step": step,
               "case_id": case_id, "start_action_log": log, "task_text": text, "change_text": "",
               "approval_id": None, "refused": False, "state": None, "ok": False, "problem": None,
               "reply_excerpt": None}
        t0 = self.clock()
        common.ddb_put(self.ddb, self.cfg["approvals_table"], {
            "pk": f"t3-{case_id}", "case_id": case_id, "run_id": self.run_id, "unit": UNIT,
            "created_at": int(t0), "source": SCRIPT}, condition="attribute_not_exists(pk)")
        seeded = self.gw.call("team_app", "ran___seed_case", {
            "case_id": case_id, "scenario": incident["scenario"], "seed": int(incident["seed"]),
            "preset": incident["preset"], "start_action_log_json": json.dumps(log)}, timeout=120)
        if seeded["decision"] != "ALLOW" or seeded.get("tool_error"):
            row["problem"] = f"seed refused or failed: {(seeded.get('tool_error') or seeded.get('text'))[:300]}"
            return self._finish(row, seeded, t0)
        sent = self.gw.call("team_app", "front___continue_case", {"case_id": case_id, "message": text}, timeout=60)
        if sent["decision"] != "ALLOW":
            row["problem"] = f"front door refused: {sent.get('text', '')[:300]}"
            return self._finish(row, sent, t0)
        deadline = self.clock() + self.timeout_s
        status: dict[str, Any] = {}
        while True:
            self.sleep(self.poll_s)
            st = self.gw.call("team_app", "front___case_status", {"case_id": case_id}, timeout=60)
            try:
                status = json.loads(st["text"]) if st["decision"] == "ALLOW" else {}
            except ValueError:
                status = {}
            if status.get("state") not in RUNNING or self.clock() >= deadline:
                break
        row["state"] = status.get("state")
        row["reply_excerpt"] = str(status.get("reply") or "")[:400]
        if status.get("state") in RUNNING or not status:
            row["problem"] = "timed out waiting for the lead"
        else:
            row["change_text"], row["approval_id"] = proposal_of(status, case_id)
            row["refused"] = row["approval_id"] is not None
            row["ok"] = True
        return self._finish(row, sent, t0)

    def _finish(self, row, out, t0):
        row["wall_ms"] = round((self.clock() - t0) * 1000, 1)
        item = self._item(row, out)
        with self._lock:
            self.rows.append({k: v for k, v in row.items() if k not in ("wall_ms",)})
            self.items.append(item)
        self.ev.append_records(self.run_id, "item", [item])
        self.log(f"{row['incident_id']} s{row['step']} {row['case_id']}: "
                 f"{row['change_text'] or '(no change)'}{'  PROBLEM ' + row['problem'] if row['problem'] else ''}")
        return row

    def run_incident(self, incident, idx, steps):
        proposals: list[str] = []
        for step in range(steps):
            row = self.run_case(incident, idx, step, proposals)
            proposals.append(row["change_text"] if row["ok"] else "")

    # ----- the run -----
    def run(self, incidents, digest, steps=3, workers=1) -> dict[str, Any]:
        decl = self.declaration(incidents, digest, steps)
        self.ev.write_declared(decl)                         # declared before the first call
        common.ddb_put(self.ddb, self.cfg["approvals_table"], {"pk": "t3-active", "run_id": self.run_id,
                                                               "unit": UNIT, "updated_at": int(self.clock())})
        if workers <= 1:
            for idx, inc in enumerate(incidents):
                self.run_incident(inc, idx, steps)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda a: self.run_incident(a[1], a[0], steps), list(enumerate(incidents))))
        self.rows.sort(key=lambda r: (r["incident_id"], r["step"]))
        body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.rows).encode()
        stored = None
        store_fn = getattr(self.ev, "default_store", None)
        if store_fn is not None:
            stored = store_fn().put_new(f"outputs/{self.run_id}/swarm-actions.jsonl", body)
        n_ok = sum(1 for it in self.items if it["passed"])
        value = n_ok / len(self.items) if self.items else 0.0
        result = envelope(
            "result", self.run_id, synthetic=self.synthetic, unit=UNIT, metric=TARGET["metric"], value=value,
            target=dict(TARGET), passed=bool(self.items) and value >= TARGET["value"], n_items=len(self.items),
            checked_at=utc_now_iso(), checker={"script": SCRIPT, "code_sha": decl["code_sha"]},
            details={"recording": stored, "recording_sha256": hashlib.sha256(body).hexdigest(),
                     "proposals": sum(1 for r in self.rows if r["change_text"]),
                     "problems": [{"case_id": r["case_id"], "problem": r["problem"]} for r in self.rows if r["problem"]],
                     "approval_policy": APPROVAL_POLICY, "workers": workers})
        self.ev.write_result(result)
        return {"result": result, "body": body}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=1, help="incidents in parallel (measure 1 first)")
    ap.add_argument("--limit", type=int, help="only the first N masked incidents (a trial; not the scored set)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--tokens-dir")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args(argv)
    targets = load_targets()
    steps = a.steps or targets["episode"]["operator_steps"]
    incidents, digest = masked_incidents()
    if a.limit:
        incidents = incidents[: a.limit]
    if not a.live:
        print(f"DRY RUN {a.run_id}: {len(incidents)} masked incidents x {steps} steps = {len(incidents) * steps} cases, "
              f"suite {digest[:12]}; per case: map, ran___seed_case, front___continue_case, poll case_status, read "
              f"the waiting change. Proposal-only (nobody approves).")
        for idx, inc in enumerate(incidents[:3]):
            print(" ", case_id_for(a.run_id, idx, 0), inc["incident_id"], json.dumps(start_log(inc, targets, [])))
        return 0
    import boto3
    from t3.suite.runner import env_config
    cfg = env_config()
    session = boto3.session.Session(region_name=cfg["region"])
    secrets = session.client("secretsmanager") if os.environ.get("T3_TOKENS_SECRET") else None
    gw = common.Gateway(cfg["gateway_url"], common.Tokens(secrets_client=secrets, tokens_dir=a.tokens_dir))
    live = LiveSwarm(run_id=a.run_id, gateway=gw, ddb=session.client("dynamodb"), evlib=common.load_evlib(),
                     cfg=cfg, targets=targets, model_id=os.environ.get("T3_SWARM_MODEL_ID", DEFAULT_MODEL),
                     timeout_s=a.timeout)
    out = live.run(incidents, digest, steps, a.workers)
    local = common.CALLS_DIR / f"{a.run_id}.swarm-actions.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(out["body"])
    print(json.dumps({k: out["result"][k] for k in ("value", "passed", "n_items")} |
                     {"recording": out["result"]["details"]["recording"], "local": str(local)}, indent=1))
    return 0 if out["result"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
