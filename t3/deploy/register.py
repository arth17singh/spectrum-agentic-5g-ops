"""Register Team13's RAN tools on the organisers' checkpoint. DRY RUN unless --apply.

    python -m t3.deploy.register --kit-dir <folder with kit.json and kit-state.json> \
        --function-arn arn:aws:lambda:<region>:<account>:function:team13-ran-tools [--apply] [--profile fyuz]

Run it after the kit's `deploy` (it needs the Gateway, the policy engine and the approvals table). It:
  1. puts a SEPARATELY NAMED inline policy `team13-ran-tools-invoke` on the kit's Gateway role
     (lambda:InvokeFunction on our function only). The kit rewrites its own inline policy
     `<team>-gateway` on every deploy; it never touches this one, so a kit redeploy keeps it;
  2. creates (or updates) the Gateway target `ran` for our Lambda, and TAKES OVER the kit's `nlm`
     target: points it at our Lambda with the kit's `ask_nlm` schema plus our `ask`, so the frozen
     name `nlm___ask` exists (amendment 1, A1). The kit's deploy skips targets that exist, so a
     redeploy keeps this; the kit's `<team>-nlm` Lambda is then unused. Waits for READY, because
     Cedar validates rules against the tool schemas;
  3. creates our Cedar rules (t3/policies/generate.py) with the kit's validation modes, replacing
     a rule only if its text changed. Pause rules are not created here (the trust gate or a person does).
Without --apply it makes no AWS call at all (not even reads). The dry-run plan assumes the kit has
deployed (its `nlm` target exists) and that none of our targets or rules exist yet.

It never creates IAM roles, tables or Lambda functions: service roles in this account are created in
CloudShell under /bb26/ with the event boundary, and case rows live in the kit's approvals table
(A10). It prints the permissions and settings our function needs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from t3.policies import generate
from t3.tools import catalog

INLINE_POLICY = "team13-ran-tools-invoke"
FN_ARN_RE = re.compile(r"arn:aws:lambda:([a-z0-9-]+):(\d{12}):function:([A-Za-z0-9_-]+)")
ACP = "bedrock-agentcore-control"


class Api:
    """Prints every call; executes it only when `apply` is true."""

    def __init__(self, apply: bool, client_factory: Callable[[str], Any] | None = None, out=print):
        self.apply, self.client_factory, self.out = apply, client_factory, out
        self.calls: list[tuple[str, str, dict]] = []
        self._clients: dict[str, Any] = {}

    def __call__(self, service: str, op: str, read: bool = False, **params):
        self.calls.append((service, op, params))
        verb = ("READ" if read else "CALL") if self.apply else ("WOULD READ" if read else "WOULD CALL")
        self.out(f"{verb} {service}.{op} {json.dumps(params, default=str, sort_keys=True)}")
        if not self.apply:
            return None
        if service not in self._clients:
            self._clients[service] = self.client_factory(service)
        return getattr(self._clients[service], op)(**params)


def load_kit(kit_dir: Path | None) -> tuple[dict, dict]:
    if kit_dir is None:
        return {}, {}
    cfg = json.loads((kit_dir / "kit.json").read_text()) if (kit_dir / "kit.json").exists() else {}
    state = json.loads((kit_dir / "kit-state.json").read_text()) if (kit_dir / "kit-state.json").exists() else {}
    return cfg, state


def preflight_warnings(cfg: dict, team: str) -> list[str]:
    """Kit settings our rules and tools depend on."""
    warn = []
    agents = [cfg.get("lead")] + list(cfg.get("specialists") or [])
    for agent in generate.AGENT_TOOLS:
        if cfg and agent not in agents:
            warn.append(f"agent {agent} is not in kit.json (lead/specialists): it has no token, so it cannot call")
    may_change = cfg.get("may_change") or ([cfg["lead"]] if cfg.get("lead") else [])
    for agent in generate.APPLY_AGENTS:
        if cfg and agent not in may_change:
            warn.append(f"{agent} is not in kit.json may_change: the interceptor marks its apply NOT_ALLOWED and "
                        "never asks a person, so every ran___apply_change is refused")
    if cfg and cfg.get("name") and cfg["name"] != team:
        warn.append(f"kit.json name is {cfg['name']!r} but --team is {team!r}")
    return warn


def function_settings(fn_arn: str, team: str, region: str, account: str) -> dict:
    """What our function and its execution role need (printed for the lead; not created here)."""
    return {
        "role_policy": {"Version": "2012-10-17", "Statement": [
            {"Sid": "CasesAndApprovals", "Effect": "Allow",  # TransactWriteItems needs the per-item actions
             "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"],
             "Resource": f"arn:aws:dynamodb:{region}:{account}:table/{team}-approvals"},
            {"Sid": "EvidenceRecords", "Effect": "Allow", "Action": "s3:PutObject",
             "Resource": "arn:aws:s3:::<evidence-bucket>/evidence/records/*"},
            {"Sid": "TrustGateAsync", "Effect": "Allow", "Action": "lambda:InvokeFunction",
             "Resource": f"arn:aws:lambda:{region}:{account}:function:{team}-trust-gate"},
            {"Sid": "TunedNlm", "Effect": "Allow", "Action": "sagemaker:InvokeEndpoint",
             "Resource": f"arn:aws:sagemaker:{region}:{account}:endpoint/<NLM_ENDPOINT>"},
            {"Sid": "StandInNlm", "Effect": "Allow", "Action": "bedrock:InvokeModel",
             "Resource": f"arn:aws:bedrock:{region}::foundation-model/<NLM_STANDIN_MODEL>"}]},
        "managed_policies": ["arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"],
        "no_policy_write_rights": "the tools never create or delete Cedar policies (A7)",
        "function": {"arn": fn_arn, "image_cmd": "t3.tools.app.handler", "memory_mb": 3008, "timeout_s": 120,
                     "log_group": f"/aws/lambda/{team}-ran-tools"},
        "environment": {"APPROVALS_TABLE": f"{team}-approvals", "T3_EVIDENCE": "s3://<evidence-bucket>/evidence",
                        "T3_TRUST_FUNCTION": f"{team}-trust-gate", "T2_PREDICTOR_PATH": "<path in image>",
                        "T1_POLICY_PATH": "<path in image>", "NLM_ENDPOINT": "<endpoint name, when it exists>",
                        "NLM_STANDIN_MODEL": "<Bedrock model id, until then>",
                        "T3_SCENARIO": "nominal", "T3_SEED": "0", "T3_TWIN_CONFIG": "official"},
        "never_in_aws": ["T3_TOOL_NAME_FALLBACK", "T3_STORE"],
    }


def _items(response) -> list:
    if not response:
        return []
    for key, value in response.items():
        if key != "ResponseMetadata" and isinstance(value, list):
            return value
    return []


def register(api: Api, *, team: str, fn_arn: str, gateway_id: str, gateway_arn: str, engine_id: str,
             wait: Callable[[float], None] = time.sleep, out=print) -> int:
    failed = 0
    # 1. separately named inline policy on the kit's Gateway role
    out(f"\n== Gateway role inline policy {INLINE_POLICY}")
    api("iam", "put_role_policy", RoleName=f"{team}-gateway", PolicyName=INLINE_POLICY,
        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{
            "Sid": "InvokeTeam13RanTools", "Effect": "Allow", "Action": "lambda:InvokeFunction",
            "Resource": [fn_arn, fn_arn + ":*"]}]}, sort_keys=True))

    # 2. Gateway targets
    out("\n== Gateway targets")
    listed = api(ACP, "list_gateway_targets", read=True, gatewayIdentifier=gateway_id)
    existing = {t["name"]: t for t in _items(listed)}
    if not api.apply:  # dry run: the kit has deployed, so its nlm target exists
        existing.setdefault(catalog.NLM_TARGET, {"targetId": "<kit nlm targetId>"})
    creds = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
    for name, tools in catalog.targets().items():
        config = {"mcp": {"lambda": {"lambdaArn": fn_arn, "toolSchema": {"inlinePayload": tools}}}}
        if name in existing:
            if name == catalog.NLM_TARGET:
                out(f"   TAKE OVER the kit's target {name}: point it at our function "
                    f"({', '.join(t['name'] for t in tools)})")
            api(ACP, "update_gateway_target", gatewayIdentifier=gateway_id, targetId=existing[name]["targetId"],
                name=name, targetConfiguration=config, credentialProviderConfigurations=creds)
        else:
            api(ACP, "create_gateway_target", gatewayIdentifier=gateway_id, name=name,
                targetConfiguration=config, credentialProviderConfigurations=creds)
    if api.apply:
        ours = set(catalog.targets())
        for _ in range(60):
            states = {t["name"]: t.get("status") for t in _items(api(
                ACP, "list_gateway_targets", read=True, gatewayIdentifier=gateway_id))}
            if any("FAIL" in str(states.get(n)) for n in ours):
                out(f"targets failed: {states}")
                return 1
            if all(states.get(n) == "READY" for n in ours):
                break
            wait(5)
        else:
            out("targets did not become READY")
            return 1
        wait(20)  # the guide: rules are checked against the tool schemas, so let the schema settle

    # 3. Cedar rules
    out("\n== Cedar rules")
    current = {p["name"]: p for p in _items(api(ACP, "list_policies", read=True, policyEngineId=engine_id))}
    for rule in generate.rules(team):
        statement = generate.fill(rule["statement"], gateway_arn)
        have = current.get(rule["name"])
        if have:
            got = api(ACP, "get_policy", read=True, policyEngineId=engine_id, policyId=have["policyId"]) or {}
            text = ((got.get("definition") or {}).get("policy") or (got.get("definition") or {}).get("cedar") or {})
            if got.get("status") == "ACTIVE" and " ".join(text.get("statement", "").split()) == " ".join(statement.split()):
                out(f"   unchanged {rule['name']}")
                continue
            api(ACP, "delete_policy", policyEngineId=engine_id, policyId=have["policyId"])
            if api.apply:
                for _ in range(60):
                    try:
                        api(ACP, "get_policy", read=True, policyEngineId=engine_id, policyId=have["policyId"])
                        wait(3)
                    except Exception:  # noqa: BLE001 — gone
                        break
        created = api(ACP, "create_policy", policyEngineId=engine_id, name=rule["name"],
                      validationMode=rule["mode"], description="team13 RAN tools (t3/policies/generate.py)",
                      definition={"policy": {"statement": statement}})
        if api.apply:
            status, p = "CREATING", {}
            for _ in range(100):
                p = api(ACP, "get_policy", read=True, policyEngineId=engine_id, policyId=created["policyId"])
                status = p["status"]
                if status == "ACTIVE" or "FAIL" in status:
                    break
                wait(3)
            out(f"   {status:<8} {rule['name']}")
            if status != "ACTIVE":
                failed += 1
                out(json.dumps({k: v for k, v in p.items() if k not in ("definition", "ResponseMetadata")},
                               default=str, indent=2))
    return failed


def main(argv: list[str] | None = None, client_factory: Callable[[str], Any] | None = None, out=print,
         wait: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(description="Register Team13's RAN tools on the kit's Gateway (dry run by default).")
    ap.add_argument("--kit-dir", type=Path, help="folder with the kit's kit.json and kit-state.json")
    ap.add_argument("--function-arn", help="our tool Lambda's ARN")
    ap.add_argument("--team", help="the kit's name (default: kit.json name, else team13)")
    ap.add_argument("--profile", help="AWS profile for --apply (e.g. fyuz)")
    ap.add_argument("--apply", action="store_true", help="really make the calls (default: print them only)")
    a = ap.parse_args(argv)

    cfg, state = load_kit(a.kit_dir)
    team = a.team or cfg.get("name") or "team13"
    region = cfg.get("region", "us-east-2")
    fn_arn = a.function_arn or f"arn:aws:lambda:{region}:000000000000:function:{team}-ran-tools"
    m = FN_ARN_RE.fullmatch(fn_arn)
    if not m:
        out(f"--function-arn {fn_arn!r} is not a Lambda function ARN")
        return 2
    if a.apply:
        missing = [k for k in ("gateway_id", "gateway_arn", "engine_id") if not state.get(k)]
        if missing or not a.function_arn:
            out(f"--apply needs the kit's kit-state.json (missing: {', '.join(missing) or 'none'}) "
                "and a real --function-arn")
            return 2
        if m.group(1) != region:
            out(f"function region {m.group(1)} != kit region {region}")
            return 2
    problems = generate.check(generate.rules(team), team)
    if problems:
        out("Cedar rules fail the offline checks:\n  " + "\n  ".join(problems))
        return 1
    for w in preflight_warnings(cfg, team):
        out(f"WARNING {w}")

    if a.apply and client_factory is None:
        import boto3

        client_factory = boto3.Session(profile_name=a.profile, region_name=region).client
    api = Api(a.apply, client_factory, out)
    out(f"{'APPLYING' if a.apply else 'DRY RUN (no AWS calls; add --apply to act)'}: team={team} region={region}")
    gateway_arn = state.get("gateway_arn") or \
        f"arn:aws:bedrock-agentcore:{region}:000000000000:gateway/{team}-checkpoint-placeholder"
    failed = register(api, team=team, fn_arn=fn_arn, gateway_id=state.get("gateway_id", "<gateway_id>"),
                      gateway_arn=gateway_arn, engine_id=state.get("engine_id", "<engine_id>"), wait=wait, out=out)
    out("\n== our function needs (create the role in CloudShell under /bb26/ with the boundary):")
    out(json.dumps(function_settings(fn_arn, team, region, m.group(2)), indent=2))
    if failed:
        out(f"\n{failed} rule(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
