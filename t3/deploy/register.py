"""Register Team13's RAN tools on the organisers' checkpoint. DRY RUN unless --apply.

    python -m t3.deploy.register --kit-dir <folder with kit.json and kit-state.json> \
        --function-arn arn:aws:lambda:<region>:<account>:function:<name> [--apply] [--profile fyuz]

Run it after the kit's `deploy` (it needs the Gateway and the policy engine). It:
  1. creates the cases table `<team>-ran-cases` if it is missing (partition key case_id);
  2. puts a SEPARATELY NAMED inline policy `team13-ran-tools-invoke` on the kit's Gateway role
     (lambda:InvokeFunction on our function only). The kit rewrites its own inline policy
     `<team>-gateway` on every deploy; it never touches this one, so a kit redeploy keeps it;
  3. creates (or updates) the Gateway targets `ran` and the NLM target (default `nlm13`) for our
     Lambda, and waits until they are READY (Cedar validates rules against the tool schemas);
  4. creates our Cedar rules (t3/policies/generate.py) with the kit's validation modes,
     replacing a rule only if its text changed. Pause rules are not created here.
It prints every AWS call it would make. Without --apply it makes no AWS call at all (it
does not even read), so "exists?" checks are shown as reads and the plan assumes nothing exists.

It never creates IAM roles or Lambda functions: service roles in this account are created in
CloudShell under /bb26/ with the event boundary. It prints the permissions our function's role needs.
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


class Api:
    """Prints every call; executes it only when `apply` is true."""

    def __init__(self, apply: bool, client_factory: Callable[[str], Any] | None = None, out=print):
        self.apply, self.client_factory, self.out = apply, client_factory, out
        self.calls: list[tuple[str, str, dict]] = []
        self._clients: dict[str, Any] = {}

    def __call__(self, service: str, op: str, read: bool = False, **params):
        self.calls.append((service, op, params))
        tag = "READ " if read else "CALL "
        self.out(f"{tag if self.apply else 'WOULD ' + tag.strip() + ' '}{service}.{op} "
                 f"{json.dumps(params, default=str, sort_keys=True)}")
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


def lambda_role_permissions(fn_arn: str, team: str, region: str, account: str) -> dict:
    """What our function's execution role needs (printed for the lead; not created here)."""
    return {"Version": "2012-10-17", "Statement": [
        {"Sid": "Cases", "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem"],
         "Resource": f"arn:aws:dynamodb:{region}:{account}:table/{team}-ran-cases"},
        {"Sid": "Approvals", "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:UpdateItem"],
         "Resource": f"arn:aws:dynamodb:{region}:{account}:table/{team}-approvals"},
        {"Sid": "Evidence", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
         "Resource": "arn:aws:s3:::<evidence-bucket>/evidence/records/*"},
        {"Sid": "TunedNlm", "Effect": "Allow", "Action": "sagemaker:InvokeEndpoint",
         "Resource": f"arn:aws:sagemaker:{region}:{account}:endpoint/<NLM_ENDPOINT>"},
        {"Sid": "StandInNlm", "Effect": "Allow", "Action": ["bedrock:InvokeModel"],
         "Resource": f"arn:aws:bedrock:{region}::foundation-model/<NLM_STANDIN_MODEL>"},
        {"Sid": "TrustGatePause", "Effect": "Allow",
         "Action": ["bedrock-agentcore:CreatePolicy", "bedrock-agentcore:GetPolicy", "bedrock-agentcore:ListPolicies"],
         "Resource": "*"},
    ], "_also": ["AWSLambdaBasicExecutionRole", f"LoggingConfig.LogGroup = /{team}/checkpoint (so the kit's "
                 "`trace case-...` finds our records)", f"function ARN {fn_arn}"]}


def register(api: Api, *, team: str, fn_arn: str, gateway_id: str, gateway_arn: str, engine_id: str,
             change_type: str = "object", wait: Callable[[float], None] = time.sleep, out=print) -> int:
    failed = 0
    # 1. cases table
    out("\n== cases table")
    table = f"{team}-ran-cases"
    try:
        exists = api("dynamodb", "describe_table", read=True, TableName=table) is not None
    except Exception as exc:  # noqa: BLE001
        if "ResourceNotFound" not in type(exc).__name__ + str(exc):
            raise
        exists = False
    if not exists:
        api("dynamodb", "create_table", TableName=table, BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": "case_id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "case_id", "KeyType": "HASH"}])

    # 2. separately named inline policy on the kit's Gateway role
    out(f"\n== Gateway role inline policy {INLINE_POLICY}")
    api("iam", "put_role_policy", RoleName=f"{team}-gateway", PolicyName=INLINE_POLICY,
        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{
            "Sid": "InvokeTeam13RanTools", "Effect": "Allow", "Action": "lambda:InvokeFunction",
            "Resource": [fn_arn, fn_arn + ":*"]}]}, sort_keys=True))

    # 3. Gateway targets
    out("\n== Gateway targets")
    listed = api("bedrock-agentcore-control", "list_gateway_targets", read=True, gatewayIdentifier=gateway_id)
    existing = {t["name"]: t for t in _items(listed)}
    for name, tools in catalog.targets(change_type).items():
        config = {"mcp": {"lambda": {"lambdaArn": fn_arn, "toolSchema": {"inlinePayload": tools}}}}
        creds = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
        if name in existing:
            api("bedrock-agentcore-control", "update_gateway_target", gatewayIdentifier=gateway_id,
                targetId=existing[name]["targetId"], name=name, targetConfiguration=config,
                credentialProviderConfigurations=creds)
        else:
            api("bedrock-agentcore-control", "create_gateway_target", gatewayIdentifier=gateway_id, name=name,
                targetConfiguration=config, credentialProviderConfigurations=creds)
    if api.apply:
        ours = set(catalog.targets(change_type))
        for _ in range(60):
            states = {t["name"]: t.get("status") for t in _items(api(
                "bedrock-agentcore-control", "list_gateway_targets", read=True, gatewayIdentifier=gateway_id))}
            if any("FAIL" in str(states.get(n)) for n in ours):
                out(f"targets failed: {states}")
                return 1
            if all(states.get(n) == "READY" for n in ours):
                break
            wait(5)
        wait(20)  # the guide: rules are checked against the tool schemas, so let the schema settle

    # 4. Cedar rules
    out("\n== Cedar rules")
    current = {p["name"]: p for p in _items(api("bedrock-agentcore-control", "list_policies", read=True,
                                                policyEngineId=engine_id))}
    for rule in generate.rules(team):
        statement = generate.fill(rule["statement"], gateway_arn)
        have = current.get(rule["name"])
        if have:
            got = api("bedrock-agentcore-control", "get_policy", read=True, policyEngineId=engine_id,
                      policyId=have["policyId"]) or {}
            text = ((got.get("definition") or {}).get("policy") or (got.get("definition") or {}).get("cedar") or {})
            if got.get("status") == "ACTIVE" and " ".join(text.get("statement", "").split()) == " ".join(statement.split()):
                out(f"   unchanged {rule['name']}")
                continue
            api("bedrock-agentcore-control", "delete_policy", policyEngineId=engine_id, policyId=have["policyId"])
            if api.apply:
                for _ in range(60):
                    try:
                        api("bedrock-agentcore-control", "get_policy", read=True, policyEngineId=engine_id,
                            policyId=have["policyId"])
                        wait(3)
                    except Exception:  # noqa: BLE001 — gone
                        break
        created = api("bedrock-agentcore-control", "create_policy", policyEngineId=engine_id, name=rule["name"],
                      validationMode=rule["mode"], description="team13 RAN tools (t3/policies/generate.py)",
                      definition={"policy": {"statement": statement}})
        if api.apply:
            status = "CREATING"
            for _ in range(100):
                p = api("bedrock-agentcore-control", "get_policy", read=True, policyEngineId=engine_id,
                        policyId=created["policyId"])
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


def _items(response) -> list:
    if not response:
        return []
    for key, value in response.items():
        if key != "ResponseMetadata" and isinstance(value, list):
            return value
    return []


def main(argv: list[str] | None = None, client_factory: Callable[[str], Any] | None = None, out=print,
         wait: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(description="Register Team13's RAN tools on the kit's Gateway (dry run by default).")
    ap.add_argument("--kit-dir", type=Path, help="folder with the kit's kit.json and kit-state.json")
    ap.add_argument("--function-arn", help="our tool Lambda's ARN")
    ap.add_argument("--team", help="the kit's name (default: kit.json name, else team13)")
    ap.add_argument("--change-type", choices=catalog.CHANGE_TYPES, default="object",
                    help="how `change` is declared to the Gateway (string = JSON text fallback)")
    ap.add_argument("--profile", help="AWS profile for --apply (e.g. fyuz)")
    ap.add_argument("--apply", action="store_true", help="really make the calls (default: print them only)")
    a = ap.parse_args(argv)

    cfg, state = load_kit(a.kit_dir)
    team = a.team or cfg.get("name") or "team13"
    region = cfg.get("region", "<region>")
    fn_arn = a.function_arn or f"arn:aws:lambda:{region}:<account>:function:{team}-ran-tools"
    m = FN_ARN_RE.fullmatch(fn_arn)
    if a.apply:
        missing = [k for k in ("gateway_id", "gateway_arn", "engine_id") if not state.get(k)]
        if missing or not m:
            out(f"--apply needs the kit's kit-state.json ({', '.join(missing) or 'ok'}) and a real --function-arn")
            return 2
        if cfg.get("region") and m.group(1) != cfg["region"]:
            out(f"function region {m.group(1)} != kit region {cfg['region']}")
            return 2
    problems = generate.check(generate.rules(team), team)
    if problems:
        out("Cedar rules fail the offline checks:\n  " + "\n  ".join(problems))
        return 1
    for w in preflight_warnings(cfg, team):
        out(f"WARNING {w}")

    if a.apply and client_factory is None:
        import boto3

        session = boto3.Session(profile_name=a.profile, region_name=region)
        client_factory = session.client
    api = Api(a.apply, client_factory, out)
    out(f"{'APPLYING' if a.apply else 'DRY RUN (no AWS calls; add --apply to act)'}: team={team} region={region}")
    gateway_arn = state.get("gateway_arn") or f"arn:aws:bedrock-agentcore:{region if region != '<region>' else 'us-east-2'}:000000000000:gateway/{team}-checkpoint-placeholder"
    failed = register(api, team=team, fn_arn=fn_arn, gateway_id=state.get("gateway_id", "<gateway_id>"),
                      gateway_arn=gateway_arn, engine_id=state.get("engine_id", "<engine_id>"),
                      change_type=a.change_type, wait=wait, out=out)
    account = m.group(2) if m else "<account>"
    out("\n== our function's execution role needs (create it in CloudShell under /bb26/ with the boundary):")
    out(json.dumps(lambda_role_permissions(fn_arn, team, region, account), indent=2))
    out("\nfunction environment: CASES_TABLE=" + f"{team}-ran-cases APPROVALS_TABLE={team}-approvals "
        "T3_EVIDENCE=s3://<bucket>/evidence T3_NLM_TARGET=" + catalog.nlm_target() +
        " NLM_ENDPOINT=... or NLM_STANDIN_MODEL=... T2_PREDICTOR_PATH=... T1_POLICY_PATH=...")
    if failed:
        out(f"\n{failed} rule(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
