"""Deploy Team13's RAN tools onto the organisers' checkpoint. DRY RUN unless --apply.

    python -m t3.deploy.register --kit-dir <folder with kit.json and kit-state.json> \
        --evidence-bucket <team bucket> [--image-uri <ecr uri>:<tag>] [--apply] [--profile fyuz]

Order: kit `deploy` -> `t3/deploy/iam_cloudshell.sh` in CloudShell (roles and the Gateway-role
inline policy; this session cannot do IAM) -> push W3b's image -> this script. It makes NO IAM
calls; it assumes the roles /bb26/team13-ran-tools and /bb26/team13-trust-gate and the inline
policy team13-ran-tools-invoke on team13-gateway exist. It:
  1. creates the ECR repository `team13-ran-tools` if missing, with a repository policy that lets
     Lambda pull our functions' images;
  2. creates or updates the Lambdas `team13-ran-tools` (handler t3.tools.app.handler, 3008 MB,
     120 s) and `team13-trust-gate` (same image, command t3.trust.handler.handler);
  3. creates or updates the Gateway target `ran`, and TAKES OVER the kit's `nlm` target: points
     it at our Lambda with the kit's `ask_nlm` schema plus our `ask`, so the frozen name
     `nlm___ask` exists (amendment 1, A1). The kit's deploy skips existing targets, so a redeploy
     keeps this; the kit's `team13-nlm` Lambda is then unused. Waits for READY, because Cedar
     validates rules against the tool schemas;
  4. creates our Cedar rules (t3/policies/generate.py) with the kit's validation modes, replacing
     a rule only if its text changed. Pause rules are not created here (the trust gate or a person does).
Without --apply it makes no AWS call at all (not even reads). The dry-run plan assumes the kit has
deployed (its `nlm` target exists) and that none of our resources exist yet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from t3.policies import generate
from t3.tools import catalog

ACP = "bedrock-agentcore-control"
TOOLS_HANDLER = "t3.tools.app.handler"
TRUST_HANDLER = "t3.trust.handler.handler"
STANDIN_DEFAULT = "openai.gpt-oss-20b-1:0"


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


def _code(exc: Exception) -> str | None:
    return (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")


def _exists(api: Api, service: str, op: str, missing: tuple[str, ...], **params) -> Any:
    """The read's answer, or None if the resource does not exist (dry run: always None)."""
    try:
        return api(service, op, read=True, **params)
    except Exception as exc:  # noqa: BLE001
        if _code(exc) in missing:
            return None
        raise


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


def _items(response) -> list:
    if not response:
        return []
    for key, value in response.items():
        if key != "ResponseMetadata" and isinstance(value, list):
            return value
    return []


def function_specs(*, team: str, region: str, account: str, image_uri: str, arch: str, bucket: str,
                   gateway_arn: str, engine_id: str, standin: str, nlm_endpoint: str, predictor_path: str,
                   policy_path: str, trust_live: bool) -> list[dict]:
    """create_function arguments for our two Lambdas (roles come from iam_cloudshell.sh)."""
    def role(name):
        return f"arn:aws:iam::{account}:role/bb26/{name}"
    tools_env = {"APPROVALS_TABLE": f"{team}-approvals", "T3_EVIDENCE": f"s3://{bucket}/evidence",
                 "T3_TRUST_FUNCTION": f"{team}-trust-gate", "NLM_STANDIN_MODEL": standin,
                 "T3_SCENARIO": "nominal", "T3_SEED": "0", "T3_TWIN_CONFIG": "official"}
    for key, value in (("NLM_ENDPOINT", nlm_endpoint), ("T2_PREDICTOR_PATH", predictor_path),
                       ("T1_POLICY_PATH", policy_path)):
        if value:
            tools_env[key] = value
    trust_env = {"T3_TRUST_STORE": f"ddb:{team}-approvals", "T3_TRUST_LIVE": "1" if trust_live else "0",
                 "T3_GATEWAY_ARN": gateway_arn, "T3_POLICY_ENGINE_ID": engine_id,
                 "T3_EVIDENCE": f"s3://{bucket}/evidence"}
    common = {"PackageType": "Image", "Code": {"ImageUri": image_uri}, "Architectures": [arch]}
    return [
        {**common, "FunctionName": f"{team}-ran-tools", "Role": role(f"{team}-ran-tools"),
         "Description": "Team13 governed RAN tools (AgentCore Gateway target)",
         "MemorySize": 3008, "Timeout": 120, "ImageConfig": {"Command": [TOOLS_HANDLER]},
         "Environment": {"Variables": tools_env}},
        {**common, "FunctionName": f"{team}-trust-gate", "Role": role(f"{team}-trust-gate"),
         "Description": "Team13 trust gate (invoked asynchronously by ran___apply_change)",
         "MemorySize": 1024, "Timeout": 60, "ImageConfig": {"Command": [TRUST_HANDLER]},
         "Environment": {"Variables": trust_env}},
    ]


def _wait_function(api: Api, name: str, wait: Callable[[float], None]) -> None:
    if not api.apply:
        return
    for _ in range(120):
        cfg = (api("lambda", "get_function", read=True, FunctionName=name) or {}).get("Configuration", {})
        if cfg.get("State") == "Failed" or cfg.get("LastUpdateStatus") == "Failed":
            raise RuntimeError(f"{name}: {cfg.get('StateReason') or cfg.get('LastUpdateStatusReason')}")
        if cfg.get("State") == "Active" and cfg.get("LastUpdateStatus") in (None, "Successful"):
            return
        wait(5)
    raise RuntimeError(f"{name} did not become Active")


def deploy_functions(api: Api, *, team: str, region: str, account: str, specs: list[dict],
                     wait: Callable[[float], None], out=print) -> str:
    out("\n== ECR repository")
    repo = f"{team}-ran-tools"
    if not _exists(api, "ecr", "describe_repositories", ("RepositoryNotFoundException",), repositoryNames=[repo]):
        api("ecr", "create_repository", repositoryName=repo, imageTagMutability="MUTABLE",
            imageScanningConfiguration={"scanOnPush": True})
    api("ecr", "set_repository_policy", repositoryName=repo, policyText=json.dumps({
        "Version": "2012-10-17", "Statement": [{
            "Sid": "LambdaPullForTeam13Functions", "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
            "Condition": {"StringLike": {"aws:sourceArn": f"arn:aws:lambda:{region}:{account}:function:{team}-*"}}}]},
        sort_keys=True))

    out("\n== Lambda functions")
    tools_arn = ""
    for spec in specs:
        name = spec["FunctionName"]
        found = _exists(api, "lambda", "get_function", ("ResourceNotFoundException",), FunctionName=name)
        if found:
            api("lambda", "update_function_code", FunctionName=name, ImageUri=spec["Code"]["ImageUri"],
                Architectures=spec["Architectures"])
            _wait_function(api, name, wait)
            api("lambda", "update_function_configuration",
                **{k: v for k, v in spec.items() if k not in ("PackageType", "Code", "Architectures")})
        else:
            api("lambda", "create_function", **spec)
        _wait_function(api, name, wait)
        if name == f"{team}-ran-tools":
            tools_arn = f"arn:aws:lambda:{region}:{account}:function:{name}"
    return tools_arn


def register(api: Api, *, team: str, fn_arn: str, gateway_id: str, gateway_arn: str, engine_id: str,
             wait: Callable[[float], None] = time.sleep, out=print) -> int:
    failed = 0
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
    ap = argparse.ArgumentParser(description="Deploy Team13's RAN tools onto the kit's Gateway (dry run by default).")
    ap.add_argument("--kit-dir", type=Path, help="folder with the kit's kit.json and kit-state.json")
    ap.add_argument("--team", help="the kit's name (default: kit.json name, else team13)")
    ap.add_argument("--image-uri", help="our image in ECR (default <account>.dkr.ecr.<region>.amazonaws.com/<team>-ran-tools:latest)")
    ap.add_argument("--arch", choices=("arm64", "x86_64"), default="arm64", help="the image's architecture")
    ap.add_argument("--evidence-bucket", default="<evidence-bucket>", help="team bucket for evidence/records/")
    ap.add_argument("--standin-model", default=STANDIN_DEFAULT, help="Bedrock stand-in for the NLM")
    ap.add_argument("--nlm-endpoint", default="", help="SageMaker endpoint of our tuned model, when it exists")
    ap.add_argument("--predictor-path", default="", help="T2_PREDICTOR_PATH inside the image")
    ap.add_argument("--policy-path", default="", help="T1_POLICY_PATH inside the image")
    ap.add_argument("--trust-live", action="store_true", help="let the trust gate create real pause rules")
    ap.add_argument("--profile", help="AWS profile for --apply (e.g. fyuz)")
    ap.add_argument("--apply", action="store_true", help="really make the calls (default: print them only)")
    a = ap.parse_args(argv)

    cfg, state = load_kit(a.kit_dir)
    team = a.team or cfg.get("name") or "team13"
    region = cfg.get("region", "us-east-2")
    if a.apply:
        missing = [k for k in ("gateway_id", "gateway_arn", "engine_id") if not state.get(k)]
        if missing or a.evidence_bucket.startswith("<"):
            out(f"--apply needs the kit's kit-state.json (missing: {', '.join(missing) or 'none'}) "
                "and --evidence-bucket")
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
    out("Assumes iam_cloudshell.sh has run: roles /bb26/{0}-ran-tools, /bb26/{0}-trust-gate and the inline "
        "policy {0}-ran-tools-invoke on {0}-gateway. This script makes no IAM calls.".format(team))
    account = (api("sts", "get_caller_identity", read=True) or {}).get("Account", "000000000000")
    image_uri = a.image_uri or f"{account}.dkr.ecr.{region}.amazonaws.com/{team}-ran-tools:latest"
    gateway_arn = state.get("gateway_arn") or \
        f"arn:aws:bedrock-agentcore:{region}:000000000000:gateway/{team}-checkpoint-placeholder"
    engine_id = state.get("engine_id", "<engine_id>")
    specs = function_specs(team=team, region=region, account=account, image_uri=image_uri, arch=a.arch,
                           bucket=a.evidence_bucket, gateway_arn=gateway_arn, engine_id=engine_id,
                           standin=a.standin_model, nlm_endpoint=a.nlm_endpoint, predictor_path=a.predictor_path,
                           policy_path=a.policy_path, trust_live=a.trust_live)
    fn_arn = deploy_functions(api, team=team, region=region, account=account, specs=specs, wait=wait, out=out)
    failed = register(api, team=team, fn_arn=fn_arn, gateway_id=state.get("gateway_id", "<gateway_id>"),
                      gateway_arn=gateway_arn, engine_id=engine_id, wait=wait, out=out)
    if failed:
        out(f"\n{failed} rule(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
