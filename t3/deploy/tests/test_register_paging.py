"""register.py reads EVERY page of list_policies / list_gateway_targets (nextToken). On 3 Nov the engine held
about 27 policies; register read only page 1, missed team13_ran_allow_propose, tried to create it again,
and stopped without saying why. Re-running against an engine where every rule exists must create nothing,
and a failing call must be named with its exception."""
from __future__ import annotations

from botocore.exceptions import ClientError

from t3.deploy import register
from t3.deploy.tests.test_register import GW, ValidatingFake, kit_dir  # noqa: F401 - fixture
from t3.policies import generate

IMAGE = "000000000000.dkr.ecr.us-east-2.amazonaws.com/team13-ran-tools:0123456789ab"
PAGE = 5   # the fake engine answers 5 policies per page, like a small page size


def paged_world(policies: dict[str, str], targets: list[str], kit_policies: int = 12):
    """Every one of our rules already exists, ACTIVE, with the generated text; plus the kit's own rules."""
    allp = [(f"team13_kit_rule_{i}", "permit (principal, action, resource);") for i in range(kit_policies)]
    allp += list(policies.items())

    def page(items, token):
        start = int(token or 0)
        nxt = start + PAGE
        out = {"items": items[start:nxt]}
        if nxt < len(items):
            out["nextToken"] = str(nxt)
        return out

    def world(service, op, params):
        if op == "get_caller_identity":
            return {"Account": "000000000000"}
        if op == "describe_repositories":
            return {"repositories": [{"repositoryName": "team13-ran-tools"}]}
        if op == "get_function":
            return {"Configuration": {"State": "Active", "LastUpdateStatus": "Successful"}}
        if op == "list_gateway_targets":
            return page([{"targetId": f"t-{n}", "name": n, "status": "READY"} for n in targets],
                        params.get("nextToken"))
        if op == "list_policies":
            got = page([{"name": n, "policyId": f"p-{n}"} for n, _ in allp], params.get("nextToken"))
            return {"policies": got["items"], **({"nextToken": got["nextToken"]} if "nextToken" in got else {})}
        if op == "get_policy":
            name = params["policyId"][2:]
            return {"status": "ACTIVE", "definition": {"policy": {"statement": dict(allp)[name]}}}
        if op == "create_policy":
            raise ClientError({"Error": {"Code": "ConflictException", "Message": "policy name exists"}},
                              "CreatePolicy")
        return {}
    return world


def run(kit_dir, world):  # noqa: F811
    log, lines = [], []
    rc = register.main(["--kit-dir", str(kit_dir), "--image-uri", IMAGE, "--evidence-bucket", "team-bucket",
                        "--apply"], client_factory=lambda svc: ValidatingFake(svc, log, world),
                       out=lines.append, wait=lambda s: None)
    return rc, log, lines


def test_every_page_is_read_and_an_up_to_date_engine_gets_no_create(kit_dir):  # noqa: F811
    ours = {r["name"]: generate.fill(r["statement"], GW) for r in generate.rules()}
    targets = ["case", "work", "nlm", "swarm", "front", "ran", "extra1", "extra2"]   # 8 targets: 2 pages
    rc, log, lines = run(kit_dir, paged_world(ours, targets))
    assert rc == 0, "\n".join(lines[-5:])
    ops = [op for _, op, _ in log]
    assert "create_policy" not in ops and "delete_policy" not in ops
    pages = [p for _, op, p in log if op == "list_policies"]
    assert len(pages) == -(-(12 + len(ours)) // PAGE) and pages[1].get("nextToken") == str(PAGE)
    assert sum(1 for _, op, p in log if op == "list_gateway_targets" and p.get("nextToken")) >= 1
    assert sum(1 for line in lines if "unchanged team13_ran_" in line) == len(ours)


def test_a_failing_call_is_named_with_its_exception(kit_dir):  # noqa: F811
    ours = {r["name"]: generate.fill(r["statement"], GW) for r in generate.rules()}
    missing = dict(ours)
    missing.pop("team13_ran_seed_app_only")       # one rule absent -> create -> the fake refuses it
    rc, log, lines = run(kit_dir, paged_world(missing, ["case", "work", "nlm", "swarm", "front", "ran"]))
    assert rc == 1
    assert any("FAILED at bedrock-agentcore-control.create_policy: ClientError" in line and "policy name exists"
               in line for line in lines)
