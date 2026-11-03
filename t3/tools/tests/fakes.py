"""Fakes for the RAN tool tests: DynamoDB (low-level client), S3, the twin, the predictor, the
policy, the trust-gate invoker and the AWS model clients. No network. Requests to the DynamoDB and
S3 fakes are validated against botocore's real service models, offline."""
from __future__ import annotations

import copy
import io
import json
import types

import boto3
from boto3.dynamodb.types import TypeDeserializer
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

N_CELLS = 4
_SESSION = boto3.Session(region_name="us-east-2", aws_access_key_id="x", aws_secret_access_key="x")
_DE = TypeDeserializer()


_MODELS: dict = {}


def _validate(service: str, op: str, params: dict) -> None:
    if service not in _MODELS:
        _MODELS[service] = _SESSION.client(service).meta.service_model
    model = _MODELS[service]
    validate_parameters(params, model.operation_model(op).input_shape)


def _err(code: str, op: str, **extra) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}, **extra}, op)


class FakeDynamo:
    """The low-level DynamoDB calls the store makes, with their condition expressions."""

    def __init__(self, table: str = "team13-approvals"):
        self.table, self.items, self.calls = table, {}, []

    # helpers for tests (what the kit's interceptor and `approve` do)
    def plain(self, pk: str) -> dict | None:
        item = self.items.get(pk)
        return {k: _DE.deserialize(v) for k, v in item.items()} if item else None

    def set_plain(self, pk: str, **attrs) -> None:
        item = {"pk": {"S": pk}}
        for k, v in attrs.items():
            item[k] = {"N": str(v)} if isinstance(v, int) and not isinstance(v, bool) else {"S": str(v)}
        self.items[pk] = item

    def _check(self, cond: str | None, cur: dict | None, values: dict | None, names: dict | None) -> bool:
        if cond is None:
            return True
        if cond == "attribute_not_exists(pk)":
            return cur is None
        if cond == "attribute_not_exists(pk) OR attribute_not_exists(doc)":
            return cur is None or "doc" not in cur
        if cond == "version = :v":
            return cur is not None and "version" in cur and cur["version"] == values[":v"]
        if cond == "#s = :approved":
            assert names == {"#s": "status"}
            return cur is not None and cur.get("status") == values[":approved"]
        raise AssertionError(f"unexpected condition {cond}")

    def get_item(self, **kw):
        _validate("dynamodb", "GetItem", kw)
        self.calls.append(("get_item", kw))
        assert kw["TableName"] == self.table
        item = self.items.get(kw["Key"]["pk"]["S"])
        return {"Item": copy.deepcopy(item)} if item else {}

    def put_item(self, **kw):
        _validate("dynamodb", "PutItem", kw)
        self.calls.append(("put_item", kw))
        pk = kw["Item"]["pk"]["S"]
        if not self._check(kw.get("ConditionExpression"), self.items.get(pk), kw.get("ExpressionAttributeValues"), None):
            raise _err("ConditionalCheckFailedException", "PutItem")
        self.items[pk] = copy.deepcopy(kw["Item"])

    def transact_write_items(self, **kw):
        _validate("dynamodb", "TransactWriteItems", kw)
        self.calls.append(("transact_write_items", kw))
        reasons, ok = [], True
        for t in kw["TransactItems"]:
            op = t.get("Update") or t.get("Put")
            pk = (op.get("Key") or op.get("Item"))["pk"]["S"]
            cur = self.items.get(pk)
            if self._check(op.get("ConditionExpression"), cur, op.get("ExpressionAttributeValues"),
                           op.get("ExpressionAttributeNames")):
                reasons.append({"Code": "None"})
            else:
                ok = False
                r = {"Code": "ConditionalCheckFailed", "Message": "The conditional request failed"}
                if op.get("ReturnValuesOnConditionCheckFailure") == "ALL_OLD" and cur:
                    r["Item"] = copy.deepcopy(cur)
                reasons.append(r)
        if not ok:
            raise _err("TransactionCanceledException", "TransactWriteItems", CancellationReasons=reasons)
        for t in kw["TransactItems"]:
            if "Put" in t:
                self.items[t["Put"]["Item"]["pk"]["S"]] = copy.deepcopy(t["Put"]["Item"])
            else:
                u = t["Update"]
                assert u["UpdateExpression"] == "SET #s = :applied, applied_at = :t"
                item = self.items[u["Key"]["pk"]["S"]]
                item["status"] = u["ExpressionAttributeValues"][":applied"]
                item["applied_at"] = u["ExpressionAttributeValues"][":t"]


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, **kw):
        _validate("s3", "PutObject", kw)
        if kw.get("IfNoneMatch") == "*" and (kw["Bucket"], kw["Key"]) in self.objects:
            raise _err("PreconditionFailed", "PutObject")
        self.objects[(kw["Bucket"], kw["Key"])] = kw["Body"]
        return {}


class FakeTwin:
    """Deterministic toy twin with W2's ReplayTwin surface: reset(), apply(change) (through a safety
    filter that clips each delta to +-3), action_log entries {step, change, filtered, rules, clamped,
    kpis}, and a resolved `config`."""

    instances = 0

    def __init__(self, scenario, seed, config):
        FakeTwin.instances += 1
        self.scenario, self.seed = scenario, seed
        self.config = {"preset": config, "step_seconds": 1.0} if isinstance(config, str) else config
        self.action_log = []

    def reset(self):
        self.state = {"step": 0, "power": [40.0 + (self.seed % 3)] * N_CELLS, "tilt": [6.0] * N_CELLS}
        self.action_log = []
        return copy.deepcopy(self.state)

    def apply(self, change):
        filtered, rules = {"power": [0.0] * N_CELLS, "tilt": [0.0] * N_CELLS}, []
        for c in change:
            for key, src in (("power", "d_power_db"), ("tilt", "d_tilt_deg")):
                v = float(c.get(src, 0.0))
                clipped = max(-3.0, min(3.0, v))
                if clipped != v:
                    rules.append(f"clip_{key}")
                filtered[key][c["cell"]] += clipped
        for i in range(N_CELLS):
            self.state["power"][i] += filtered["power"][i]
            self.state["tilt"][i] += filtered["tilt"][i]
        self.state["step"] += 1
        self.action_log.append({"step": len(self.action_log), "change": copy.deepcopy(change),
                                "filtered": filtered, "rules": rules, "clamped": bool(rules),
                                "kpis": {k: v for k, v in fake_kpis(self.state).items() if k != "per_cell"}})
        return copy.deepcopy(self.state)


class FakeTwinCF(FakeTwin):
    """FakeTwin plus W2's `counterfactual(change)` (int/v4): both futures from a snapshot, live twin untouched."""

    def counterfactual(self, change):
        saved_state, saved_log = copy.deepcopy(self.state), copy.deepcopy(self.action_log)
        after = fake_kpis(self.apply(change))
        entry = self.action_log[-1]
        self.state, self.action_log = copy.deepcopy(saved_state), copy.deepcopy(saved_log)
        base = fake_kpis(self.apply([]))
        self.state, self.action_log = saved_state, saved_log
        keys = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
        return {"baseline": {k: base[k] for k in keys}, "after": {k: after[k] for k in keys},
                "delta": {k: after[k] - base[k] for k in keys}, "clamped": entry["clamped"],
                "rules": entry["rules"], "filtered": entry["filtered"]}


def fake_kpis(state) -> dict:
    p, t, step = state["power"], state["tilt"], state["step"]
    cov = min(1.0, 0.5 + sum(p) / 400.0 - sum(abs(x - 8.0) for x in t) / 200.0 + 0.001 * step)
    return {"coverage_ratio": cov, "sla_ratio": cov * 0.9, "sinr_p10_db": sum(t) / 4.0 - 3.0 + 0.01 * step,
            "sinr_mean_db": sum(t) / 4.0 + 5.0, "rsrp_p10_dbm": -100.0 + sum(p) / 40.0,
            "per_cell": [{"cell": i, "power_dbm": p[i], "tilt_deg": t[i], "ue_count": 10 + i} for i in range(N_CELLS)]}


def true_delta(state, change) -> dict:
    """The exact counterfactual delta of the toy twin (what a perfect predictor would say)."""
    base = copy.deepcopy(state)
    base["step"] += 1
    after = copy.deepcopy(base)
    for c in change:
        after["power"][c["cell"]] += max(-3.0, min(3.0, c["d_power_db"]))
        after["tilt"][c["cell"]] += max(-3.0, min(3.0, c["d_tilt_deg"]))
    b, a = fake_kpis(base), fake_kpis(after)
    return {k: a[k] - b[k] for k in ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")}


class FakePredictor:
    """Predicts the true delta plus a bias; flip=True predicts the opposite direction."""

    model_id = "t2-response"

    def __init__(self, bias: float = 0.0, flip: bool = False, version: str = "v0-test"):
        self.bias, self.flip, self.version, self.calls = bias, flip, version, []

    def predict(self, state, change):
        self.calls.append((copy.deepcopy(state), change))
        d = true_delta(state, change)
        return {k: (-v if self.flip else v) + self.bias for k, v in d.items()}


class FakePolicy:
    model_id, version = "ppo-maveric", "v0-test"

    def propose(self, state):
        return [{"cell": 2, "d_power_db": 0.04, "d_tilt_deg": 2.0}]   # rounded to one decimal by the tool


class FakeTrustInvoker:
    """The trust gate answering synchronously, like W5's handler."""

    def __init__(self, mode: str = "ok"):
        self.mode, self.payloads, self.trust = mode, [], 1.0

    @property
    def records(self):
        return [p["record"] for p in self.payloads]

    def __call__(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        if self.mode == "timeout":
            from botocore.exceptions import ReadTimeoutError
            raise ReadTimeoutError(endpoint_url="https://lambda.us-east-2.amazonaws.com/")
        if self.mode == "error":
            raise RuntimeError("team13-trust-gate Unhandled: KeyError('scales')")
        before = self.trust
        rec = payload["record"]
        self.trust = before * (1.0 if rec["direction_agree"] else 0.5)
        return {"trust_before": before, "trust_after": self.trust, "e": rec["error"]["summary"],
                "score": 1.0 if rec["direction_agree"] else 0.0,
                "action": None if rec["direction_agree"] else "pause",
                **({} if rec["direction_agree"] else {"control_record_key": "ctl-0001"})}


class MemoryWriter:
    def __init__(self):
        self.records = {}

    def put_record(self, run_id, kind, record_key, record):
        path = f"records/{run_id}/{kind}/{record_key}.json"
        assert path not in self.records
        self.records[path] = json.loads(json.dumps(record, allow_nan=False))
        return f"memory://{path}"


class FakeSageMaker:
    def __init__(self, response):
        self.response, self.calls = response, []

    def invoke_endpoint(self, **kw):
        self.calls.append(kw)
        return {"Body": io.BytesIO(json.dumps(self.response).encode())}


class FakeBedrock:
    def __init__(self, text):
        self.text, self.calls = text, []

    def converse(self, **kw):
        self.calls.append(kw)
        return {"output": {"message": {"role": "assistant", "content": [{"text": self.text}]}}}


def context(tool_name: str):
    """A Lambda context as the Gateway sends it."""
    return types.SimpleNamespace(client_context=types.SimpleNamespace(custom={"bedrockAgentCoreToolName": tool_name}))
