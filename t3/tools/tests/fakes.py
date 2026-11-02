"""Fakes for the RAN tool tests: DynamoDB tables, S3, the twin, the predictor, the policy, the
trust gate and the AWS model clients. No network, no boto3 sessions."""
from __future__ import annotations

import copy
import io
import json
import types

from botocore.exceptions import ClientError

N_CELLS = 4


def conditional_failed(op: str) -> ClientError:
    return ClientError({"Error": {"Code": "ConditionalCheckFailedException", "Message": "The conditional request failed"}}, op)


class FakeTable:
    """Just the DynamoDB Table calls and condition expressions the tools use."""

    def __init__(self, key: str):
        self.key, self.items, self.calls = key, {}, []

    def get_item(self, Key, ConsistentRead=False):
        self.calls.append(("get_item", Key))
        item = self.items.get(Key[self.key])
        return {"Item": copy.deepcopy(item)} if item else {}

    def put_item(self, Item, ConditionExpression=None, ExpressionAttributeValues=None):
        self.calls.append(("put_item", Item, ConditionExpression))
        cur = self.items.get(Item[self.key])
        vals = ExpressionAttributeValues or {}
        if ConditionExpression == "attribute_not_exists(case_id)":
            ok = cur is None
        elif ConditionExpression == "attribute_not_exists(version) OR version = :v":
            ok = cur is not None and (cur.get("version") is None or cur.get("version") == vals[":v"])
        elif ConditionExpression == "version = :v":
            ok = cur is not None and cur.get("version") == vals[":v"]
        elif ConditionExpression is None:
            ok = True
        else:
            raise AssertionError(f"unexpected condition {ConditionExpression}")
        if not ok:
            raise conditional_failed("PutItem")
        self.items[Item[self.key]] = copy.deepcopy(Item)

    def update_item(self, Key, UpdateExpression, ConditionExpression, ExpressionAttributeNames,
                    ExpressionAttributeValues):
        self.calls.append(("update_item", Key, UpdateExpression, ConditionExpression))
        assert UpdateExpression == "SET #s = :applied, applied_at = :t"
        assert ConditionExpression == "#s = :approved" and ExpressionAttributeNames == {"#s": "status"}
        cur = self.items.get(Key[self.key])
        if not cur or cur.get("status") != ExpressionAttributeValues[":approved"]:
            raise conditional_failed("UpdateItem")
        cur.update(status=ExpressionAttributeValues[":applied"], applied_at=ExpressionAttributeValues[":t"])


class FakeS3:
    def __init__(self):
        self.objects, self.puts = {}, []

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        body, etag = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ETag": etag}

    def put_object(self, Bucket, Key, Body, ContentType, IfMatch=None, IfNoneMatch=None):
        self.puts.append((Key, IfMatch, IfNoneMatch))
        cur = self.objects.get((Bucket, Key))
        if (IfNoneMatch == "*" and cur) or (IfMatch and (not cur or cur[1] != IfMatch)):
            raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "etag"}}, "PutObject")
        self.objects[(Bucket, Key)] = (Body, f'"{len(self.puts)}"')


class FakeTwin:
    """Deterministic toy twin with the W2 ReplayTwin surface: reset() and apply(change) -> state."""

    instances = 0

    def __init__(self, scenario, seed, config):
        FakeTwin.instances += 1
        self.scenario, self.seed, self.config = scenario, seed, config
        self.action_log = []

    def reset(self):
        self.state = {"step": 0, "power": [40.0 + (self.seed % 3)] * N_CELLS, "tilt": [6.0] * N_CELLS}
        self.action_log = []
        return copy.deepcopy(self.state)

    def apply(self, change):
        for c in change:
            self.state["power"][c["cell"]] += c["d_power_db"]
            self.state["tilt"][c["cell"]] += c["d_tilt_deg"]
        self.state["step"] += 1
        self.action_log.append(change)
        return copy.deepcopy(self.state)


def fake_kpis(state) -> dict:
    p, t = state["power"], state["tilt"]
    cov = min(1.0, 0.5 + sum(p) / 400.0 - sum(abs(x - 8.0) for x in t) / 200.0)
    return {"coverage_ratio": cov, "sla_ratio": cov * 0.9, "sinr_p10_db": sum(t) / 4.0 - 3.0,
            "sinr_mean_db": sum(t) / 4.0 + 5.0, "rsrp_p10_dbm": -100.0 + sum(p) / 40.0,
            "per_cell": [{"cell": i, "power_dbm": p[i], "tilt_deg": t[i]} for i in range(N_CELLS)]}


class FakePredictor:
    """Predicts with the true toy model plus a bias; flip=True predicts the wrong direction."""

    model_id, version = "t2-response", "v0-test"

    def __init__(self, bias: float = 0.0, flip: bool = False):
        self.bias, self.flip, self.calls = bias, flip, []

    def predict(self, state, change):
        self.calls.append((copy.deepcopy(state), change))
        before = fake_kpis(state)
        s = copy.deepcopy(state)
        for c in change:
            s["power"][c["cell"]] += c["d_power_db"]
            s["tilt"][c["cell"]] += c["d_tilt_deg"]
        after = fake_kpis(s)
        out = {}
        for k in ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db"):
            d = after[k] - before[k]
            out[k] = before[k] + (-d if self.flip else d) + self.bias
        return out


class FakePolicy:
    model_id, version = "ppo-maveric", "v0-test"

    def propose(self, state):
        return [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 2.0}]


class FakeTrust:
    def __init__(self, pause_on_disagree: bool = True):
        self.updates, self.trust = [], 1.0
        self.pause_on_disagree = pause_on_disagree

    def update(self, record):
        self.updates.append(copy.deepcopy(record))
        before = self.trust
        self.trust = self.trust * (1.0 if record["direction_agree"] else 0.5)
        action = "pause" if (self.pause_on_disagree and not record["direction_agree"]) else None
        return {"trust_before": before, "trust_after": self.trust, "action": action}


class ListWriter:
    def __init__(self):
        self.records = []

    def append_records(self, run_id, kind, records):
        self.records += [(run_id, kind, copy.deepcopy(r)) for r in records]
        return f"memory://{run_id}/{kind}"


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
