from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices._shared.redact import PERSON, RedactMiddleware, redact

H = "e04d8c907ca30fd3c876121b924c2836157cb546b1890a984184bb52f37484c1"
A = "123456789012"


def test_redact_strings_only():
    d = {"uri": f"s3://bb26-p-team13-data-{A}-use2/evidence/x.json",
         "arn": f"arn:aws:iam::{A}:role/bb26/team13-x", "sha": H, "n": 123456789012, "t": "20261103T043115Z",
         "job": f"arn:aws:sagemaker:us-east-2:{A}:training-job/t1-x-1",
         "who": f"arn:aws:sts::{A}:assumed-role/AWSReservedSSO_P_0123/jane.doe@example.com",
         "mail": "ops jane.doe@example.com",
         "img": f"{A}.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.9.0",
         "RAN_RL_TWIN_URL": f"arn:aws:secretsmanager:us-east-2:{A}:secret:AmazonSageMaker-team13-twin-url-AbCdEf",
         "bare": "AmazonSageMaker-team13-twin-token-AbCdEf",
         "lambda": f"arn:aws:lambda:us-east-2:{A}:function:team13-ran-tools"}
    out = redact(d)
    assert out["uri"] == "s3://bb26-p-team13-data-<acct>-use2/evidence/x.json"
    assert out["arn"] == "<arn:iam>/role/team13-x"
    assert out["job"] == "<arn:sagemaker>/t1-x-1"
    assert out["who"] == PERSON and out["mail"] == f"ops {PERSON}"
    assert out["img"] == "<ecr>/pytorch-training:2.9.0"
    assert out["RAN_RL_TWIN_URL"] == "<secret>" and out["bare"] == "<secret>"
    assert out["lambda"] == "<arn:lambda>"
    assert out["sha"] == H and out["n"] == 123456789012 and out["t"] == d["t"]


def test_signed_ledger_records_pass_through():
    rec = {"seq": 1, "hash": H, "sig": "ab", "payload": {"note": f"arn:aws:iam::{A}:role/r"}}
    assert redact([rec]) == [rec]


def test_middleware_rewrites_api_json_but_not_the_chain():
    app = FastAPI()
    app.add_middleware(RedactMiddleware)

    @app.get("/api/scorecard/x")
    def a():
        return {"arn": f"arn:aws:sagemaker:us-east-2:{A}:training-job/j"}

    @app.get("/api/evidence/chain")
    def b():
        return {"records": [{"x": A}]}

    c = TestClient(app)
    assert c.get("/api/scorecard/x").json() == {"arn": "<arn:sagemaker>/j"}
    assert c.get("/api/evidence/chain").json() == {"records": [{"x": A}]}
