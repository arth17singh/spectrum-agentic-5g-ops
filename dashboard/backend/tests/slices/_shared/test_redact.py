from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices._shared.redact import RedactMiddleware, redact

H = "e04d8c907ca30fd3c876121b924c2836157cb546b1890a984184bb52f37484c1"


def test_redact_strings_only():
    d = {"uri": "s3://bb26-p-team13-data-123456789012-use2/evidence/x.json",
         "arn": "arn:aws:iam::123456789012:role/r", "sha": H, "n": 123456789012, "t": "20261103T043115Z"}
    out = redact(d)
    assert out["uri"] == "s3://bb26-p-team13-data-<acct>-use2/evidence/x.json"
    assert out["arn"] == "iam-role:r"
    assert out["sha"] == H and out["n"] == 123456789012 and out["t"] == d["t"]


def test_middleware_rewrites_slice_json_only():
    app = FastAPI()
    app.add_middleware(RedactMiddleware)

    @app.get("/api/scorecard/x")
    def a():
        return {"arn": "arn:aws:sagemaker:us-east-2:123456789012:training-job/j"}

    @app.get("/api/other")
    def b():
        return {"arn": "123456789012"}

    c = TestClient(app)
    assert c.get("/api/scorecard/x").json() == {"arn": "arn:aws:sagemaker:us-east-2:<acct>:training-job/j"}
    assert c.get("/api/other").json() == {"arn": "123456789012"}
