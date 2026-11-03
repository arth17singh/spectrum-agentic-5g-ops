import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TOOLS), str(TOOLS / "tests")]


@pytest.fixture(autouse=True)
def _no_aws_env(monkeypatch):
    """Tests never reach AWS: no evidence targets or profiles leak in from the shell, and boto3 (if anything
    did create a client) would get credentials that cannot work."""
    for k in ("EVIDENCE_S3", "EVIDENCE_DIR", "EVIDENCE_LOG_GROUPS", "EVIDENCE_CASES_TABLE", "AWS_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing-not-a-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing-not-a-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    yield
