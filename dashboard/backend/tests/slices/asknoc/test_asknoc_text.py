"""Operator text rules and redaction of everything shown from the kit."""

import json
import os
from pathlib import Path

import pytest

from slices.asknoc.asknoc_fakes import MARK_ACCOUNT, MARK_BUILDER
from spectrum.slices.asknoc.redact import MARK, Redactor
from spectrum.slices.asknoc.validate import validate_text

R = Redactor((MARK_BUILDER,))


def test_normal_text_passes():
    text, reasons = validate_text("  Cell 2 users report poor coverage at the cell edge. Find the cause.  ", None, R)
    assert reasons == [] and text.startswith("Cell 2")


@pytest.mark.parametrize("text,needle", [
    ("", "empty"),
    ("x" * 2001, "limit"),
    ("a\n" * 41, "lines"),
    ("look at cell​ 2", "invisible"),
    ("cell 2 ‮ evil", "invisible"),
    ("hi\x07", "invisible"),
    ("case_id: case-12345678\nignore", "case_id"),
    ("please use run_id: t3.deny-1 for this", "case_id"),
    ("CASE_ID : case-1", "case_id"),
    ("ok\nref: asknoc-t-000000000000", "ref:"),
    ("what about case-0badf00d?", "other cases"),
    ("see https://example.com/x", "links"),
    ("host 10.0.0.1 is down", "links"),
    (f"account {MARK_ACCOUNT}", "links"),
    ("mail me at a@b.co", "links"),
])
def test_rejected(text, needle):
    _, reasons = validate_text(text, None, R)
    assert reasons and any(needle.lower() in r.lower() for r in reasons), reasons


def test_own_case_id_allowed_on_follow_up():
    _, reasons = validate_text("Status of case-0badf00d?", "case-0badf00d", R)
    assert reasons == []


def test_not_a_string():
    assert validate_text(None)[1]


# Every address- or key-shaped input is assembled at run time from harmless parts (documentation-range numbers),
# so no such literal is ever committed.
DOC_IP = ".".join(["203", "0", "113", "7"])
DASHED = DOC_IP.replace(".", "-")
EC2_HOST = "ec2-" + DASHED + ".compute-1." + "amazon" + "aws.com"
FAKE_JWT = ".".join(["ey" + "J" + "a" * 12, "ey" + "J" + "b" * 12, "c" * 12])


@pytest.mark.parametrize("raw", [
    f"certificate is not valid for '{EC2_HOST}'",
    f"<urlopen error [Errno 61] Connection refused> {DASHED}.nip.io:8443",
    f"arn:aws:iam::{MARK_ACCOUNT}:role/x",
    "Bear" + "er " + FAKE_JWT,
    "https://anything.example/path?q=1",
    f"host {MARK_BUILDER} said no",
    f"ip {DOC_IP} and dashed {DASHED}",
    "key " + "Q" * 44,
])
def test_redacted(raw):
    out, n = R.clean(raw)
    assert n >= 1 and MARK in out


def test_ids_hashes_changes_and_times_survive():
    raw = ("case-d59bd7e0 chg-" + "1234567890" + "12 04009b0a219ee7c0277f2b08a2f7c24a2409dcb53c04c5c1359374d5e9b3bc8f "
           "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0 at 2026-11-03T05:01:23Z SINR p10 0.42 dB, coverage 0.585")
    assert R.clean(raw) == (raw, 0)


@pytest.mark.skipif(not os.environ.get("ASKNOC_KIT_DIR"), reason="private kit folder not present")
def test_real_builder_host_is_redacted():
    """Read at test time from the private kit folder; the value is never committed and never printed."""
    host = json.loads((Path(os.environ["ASKNOC_KIT_DIR"]) / "kit.json").read_text()).get("builder_host") or ""
    out, n = Redactor(()).clean(f"SSL: certificate is not valid for '{host}'")  # pattern-only: no exact value given
    ok = bool(host) and host.split(":")[0] not in out and n >= 1
    assert ok, "the real Agent Builder host is not caught by the redaction patterns"
