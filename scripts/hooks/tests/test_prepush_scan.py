"""Tests for the pre-push guard. Every sensitive-looking value here is FAKE and is assembled at run time,
so this file contains no literal that the guard itself would flag.

Run:  python3 -m pytest -q scripts/hooks/tests
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOKS))
import prepush_scan as ps  # noqa: E402

SCAN = HOOKS / "prepush_scan.py"
HOOK = HOOKS / "pre-push"

# --- fake values, assembled so no source literal matches a rule --------------------------------------------
FAKE_AKIA = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
FAKE_ASIA = "AS" + "IA" + "ABCDEFGHIJKLMNOP"
FAKE_PRIVATE = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIfake\n-----END " + "RSA PRIVATE KEY-----\n"
FAKE_JWT = "ey" + "Jhbgfake1." + "ey" + "Jzdwfake2." + "sigfake3abc"
FAKE_BEARER = "Bear" + "er " + "abcdefghijklmnopqrstuvwxyz012345"
FAKE_IP = ".".join(["203", "0", "114", "7"])  # not in any allowed block
FAKE_ACCOUNT = "9" * 4 + "8" * 4 + "7" * 4
FAKE_COGNITO = "https://x." + "auth.us-east-2." + "amazon" + "cognito.com/login"
FAKE_SIG = "X-Amz-" + "Signature=abc"
FAKE_EC2 = "ec2-1-2-3-4.us-east-2.compute." + "amazon" + "aws.com"
KIT_SECRET = "kitsecret-" + "Zq8WmN3rT5yV1xB7cD9eF2gH"


def sh(*args, cwd, env=None, check=True):
    e = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
         "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid", **(env or {})}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=check, env=e)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    sh("git", "init", "-q", "-b", "main", cwd=r)
    return r


def commit(repo: Path, files: dict[str, str | bytes], msg="c"):
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    sh("git", "add", "-A", "-f", cwd=repo)
    sh("git", "commit", "-q", "-m", msg, cwd=repo)
    return sh("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()


def scan(repo: Path, *extra, kit=None, env=None):
    cmd = [sys.executable, "-I", str(SCAN), "--repo", str(repo), "--kit-dir", str(kit or repo / "nokit"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})


# --- the pattern rules ----------------------------------------------------------------------------------
@pytest.mark.parametrize("name,content,expect", [
    ("aws key", f"key = '{FAKE_AKIA}'\n", "AWS access key id"),
    ("aws session key", f"key = '{FAKE_ASIA}'\n", "AWS access key id"),
    ("private key", FAKE_PRIVATE, "private key block"),
    ("jwt", f"t = '{FAKE_JWT}'\n", "JWT"),
    ("bearer", f"Authorization: {FAKE_BEARER}\n", "Bearer token"),
    ("ip", f"host = {FAKE_IP}\n", "IPv4 address"),
    ("account", f"arn:aws:iam::{FAKE_ACCOUNT}:role/x\n", "12-digit account id"),
    ("cognito", f"url = {FAKE_COGNITO}\n", "amazoncognito"),
    ("signature", f"u?{FAKE_SIG}\n", "X-Amz-Signature"),
    ("ec2 host", f"ssh {FAKE_EC2}\n", "EC2 public host name"),
])
def test_content_rule_blocks_and_never_prints_the_match(repo, name, content, expect):
    commit(repo, {"ok.txt": "fine\n"})
    commit(repo, {"src/leak.txt": content})
    res = scan(repo, "--refs", "HEAD", "--no-evidence")
    assert res.returncode == 1, res.stderr
    assert expect in res.stderr and "src/leak.txt" in res.stderr
    for secret in (FAKE_AKIA, FAKE_ASIA, FAKE_JWT, FAKE_IP, FAKE_ACCOUNT, "abcdefghijklmnopqrstuvwxyz012345", "MIIfake"):
        assert secret not in res.stderr + res.stdout


@pytest.mark.parametrize("value", [
    "127.0.0.1", "0.0.0.0", "10.0.0.5", "10.0.0.200", "192.0.2.1", "198.51.100.9", "203.0.113.4", "255.255.255.255",
])
def test_allowlisted_ip_values_pass(repo, value):
    commit(repo, {"fixtures/net.json": json.dumps({"ip": value}) + "\n"})
    assert scan(repo, "--refs", "HEAD", "--no-evidence").returncode == 0


def test_version_strings_and_hashes_are_not_ips_or_accounts(repo):
    commit(repo, {"x.txt": "version 1.2.3.4.5 and 0.123456789012345 and sha 00112233445566778899aabbccddeeff\n"
                           "run 20261102T234623Z id 1762129583123\n"})
    assert scan(repo, "--refs", "HEAD", "--no-evidence").returncode == 0


def test_documented_example_account_ids_pass(repo):
    commit(repo, {"x.txt": "arn:aws:iam::123456789012:role/x and 000000000000\n"})
    assert scan(repo, "--refs", "HEAD", "--no-evidence").returncode == 0


def test_binary_files_skip_text_only_rules(repo):
    blob = b"\0\x01\x02" + FAKE_IP.encode() + b"\0" + FAKE_ACCOUNT.encode() + b"\0"
    commit(repo, {"data.bin": blob})
    assert scan(repo, "--refs", "HEAD", "--no-evidence").returncode == 0


def test_commit_message_is_scanned(repo):
    commit(repo, {"a.txt": "x\n"}, msg=f"oops {FAKE_AKIA}")
    res = scan(repo, "--refs", "HEAD", "--no-evidence")
    assert res.returncode == 1 and "in commit message" in res.stderr and FAKE_AKIA not in res.stderr


# --- path rules, size -------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [
    "checkpoint_kit.py", "kit/checkpoint_kit.py", "pkg/ran_rl-1.2.9-py3-none-any.whl", "ran-rl-starter/README.md",
    "ran-rl-starter-1.2.9.zip", "twin-host/app.py", "hackathon_docs/guide.md", "tokens/noc_lead.txt", "kit.json",
    "kit-state.json", "a2a_token.txt", "builder_admin_password.txt", ".env", ".env.local", "deploy/host.env",
    "keys/x.pem", ".evidence-key", "dashboard/backend/.evidence-key",
])
def test_forbidden_path_blocks(repo, path):
    commit(repo, {path: "harmless\n"})
    res = scan(repo, "--refs", "HEAD", "--no-evidence")
    assert res.returncode == 1 and "BLOCKED" in res.stderr and path in res.stderr


@pytest.mark.parametrize("path", [".env.example", "t3/agents/kit.json.template", "docs/tokens.html", "t3/tokens_notes.md"])
def test_lookalike_paths_pass(repo, path):
    commit(repo, {path: "harmless\n"})
    assert scan(repo, "--refs", "HEAD", "--no-evidence").returncode == 0


def test_file_over_5mb_blocks_and_under_passes(repo):
    commit(repo, {"big.dat": b"a" * (5 * 1024 * 1024 + 1)})
    res = scan(repo, "--refs", "HEAD", "--no-evidence")
    assert res.returncode == 1 and "file over 5 MB" in res.stderr
    repo2 = repo.parent / "repo2"
    repo2.mkdir()
    sh("git", "init", "-q", cwd=repo2)
    commit(repo2, {"ok.dat": b"a" * (5 * 1024 * 1024 - 10)})
    assert scan(repo2, "--refs", "HEAD", "--no-evidence").returncode == 0


# --- kit secrets ------------------------------------------------------------------------------------------
def make_kit(tmp_path: Path) -> Path:
    kit = tmp_path / "kit"
    (kit / "tokens").mkdir(parents=True)
    (kit / "tokens" / "agent_a.txt").write_text(KIT_SECRET + "-token\n")
    (kit / "a2a_token.txt").write_text(KIT_SECRET + "-a2a\n")
    (kit / "builder_admin_password.txt").write_text(KIT_SECRET + "-pw\n")
    (kit / "kit-state.json").write_text(json.dumps({"gateway_url": "https://gw." + KIT_SECRET + ".example.invalid/mcp",
                                                    "bedrock_log_group": "/team13/bedrock-invocations"}))
    return kit


@pytest.mark.parametrize("suffix", ["-token", "-a2a", "-pw"])
def test_kit_secret_blocks_by_label_not_value(repo, tmp_path, suffix):
    kit = make_kit(tmp_path)
    commit(repo, {"notes.txt": f"remember {KIT_SECRET}{suffix}\n"})
    res = scan(repo, "--refs", "HEAD", "--no-evidence", kit=kit)
    assert res.returncode == 1 and "kit secret" in res.stderr and "notes.txt" in res.stderr
    assert KIT_SECRET not in res.stderr + res.stdout


def test_kit_state_value_blocks_but_public_key_does_not(repo, tmp_path):
    kit = make_kit(tmp_path)
    commit(repo, {"a.txt": "log group /team13/bedrock-invocations is fine\n"})
    assert scan(repo, "--refs", "HEAD", "--no-evidence", kit=kit).returncode == 0
    commit(repo, {"b.txt": "https://gw." + KIT_SECRET + ".example.invalid/mcp\n"})
    res = scan(repo, "--refs", "HEAD", "--no-evidence", kit=kit)
    assert res.returncode == 1 and "kit-state.json:gateway_url" in res.stderr and KIT_SECRET not in res.stderr


def test_missing_kit_warns_and_can_be_made_fatal(repo, tmp_path):
    commit(repo, {"a.txt": "x\n"})
    res = scan(repo, "--refs", "HEAD", "--no-evidence", kit=tmp_path / "absent")
    assert res.returncode == 0 and "WARNING kit folder not found" in res.stderr
    res = scan(repo, "--refs", "HEAD", "--no-evidence", kit=tmp_path / "absent", env={"PREPUSH_REQUIRE_KIT": "1"})
    assert res.returncode == 1


# --- evidence/ ---------------------------------------------------------------------------------------------
def test_evidence_folder_is_scanned_even_if_untracked(repo, tmp_path):
    kit = make_kit(tmp_path)
    commit(repo, {"a.txt": "x\n"})
    ev = repo / "evidence" / "records" / "run1"
    ev.mkdir(parents=True)
    (ev / "item.jsonl").write_text(json.dumps({"host": FAKE_IP, "note": KIT_SECRET + "-token"}) + "\n")
    res = scan(repo, "--evidence-only", kit=kit)
    assert res.returncode == 1
    assert "evidence/records/run1/item.jsonl" in res.stderr
    assert "IPv4 address" in res.stderr and "kit secret" in res.stderr
    assert FAKE_IP not in res.stderr and KIT_SECRET not in res.stderr


def test_clean_evidence_passes(repo):
    commit(repo, {"a.txt": "x\n"})
    ev = repo / "evidence" / "declared"
    ev.mkdir(parents=True)
    (ev / "run1.json").write_text(json.dumps({"run_id": "run1", "twin": {"host": "10.0.0.7"}}))
    assert scan(repo, "--evidence-only").returncode == 0


# --- the push range ---------------------------------------------------------------------------------------
def test_range_scans_only_new_commits_but_includes_a_secret_added_then_removed(repo):
    commit(repo, {"old.txt": f"{FAKE_AKIA}\n"}, msg="old leak, already pushed")
    base = sh("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    commit(repo, {"new.txt": "fine\n"})
    assert scan(repo, "--range", f"{base}..HEAD", "--no-evidence").returncode == 0
    assert scan(repo, "--refs", "HEAD", "--no-evidence").returncode == 1
    # a leak committed and deleted inside the pushed range is still pushed, so it still blocks
    commit(repo, {"tmp.txt": f"scratch {FAKE_AKIA}\n"})  # different bytes: an identical blob is already on the remote
    sh("git", "rm", "-q", "tmp.txt", cwd=repo)
    sh("git", "commit", "-q", "-m", "remove it", cwd=repo)
    res = scan(repo, "--range", f"{base}..HEAD", "--no-evidence")
    assert res.returncode == 1 and "tmp.txt" in res.stderr


# --- the real hook, through `git push` ---------------------------------------------------------------------
@pytest.fixture
def pushable(tmp_path):
    bare = tmp_path / "remote.git"
    sh("git", "init", "-q", "--bare", str(bare), cwd=tmp_path)
    work = tmp_path / "work"
    sh("git", "clone", "-q", str(bare), str(work), cwd=tmp_path)
    sh("git", "checkout", "-q", "-b", "main", cwd=work)
    sh("git", "config", "core.hooksPath", str(HOOKS), cwd=work)
    assert os.access(HOOK, os.X_OK), "scripts/hooks/pre-push must be executable"
    return work, bare


def push(work: Path, *refspec, env=None):
    return sh("git", "push", "origin", *refspec, cwd=work, check=False, env={"TEAM13_KIT_DIR": str(work / "nokit"), **(env or {})})


def test_hook_blocks_a_bad_push_and_allows_a_clean_one(pushable, tmp_path):
    work, bare = pushable
    commit(work, {"README.md": "hello\n"})
    ok = push(work, "main")
    assert ok.returncode == 0, ok.stderr
    assert "pre-push scan: clean" in ok.stderr

    base = sh("git", "rev-parse", "HEAD", cwd=work).stdout.strip()
    commit(work, {"src/leak.py": f"KEY = '{FAKE_AKIA}'\n"})
    bad = push(work, "main")
    assert bad.returncode != 0 and "BLOCKED" in bad.stderr and "src/leak.py" in bad.stderr
    assert FAKE_AKIA not in bad.stderr + bad.stdout
    # nothing reached the remote
    assert sh("git", "rev-parse", "main", cwd=bare).stdout.strip() == base

    sh("git", "reset", "-q", "--hard", base, cwd=work)
    commit(work, {"src/fine.py": "x = 1\n"})
    assert push(work, "main").returncode == 0


def test_hook_scans_a_new_branch_against_what_the_remote_already_has(pushable):
    work, _ = pushable
    commit(work, {"README.md": "hello\n"})
    assert push(work, "main").returncode == 0
    sh("git", "checkout", "-q", "-b", "feature", cwd=work)
    commit(work, {"f.txt": "fine\n"})
    assert push(work, "feature").returncode == 0
    sh("git", "checkout", "-q", "-b", "feature2", cwd=work)
    commit(work, {"tokens/x.txt": "harmless\n"})
    assert push(work, "feature2").returncode != 0


def test_hook_ignores_a_branch_deletion(pushable):
    work, _ = pushable
    commit(work, {"README.md": "hello\n"})
    assert push(work, "main").returncode == 0
    sh("git", "branch", "gone", cwd=work)
    assert push(work, "gone").returncode == 0
    assert push(work, "--delete", "gone").returncode == 0


def test_scan_fails_closed_when_it_cannot_run(tmp_path):
    res = subprocess.run([sys.executable, "-I", str(SCAN), "--repo", str(tmp_path / "not-a-repo"), "--refs", "HEAD"],
                         capture_output=True, text=True)
    assert res.returncode == 1 and "blocking the push" in res.stderr
