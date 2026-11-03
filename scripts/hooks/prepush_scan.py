#!/usr/bin/env python3
"""Pre-push safety scan for team13's repository.

Scans every object a push would add (commits, with their messages, and the files in them) plus the
working-tree `evidence/` folder. It prints the rule and the file for each finding, and NEVER the matched
text: kit secrets are loaded from the private kit folder into memory and compared there.

Exit codes: 0 clean, 1 findings (or the scan could not complete: it fails closed), 2 bad arguments.

How it is used
  as the git hook (scripts/hooks/pre-push calls it):   --hook <remote> <url>        (ref list on stdin)
  audit a range:                                       --range origin/main..HEAD
  audit a whole history:                               --refs int/v4 [other refs]
  audit evidence/ only:                                --evidence-only

The kit folder defaults to ~/firstmate-homes/team13-kit; set TEAM13_KIT_DIR or pass --kit-dir to move it.
If it is missing, the secret-value check is skipped with a warning (the pattern rules still run); set
PREPUSH_REQUIRE_KIT=1 to make a missing kit a failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024  # a file over this is refused
MAX_SCAN_BYTES = 40 * 1024 * 1024  # content of anything larger is not read into memory
MIN_SECRET_LEN = 8
MIN_STATE_VALUE_LEN = 12
ZERO_SHA = "0" * 40

# --- allow-lists: every exemption is written here, explicitly -------------------------------------------
# IPv4 values that are obviously not ours: loopback, "any", the broadcast mask, 10.0.0.x test addresses,
# the three RFC 5737 documentation ranges, and the link-local instance metadata address.
ALLOWED_IPV4 = [
    re.compile(r"^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$"),  # the loopback block, 127.0.0.1 included
    re.compile(r"^0\.0\.0\.0$"),
    re.compile(r"^255\.255\.255\.(0|255)$"),
    re.compile(r"^10\.0\.0\.\d{1,3}$"),
    re.compile(r"^192\.0\.2\.\d{1,3}$"),
    re.compile(r"^198\.51\.100\.\d{1,3}$"),
    re.compile(r"^203\.0\.113\.\d{1,3}$"),
    re.compile(r"^169\.254\.169\.254$"),
    re.compile(r"^10\.255\.255\.255$"),  # dashboard/backend main.py: the unrouted UDP probe that finds the LAN address
    re.compile(r"^192\.168\.1\.20$"),  # dashboard/backend tests: the fake phone address
]
# A rule may be waived for one file when the file is known to contain look-alikes. Each entry is (rule, path regex).
PATH_WAIVERS = [
    # GSMA ot-lite text: 3GPP clause numbers with four dotted parts parse as four octets.
    ("IPv4 address", re.compile(r"^t1/evals/items/knowledge\.jsonl$")),
    # The guard's own source, README and tests have to name the patterns they look for.
    ("amazoncognito", re.compile(r"^scripts/hooks/(README\.md|prepush_scan\.py|tests/test_prepush_scan\.py)$")),
    ("presigned URL signature (X-Amz-Signature)", re.compile(r"^scripts/hooks/(README\.md|prepush_scan\.py|tests/test_prepush_scan\.py)$")),
    ("EC2 public host name", re.compile(r"^scripts/hooks/(README\.md|prepush_scan\.py|tests/test_prepush_scan\.py)$")),
]
# 12-digit values that are the documented AWS example ids.
ALLOWED_ACCOUNT_IDS = {"123456789012", "000000000000", "111111111111", "111122223333"}
# kit.json / kit-state.json keys whose values are public names, already written in our docs and code.
PUBLIC_KIT_KEYS = {"bedrock_log_group", "specialists", "lead", "may_change", "name", "region"}

# --- path rules (applied to every path a pushed object has) ---------------------------------------------
PATH_RULES = [
    ("organiser docs folder", re.compile(r"(^|/)hackathon_docs/")),
    ("organiser kit file (checkpoint_kit.py)", re.compile(r"(^|/)checkpoint_kit\.py$")),
    ("python wheel", re.compile(r"\.whl$")),
    ("organiser starter or bundle (ran-rl-starter*)", re.compile(r"(^|/)ran-rl-(starter|staging)")),
    ("twin-host source", re.compile(r"(^|/)twin[-_]?host", re.I)),
    ("evidence signing key file", re.compile(r"(^|/)\.evidence-key$")),
    ("kit tokens or kit state", re.compile(r"(^|/)(tokens/|kit\.json$|kit-state\.json$|a2a_token|builder_admin_password)")),
    ("env file", re.compile(r"(^|/)\.env(\.(?!example$)[^/]*)?$")),
    ("host.env file", re.compile(r"(^|/)host\.env$")),
    ("key or certificate file", re.compile(r"\.(pem|p12|pfx)$|(^|/)id_(rsa|ed25519|ecdsa)$")),
]

# --- content rules (bytes) -------------------------------------------------------------------------------
# kind "any": checked in every file; kind "text": skipped in binary files, where random bytes could match.
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
CONTENT_RULES = [
    ("AWS access key id", "any", re.compile(rb"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])")),
    ("AWS secret key assignment", "text", re.compile(rb"aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{30,}", re.I)),
    ("private key block", "any", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")),
    ("JWT", "any", re.compile(rb"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("Bearer token", "text", re.compile(rb"Bearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("IPv4 address", "text", re.compile(rb"(?<![\d.])" + _OCTET.encode() + rb"(?:\." + _OCTET.encode() + rb"){3}(?!\d|\.\d)")),
    ("12-digit account id", "text", re.compile(rb"(?<![0-9A-Fa-f.])\d{12}(?![0-9A-Fa-f]|\.\d)")),
    ("amazoncognito", "any", re.compile(rb"amazoncognito", re.I)),
    ("presigned URL signature (X-Amz-Signature)", "any", re.compile(rb"X-Amz-Signature", re.I)),
    ("EC2 public host name", "any", re.compile(rb"\bec2-[0-9a-z.-]+\.amazonaws\.com", re.I)),
]


class Finding:
    __slots__ = ("rule", "where", "line")

    def __init__(self, rule: str, where: str, line: int | None = None):
        self.rule, self.where, self.line = rule, where, line

    def key(self):
        return (self.rule, self.where, self.line or 0)

    def __str__(self):
        loc = f"{self.where}:{self.line}" if self.line else self.where
        return f"{self.rule:46} {loc}"


# --- kit secrets ------------------------------------------------------------------------------------------
def _strings_in(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings_in(v)


def _labelled_strings(obj, label, skip_keys=frozenset()):
    """(label, string) for every string in a JSON value; the label is the top-level key, never the value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            for s in _strings_in(v):
                yield f"{label}:{k}", s
    else:
        for s in _strings_in(obj):
            yield label, s


def load_kit_secrets(kit_dir: str | None) -> tuple[dict[bytes, str], str]:
    """Return ({secret bytes: label}, status). Values stay in memory; labels are file or key names only."""
    secrets: dict[bytes, str] = {}
    if not kit_dir or not os.path.isdir(kit_dir):
        return secrets, "missing"

    def add(value: str, label: str, min_len: int):
        value = value.strip()
        if len(value) >= min_len:
            secrets.setdefault(value.encode("utf-8", "ignore"), label)

    for name in ("a2a_token.txt", "builder_admin_password.txt"):
        p = os.path.join(kit_dir, name)
        if os.path.isfile(p):
            add(Path(p).read_text(errors="ignore"), name, MIN_SECRET_LEN)
    tok_dir = os.path.join(kit_dir, "tokens")
    if os.path.isdir(tok_dir):
        for f in sorted(os.listdir(tok_dir)):
            p = os.path.join(tok_dir, f)
            if not os.path.isfile(p):
                continue
            raw = Path(p).read_text(errors="ignore")
            add(raw, f"tokens/{f}", MIN_SECRET_LEN)
            try:
                for lbl, s in _labelled_strings(json.loads(raw), f"tokens/{f}"):
                    add(s, lbl, MIN_SECRET_LEN)
            except ValueError:
                pass
    for name in ("kit-state.json", "kit.json"):
        p = os.path.join(kit_dir, name)
        if os.path.isfile(p):
            try:
                for lbl, s in _labelled_strings(json.loads(Path(p).read_text()), name, PUBLIC_KIT_KEYS):
                    add(s, lbl, MIN_STATE_VALUE_LEN)
            except ValueError:
                pass
    return secrets, "loaded"


def load_local_key_files(root: str) -> dict[bytes, str]:
    """The evidence ledger's private key, wherever it sits in the working tree."""
    out: dict[bytes, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", ".venv")]
        if ".evidence-key" in filenames:
            v = Path(dirpath, ".evidence-key").read_text(errors="ignore").strip()
            if len(v) >= MIN_SECRET_LEN:
                out.setdefault(v.encode(), "evidence ledger private key")
    return out


# --- scanning ---------------------------------------------------------------------------------------------
def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def scan_path(path: str) -> list[str]:
    return [rule for rule, rx in PATH_RULES if rx.search(path)]


def scan_content(data: bytes, secrets: dict[bytes, str], path: str = "") -> list[tuple[str, int | None]]:
    """(rule, line) pairs. The matched text is never returned."""
    out: list[tuple[str, int | None]] = []
    binary = _is_binary(data)

    def line_of(pos: int) -> int | None:
        return None if binary else data.count(b"\n", 0, pos) + 1

    for rule, kind, rx in CONTENT_RULES:
        if kind == "text" and binary:
            continue
        if any(r == rule and rx2.search(path) for r, rx2 in PATH_WAIVERS):
            continue
        for m in rx.finditer(data):
            text = m.group(0)
            if rule == "IPv4 address" and any(a.match(text.decode()) for a in ALLOWED_IPV4):
                continue
            if rule == "12-digit account id" and text.decode() in ALLOWED_ACCOUNT_IDS:
                continue
            out.append((rule, line_of(m.start())))
            break  # one finding per rule per file is enough to act on
    for value, label in secrets.items():
        pos = data.find(value)
        if pos >= 0:
            out.append((f"kit secret ({label})", line_of(pos)))
    return out


def _git(repo: str, *args: str, inp: bytes | None = None) -> bytes:
    r = subprocess.run(["git", "-C", repo, *args], input=inp, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed (exit {r.returncode})")
    return r.stdout


def scan_objects(repo: str, rev_args: list[str], secrets: dict[bytes, str]) -> tuple[list[Finding], int]:
    """Scan every commit message and blob that `git rev-list --objects <rev_args>` lists."""
    listing = _git(repo, "rev-list", "--objects", *rev_args).decode("utf-8", "replace").splitlines()
    paths: dict[str, set[str]] = {}
    for ln in listing:
        sha, _, path = ln.partition(" ")
        if sha:
            paths.setdefault(sha, set())
            if path:
                paths[sha].add(path)
    shas = list(paths)
    findings: list[Finding] = []
    n_blobs = 0
    for i in range(0, len(shas), 400):
        batch = shas[i : i + 400]
        out = _git(repo, "cat-file", "--batch", inp=("\n".join(batch) + "\n").encode())
        pos = 0
        for sha in batch:
            nl = out.index(b"\n", pos)
            header = out[pos:nl].split()
            if len(header) < 3:
                raise RuntimeError("unexpected git cat-file output")
            otype, size = header[1].decode(), int(header[2])
            body = out[nl + 1 : nl + 1 + size]
            pos = nl + 1 + size + 1
            if otype == "commit":
                msg = body.split(b"\n\n", 1)[-1]
                for rule, _line in scan_content(msg, secrets):
                    findings.append(Finding(f"{rule} in commit message", f"commit {sha[:9]}"))
                continue
            if otype != "blob":
                continue
            n_blobs += 1
            names = sorted(paths[sha]) or [f"(blob {sha[:9]})"]
            where = f"{names[0]}" + (f" (+{len(names) - 1} more paths)" if len(names) > 1 else "") + f" [blob {sha[:9]}]"
            for p in names:
                for rule in scan_path(p):
                    findings.append(Finding(rule, f"{p} [blob {sha[:9]}]"))
            if size > MAX_FILE_BYTES:
                findings.append(Finding("file over 5 MB", where))
            if size <= MAX_SCAN_BYTES:
                for rule, line in scan_content(body, secrets, names[0]):
                    findings.append(Finding(rule, where, line))
    return findings, n_blobs


def scan_dir(directory: str, root: str, secrets: dict[bytes, str]) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    n = 0
    if not os.path.isdir(directory):
        return findings, 0
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if os.path.islink(full) or not os.path.isfile(full):
                continue
            n += 1
            for rule in scan_path(rel):
                findings.append(Finding(rule, rel))
            size = os.path.getsize(full)
            if size > MAX_FILE_BYTES:
                findings.append(Finding("file over 5 MB", rel))
            if size <= MAX_SCAN_BYTES:
                data = Path(full).read_bytes()
                for rule, line in scan_content(data, secrets, rel):
                    findings.append(Finding(rule, rel, line))
    return findings, n


# --- hook protocol ----------------------------------------------------------------------------------------
def _have(repo: str, sha: str) -> bool:
    return subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def rev_args_for_hook(repo: str, remote: str, lines: list[str]) -> list[list[str]]:
    """One rev-list argument set per pushed ref: <remote sha>..<local sha>, or, for a new branch, the
    commits no ref of this remote has yet."""
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        _lref, lsha, _rref, rsha = parts[:4]
        if lsha == ZERO_SHA:
            continue  # a deletion pushes no objects
        if rsha != ZERO_SHA and _have(repo, rsha):
            out.append([lsha, "--not", rsha])
        else:
            out.append([lsha, "--not", f"--remotes={remote}"])
    return out


def toplevel(start: str | None = None) -> str:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=start or os.getcwd())
    if r.returncode != 0:
        raise RuntimeError("not inside a git repository")
    return r.stdout.strip()


def report(findings: list[Finding], stats: str, out=sys.stderr) -> int:
    seen, uniq = set(), []
    for f in findings:
        if f.key() not in seen:
            seen.add(f.key())
            uniq.append(f)
    print(f"pre-push scan: {stats}", file=out)
    if not uniq:
        print("pre-push scan: clean", file=out)
        return 0
    print(f"pre-push scan: BLOCKED, {len(uniq)} finding(s). Rule and file only; matched text is never printed.", file=out)
    for f in sorted(uniq, key=Finding.key):
        print(f"  {f}", file=out)
    print("Fix the file, rewrite the commit, then push again. (Bypass is not offered here; talk to the lead.)", file=out)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hook", nargs="*", metavar="ARG", help="git pre-push mode: <remote> <url>; ref list on stdin")
    ap.add_argument("--range", action="append", default=[], help="scan the objects of this rev range (repeatable)")
    ap.add_argument("--refs", nargs="+", help="scan the whole history reachable from these refs")
    ap.add_argument("--evidence-only", action="store_true")
    ap.add_argument("--no-evidence", action="store_true", help="skip the working-tree evidence/ scan")
    ap.add_argument("--evidence-dir", default=None, help="default: <repo>/evidence")
    ap.add_argument("--kit-dir", default=os.environ.get("TEAM13_KIT_DIR") or os.path.expanduser("~/firstmate-homes/team13-kit"))
    ap.add_argument("--repo", default=None)
    args = ap.parse_args(argv)

    try:
        repo = args.repo or toplevel()
        secrets, kit_status = load_kit_secrets(args.kit_dir)
        secrets.update(load_local_key_files(repo))
        if kit_status == "missing":
            print("pre-push scan: WARNING kit folder not found; kit secret values were NOT checked "
                  "(pattern rules still run). Set TEAM13_KIT_DIR to enable.", file=sys.stderr)
            if os.environ.get("PREPUSH_REQUIRE_KIT") == "1":
                print("pre-push scan: PREPUSH_REQUIRE_KIT=1, refusing to continue without the kit.", file=sys.stderr)
                return 1

        sets: list[list[str]] = []
        if args.hook is not None:
            remote = args.hook[0] if args.hook else "origin"
            lines = sys.stdin.read().splitlines()
            sets += rev_args_for_hook(repo, remote, lines)
        for r in args.range:
            sets.append([r])
        if args.refs:
            sets.append(list(args.refs))
        findings: list[Finding] = []
        n_blobs = 0
        if not args.evidence_only:
            for rev in sets:
                f, n = scan_objects(repo, rev, secrets)
                findings += f
                n_blobs += n
        n_ev = 0
        if not args.no_evidence:
            ev = args.evidence_dir or os.path.join(repo, "evidence")
            f, n_ev = scan_dir(ev, repo, secrets)
            findings += f
        stats = (f"{len(sets)} ref set(s), {n_blobs} files in pushed objects, {n_ev} files in evidence/, "
                 f"{len(secrets)} kit secret values checked ({kit_status})")
        return report(findings, stats)
    except Exception as exc:  # fail closed: an unscannable push is a blocked push
        print(f"pre-push scan: could not complete ({type(exc).__name__}: {exc}); blocking the push.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
