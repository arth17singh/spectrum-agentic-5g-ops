"""The job's source: `git archive <sha> -- <pathspec>` plus the generated requirements.txt, gzipped.

The archive holds the committed tree only, so a dirty working tree never reaches a job. Every tracked
member is byte-identical to `git cat-file blob <sha>:<path>`; check_unit part b verifies that against
the tarball the job actually downloaded.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path


class PackageError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> bytes:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    if r.returncode != 0:
        raise PackageError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def repo_root(start: Path | None = None) -> Path:
    here = start or Path(__file__).resolve().parent
    return Path(git(here, "rev-parse", "--show-toplevel").decode().strip())


def resolve_commit(repo: Path, rev: str = "HEAD") -> str:
    return git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()


def tree_dirty(repo: Path) -> list[str]:
    """Tracked files with uncommitted changes (untracked files never ship, so they do not count)."""
    out = git(repo, "status", "--porcelain", "--untracked-files=no").decode()
    return [line[3:] for line in out.splitlines() if line.strip()]


def tracked_files(repo: Path, sha: str, paths: tuple[str, ...] | list[str]) -> list[str]:
    out = git(repo, "ls-tree", "-r", "--name-only", "-z", sha, "--", *paths)
    return sorted(p for p in out.decode("utf-8").split("\0") if p)


def blob(repo: Path, sha: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "cat-file", "blob", f"{sha}:{path}"], cwd=repo, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def commit_epoch(repo: Path, sha: str) -> int:
    return int(git(repo, "show", "-s", "--format=%ct", sha).decode().strip())


def build_source(repo: Path, sha: str, paths: tuple[str, ...] | list[str], extra: dict[str, bytes]) -> bytes:
    """sourcedir.tar.gz bytes. Deterministic for one machine: member order from git, fixed mtimes, gzip
    mtime 0 (other zlib builds may compress differently, which is why the contents are what part b checks)."""
    if not paths:
        raise PackageError("empty pathspec")
    files = tracked_files(repo, sha, paths)
    if not files:
        raise PackageError(f"pathspec {list(paths)} matches no tracked file at {sha[:12]}")
    clash = sorted(set(extra) & set(files))
    if clash:
        raise PackageError(f"generated file(s) {clash} would replace tracked files")
    raw = git(repo, "archive", "--format=tar", sha, "--", *paths)
    mtime = commit_epoch(repo, sha)
    buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw)) as src, \
            tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as dst:
        for m in src.getmembers():
            m.pax_headers = {}
            if m.isfile():
                dst.addfile(m, src.extractfile(m))
            elif m.isdir() or m.issym():
                dst.addfile(m)
        for name in sorted(extra):
            data = extra[name]
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode, ti.uname, ti.gname = len(data), mtime, 0o644, "root", "root"
            dst.addfile(ti, io.BytesIO(data))
    return gzip.compress(buf.getvalue(), compresslevel=9, mtime=0)


def members(tar_gz: bytes) -> dict[str, bytes]:
    """Regular files in a sourcedir.tar.gz -> bytes (symlinks as their target text)."""
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_gz), mode="r:gz") as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name] = t.extractfile(m).read()
            elif m.issym():
                out[m.name] = m.linkname.encode("utf-8")
    return out


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
