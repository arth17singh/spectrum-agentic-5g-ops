"""Generation is deterministic and the committed files are current."""

import shutil
import subprocess
from pathlib import Path

import pytest

import gen  # on sys.path via conftest.py

CONTRACTS = Path(__file__).resolve().parents[1]  # no `from conftest import`: works with --import-mode=importlib
REPO_ROOT = CONTRACTS.parent


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(rel): (root / rel).read_bytes() for rel in gen.render()}


def test_generation_is_idempotent(tmp_path):
    assert gen.main(["--out-root", str(tmp_path)]) == 0
    first = _snapshot(tmp_path)
    assert gen.main(["--out-root", str(tmp_path)]) == 0
    assert _snapshot(tmp_path) == first
    assert gen.main(["--out-root", str(tmp_path), "--check"]) == 0


def test_committed_files_are_current():
    assert gen.main(["--out-root", str(REPO_ROOT), "--check"]) == 0


def test_check_detects_stale_file(tmp_path):
    gen.main(["--out-root", str(tmp_path)])
    ts = tmp_path / gen.TS_REL
    ts.write_text(ts.read_text() + "// edited by hand\n")
    assert gen.main(["--out-root", str(tmp_path), "--check"]) == 1


TSC = REPO_ROOT / "dashboard/frontend/node_modules/.bin/tsc"


@pytest.mark.skipif(not TSC.exists() or shutil.which("node") is None, reason="frontend node_modules not linked")
def test_generated_typescript_compiles_strict():
    proc = subprocess.run(
        [
            str(TSC),
            "--noEmit",
            "--strict",
            "--target",
            "es2022",
            "--module",
            "esnext",
            "--moduleResolution",
            "bundler",
            "contracts/tests/ts/check.ts",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
