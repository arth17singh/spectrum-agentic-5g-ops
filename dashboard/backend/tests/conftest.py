"""Environment for the whole test run, set before any test imports spectrum.main.

- The evidence ledger points at a throwaway directory, so the run never writes to the real evidence/
  folder or creates the real signing key, and never enables the S3 source.
- SPECTRUM_RECORDS_ROOT (the slices' RecordsReader) points at that same throwaway directory: one data
  root. Slice tests that need data pass their own root (a tmp_path or tests/slices/<x>/ fixtures).
- SPECTRUM_OFFLINE=1: nothing may call AWS. RecordsReader refuses s3:// roots unless a client is injected.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="spectrum-evidence-test-")
os.environ["SPECTRUM_EVIDENCE_DIR"] = os.path.join(_tmp, "evidence")
os.environ["SPECTRUM_EVIDENCE_KEY"] = os.path.join(_tmp, "evidence-key")
os.environ.pop("SPECTRUM_EVIDENCE_S3", None)
os.environ["SPECTRUM_RECORDS_ROOT"] = "file://" + os.path.join(_tmp, "evidence")
os.environ["SPECTRUM_OFFLINE"] = "1"
