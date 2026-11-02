"""Point the evidence ledger at a throwaway directory before any test imports spectrum.main, so the
test run never writes to the real evidence/ folder or creates the real signing key, and never
enables the S3 source."""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="spectrum-evidence-test-")
os.environ["SPECTRUM_EVIDENCE_DIR"] = os.path.join(_tmp, "evidence")
os.environ["SPECTRUM_EVIDENCE_KEY"] = os.path.join(_tmp, "evidence-key")
os.environ.pop("SPECTRUM_EVIDENCE_S3", None)
