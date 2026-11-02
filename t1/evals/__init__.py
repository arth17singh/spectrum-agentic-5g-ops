"""Track 1 LLM evaluation: frozen item sets, deterministic graders and one harness.

Units: interface, execution, safety, knowledge, price_latency. See README.md, run.py (harness)
and freeze.py (generation and hashes). No model judges anything here.

The team's shared modules are required, never stubbed: `team13_contracts` (W1, canonical change
string and record validation), `twin.replay` (W2, the in-process Maveric twin) and `tools.evlib`
(W6, evidence writer). Run from the repository root. If `team13_contracts` is not installed, the
in-repo copy under contracts/ is put on sys.path here.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if importlib.util.find_spec("team13_contracts") is None:
    _contracts = REPO_ROOT / "contracts"
    if (_contracts / "team13_contracts" / "__init__.py").is_file():
        sys.path.insert(0, str(_contracts))
if str(REPO_ROOT) not in sys.path:  # `twin` and `tools` are top-level packages of the repo
    sys.path.insert(0, str(REPO_ROOT))
