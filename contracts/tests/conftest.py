import sys
from pathlib import Path

CONTRACTS = Path(__file__).resolve().parents[1]
REPO_ROOT = CONTRACTS.parent
FIXTURES = CONTRACTS / "fixtures"
SCHEMA_DIR = CONTRACTS / "schema"

# Import team13_contracts and gen.py without installing anything.
if str(CONTRACTS) not in sys.path:
    sys.path.insert(0, str(CONTRACTS))
