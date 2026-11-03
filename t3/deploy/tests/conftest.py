import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root, so `t3.deploy` imports as a namespace package
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
