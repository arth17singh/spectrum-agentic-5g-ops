import sys
from pathlib import Path

BRIDGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRIDGE)]
