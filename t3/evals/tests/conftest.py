import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass
