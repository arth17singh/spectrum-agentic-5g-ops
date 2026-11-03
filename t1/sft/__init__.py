"""Track 1 LLM fine-tune: base-model staging, SFT data, LoRA training. Every step runs as a SageMaker job.

    t1.sft.stage_base   CPU job: the copied fp32 .bin shards -> bf16 safetensors, sha256 of every file,
                        model record
    t1.sft.build        CPU job: SFT examples from the in-process twin (seeds < 900000) and templates
                        written here; drops anything that overlaps a frozen eval item; manifest
    t1.sft.train_lora   GPU job: PEFT LoRA on the staged base; adapter and merged model; model record

Each entry point has --dry-run (no AWS call, no twin, no torch). The job runner writes the
declaration (templates in t1/sft/jobs/) before it starts the job.

The team's shared modules are required, never stubbed: `team13_contracts` (W1), `twin.replay` (W2),
`tools.evlib` (W6) and the frozen eval contract `t1.evals` (W12). If `team13_contracts` is not
installed, the in-repo copy under contracts/ is put on sys.path here, as t1/evals does.
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
