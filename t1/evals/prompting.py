"""The single prompt file and how an item becomes (system, user) messages.

`prompt.txt` holds a `### SYSTEM` section shared by every unit and one `### TASK <template>`
section per item template. Placeholders are `<<name>>` (JSON braces stay literal). Every model
gets the same file, so base and tuned see byte-identical prompts for the same item.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

PROMPT_PATH = Path(__file__).with_name("prompt.txt")
_HEADER = re.compile(r"^### (SYSTEM|TASK [a-z_]+)\s*$", re.M)
_PLACEHOLDER = re.compile(r"<<([a-z_]+)>>")


def prompt_bytes(path: Path = PROMPT_PATH) -> bytes:
    return path.read_bytes()


def prompt_sha256(path: Path = PROMPT_PATH) -> str:
    return hashlib.sha256(prompt_bytes(path)).hexdigest()


def load_sections(path: Path = PROMPT_PATH) -> dict[str, str]:
    text = prompt_bytes(path).decode("utf-8")
    heads = list(_HEADER.finditer(text))
    if not heads or heads[0].group(1) != "SYSTEM":
        raise ValueError("prompt.txt must start with a ### SYSTEM section")
    out: dict[str, str] = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        name = "system" if h.group(1) == "SYSTEM" else h.group(1).split(" ", 1)[1]
        if name in out:
            raise ValueError(f"duplicate prompt section {name}")
        out[name] = text[h.end():end].strip("\n")
    return out


def fill(template: str, values: dict[str, Any]) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"prompt placeholder <<{key}>> has no value")
        return str(values[key])

    return _PLACEHOLDER.sub(sub, template)


def render(item: dict[str, Any], sections: dict[str, str] | None = None) -> tuple[str, str]:
    """(system, user) for an item. `item["template"]` names the TASK section, `item["input"]`
    fills it. A price_latency item carries the template and input of the item it reuses."""
    sections = sections or load_sections()
    tpl = item["template"]
    if tpl not in sections:
        raise KeyError(f"no prompt section TASK {tpl}")
    return sections["system"], fill(sections[tpl], item["input"])


def input_sha256(system: str, user: str) -> str:
    return hashlib.sha256((system + "\n\x1e\n" + user).encode("utf-8")).hexdigest()
