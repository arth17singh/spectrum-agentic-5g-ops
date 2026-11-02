"""The one output parser every model goes through (PARSER_VERSION is declared before each run).

Steps, all deterministic, no repair:
  1. drop reasoning blocks `<think>...</think>`; an unclosed `<think>` means the answer was cut off,
     so the whole output is a parse failure;
  2. drop Markdown code fences (```json ... ```), keeping their contents;
  3. return the FIRST complete JSON object in what is left (json.JSONDecoder.raw_decode tried at
     each "{" in order). Anything else is a parse failure.

Knowledge items accept one documented fallback because the ot-lite questions carry their own
answer format: telelogs `\\boxed{C3}` (last one wins) and 3gpp_tsg `{"WORKING GROUP": ...}`.
"""

from __future__ import annotations

import json
import re
from typing import Any

PARSER_VERSION = "t1-parse-v1"

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?|```")
_BOXED = re.compile(r"\\boxed\{+\s*([^{}]*?)\s*\}+")


def strip_reasoning(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    t = _THINK.sub("", text)
    if re.search(r"<think>", t, re.I):
        return None  # unclosed reasoning block: truncated output
    t = re.sub(r"</think>", "", t, flags=re.I)  # some templates emit only the closing tag
    return _FENCE.sub("", t)


def first_json_object(text: str) -> dict[str, Any] | None:
    t = strip_reasoning(text)
    if t is None:
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            obj, _ = dec.raw_decode(t, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def last_boxed(text: str) -> str | None:
    t = strip_reasoning(text)
    if t is None:
        return None
    found = _BOXED.findall(t)
    return found[-1].strip() if found else None
