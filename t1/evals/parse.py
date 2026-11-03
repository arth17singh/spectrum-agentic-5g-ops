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


# --------------------------------------------------------------------------- knowledge, plain-text answers
# t1-parse-v1+knowledge-plain-v1 (knowledge unit only; every other unit stays t1-parse-v1). The base
# OTel model answers most multiple-choice items with a bare "B" or "B. <choice text>" instead of the
# requested JSON. When no JSON answer is found, the answer is read from the START of the reply only
# (after reasoning and fences are dropped), never searched for elsewhere in the text.
KNOWLEDGE_PARSER_VERSION = "t1-parse-v1+knowledge-plain-v1"
_LEAD_LETTER = re.compile(r"^\s*(?:(?i:answer)\s*[:=]\s*)?\(?([A-H])(?=\s*$|[.):\]]|\s*\n)")
_LEAD_RCA = re.compile(r"^\s*(?:answer\s*[:=]\s*)?C\s*(\d+)(?!\d)", re.I)
_LEAD_LABEL = re.compile(r"^\s*(?:answer\s*[:=]\s*)?\"?([A-Z]{2,4}(?:\d|_AH\d))(?![A-Za-z0-9])")


def parser_version(unit: str) -> str:
    return KNOWLEDGE_PARSER_VERSION if unit == "knowledge" else PARSER_VERSION


def leading(text: str, kind: str) -> str | None:
    """The answer at the start of a plain-text reply: a choice letter (mcq), a root-cause label C<n>
    (rca) or a 3GPP working-group name (label); None when the reply does not start with one."""
    t = strip_reasoning(text)
    if t is None:
        return None
    rx = {"mcq": _LEAD_LETTER, "rca": _LEAD_RCA, "label": _LEAD_LABEL}.get(kind)
    m = rx.match(t) if rx else None
    if not m:
        return None
    if kind == "mcq":
        return m.group(1).upper()
    if kind == "rca":
        return f"C{int(m.group(1))}"
    return m.group(1)
