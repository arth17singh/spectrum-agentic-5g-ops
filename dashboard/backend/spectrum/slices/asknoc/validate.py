"""Rules for what an operator may send to noc_lead.

- 1 to 2000 characters after NFC normalisation and trimming, at most 40 lines.
- No control, format (zero-width, bidi), private-use, surrogate or unassigned characters; only \\n and \\t.
- No `case_id:` or `run_id:` anywhere and no `ref:` line: the kit prefixes the lead's message with
  `case_id: <id>` (checkpoint_kit.py L218) and the lead's rule 1 acts on `case_id:` and copies `run_id:`
  lines, so either could point the swarm at another case or run. `ref:` is the panel's own tag.
- No case id other than the current case.
- No URL, address, key or account-like string: the kit logs the request verbatim to the team's audit log
  (L153), and the panel would otherwise have to redact the operator's own words.
"""

from __future__ import annotations

import re
import unicodedata

from .redact import Redactor

MAX_CHARS = 2000
MAX_LINES = 40
BAD_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Cn"}
SPOOF = re.compile(r"\b(?:case_id|run_id)\s*:", re.I)
REF_LINE = re.compile(r"^\s*ref\s*:", re.I | re.M)
CASE_ID = re.compile(r"case-[0-9a-f]{8}")


def validate_text(raw, current_case: str | None = None, redactor: Redactor | None = None) -> tuple[str, list[str]]:
    """(normalised text, reasons). An empty reasons list means the text may be sent."""
    if not isinstance(raw, str):
        return "", ["The message must be text."]
    text = unicodedata.normalize("NFC", raw).strip()
    reasons: list[str] = []
    if not text:
        return "", ["The message is empty."]
    if len(text) > MAX_CHARS:
        reasons.append(f"The message is {len(text)} characters; the limit is {MAX_CHARS}.")
    if text.count("\n") + 1 > MAX_LINES:
        reasons.append(f"The message has more than {MAX_LINES} lines.")
    if any(unicodedata.category(c) in BAD_CATEGORIES and c not in "\n\t" for c in text):
        reasons.append("The message contains invisible or control characters (for example zero-width or "
                       "direction marks). Retype it.")
    if SPOOF.search(text):
        reasons.append("Remove `case_id:` and `run_id:`: the kit adds the case id itself, and these words "
                       "would tell noc_lead to work on another case or run.")
    if REF_LINE.search(text):
        reasons.append("Remove the line starting with `ref:`; the panel adds that tag itself.")
    others = sorted({c for c in CASE_ID.findall(text) if c != current_case})
    if others:
        reasons.append(f"Don't name other cases ({', '.join(others)}). Each case is followed on its own.")
    if redactor is not None and redactor.finds(text):
        reasons.append("Remove links, addresses, keys and account numbers: the kit stores your text verbatim "
                       "in the team's audit log.")
    return text, reasons
