"""Names and Gateway tool schemas of Team13's RAN tools.

One source of truth for the Lambda (dispatch), the Cedar generator (actions and the
`context.input` fields the rules read) and the deploy script (Gateway targets).

The Gateway names a tool `<target>___<tool>` and Cedar sees that full name as the action.
Target names allow letters, digits and single hyphens (no underscores).

The NLM tool cannot live in a target named `nlm`: the organisers' kit already owns that
target (it serves `nlm___ask_nlm`) and creates it on every deploy. Our NLM target name is
`T3_NLM_TARGET` (default `nlm13`, so the tool is `nlm13___ask`).
"""
from __future__ import annotations

import os
import re

RAN_TARGET = "ran"
DEFAULT_NLM_TARGET = "nlm13"
TARGET_NAME_RE = re.compile(r"([0-9a-zA-Z]-?){1,100}")

OBSERVE = "observe"
PROPOSE = "propose"
PREDICT = "predict"
PREFLIGHT = "preflight"
APPLY = "apply_change"
ASK = "ask"

RAN_TOOLS = (OBSERVE, PROPOSE, PREDICT, PREFLIGHT, APPLY)

# How `change` is declared to the Gateway. "object" is the contract; "string" (JSON text)
# is the fallback if the Gateway's Cedar schema generation rejects nested numbers.
CHANGE_TYPES = ("object", "string")


def nlm_target() -> str:
    name = os.environ.get("T3_NLM_TARGET") or DEFAULT_NLM_TARGET
    if not TARGET_NAME_RE.fullmatch(name) or name == "nlm":
        raise ValueError(f"T3_NLM_TARGET={name!r}: letters, digits, single hyphens; not the kit's 'nlm'")
    return name


def full(target: str, tool: str) -> str:
    return f"{target}___{tool}"


def ran(tool: str) -> str:
    return full(RAN_TARGET, tool)


def nlm_tool() -> str:
    return full(nlm_target(), ASK)


def all_tools() -> list[str]:
    """Every full tool name we serve, in a fixed order."""
    return [ran(t) for t in RAN_TOOLS] + [nlm_tool()]


# ---------- Gateway tool schemas (the subset of JSON Schema the Gateway accepts) ----------
def _p(type_: str, description: str, **extra) -> dict:
    return {"type": type_, "description": description, **extra}


CASE_ID = _p("string", "The case_id returned by open_case (case-...). Every call needs it.")
SPECIAL = _p("object", "Added by the Agent Builder. Leave it out.")
RUN_ID = _p("string", "Optional. The run_id line from your task text, if it has one (run_id: ...). Copy it as is.")
CELL = {
    "type": "object",
    "description": "One cell's adjustment.",
    "properties": {
        "cell": _p("integer", "Cell index."),
        "d_power_db": _p("number", "Transmit power change in dB."),
        "d_tilt_deg": _p("number", "Electrical tilt change in degrees."),
    },
    "required": ["cell", "d_power_db", "d_tilt_deg"],
}


def change_schema(change_type: str = "object") -> dict:
    if change_type == "string":
        return _p("string", 'The change as JSON text, e.g. {"cells": [{"cell": 2, "d_power_db": 0, '
                            '"d_tilt_deg": -2}]}. Pass it unchanged between tools.')
    if change_type != "object":
        raise ValueError(f"change_type must be one of {CHANGE_TYPES}")
    return {"type": "object",
            "description": "The change, e.g. {\"cells\": [{\"cell\": 2, \"d_power_db\": 0, \"d_tilt_deg\": -2}]}. "
                           "Pass it unchanged between propose, predict, preflight and apply_change.",
            "properties": {"cells": {"type": "array", "description": "Per-cell adjustments.", "items": CELL}},
            "required": ["cells"]}


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    # case_id is deliberately not "required": a call without it must reach Cedar, which refuses
    # it with a recorded decision (the case-id forbid), instead of failing schema validation.
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "inputSchema": schema}


def ran_tool_schemas(change_type: str = "object") -> list[dict]:
    chg = change_schema(change_type)
    base = {"case_id": CASE_ID, "run_id": RUN_ID, "special_context": SPECIAL}
    return [
        _tool(OBSERVE, "Read the RAN twin's current KPIs for this case (coverage, SLA, SINR, per cell).",
              dict(base), []),
        _tool(PROPOSE, "Ask the trained RL policy for a change for this case. Returns a change to pass on.",
              dict(base), []),
        _tool(PREDICT, "Predict the KPIs after a change with the learned response model. "
                       "The prediction is later compared with what the twin measures.",
              {**base, "change": chg}, ["change"]),
        _tool(PREFLIGHT, "Try a change on a copy of this case's twin and report the KPIs. Changes nothing.",
              {**base, "change": chg}, ["change"]),
        _tool(APPLY, "Apply a change to this case's twin. Needs a person's approval; one approval applies "
                     "one change. If refused for approval, give the person the approval id, and after they "
                     "approve call again with exactly the same values.",
              {**base,
               "item_id": _p("string", "What the change acts on, e.g. cluster-1 or cell-2."),
               "change": chg,
               "reason": _p("string", "Why this change solves the problem."),
               "approval": _p("string", "Set by the checkpoint. Leave it out.")},
              ["item_id", "change", "reason"]),
    ]


def nlm_tool_schemas() -> list[dict]:
    return [_tool(ASK, "Ask the network language model (NLM) a question. The answer says which model answered.",
                  {"case_id": CASE_ID, "question": _p("string", "The question."),
                   "run_id": RUN_ID, "special_context": SPECIAL}, ["question"])]


def targets(change_type: str = "object") -> dict[str, list[dict]]:
    """{target name: [tool schemas]} for the Gateway."""
    return {RAN_TARGET: ran_tool_schemas(change_type), nlm_target(): nlm_tool_schemas()}


def input_fields(full_name: str, change_type: str = "object") -> set[str]:
    """The top-level argument names a tool declares (what Cedar can read as context.input.*)."""
    for target, tools in targets(change_type).items():
        for t in tools:
            if full(target, t["name"]) == full_name:
                return set(t["inputSchema"]["properties"])
    raise KeyError(full_name)
