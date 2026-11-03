"""Interface items: operator requests from templates, each with its exact intent and entities.

Answer schema (prompt TASK interface): {intent, cells, parameter, direction, amount, kpi}.
Paraphrases cover verb, word order, number words, units and cell spelling ("cell 2", "cell #2",
"c2"); distractors add background sentences naming other cells and numbers, and include
no-action, out-of-scope and query requests. Direction is the operator's word (up/down), so no
radio-physics sign convention is needed to label it.
"""

from __future__ import annotations

import re
from typing import Any

from t1.evals.common import amount_text, h, item, pick

UNIT = "interface"
N_ITEMS = 320
SEED = "interface-v1"

_UP_P = ["Increase", "Raise", "Boost", "Bump up"]
_DOWN_P = ["Decrease", "Lower", "Reduce", "Drop"]

POWER_T = [
    "{V} the transmit power on {C} by {A} dB.",
    "{V} {C}'s power by {A} decibels.",
    "Please {v} the Tx power of {C} by {A} dB.",
    "{C}: {v} power {A} dB.",
    "Can you {v} power on {C} by {A} dB?",
    "I need {C} {dir_word} {A} dB in power.",
]
POWER_MULTI_T = [
    "{V} the power on {C} by {A} dB each.",
    "Please {v} transmit power by {A} dB for {C}.",
]
TILT_T = [
    "Tilt {C} {dir} by {A} degrees.",
    "{V} the antenna tilt of {C} by {A} degrees.",
    "{C} is losing coverage at the edge, tilt it {dir} {A} degrees.",
    "Please {v} the tilt on {C} by {A}°.",
    "Move {C}'s tilt {dir} {A} deg.",
]
TILT_MULTI_T = [
    "Tilt {C} {dir} by {A} degrees each.",
    "{V} the tilt by {A} degrees on {C}.",
]
QUERY_T = {
    "coverage_ratio": ["What is the coverage ratio on {C}?", "Show me the current coverage for {C}."],
    "sla_ratio": ["What's the SLA ratio for {C} right now?", "Report the SLA compliance of {C}."],
    "sinr": ["Show me the SINR on {C}.", "What SINR is {C} seeing?"],
    "rsrp": ["Report RSRP for {C}.", "What is the RSRP level on {C}?"],
    "ue_count": ["How many users are on {C}?", "Give me the UE count for {C}."],
}
QUERY_NET_T = {
    "coverage_ratio": "What is the network-wide coverage ratio?",
    "sla_ratio": "What's the overall SLA ratio across the network right now?",
    "ue_count": "How many UEs are attached across the whole network?",
}
NO_ACTION_T = [
    "{C} looks healthy today, no change needed.",
    "Thanks, that's all for now.",
    "Just confirming the change on {C} from this morning went through fine. Nothing else.",
    "FYI the maintenance window on {C} is over. No action required.",
]
OUT_OF_SCOPE_T = [
    "Reboot {C}.",
    "Change the handover offset on {C} by {A} dB.",
    "Open a ticket with the vendor about {C}'s backhaul.",
    "Switch {C} to the 2600 MHz band.",
    "Update the neighbour list of {C}.",
]
DISTRACTORS = [
    "",
    "",
    "Yesterday cell {D} had an alarm, which is now cleared. ",
    "Traffic on cell {D} is up 15% this week. ",
    "Ticket 4471: ",
    "Following up on the 3 complaints near cell {D}: ",
    "Cell {D} is fine. ",
]


def _cell_text(c: int, key: Any) -> str:
    return pick(["cell {c}", "cell #{c}", "c{c}", "Cell {c}"], "celltxt", key).format(c=c)


def _cells_text(cells: list[int], key: Any) -> str:
    names = [_cell_text(c, (key, c)) for c in cells]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _expected(intent: str, cells=(), parameter=None, direction=None, amount=None, kpi=None) -> dict:
    return {
        "intent": intent,
        "cells": sorted(set(cells)),
        "parameter": parameter,
        "direction": direction,
        "amount": amount,
        "kpi": kpi,
    }


def _one(i: int) -> tuple[str, dict, str]:
    k = (SEED, i)
    kind = pick(["power", "power", "tilt", "tilt", "multi", "query", "query_net", "no_action", "oos"], "kind", k)
    c = h("cell", k) % 4
    d = (c + 1 + h("dcell", k) % 3) % 4  # a different cell for the distractor
    distract = "" if kind == "multi" else pick(DISTRACTORS, "distract", k).format(D=d)
    up = h("dir", k) % 2 == 0
    if kind in ("power", "multi") and (kind == "power" or h("mparam", k) % 2 == 0):
        amount = pick([1, 2, 3, 4, 5, 6, 1.5, 0.5, 2.5], "amt", k) if up else pick([1, 2, 3, 4, 5, 6, 1.5, 0.5], "amt", k)
        cells = [c] if kind == "power" else sorted({c, d})
        verbs = _UP_P if up else _DOWN_P
        V = pick(verbs, "verb", k)
        tpl = pick(POWER_T if kind == "power" else POWER_MULTI_T, "tpl", k)
        C = _cell_text(c, k) if kind == "power" else _cells_text(cells, k)
        text = tpl.format(V=V, v=V.lower(), C=C, A=amount_text(amount, k), dir_word="up" if up else "down")
        return distract + _cap(text) if distract else _cap(text), _expected(
            "adjust_power", cells, "power", "up" if up else "down", float(amount)
        ), "adjust_power" if kind == "power" else "adjust_power_multi"
    if kind in ("tilt", "multi"):
        amount = pick([1, 2, 3, 4, 5, 0.5, 1.5], "amt", k)
        cells = [c] if kind == "tilt" else sorted({c, d})
        V = pick(_UP_P[:3] if up else _DOWN_P[:3], "verb", k)
        tpl = pick(TILT_T if kind == "tilt" else TILT_MULTI_T, "tpl", k)
        C = _cell_text(c, k) if kind == "tilt" else _cells_text(cells, k)
        text = tpl.format(V=V, v=V.lower(), C=C, A=amount_text(amount, k), dir="up" if up else "down")
        return distract + _cap(text) if distract else _cap(text), _expected(
            "adjust_tilt", cells, "tilt", "up" if up else "down", float(amount)
        ), "adjust_tilt" if kind == "tilt" else "adjust_tilt_multi"
    if kind == "query":
        kpi = pick(sorted(QUERY_T), "kpi", k)
        text = pick(QUERY_T[kpi], "tpl", k).format(C=_cell_text(c, k))
        return distract + _cap(text) if distract else _cap(text), _expected("query_kpi", [c], kpi=kpi), "query_kpi"
    if kind == "query_net":
        kpi = pick(sorted(QUERY_NET_T), "kpi", k)
        return distract + QUERY_NET_T[kpi], _expected("query_kpi", [], kpi=kpi), "query_kpi_network"
    if kind == "no_action":
        text = pick(NO_ACTION_T, "tpl", k).format(C=_cell_text(c, k))
        return _cap(text), _expected("no_action"), "no_action"
    text = pick(OUT_OF_SCOPE_T, "tpl", k).format(C=_cell_text(c, k), A=amount_text(pick([1, 2, 3], "amt", k), k))
    return distract + _cap(text) if distract else _cap(text), _expected("out_of_scope"), "out_of_scope"


def generate() -> list[dict]:
    out, seen, i = [], set(), 0
    while len(out) < N_ITEMS:
        text, expected, family = _one(i)
        text = re.sub(r"\b(1|one) degrees\b", r"\1 degree", text)
        i += 1
        if text in seen:
            continue
        seen.add(text)
        n = len(out)
        out.append(
            item(UNIT, f"if-{n:04d}", "interface", {"request": text}, expected,
                 {"family": family, "gen": SEED, "gen_index": i - 1})
        )
    return out
