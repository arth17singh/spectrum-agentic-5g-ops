"""Price and latency items: a fixed prompt set drawn from the other units' frozen items.

The mix mirrors the expected workload: 15 interface, 10 execution, 10 safety and 15 core
knowledge prompts, chosen by hash from each unit's test split. Each item copies the source item's
template and input, so the prompt is byte-identical to the one the source unit sends. The harness
records per-call latency, tokens and price; nothing is scored for correctness here.
"""

from __future__ import annotations

from t1.evals.common import h, item

UNIT = "price_latency"
SEED = "price-latency-v1"
MIX = {"interface": 15, "execution": 10, "safety": 10, "knowledge": 15}


def generate(units: dict[str, list[dict]]) -> list[dict]:
    out = []
    for unit, n in MIX.items():
        pool = [it for it in units[unit] if it["split"] == "test"]
        if unit == "knowledge":
            pool = [it for it in pool if it["meta"]["group"] == "core"]
        chosen = sorted(pool, key=lambda it: (h(SEED, it["item_id"]), it["item_id"]))[:n]
        for src in sorted(chosen, key=lambda it: it["item_id"]):
            out.append(
                item(UNIT, f"pl-{src['item_id']}", src["template"], dict(src["input"]), {},
                     {"source_unit": unit, "source_item_id": src["item_id"], "gen": SEED})
            )
    return out
