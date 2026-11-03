"""Test doubles for t1/sft (tests only; a real build never uses them, and its manifest names the labeler).

    fake_reference(spec)   synthetic stand-in for t1.evals.gen_execution.reference: no twin, instant,
                           deterministic; about a third of specs are "rejected by the keep rules"
    MINI_TEMPLATE          a small chat template written here with the same assistant-turn and
                           generation-prompt behaviour as the Qwen3 family (thinking on/off)
    FakeTokenizer          character tokenizer with the four special tokens chat.py relies on
"""

from __future__ import annotations

import re
from typing import Any

from t1.evals.common import h


def fake_reference(spec: dict[str, Any]) -> dict[str, Any] | None:
    from t1.evals.gen_execution import candidates

    k = (spec["scenario"], spec["seed"], "|".join(spec["prefix"]))
    if h("fake-keep", *k) % 3 == 0:
        return None
    alarm = [False] * 4
    if spec["fault"]["kind"] == "outage":
        alarm[spec["fault"]["cell"]] = True
    live = [i for i in range(4) if not alarm[i]]
    c = live[h("fake-cell", *k) % len(live)]
    state = {
        "power_dbm": [23.0 + 6 * spec["fault"].get("operating_point_steps", 0)] * 4,
        "tilt_deg": [6.0] * 4,
        "ue_count": [float(30 + h("fake-ue", *k, i) % 40) for i in range(4)],
        "rsrp_mean_dbm": [-90.0 - (h("fake-rm", *k, i) % 900) / 100 for i in range(4)],
        "rsrp_p10_dbm": [-100.0 - (h("fake-rp", *k, i) % 900) / 100 for i in range(4)],
        "sinr_mean_db": [5.0 + (h("fake-sm", *k, i) % 500) / 100 for i in range(4)],
        "sinr_p10_db": [-2.0 + (h("fake-sp", *k, i) % 400) / 100 for i in range(4)],
        "alarm": alarm, "coverage_ratio": 0.71, "sla_ratio": 0.88, "hour": 9.0,
    }
    for i in range(4):
        if alarm[i]:
            state["ue_count"][i] = 0.0
    cands = [x for x in candidates() if x.startswith(f"cell={c};")]
    change = cands[h("fake-cand", *k) % len(cands)]
    return {"state": state, "noop_kpi": 1.59, "oracle_change": change, "oracle_gain": 0.05, "rule_change": "",
            "rule_gain": 0.0, "constant_gain": 0.01}


def failing_reference(spec: dict[str, Any]) -> dict[str, Any] | None:
    if spec["seed"] % 2:
        raise RuntimeError("synthetic labeller failure")
    return fake_reference(spec)


MINI_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'assistant' and loop.last -%}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m.content + '<|im_end|>\\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}{{- '<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}"
    "{%- endif -%}"
)


class FakeTokenizer:
    SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    _SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")

    def __init__(self, template: str = MINI_TEMPLATE):
        from t1.sft.chat import jinja_renderer

        self._render = jinja_renderer(template)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids: list[int] = []
        for part in self._SPLIT.split(text):
            if part in self.SPECIAL:
                ids.append(self.convert_tokens_to_ids(part))
            else:
                ids += [ord(ch) for ch in part]
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        return 200_000 + self.SPECIAL.index(token)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw) -> str:
        return self._render(messages, add_generation_prompt=add_generation_prompt, **kw)
