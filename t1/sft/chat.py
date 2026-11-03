"""Chat-template rendering and the supervised span for SFT, thinking OFF.

The eval harness calls the model through the base model's own chat template (Qwen3 family) with
enable_thinking=False, so its prompt ends "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n".
Training renders the whole conversation with the same template; for the last assistant turn the
template writes "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{answer}<|im_end|>\\n".

Supervised span (decided 2 Nov: "an empty <think> block in the target as the template renders
it"): everything after "<|im_start|>assistant\\n", i.e. the empty think block, the JSON answer and
<|im_end|>. The model learns to stop at <|im_end|>, and to write the empty block itself if it is
ever called with thinking on. The newline the template adds after <|im_end|> is not trained.

Invariants checked for every example (ChatFormatError otherwise):
  1. P_off = render(system+user, add_generation_prompt, enable_thinking=False) is a prefix of the
     full text F: training text = exactly what the harness sends + the answer;
  2. P_on = render(system+user, add_generation_prompt) is a prefix of P_off, and
     F[len(P_on):] == "<think>\\n\\n</think>\\n\\n" + answer + "<|im_end|>\\n";
  3. token ids of P_on are a prefix of the token ids of F (no token merges across the boundary).
"""

from __future__ import annotations

from typing import Any, Callable

THINK_EMPTY = "<think>\n\n</think>\n\n"
END = "<|im_end|>"

Render = Callable[..., str]


class ChatFormatError(ValueError):
    """The template does not render the way training and the eval harness both assume."""


def jinja_renderer(template: str, **extra: Any) -> Render:
    """A renderer with the settings transformers uses for chat templates (sandboxed Jinja,
    trim_blocks, lstrip_blocks, tojson, raise_exception), for tests without transformers."""
    import json

    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg: str) -> None:
        raise TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False: json.dumps(x, indent=indent,
                                                                                ensure_ascii=ensure_ascii)
    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(template)

    def render(messages: list[dict[str, str]], add_generation_prompt: bool = False, **kw: Any) -> str:
        return tpl.render(messages=messages, add_generation_prompt=add_generation_prompt, **extra, **kw)

    return render


def tokenizer_renderer(tokenizer: Any) -> Render:
    def render(messages: list[dict[str, str]], add_generation_prompt: bool = False, **kw: Any) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                             **kw)

    return render


def split_text(render: Render, messages: list[dict[str, str]]) -> tuple[str, str]:
    """(prompt part, supervised part) of one system/user/assistant example, invariants 1-2 checked."""
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ChatFormatError("an example is exactly system, user, assistant")
    answer = messages[2]["content"]
    full = render(messages, enable_thinking=False)
    p_off = render(messages[:2], add_generation_prompt=True, enable_thinking=False)
    p_on = render(messages[:2], add_generation_prompt=True)
    if not full.startswith(p_off):
        raise ChatFormatError("the thinking-off prompt is not a prefix of the training text")
    if not p_off.startswith(p_on) or p_off[len(p_on):] != THINK_EMPTY:
        raise ChatFormatError(f"thinking-off prompt is not the generation prompt + the empty think block: "
                              f"{p_off[len(p_on):]!r}")
    target = full[len(p_on):]
    want = THINK_EMPTY + answer + END + "\n"
    if target != want:
        raise ChatFormatError(f"assistant turn renders as {target[:80]!r}..., expected {want[:80]!r}...")
    return p_on, target[: -1]  # drop the newline after <|im_end|>


def encode(tokenizer: Any, messages: list[dict[str, str]], render: Render | None = None) -> dict[str, Any]:
    """input_ids and labels (-100 on the prompt) for one example; invariant 3 checked."""
    render = render or tokenizer_renderer(tokenizer)
    prompt, target = split_text(render, messages)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(prompt + target, add_special_tokens=False)["input_ids"]
    if f_ids[: len(p_ids)] != p_ids:
        raise ChatFormatError("prompt tokens are not a prefix of the full tokens (merge across the boundary)")
    end_id = tokenizer.convert_tokens_to_ids(END)
    if f_ids[-1] != end_id:
        raise ChatFormatError(f"the last supervised token is {f_ids[-1]}, not {END} ({end_id})")
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels, "n_prompt": len(p_ids), "n_target": len(f_ids) - len(p_ids)}
