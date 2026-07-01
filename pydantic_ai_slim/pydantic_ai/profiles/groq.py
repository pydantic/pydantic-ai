from __future__ import annotations as _annotations

from typing import Literal

from . import ModelProfile


class GroqModelProfile(ModelProfile, total=False):
    """Profile for models used with GroqModel.

    ALL FIELDS MUST BE `groq_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    groq_always_has_web_search_builtin_tool: bool
    """Whether the model always has the web search built-in tool available. Default: `False`."""

    groq_supports_reasoning_disable: bool
    """Whether `thinking=False` truly disables reasoning via `reasoning_effort='none'`. Default: `False`.

    Only the qwen3 family supports this; other Groq reasoning models can at most suppress reasoning
    *output* via `reasoning_format='hidden'` while still reasoning internally.
    """

    groq_send_back_thinking_parts: Literal['auto', 'tags']
    """How thinking parts in history are sent back to the model. Default: `'auto'`.

    Groq has no native thinking round-trip, so this governs all `ThinkingPart`s:

    * `'auto'` (default): they are dropped. Groq does not re-absorb `<think>` tags from history, so
    re-rendering them as text teaches the model to mimic the format in its user-visible answers, leaking
    reasoning to end users. (Whether the leak materializes also depends on `groq_reasoning_format`: `'raw'`
    re-absorbs the tags, while `'parsed'`/`'hidden'` do not — dropping is safe for all of them.)
    * `'tags'`: they are re-rendered as text wrapped in the `thinking_tags`, the behavior from before this
    setting existed.

    See `AnthropicModelProfile.anthropic_send_back_thinking_parts` for more context."""


def groq_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Groq model."""
    # Current and legacy reasoning models on Groq
    is_reasoning_model = any(
        model_name.startswith(p)
        for p in (
            'qwen/qwen3',  # current: qwen/qwen3-32b
            'qwen-qwq',  # legacy (deprecated)
            'deepseek-r1',  # legacy (deprecated)
            'llama-4-maverick',  # legacy (deprecated)
        )
    )
    is_qwen3 = model_name.startswith('qwen/qwen3')
    return GroqModelProfile(
        groq_always_has_web_search_builtin_tool=model_name.startswith('compound-'),
        supports_thinking=is_reasoning_model,
        # qwen3 can disable reasoning with reasoning_effort='none'; legacy models can't
        thinking_always_enabled=is_reasoning_model and not is_qwen3,
        groq_supports_reasoning_disable=is_qwen3,
    )
