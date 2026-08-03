from __future__ import annotations as _annotations

from typing import Literal

from ..settings import ThinkingEffort
from . import ModelProfile

GROQ_GPT_OSS_REASONING_EFFORT_MAP: dict[ThinkingEffort, Literal['low', 'medium', 'high']] = {
    'minimal': 'low',
    'low': 'low',
    'medium': 'medium',
    'high': 'high',
    'xhigh': 'high',
}
"""Maps unified thinking effort levels to the graded `reasoning_effort` values the gpt-oss family accepts.

gpt-oss only accepts `low`/`medium`/`high` (not `none`/`default`), so `minimal` folds into `low` and
`xhigh` into `high`. `thinking=True` (bare enable) maps to `medium` in `GroqModel`, mirroring the neutral
default other providers use. See [the Groq docs](https://console.groq.com/docs/reasoning#reasoning-effort).
"""


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

    groq_supports_graded_reasoning_effort: bool
    """Whether the model accepts graded `reasoning_effort` values (`low`/`medium`/`high`). Default: `False`.

    Only the gpt-oss family supports this; unified `thinking` levels map to those values via
    [`GROQ_GPT_OSS_REASONING_EFFORT_MAP`][pydantic_ai.profiles.groq.GROQ_GPT_OSS_REASONING_EFFORT_MAP].
    The qwen3 family instead only accepts `none`/`default` (see `groq_supports_reasoning_disable`).
    """


def groq_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Groq model."""
    # Current and legacy reasoning models on Groq
    is_reasoning_model = any(
        model_name.startswith(p)
        for p in (
            'openai/gpt-oss',  # graded reasoning_effort (low/medium/high), always-on
            'qwen/qwen3',  # current: qwen/qwen3-32b
            'qwen-qwq',  # legacy (deprecated)
            'deepseek-r1',  # legacy (deprecated)
            'llama-4-maverick',  # legacy (deprecated)
        )
    )
    is_qwen3 = model_name.startswith('qwen/qwen3')
    is_gpt_oss = model_name.startswith('openai/gpt-oss')
    return GroqModelProfile(
        groq_always_has_web_search_builtin_tool=model_name.startswith('compound-'),
        supports_thinking=is_reasoning_model,
        # qwen3 can disable reasoning with reasoning_effort='none'; gpt-oss and legacy models can't
        thinking_always_enabled=is_reasoning_model and not is_qwen3,
        groq_supports_reasoning_disable=is_qwen3,
        groq_supports_graded_reasoning_effort=is_gpt_oss,
    )
