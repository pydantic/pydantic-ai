from __future__ import annotations as _annotations

from dataclasses import dataclass

from . import ModelProfile


@dataclass(kw_only=True)
class GroqModelProfile(ModelProfile):
    """Profile for models used with GroqModel.

    ALL FIELDS MUST BE `groq_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    groq_always_has_web_search_builtin_tool: bool = False
    """Whether the model always has the web search built-in tool available."""


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
    return GroqModelProfile(
        groq_always_has_web_search_builtin_tool=model_name.startswith('compound-'),
        supports_thinking=is_reasoning_model,
        # qwen3 can disable reasoning with reasoning_effort='none'; legacy models can't
        thinking_always_enabled=is_reasoning_model and not model_name.startswith('qwen/qwen3'),
    )
