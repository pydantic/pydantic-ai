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
    is_reasoning_model = any(model_name.startswith(p) for p in ('qwen-qwq', 'deepseek-r1', 'llama-4-maverick'))
    return GroqModelProfile(
        groq_always_has_web_search_builtin_tool=model_name.startswith('compound-'),
        supports_thinking=is_reasoning_model,
        thinking_always_enabled=is_reasoning_model,
    )
