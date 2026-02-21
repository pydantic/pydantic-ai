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


# Models that support reasoning on Groq
_GROQ_REASONING_MODELS = (
    'deepseek-r1',
    'qwen-qwq',
    'qwq',
)


def groq_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Groq model."""
    model_lower = model_name.lower()

    # Check for reasoning model support
    is_reasoning = any(name in model_lower for name in _GROQ_REASONING_MODELS)

    return GroqModelProfile(
        groq_always_has_web_search_builtin_tool=model_name.startswith('compound-'),
        supports_thinking=is_reasoning,
        # Groq's reasoning models (DeepSeek R1, QwQ) are "thinking-only mode" - cannot disable thinking
        thinking_always_enabled=is_reasoning,
    )
