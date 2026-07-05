from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import OpenAIModelProfile


def nvidia_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an NVIDIA model."""
    # Nemotron servings on OpenAI-compatible providers (e.g. Together, reached directly or
    # via OpenRouter) answer `tool_choice: 'required'` with the tool-call JSON rendered as
    # content text (a `[{"name": ..., "parameters": ...}]` array) that is never mapped back
    # into `tool_calls`, so forced tool use fails output validation. The same servings
    # tool-call correctly under `tool_choice: 'auto'`.
    return OpenAIModelProfile(openai_supports_tool_choice_required=False)
