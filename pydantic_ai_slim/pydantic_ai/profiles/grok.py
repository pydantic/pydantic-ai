from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile


def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer, openai_supports_strict_tool_definition=False
    )
