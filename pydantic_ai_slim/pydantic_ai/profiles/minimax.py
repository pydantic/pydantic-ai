from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import OpenAIJsonSchemaTransformer


def minimax_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MiniMax model.

    MiniMax models do not support JSON schema or JSON object output modes.
    """
    return ModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_schema_output=False,
        supports_json_object_output=False,
    )
