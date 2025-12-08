from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import OpenAIJsonSchemaTransformer


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    return ModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )
