from __future__ import annotations as _annotations

from . import InlineDefsJsonSchemaTransformer, ModelProfile


def amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model."""
    # Nova 2 models (nova-2-lite, nova-2-pro) support reasoning
    supports_thinking = 'nova-2' in model_name
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        supports_thinking=supports_thinking,
    )
