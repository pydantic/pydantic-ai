from __future__ import annotations as _annotations

from . import InlineDefsJsonSchemaTransformer, ModelProfile


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)
