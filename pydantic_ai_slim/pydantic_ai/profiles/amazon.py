from __future__ import annotations as _annotations

from . import InlineDefsJsonSchemaTransformer, ModelProfile


def amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model."""
    # Nova Micro is the text-only model in the family, so it is the one Nova that reads no video; Lite, Pro and
    # Premier all do. https://docs.aws.amazon.com/nova/latest/userguide/modalities.html
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        supports_video_input='nova' in model_name and 'nova-micro' not in model_name,
    )
