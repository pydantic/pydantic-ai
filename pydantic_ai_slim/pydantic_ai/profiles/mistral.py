from __future__ import annotations as _annotations

from .._file_input import KNOWN_IMAGE_MEDIA_TYPES
from . import InlineDefsJsonSchemaTransformer, ModelProfile, merge_profile

_MISTRAL_IMAGE_MEDIA_TYPES = KNOWN_IMAGE_MEDIA_TYPES | {'image/avif', 'image/bmp', 'image/heic', 'image/tiff'}
"""Images Mistral's vision models read, as checked live with `mistral-medium-latest` (2026-11-04)."""


def mistral_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Mistral model."""
    # Mistral's tool-calling chat template uses tool call IDs of exactly 9 alphanumerics, and a model that
    # validates them rejects any other ID with a 400: `voxtral-small` on La Plateforme does, and so do
    # self-hosted deployments of the template. Most models don't check, but IDs that already fit are sent
    # unchanged, so this is set for every Mistral model rather than tracking which deployments validate.
    # See https://github.com/pydantic/pydantic-ai/issues/9096.
    profile = ModelProfile(tool_call_id_length=9, supported_image_media_types=_MISTRAL_IMAGE_MEDIA_TYPES)
    if model_name == 'zai-glm-5-3' or model_name.startswith('zai-glm-5-3-'):
        # GLM-5.3 stringifies nested tool arguments when the field schema is a `$ref` into `$defs`.
        # See https://github.com/pydantic/pydantic-ai/issues/10243 and `zai_model_profile`.
        profile = merge_profile(profile, ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer))
    if model_name.startswith('magistral'):
        return merge_profile(profile, ModelProfile(supports_thinking=True, thinking_always_enabled=True))
    return profile
