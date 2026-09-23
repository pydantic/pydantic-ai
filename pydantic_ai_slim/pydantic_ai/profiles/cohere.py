from __future__ import annotations as _annotations

from . import ModelProfile


def cohere_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Cohere model."""
    is_reasoning = 'reasoning' in model_name
    # Cohere's chat API takes text only.
    return ModelProfile(
        supports_thinking=is_reasoning,
        thinking_always_enabled=is_reasoning,
        supports_image_input=False,
        supports_document_input=False,
    )
