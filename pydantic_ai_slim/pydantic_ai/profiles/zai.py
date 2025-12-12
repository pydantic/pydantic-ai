from __future__ import annotations as _annotations

from . import ModelProfile


def zai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for ZAI models.

    Currently returns None as ZAI model-specific properties are handled at the provider level.
    """
    return None
