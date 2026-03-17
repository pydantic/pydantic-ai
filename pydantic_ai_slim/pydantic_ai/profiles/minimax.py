from __future__ import annotations as _annotations

from . import ModelProfile


def minimax_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MiniMax model.

    MiniMax models do not support JSON schema or JSON object output modes.
    """
    return ModelProfile(
        supports_json_schema_output=False,
        supports_json_object_output=False,
    )
