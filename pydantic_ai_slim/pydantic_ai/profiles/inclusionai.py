from __future__ import annotations as _annotations

from . import ModelProfile


def inclusionai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for an InclusionAI Ling model, matched by its published id.

    Ling 3 (`ling-3.0-flash`, `ling-3.1-flash`, ...) thinks by default and can be turned off. Other ids return `None`.
    """
    if model_name.lower().startswith('ling-3'):
        return ModelProfile(supports_thinking=True, thinking_enabled_by_default=True)
    return None
