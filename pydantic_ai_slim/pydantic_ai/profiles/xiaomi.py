from __future__ import annotations as _annotations

from . import ModelProfile


def xiaomi_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for a Xiaomi MiMo model, matched by its published id.

    MiMo V2.5 and V2.6 (`mimo-v2.5-pro`, `mimo-v2.6-flash`, ...) think when asked to. Other ids return `None`.
    """
    if model_name.lower().startswith(('mimo-v2.5', 'mimo-v2.6')):
        return ModelProfile(supports_thinking=True)
    return None
