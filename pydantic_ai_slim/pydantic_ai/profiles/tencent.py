from __future__ import annotations as _annotations

from . import ModelProfile


def tencent_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for a Tencent Hy model, matched by its published id.

    Hy3 and Hy4 (`hy4`, `hy4-flash`, ...) think by default and can be turned off. Other Tencent models, including
    ones that don't reason such as the `hy-mt2` translation models, return `None`.
    """
    if model_name.lower().startswith(('hy3', 'hy4')):
        return ModelProfile(supports_thinking=True, thinking_enabled_by_default=True)
    return None
