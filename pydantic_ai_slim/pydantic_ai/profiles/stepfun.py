from __future__ import annotations as _annotations

from . import ModelProfile


def stepfun_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for a StepFun Step model, matched by its published id.

    Step 5 (`step-5-preview`) and Step 3.7 (`step-3.7-flash`) always think. Other ids return `None`.
    """
    if model_name.lower().startswith(('step-5', 'step-3.7')):
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)
    return None
