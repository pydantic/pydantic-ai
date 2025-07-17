"""MoonshotAI model profile.

This profile is intentionally minimal since the MoonshotAI platform exposes
OpenAI-compatible models.  Custom behaviour can be added here in the future
should Moonshot-specific quirks arise.
"""

from __future__ import annotations as _annotations

from . import ModelProfile


def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Return the profile for the given MoonshotAI *model_name*.

    At the time of writing, MoonshotAI follows the OpenAI chat-completions
    specification closely, so we don't need any adjustments beyond the
    defaults.  A placeholder function is provided for symmetry with other
    providers and to allow future customisation without breaking imports.
    """
    return None
