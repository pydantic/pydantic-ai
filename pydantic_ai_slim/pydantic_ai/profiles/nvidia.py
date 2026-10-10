from __future__ import annotations as _annotations

from . import ModelProfile
from .meta import meta_model_profile


def nvidia_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an NVIDIA model, such as Nemotron.

    The Llama Nemotron models (`llama-3.1-nemotron-70b-instruct`, `llama-3.1-nemotron-ultra-253b-v1`, ...) are
    fine-tuned from Meta Llama, so they share the Meta model profile.
    """
    if model_name.lower().startswith('llama'):
        return meta_model_profile(model_name)
    return None
