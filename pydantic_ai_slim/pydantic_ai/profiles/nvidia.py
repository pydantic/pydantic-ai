from __future__ import annotations as _annotations

from . import ModelProfile

_HYBRID_THINKING_PREFIXES = ('nemotron-3-super', 'nemotron-3-ultra')
_OPT_IN_THINKING_PREFIXES = ('nemotron-3.5-super', 'nemotron-3.5-lightning')


def nvidia_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for an NVIDIA Nemotron model, matched by its published id.

    The function takes the id gateways publish (`nemotron-3-super-120b-a12b`) or NVIDIA's Hugging Face repo name
    (`NVIDIA-Nemotron-3-Super-120B-A12B-FP8`). Nemotron 3 Super and Ultra think by default and can be turned off;
    Nemotron 3.5 Super and Lightning think when asked to. Other NVIDIA models, including ones that don't reason such
    as `nemotron-mini-4b-instruct`, return `None`.
    """
    model_name = model_name.lower().removeprefix('nvidia-')
    if model_name.startswith(_HYBRID_THINKING_PREFIXES):
        return ModelProfile(supports_thinking=True, thinking_enabled_by_default=True)
    if model_name.startswith(_OPT_IN_THINKING_PREFIXES):
        return ModelProfile(supports_thinking=True)
    return None
