from __future__ import annotations as _annotations

from . import ModelProfile


class MistralModelProfile(ModelProfile, total=False):
    """Profile for models used with `MistralModel`.

    ALL FIELDS MUST BE `mistral_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    mistral_supports_media_in_tool_returns: bool
    """Whether the model accepts a content-chunk array as `ToolMessage.content`. Default: `True`.

    Mistral's API schema accepts either a string or a chunk array for every model, and the docs
    state no per-model constraint, so this defaults on; it is verified live against
    `mistral-medium-latest`. Set it to `False` for a model found to reject chunked tool content,
    which sends the media as prompt content behind the provenance marker instead.
    """


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    if model_name.startswith('magistral'):
        return MistralModelProfile(supports_thinking=True, thinking_always_enabled=True)
    return None
