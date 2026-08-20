from __future__ import annotations as _annotations

from . import ModelProfile


class MistralModelProfile(ModelProfile, total=False):
    """Profile for models used with `MistralModel`.

    ALL FIELDS MUST BE `mistral_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    mistral_supports_media_in_tool_returns: bool
    """Whether the model accepts a content-chunk array as `ToolMessage.content`. Default: `True`.

    Mistral's OpenAPI schema types `ToolMessage.content` as `string | null | ContentChunk[]` with no
    model discriminator, so this defaults on for every model; the chunk array is verified live
    against `mistral-medium-latest`. Turning it off sends the media as prompt content behind the
    provenance marker instead, which is how providers without native tool-result media carry it —
    note that this is a different rendering, not a way to restore pre-provenance output.

    A `bool` rather than the allow-list its Google and Bedrock counterparts use
    ([`google_supported_mime_types_in_tool_returns`][pydantic_ai.profiles.google.GoogleModelProfile.google_supported_mime_types_in_tool_returns],
    `bedrock_supported_media_kinds_in_tool_returns`): those enumerate a per-model split their
    providers publish, and Mistral publishes none, so there is nothing to enumerate.
    """


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    if model_name.startswith('magistral'):
        return MistralModelProfile(supports_thinking=True, thinking_always_enabled=True)
    return None
