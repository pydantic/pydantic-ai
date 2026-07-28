from __future__ import annotations as _annotations

from typing_extensions import TypedDict

from .input import EmbeddingModality

__all__ = ('EmbeddingModelProfile', 'DEFAULT_EMBEDDING_PROFILE')


class EmbeddingModelProfile(TypedDict, total=False):
    """What an embedding model can accept, independent of the model and provider classes used.

    Support is per model rather than per provider: one class often covers models that differ here,
    so a profile is keyed by model name. Checked in
    [`prepare_embed()`][pydantic_ai.embeddings.EmbeddingModel.prepare_embed], so an input a model
    can't take raises a [`UserError`][pydantic_ai.exceptions.UserError] instead of a provider error.

    All fields are optional; absent keys mean "use the documented default".
    """

    supported_modalities: frozenset[EmbeddingModality]
    """The modalities this model can embed. Default: `{'text'}`."""

    supports_grouped_inputs: bool
    """Whether the model can embed an [`EmbeddingGroup`][pydantic_ai.embeddings.EmbeddingGroup] of
    several parts into a single vector. Default: `False`.

    Independent of `supported_modalities`: a model can accept every modality and still embed only one
    part per request, as `amazon.nova-2-multimodal-embeddings-v1:0` does. A group holding exactly one
    part is accepted either way, as there is nothing to combine.
    """


DEFAULT_EMBEDDING_PROFILE: EmbeddingModelProfile = {
    'supported_modalities': frozenset({'text'}),
    'supports_grouped_inputs': False,
}
"""Fully populated default `EmbeddingModelProfile`: text only, one part per input."""
