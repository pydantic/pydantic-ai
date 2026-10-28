"""Seeing through model wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .wrapper import WrapperModel

if TYPE_CHECKING:
    from . import Model

__all__ = ('unwrap_model',)


def unwrap_model(model: Model) -> Model:
    """Strip [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel] layers to the underlying model.

    For telling which model a request is really for, whatever wraps it: an
    [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] capability's
    [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel], or a durability wrapper. Durability
    capabilities compare the unwrapped run model with the one they were built for, by identity, to detect a model
    passed at run time: `model_id` strings are too coarse, as two instances on different providers can share one.
    `OpenAICompaction` checks that the unwrapped model is an `OpenAIResponsesModel`.
    """
    while isinstance(model, WrapperModel):
        model = model.wrapped
    return model
