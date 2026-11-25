"""What a model is sent for a history, for harness tests that check how a capability part reaches the model."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

_MODEL = FunctionModel(
    lambda messages, info: ModelResponse(parts=[]), profile=ModelProfile(supports_inline_system_prompts=True)
)


def sent_messages(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """`messages` as a model with inline system prompts is sent them, each capability part rendered as plain parts."""
    return _MODEL.prepare_messages(list(messages))
