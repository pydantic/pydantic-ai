"""Logic related to making requests to an LLM.

The aim here is to make a common interface
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from ..messages import LLMMessage, Message

if TYPE_CHECKING:
    from .._utils import ObjectJsonSchema
    from ..agent import KnownModelName


class Model(ABC):
    """Abstract class for a model."""

    @abstractmethod
    def agent_model(self, allow_plain_message: bool, tools: list[AbstractToolDefinition]) -> AgentModel:
        """Create an agent model."""
        raise NotImplementedError()


class AgentModel(ABC):
    """Model set up for a specific agent."""

    @abstractmethod
    async def request(self, messages: list[Message]) -> LLMMessage:
        """Request a response from the model."""
        raise NotImplementedError()

    # TODO streamed response


def infer_model(model: Model | KnownModelName) -> Model:
    """Infer the model from the name."""
    if isinstance(model, Model):
        return model
    elif model.startswith('openai:'):
        from .openai import OpenAIModel

        return OpenAIModel(model[7:])  # pyright: ignore[reportArgumentType]
    else:
        raise TypeError(f'Invalid model: {model}')


class AbstractToolDefinition(Protocol):
    """Abstract definition of a function/tool."""

    name: str
    description: str
    json_schema: ObjectJsonSchema
