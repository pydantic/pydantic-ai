"""Logic related to making requests to an LLM.

The aim here is to make a common interface
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, Protocol

from httpx import AsyncClient as AsyncHTTPClient

from ..messages import LLMMessage, Message

if TYPE_CHECKING:
    from .._utils import ObjectJsonSchema
    from ..agent import KnownModelName


class Model(ABC):
    """Abstract class for a model."""

    @abstractmethod
    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tool: AbstractToolDefinition | None,
    ) -> AgentModel:
        """Create an agent model.

        Args:
            retrievers: The retrievers available to the agent.
            allow_text_result: Whether a plain text final response/result is permitted.
            result_tool: Tool definition for the final result tool, if any.

        Returns:
            An agent model.
        """
        raise NotImplementedError()


class AgentModel(ABC):
    """Model configured for a specific agent."""

    @abstractmethod
    async def request(self, messages: list[Message]) -> LLMMessage:
        """Request a response from the model."""
        raise NotImplementedError()

    # TODO streamed response
    # TODO support for non JSON tool calls


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
    """Abstract definition of a function/tool.

    These are generally retrievers, but can also include the response function if one exists.
    """

    name: str
    description: str
    json_schema: ObjectJsonSchema


@cache
def cached_async_http_client() -> AsyncHTTPClient:
    """
    There are good reasons why in production you should use a `AsyncHTTPClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful, this allows multiple Agents to use a single client.
    """
    return AsyncHTTPClient()
