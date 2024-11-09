"""Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the
specific LLM being used.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from functools import cache
from typing import TYPE_CHECKING, Protocol, Union

from httpx import AsyncClient as AsyncHTTPClient

from ..messages import LLMMessage, LLMToolCalls, Message

if TYPE_CHECKING:
    from .._utils import ObjectJsonSchema
    from ..agent import KnownModelName
    from ..result import Cost


class Model(ABC):
    """Abstract class for a model."""

    @abstractmethod
    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> AgentModel:
        """Create an agent model.

        Args:
            retrievers: The retrievers available to the agent.
            allow_text_result: Whether a plain text final response/result is permitted.
            result_tools: Tool definitions for the final result tool(s), if any.

        Returns:
            An agent model.
        """
        raise NotImplementedError()

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()


class AgentModel(ABC):
    """Model configured for a specific agent."""

    @abstractmethod
    async def request(self, messages: list[Message]) -> tuple[LLMMessage, Cost]:
        """Make a request to the model."""
        raise NotImplementedError()

    @asynccontextmanager
    async def request_stream(self, messages: list[Message]) -> AsyncIterator[EitherStreamedResponse]:
        """Make a request to the model and return a streaming response."""
        raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover


class StreamTextResponse(ABC):
    """Streamed response from an LLM when returning text."""

    def __aiter__(self) -> AsyncIterator[str]:
        """Stream the response as an async iterable of string chunks."""
        return self

    @abstractmethod
    async def __anext__(self) -> str:
        """Process the next chunk of the response, and return the string delta."""
        raise NotImplementedError()

    @abstractmethod
    def cost(self) -> Cost:
        """Return the cost of the request.

        NOTE: this won't return the ful cost until the stream is finished.
        """
        raise NotImplementedError()


class StreamToolCallResponse(ABC):
    """Streamed response from an LLM when calling a tool."""

    def __aiter__(self) -> AsyncIterator[None]:
        """Stream the response as an async iterable, building up the tool call as it goes.

        This is an async iterator that yields `None` to avoid doing the work of building the final tool call when
        it will often be thrown away.
        """
        return self

    @abstractmethod
    async def __anext__(self) -> None:
        """Process the next chunk of the response, see above for why this returns `None`."""
        raise NotImplementedError()

    @abstractmethod
    def get(self) -> LLMToolCalls:
        """Get the `LLMToolCalls` at this point.

        The `LLMToolCalls` may or may not be complete, depending on whether the stream is finished.
        """
        raise NotImplementedError()

    @abstractmethod
    def cost(self) -> Cost:
        """Get the cost of the request.

        NOTE: this won't return the full cost until the stream is finished.
        """
        raise NotImplementedError()


EitherStreamedResponse = Union[StreamTextResponse, StreamToolCallResponse]


def infer_model(model: Model | KnownModelName) -> Model:
    """Infer the model from the name."""
    if isinstance(model, Model):
        return model
    elif model.startswith('openai:'):
        from .openai import OpenAIModel

        return OpenAIModel(model[7:])  # pyright: ignore[reportArgumentType]
    elif model.startswith('gemini'):
        from .gemini import GeminiModel

        # noinspection PyTypeChecker
        return GeminiModel(model)  # pyright: ignore[reportArgumentType]
    else:
        from ..exceptions import UserError

        raise UserError(f'Unknown model: {model}')


class AbstractToolDefinition(Protocol):
    """Abstract definition of a function/tool.

    These are generally retrievers, but can also include the response function if one exists.
    """

    name: str
    description: str
    json_schema: ObjectJsonSchema
    # can only be true for the final result tool
    outer_typed_dict_key: str | None


@cache
def cached_async_http_client() -> AsyncHTTPClient:
    """Cached HTTPX async client so multiple agents and calls can share the same client.

    There are good reasons why in production you should use a `httpx.AsyncClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful not to, this allows multiple Agents to use a single client.
    """
    return AsyncHTTPClient(timeout=30)
