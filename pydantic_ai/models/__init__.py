"""Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the
specific LLM being used.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from functools import cache
from typing import TYPE_CHECKING, Literal, Protocol, Union

import httpx

from ..messages import Message, ModelAnyResponse, ModelStructuredResponse

if TYPE_CHECKING:
    from .._utils import ObjectJsonSchema
    from ..result import Cost


KnownModelName = Literal[
    'openai:gpt-4o',
    'openai:gpt-4o-mini',
    'openai:gpt-4-turbo',
    'openai:gpt-4',
    'openai:o1-preview',
    'openai:o1-mini',
    'openai:gpt-3.5-turbo',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'test',
]
"""Known model names that can be used with the `model` parameter of [`Agent`][pydantic_ai.Agent].

`KnownModelName` is provided as a concise way to specify a model.
"""


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
    async def request(self, messages: list[Message]) -> tuple[ModelAnyResponse, Cost]:
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

    def __aiter__(self) -> AsyncIterator[None]:
        """Stream the response as an async iterable, building up the text as it goes.

        This is an async iterator that yields `None` to avoid doing the work of validating the input and
        extracting the text field when it will often be thrown away.
        """
        return self

    @abstractmethod
    async def __anext__(self) -> None:
        """Process the next chunk of the response, see above for why this returns `None`."""
        raise NotImplementedError()

    @abstractmethod
    def get(self, *, final: bool = False) -> Iterable[str]:
        """Returns an iterable of text since the last call to `get()` — e.g. the text delta.

        Args:
            final: If True, this is the final call, after iteration is complete, the response should be fully validated
                and all text extracted.
        """
        raise NotImplementedError()

    @abstractmethod
    def cost(self) -> Cost:
        """Return the cost of the request.

        NOTE: this won't return the ful cost until the stream is finished.
        """
        raise NotImplementedError()

    @abstractmethod
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        raise NotImplementedError()


class StreamStructuredResponse(ABC):
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
    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        """Get the `ModelStructuredResponse` at this point.

        The `ModelStructuredResponse` may or may not be complete, depending on whether the stream is finished.

        Args:
            final: If True, this is the final call, after iteration is complete, the response should be fully validated.
        """
        raise NotImplementedError()

    @abstractmethod
    def cost(self) -> Cost:
        """Get the cost of the request.

        NOTE: this won't return the full cost until the stream is finished.
        """
        raise NotImplementedError()

    @abstractmethod
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        raise NotImplementedError()


EitherStreamedResponse = Union[StreamTextResponse, StreamStructuredResponse]


ALLOW_MODEL_REQUESTS = True
"""Whether to allow requests to models.

This global setting allows you to disable request to most models, e.g. to make sure you don't accidentally
make costly requests to a model during tests.

The testing models [`TestModel`][pydantic_ai.models.test.TestModel] and
[`FunctionModel`][pydantic_ai.models.function.FunctionModel] are no affected by this setting.
"""


def check_allow_model_requests() -> None:
    """Check if model requests are allowed.

    If you're defining your own models that have cost or latency associated with their use, you should call this in
    [`Model.agent_model`][pydantic_ai.models.Model.agent_model].

    Raises:
        RuntimeError: If model requests are not allowed.
    """
    if not ALLOW_MODEL_REQUESTS:
        raise RuntimeError('Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False')


@contextmanager
def override_allow_model_requests(allow_model_requests: bool) -> Iterator[None]:
    """Context manager to temporarily override [`ALLOW_MODEL_REQUESTS`][pydantic_ai.models.ALLOW_MODEL_REQUESTS].

    Args:
        allow_model_requests: Whether to allow model requests within the context.
    """
    global ALLOW_MODEL_REQUESTS
    old_value = ALLOW_MODEL_REQUESTS
    ALLOW_MODEL_REQUESTS = allow_model_requests  # pyright: ignore[reportConstantRedefinition]
    try:
        yield
    finally:
        ALLOW_MODEL_REQUESTS = old_value  # pyright: ignore[reportConstantRedefinition]


def infer_model(model: Model | KnownModelName) -> Model:
    """Infer the model from the name."""
    if isinstance(model, Model):
        return model
    elif model == 'test':
        from .test import TestModel

        return TestModel()
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

    This is used for both retrievers and result tools.
    """

    name: str
    """The name of the tool."""
    description: str
    """The description of the tool."""
    json_schema: ObjectJsonSchema
    """The JSON schema for the tool's arguments."""
    outer_typed_dict_key: str | None
    """The key in the outer [TypedDict] that wraps a result tool.

    This will only be set for result tools which don't have an `object` JSON schema.
    """


@cache
def cached_async_http_client(timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    """Cached HTTPX async client so multiple agents and calls can share the same client.

    There are good reasons why in production you should use a `httpx.AsyncClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful not to, this allows multiple Agents to use a single client.

    The default timeouts match those of OpenAI,
    see <https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9>.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


@cache
def get_user_agent() -> str:
    """Get the user agent string for the HTTP client."""
    from .. import __version__

    return f'pydantic-ai/{__version__}'
