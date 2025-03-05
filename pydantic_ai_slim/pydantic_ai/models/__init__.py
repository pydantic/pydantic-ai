"""Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the
specific LLM being used.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from typing import TYPE_CHECKING

import httpx
from typing_extensions import Literal

from .._parts_manager import ModelResponsePartsManager
from ..exceptions import UserError
from ..messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from ..settings import ModelSettings
from ..usage import Usage

if TYPE_CHECKING:
    from ..tools import ToolDefinition


KnownModelName = Literal[
    'anthropic:claude-3-7-sonnet-latest',
    'anthropic:claude-3-5-haiku-latest',
    'anthropic:claude-3-5-sonnet-latest',
    'anthropic:claude-3-opus-latest',
    'claude-3-7-sonnet-latest',
    'claude-3-5-haiku-latest',
    'claude-3-5-sonnet-latest',
    'claude-3-opus-latest',
    'cohere:c4ai-aya-expanse-32b',
    'cohere:c4ai-aya-expanse-8b',
    'cohere:command',
    'cohere:command-light',
    'cohere:command-light-nightly',
    'cohere:command-nightly',
    'cohere:command-r',
    'cohere:command-r-03-2024',
    'cohere:command-r-08-2024',
    'cohere:command-r-plus',
    'cohere:command-r-plus-04-2024',
    'cohere:command-r-plus-08-2024',
    'cohere:command-r7b-12-2024',
    'deepseek:deepseek-chat',
    'deepseek:deepseek-reasoner',
    'google-gla:gemini-1.0-pro',
    'google-gla:gemini-1.5-flash',
    'google-gla:gemini-1.5-flash-8b',
    'google-gla:gemini-1.5-pro',
    'google-gla:gemini-2.0-flash-exp',
    'google-gla:gemini-2.0-flash-thinking-exp-01-21',
    'google-gla:gemini-exp-1206',
    'google-gla:gemini-2.0-flash',
    'google-gla:gemini-2.0-flash-lite-preview-02-05',
    'google-gla:gemini-2.0-pro-exp-02-05',
    'google-vertex:gemini-1.0-pro',
    'google-vertex:gemini-1.5-flash',
    'google-vertex:gemini-1.5-flash-8b',
    'google-vertex:gemini-1.5-pro',
    'google-vertex:gemini-2.0-flash-exp',
    'google-vertex:gemini-2.0-flash-thinking-exp-01-21',
    'google-vertex:gemini-exp-1206',
    'google-vertex:gemini-2.0-flash',
    'google-vertex:gemini-2.0-flash-lite-preview-02-05',
    'google-vertex:gemini-2.0-pro-exp-02-05',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-1106',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-16k-0613',
    'gpt-4',
    'gpt-4-0125-preview',
    'gpt-4-0314',
    'gpt-4-0613',
    'gpt-4-1106-preview',
    'gpt-4-32k',
    'gpt-4-32k-0314',
    'gpt-4-32k-0613',
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
    'gpt-4-turbo-preview',
    'gpt-4-vision-preview',
    'gpt-4.5-preview',
    'gpt-4.5-preview-2025-02-27',
    'gpt-4o',
    'gpt-4o-2024-05-13',
    'gpt-4o-2024-08-06',
    'gpt-4o-2024-11-20',
    'gpt-4o-audio-preview',
    'gpt-4o-audio-preview-2024-10-01',
    'gpt-4o-audio-preview-2024-12-17',
    'gpt-4o-mini',
    'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini-audio-preview',
    'gpt-4o-mini-audio-preview-2024-12-17',
    'groq:gemma2-9b-it',
    'groq:llama-3.1-8b-instant',
    'groq:llama-3.2-11b-vision-preview',
    'groq:llama-3.2-1b-preview',
    'groq:llama-3.2-3b-preview',
    'groq:llama-3.2-90b-vision-preview',
    'groq:llama-3.3-70b-specdec',
    'groq:llama-3.3-70b-versatile',
    'groq:llama3-70b-8192',
    'groq:llama3-8b-8192',
    'groq:mixtral-8x7b-32768',
    'mistral:codestral-latest',
    'mistral:mistral-large-latest',
    'mistral:mistral-moderation-latest',
    'mistral:mistral-small-latest',
    'o1',
    'o1-2024-12-17',
    'o1-mini',
    'o1-mini-2024-09-12',
    'o1-preview',
    'o1-preview-2024-09-12',
    'o3-mini',
    'o3-mini-2025-01-31',
    'openai:chatgpt-4o-latest',
    'openai:gpt-3.5-turbo',
    'openai:gpt-3.5-turbo-0125',
    'openai:gpt-3.5-turbo-0301',
    'openai:gpt-3.5-turbo-0613',
    'openai:gpt-3.5-turbo-1106',
    'openai:gpt-3.5-turbo-16k',
    'openai:gpt-3.5-turbo-16k-0613',
    'openai:gpt-4',
    'openai:gpt-4-0125-preview',
    'openai:gpt-4-0314',
    'openai:gpt-4-0613',
    'openai:gpt-4-1106-preview',
    'openai:gpt-4-32k',
    'openai:gpt-4-32k-0314',
    'openai:gpt-4-32k-0613',
    'openai:gpt-4-turbo',
    'openai:gpt-4-turbo-2024-04-09',
    'openai:gpt-4-turbo-preview',
    'openai:gpt-4-vision-preview',
    'openai:gpt-4.5-preview',
    'openai:gpt-4.5-preview-2025-02-27',
    'openai:gpt-4o',
    'openai:gpt-4o-2024-05-13',
    'openai:gpt-4o-2024-08-06',
    'openai:gpt-4o-2024-11-20',
    'openai:gpt-4o-audio-preview',
    'openai:gpt-4o-audio-preview-2024-10-01',
    'openai:gpt-4o-audio-preview-2024-12-17',
    'openai:gpt-4o-mini',
    'openai:gpt-4o-mini-2024-07-18',
    'openai:gpt-4o-mini-audio-preview',
    'openai:gpt-4o-mini-audio-preview-2024-12-17',
    'openai:o1',
    'openai:o1-2024-12-17',
    'openai:o1-mini',
    'openai:o1-mini-2024-09-12',
    'openai:o1-preview',
    'openai:o1-preview-2024-09-12',
    'openai:o3-mini',
    'openai:o3-mini-2025-01-31',
    'test',
]
"""Known model names that can be used with the `model` parameter of [`Agent`][pydantic_ai.Agent].

`KnownModelName` is provided as a concise way to specify a model.
"""


@dataclass
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and result handling."""

    function_tools: list[ToolDefinition]
    allow_text_result: bool
    result_tools: list[ToolDefinition]


class Model(ABC):
    """Abstract class for a model."""

    @abstractmethod
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the model."""
        raise NotImplementedError()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a request to the model and return a streaming response."""
        # This method is not required, but you need to implement it if you want to support streamed responses
        raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def system(self) -> str | None:
        """The system / model provider, ex: openai."""
        raise NotImplementedError()


@dataclass
class StreamedResponse(ABC):
    """Streamed response from an LLM when calling a tool."""

    _parts_manager: ModelResponsePartsManager = field(default_factory=ModelResponsePartsManager, init=False)
    _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
    _usage: Usage = field(default_factory=Usage, init=False)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream the response as an async iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
        if self._event_iterator is None:
            self._event_iterator = self._get_event_iterator()
        return self._event_iterator

    @abstractmethod
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        pydantic_ai-format events.

        It should use the `_parts_manager` to handle deltas, and should update the `_usage` attributes as it goes.
        """
        raise NotImplementedError()
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        """Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
        return ModelResponse(
            parts=self._parts_manager.get_parts(), model_name=self.model_name, timestamp=self.timestamp
        )

    def usage(self) -> Usage:
        """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
        return self._usage

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name of the response."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        raise NotImplementedError()


ALLOW_MODEL_REQUESTS = True
"""Whether to allow requests to models.

This global setting allows you to disable request to most models, e.g. to make sure you don't accidentally
make costly requests to a model during tests.

The testing models [`TestModel`][pydantic_ai.models.test.TestModel] and
[`FunctionModel`][pydantic_ai.models.function.FunctionModel] are no affected by this setting.
"""


def check_allow_model_requests() -> None:
    """Check if model requests are allowed.

    If you're defining your own models that have costs or latency associated with their use, you should call this in
    [`Model.request`][pydantic_ai.models.Model.request] and [`Model.request_stream`][pydantic_ai.models.Model.request_stream].

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

    try:
        provider, model_name = model.split(':')
    except ValueError:
        model_name = model
        # TODO(Marcelo): We should deprecate this way.
        if model_name.startswith(('gpt', 'o1', 'o3')):
            provider = 'openai'
        elif model_name.startswith('claude'):
            provider = 'anthropic'
        elif model_name.startswith('gemini'):
            provider = 'google-gla'
        else:
            raise UserError(f'Unknown model: {model}')

    if provider == 'vertexai':
        provider = 'google-vertex'

    if provider == 'cohere':
        from .cohere import CohereModel

        # TODO(Marcelo): Missing provider API.
        return CohereModel(model_name)
    elif provider in ('deepseek', 'openai'):
        from .openai import OpenAIModel

        return OpenAIModel(model_name, provider=provider)
    elif provider in ('google-gla', 'google-vertex'):
        from .gemini import GeminiModel

        return GeminiModel(model_name, provider=provider)
    elif provider == 'groq':
        from .groq import GroqModel

        # TODO(Marcelo): Missing provider API.
        return GroqModel(model_name)
    elif provider == 'mistral':
        from .mistral import MistralModel

        # TODO(Marcelo): Missing provider API.
        return MistralModel(model_name)
    elif provider == 'anthropic':
        from .anthropic import AnthropicModel

        # TODO(Marcelo): Missing provider API.
        return AnthropicModel(model_name)
    else:
        raise UserError(f'Unknown model: {model}')


def cached_async_http_client(timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    """Cached HTTPX async client so multiple agents and calls can share the same client.

    There are good reasons why in production you should use a `httpx.AsyncClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful not to, this allows multiple Agents to use a single client.

    The default timeouts match those of OpenAI,
    see <https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9>.
    """
    client = _cached_async_http_client(timeout=timeout, connect=connect)
    if client.is_closed:
        # This happens if the context manager is used, so we need to create a new client.
        _cached_async_http_client.cache_clear()
        client = _cached_async_http_client(timeout=timeout, connect=connect)
    return client


@cache
def _cached_async_http_client(timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


@cache
def get_user_agent() -> str:
    """Get the user agent string for the HTTP client."""
    from .. import __version__

    return f'pydantic-ai/{__version__}'
