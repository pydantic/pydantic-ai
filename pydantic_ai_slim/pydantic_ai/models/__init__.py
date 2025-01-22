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
from typing import TYPE_CHECKING, Literal

import httpx

from .._parts_manager import ModelResponsePartsManager
from ..exceptions import UserError
from ..messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from ..settings import ModelSettings
from ..usage import Usage

if TYPE_CHECKING:
    from ..tools import ToolDefinition


KnownModelName = Literal[
    'openai:gpt-4o',
    'openai:gpt-4o-mini',
    'openai:gpt-4-turbo',
    'openai:gpt-4',
    'openai:o1-preview',
    'openai:o1-mini',
    'openai:o1',
    'openai:gpt-3.5-turbo',
    'groq:llama-3.3-70b-versatile',
    'groq:llama-3.1-70b-versatile',
    'groq:llama3-groq-70b-8192-tool-use-preview',
    'groq:llama3-groq-8b-8192-tool-use-preview',
    'groq:llama-3.1-70b-specdec',
    'groq:llama-3.1-8b-instant',
    'groq:llama-3.2-1b-preview',
    'groq:llama-3.2-3b-preview',
    'groq:llama-3.2-11b-vision-preview',
    'groq:llama-3.2-90b-vision-preview',
    'groq:llama3-70b-8192',
    'groq:llama3-8b-8192',
    'groq:mixtral-8x7b-32768',
    'groq:gemma2-9b-it',
    'groq:gemma-7b-it',
    'google-gla:gemini-1.5-flash',
    'google-gla:gemini-1.5-pro',
    'google-gla:gemini-2.0-flash-exp',
    'google-vertex:gemini-1.5-flash',
    'google-vertex:gemini-1.5-pro',
    'google-vertex:gemini-2.0-flash-exp',
    'mistral:mistral-small-latest',
    'mistral:mistral-large-latest',
    'mistral:codestral-latest',
    'mistral:mistral-moderation-latest',
    'ollama:codellama',
    'ollama:deepseek-r1',
    'ollama:gemma',
    'ollama:gemma2',
    'ollama:llama3',
    'ollama:llama3.1',
    'ollama:llama3.2',
    'ollama:llama3.2-vision',
    'ollama:llama3.3',
    'ollama:mistral',
    'ollama:mistral-nemo',
    'ollama:mixtral',
    'ollama:phi3',
    'ollama:phi4',
    'ollama:qwq',
    'ollama:qwen',
    'ollama:qwen2',
    'ollama:qwen2.5',
    'ollama:starcoder2',
    'anthropic:claude-3-5-haiku-latest',
    'anthropic:claude-3-5-sonnet-latest',
    'anthropic:claude-3-opus-latest',
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
    'test',
]
"""Known model names that can be used with the `model` parameter of [`Agent`][pydantic_ai.Agent].

`KnownModelName` is provided as a concise way to specify a model.
"""


class Model(ABC):
    """Abstract class for a model."""

    @abstractmethod
    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create an agent model, this is called for each step of an agent run.

        This is async in case slow/async config checks need to be performed that can't be done in `__init__`.

        Args:
            function_tools: The tools available to the agent.
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
    """Model configured for each step of an Agent run."""

    @abstractmethod
    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the model."""
        raise NotImplementedError()

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[StreamedResponse]:
        """Make a request to the model and return a streaming response."""
        # This method is not required, but you need to implement it if you want to support streamed responses
        raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover


@dataclass
class StreamedResponse(ABC):
    """Streamed response from an LLM when calling a tool."""

    _model_name: str
    _usage: Usage = field(default_factory=Usage, init=False)
    _parts_manager: ModelResponsePartsManager = field(default_factory=ModelResponsePartsManager, init=False)
    _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)

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
        """
        raise NotImplementedError()
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        """Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
        return ModelResponse(
            parts=self._parts_manager.get_parts(), model_name=self._model_name, timestamp=self.timestamp()
        )

    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    def usage(self) -> Usage:
        """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
        return self._usage

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
    elif model.startswith('cohere:'):
        from .cohere import CohereModel

        return CohereModel(model[7:])
    elif model.startswith('openai:'):
        from .openai import OpenAIModel

        return OpenAIModel(model[7:])
    elif model.startswith(('gpt', 'o1')):
        from .openai import OpenAIModel

        return OpenAIModel(model)
    elif model.startswith('google-gla'):
        from .gemini import GeminiModel

        return GeminiModel(model[11:])  # pyright: ignore[reportArgumentType]
    # backwards compatibility with old model names (ex, gemini-1.5-flash -> google-gla:gemini-1.5-flash)
    elif model.startswith('gemini'):
        from .gemini import GeminiModel

        # noinspection PyTypeChecker
        return GeminiModel(model)  # pyright: ignore[reportArgumentType]
    elif model.startswith('groq:'):
        from .groq import GroqModel

        return GroqModel(model[5:])  # pyright: ignore[reportArgumentType]
    elif model.startswith('google-vertex'):
        from .vertexai import VertexAIModel

        return VertexAIModel(model[14:])  # pyright: ignore[reportArgumentType]
    # backwards compatibility with old model names (ex, vertexai:gemini-1.5-flash -> google-vertex:gemini-1.5-flash)
    elif model.startswith('vertexai:'):
        from .vertexai import VertexAIModel

        return VertexAIModel(model[9:])  # pyright: ignore[reportArgumentType]
    elif model.startswith('mistral:'):
        from .mistral import MistralModel

        return MistralModel(model[8:])
    elif model.startswith('ollama:'):
        from .ollama import OllamaModel

        return OllamaModel(model[7:])
    elif model.startswith('anthropic'):
        from .anthropic import AnthropicModel

        return AnthropicModel(model[10:])
    # backwards compatibility with old model names (ex, claude-3-5-sonnet-latest -> anthropic:claude-3-5-sonnet-latest)
    elif model.startswith('claude'):
        from .anthropic import AnthropicModel

        return AnthropicModel(model)
    else:
        raise UserError(f'Unknown model: {model}')


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
