"""Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the
specific LLM being used.
"""

from __future__ import annotations as _annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cache, cached_property
from typing import Generic, TypeVar, overload

import httpx
from typing_extensions import Literal, TypeAliasType, TypedDict

from pydantic_ai.profiles import DEFAULT_PROFILE, ModelProfile, ModelProfileSpec

from .. import _utils
from .._output import OutputObjectDefinition
from .._parts_manager import ModelResponsePartsManager
from ..exceptions import UserError
from ..messages import FileUrl, ModelMessage, ModelRequest, ModelResponse, ModelResponseStreamEvent, VideoUrl
from ..output import OutputMode
from ..profiles._json_schema import JsonSchemaTransformer
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ..usage import Usage

KnownModelName = TypeAliasType(
    'KnownModelName',
    Literal[
        'anthropic:claude-2.0',
        'anthropic:claude-2.1',
        'anthropic:claude-3-5-haiku-20241022',
        'anthropic:claude-3-5-haiku-latest',
        'anthropic:claude-3-5-sonnet-20240620',
        'anthropic:claude-3-5-sonnet-20241022',
        'anthropic:claude-3-5-sonnet-latest',
        'anthropic:claude-3-7-sonnet-20250219',
        'anthropic:claude-3-7-sonnet-latest',
        'anthropic:claude-3-haiku-20240307',
        'anthropic:claude-3-opus-20240229',
        'anthropic:claude-3-opus-latest',
        'anthropic:claude-3-sonnet-20240229',
        'anthropic:claude-4-opus-20250514',
        'anthropic:claude-4-sonnet-20250514',
        'anthropic:claude-opus-4-0',
        'anthropic:claude-opus-4-20250514',
        'anthropic:claude-sonnet-4-0',
        'anthropic:claude-sonnet-4-20250514',
        'bedrock:amazon.titan-tg1-large',
        'bedrock:amazon.titan-text-lite-v1',
        'bedrock:amazon.titan-text-express-v1',
        'bedrock:us.amazon.nova-pro-v1:0',
        'bedrock:us.amazon.nova-lite-v1:0',
        'bedrock:us.amazon.nova-micro-v1:0',
        'bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock:anthropic.claude-3-5-haiku-20241022-v1:0',
        'bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'bedrock:anthropic.claude-instant-v1',
        'bedrock:anthropic.claude-v2:1',
        'bedrock:anthropic.claude-v2',
        'bedrock:anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock:anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock:us.anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock:anthropic.claude-3-opus-20240229-v1:0',
        'bedrock:us.anthropic.claude-3-opus-20240229-v1:0',
        'bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0',
        'bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0',
        'bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'bedrock:anthropic.claude-opus-4-20250514-v1:0',
        'bedrock:us.anthropic.claude-opus-4-20250514-v1:0',
        'bedrock:anthropic.claude-sonnet-4-20250514-v1:0',
        'bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0',
        'bedrock:cohere.command-text-v14',
        'bedrock:cohere.command-r-v1:0',
        'bedrock:cohere.command-r-plus-v1:0',
        'bedrock:cohere.command-light-text-v14',
        'bedrock:meta.llama3-8b-instruct-v1:0',
        'bedrock:meta.llama3-70b-instruct-v1:0',
        'bedrock:meta.llama3-1-8b-instruct-v1:0',
        'bedrock:us.meta.llama3-1-8b-instruct-v1:0',
        'bedrock:meta.llama3-1-70b-instruct-v1:0',
        'bedrock:us.meta.llama3-1-70b-instruct-v1:0',
        'bedrock:meta.llama3-1-405b-instruct-v1:0',
        'bedrock:us.meta.llama3-2-11b-instruct-v1:0',
        'bedrock:us.meta.llama3-2-90b-instruct-v1:0',
        'bedrock:us.meta.llama3-2-1b-instruct-v1:0',
        'bedrock:us.meta.llama3-2-3b-instruct-v1:0',
        'bedrock:us.meta.llama3-3-70b-instruct-v1:0',
        'bedrock:mistral.mistral-7b-instruct-v0:2',
        'bedrock:mistral.mixtral-8x7b-instruct-v0:1',
        'bedrock:mistral.mistral-large-2402-v1:0',
        'bedrock:mistral.mistral-large-2407-v1:0',
        'claude-2.0',
        'claude-2.1',
        'claude-3-5-haiku-20241022',
        'claude-3-5-haiku-latest',
        'claude-3-5-sonnet-20240620',
        'claude-3-5-sonnet-20241022',
        'claude-3-5-sonnet-latest',
        'claude-3-7-sonnet-20250219',
        'claude-3-7-sonnet-latest',
        'claude-3-haiku-20240307',
        'claude-3-opus-20240229',
        'claude-3-opus-latest',
        'claude-3-sonnet-20240229',
        'claude-4-opus-20250514',
        'claude-4-sonnet-20250514',
        'claude-opus-4-0',
        'claude-opus-4-20250514',
        'claude-sonnet-4-0',
        'claude-sonnet-4-20250514',
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
        'google-gla:gemini-2.0-flash',
        'google-gla:gemini-2.0-flash-lite',
        'google-gla:gemini-2.5-flash',
        'google-gla:gemini-2.5-flash-lite-preview-06-17',
        'google-gla:gemini-2.5-pro',
        'google-vertex:gemini-2.0-flash',
        'google-vertex:gemini-2.0-flash-lite',
        'google-vertex:gemini-2.5-flash',
        'google-vertex:gemini-2.5-flash-lite-preview-06-17',
        'google-vertex:gemini-2.5-pro',
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
        'gpt-4.1',
        'gpt-4.1-2025-04-14',
        'gpt-4.1-mini',
        'gpt-4.1-mini-2025-04-14',
        'gpt-4.1-nano',
        'gpt-4.1-nano-2025-04-14',
        'gpt-4o',
        'gpt-4o-2024-05-13',
        'gpt-4o-2024-08-06',
        'gpt-4o-2024-11-20',
        'gpt-4o-audio-preview',
        'gpt-4o-audio-preview-2024-10-01',
        'gpt-4o-audio-preview-2024-12-17',
        'gpt-4o-audio-preview-2025-06-03',
        'gpt-4o-mini',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o-mini-audio-preview',
        'gpt-4o-mini-audio-preview-2024-12-17',
        'gpt-4o-mini-search-preview',
        'gpt-4o-mini-search-preview-2025-03-11',
        'gpt-4o-search-preview',
        'gpt-4o-search-preview-2025-03-11',
        'grok:grok-4',
        'grok:grok-4-0709',
        'grok:grok-3',
        'grok:grok-3-mini',
        'grok:grok-3-fast',
        'grok:grok-3-mini-fast',
        'grok:grok-2-vision-1212',
        'grok:grok-2-image-1212',
        'groq:distil-whisper-large-v3-en',
        'groq:gemma2-9b-it',
        'groq:llama-3.3-70b-versatile',
        'groq:llama-3.1-8b-instant',
        'groq:llama-guard-3-8b',
        'groq:llama3-70b-8192',
        'groq:llama3-8b-8192',
        'groq:moonshotai/kimi-k2-instruct',
        'groq:whisper-large-v3',
        'groq:whisper-large-v3-turbo',
        'groq:playai-tts',
        'groq:playai-tts-arabic',
        'groq:qwen-qwq-32b',
        'groq:mistral-saba-24b',
        'groq:qwen-2.5-coder-32b',
        'groq:qwen-2.5-32b',
        'groq:deepseek-r1-distill-qwen-32b',
        'groq:deepseek-r1-distill-llama-70b',
        'groq:llama-3.3-70b-specdec',
        'groq:llama-3.2-1b-preview',
        'groq:llama-3.2-3b-preview',
        'groq:llama-3.2-11b-vision-preview',
        'groq:llama-3.2-90b-vision-preview',
        'heroku:claude-3-5-haiku',
        'heroku:claude-3-5-sonnet-latest',
        'heroku:claude-3-7-sonnet',
        'heroku:claude-4-sonnet',
        'heroku:claude-3-haiku',
        'huggingface:Qwen/QwQ-32B',
        'huggingface:Qwen/Qwen2.5-72B-Instruct',
        'huggingface:Qwen/Qwen3-235B-A22B',
        'huggingface:Qwen/Qwen3-32B',
        'huggingface:deepseek-ai/DeepSeek-R1',
        'huggingface:meta-llama/Llama-3.3-70B-Instruct',
        'huggingface:meta-llama/Llama-4-Maverick-17B-128E-Instruct',
        'huggingface:meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'mistral:codestral-latest',
        'mistral:mistral-large-latest',
        'mistral:mistral-moderation-latest',
        'mistral:mistral-small-latest',
        'moonshotai:moonshot-v1-8k',
        'moonshotai:moonshot-v1-32k',
        'moonshotai:moonshot-v1-128k',
        'moonshotai:moonshot-v1-8k-vision-preview',
        'moonshotai:moonshot-v1-32k-vision-preview',
        'moonshotai:moonshot-v1-128k-vision-preview',
        'moonshotai:kimi-latest',
        'moonshotai:kimi-thinking-preview',
        'moonshotai:kimi-k2-0711-preview',
        'o1',
        'o1-2024-12-17',
        'o1-mini',
        'o1-mini-2024-09-12',
        'o1-preview',
        'o1-preview-2024-09-12',
        'o1-pro',
        'o1-pro-2025-03-19',
        'o3',
        'o3-2025-04-16',
        'o3-deep-research',
        'o3-deep-research-2025-06-26',
        'o3-mini',
        'o3-mini-2025-01-31',
        'o3-pro',
        'o3-pro-2025-06-10',
        'openai:chatgpt-4o-latest',
        'openai:codex-mini-latest',
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
        'openai:gpt-4.1',
        'openai:gpt-4.1-2025-04-14',
        'openai:gpt-4.1-mini',
        'openai:gpt-4.1-mini-2025-04-14',
        'openai:gpt-4.1-nano',
        'openai:gpt-4.1-nano-2025-04-14',
        'openai:gpt-4o',
        'openai:gpt-4o-2024-05-13',
        'openai:gpt-4o-2024-08-06',
        'openai:gpt-4o-2024-11-20',
        'openai:gpt-4o-audio-preview',
        'openai:gpt-4o-audio-preview-2024-10-01',
        'openai:gpt-4o-audio-preview-2024-12-17',
        'openai:gpt-4o-audio-preview-2025-06-03',
        'openai:gpt-4o-mini',
        'openai:gpt-4o-mini-2024-07-18',
        'openai:gpt-4o-mini-audio-preview',
        'openai:gpt-4o-mini-audio-preview-2024-12-17',
        'openai:gpt-4o-mini-search-preview',
        'openai:gpt-4o-mini-search-preview-2025-03-11',
        'openai:gpt-4o-search-preview',
        'openai:gpt-4o-search-preview-2025-03-11',
        'openai:o1',
        'openai:o1-2024-12-17',
        'openai:o1-mini',
        'openai:o1-mini-2024-09-12',
        'openai:o1-preview',
        'openai:o1-preview-2024-09-12',
        'openai:o1-pro',
        'openai:o1-pro-2025-03-19',
        'openai:o3',
        'openai:o3-2025-04-16',
        'openai:o3-deep-research',
        'openai:o3-deep-research-2025-06-26',
        'openai:o3-mini',
        'openai:o3-mini-2025-01-31',
        'openai:o4-mini',
        'openai:o4-mini-2025-04-16',
        'openai:o4-mini-deep-research',
        'openai:o4-mini-deep-research-2025-06-26',
        'openai:o3-pro',
        'openai:o3-pro-2025-06-10',
        'openai:computer-use-preview',
        'openai:computer-use-preview-2025-03-11',
        'test',
    ],
)
"""Known model names that can be used with the `model` parameter of [`Agent`][pydantic_ai.Agent].

`KnownModelName` is provided as a concise way to specify a model.
"""


@dataclass(repr=False)
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and output handling."""

    function_tools: list[ToolDefinition] = field(default_factory=list)

    output_mode: OutputMode = 'text'
    output_object: OutputObjectDefinition | None = None
    output_tools: list[ToolDefinition] = field(default_factory=list)
    allow_text_output: bool = True

    __repr__ = _utils.dataclasses_no_defaults_repr


class Model(ABC):
    """Abstract class for a model."""

    _profile: ModelProfileSpec | None = None
    _settings: ModelSettings | None = None

    def __init__(
        self,
        *,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        """Initialize the model with optional settings and profile.

        Args:
            settings: Model-specific settings that will be used as defaults for this model.
            profile: The model profile to use.
        """
        self._settings = settings
        self._profile = profile

    @property
    def settings(self) -> ModelSettings | None:
        """Get the model settings."""
        return self._settings

    @abstractmethod
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
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

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        """Customize the request parameters for the model.

        This method can be overridden by subclasses to modify the request parameters before sending them to the model.
        In particular, this method can be used to make modifications to the generated tool JSON schemas if necessary
        for vendor/model-specific reasons.
        """
        if transformer := self.profile.json_schema_transformer:
            model_request_parameters = replace(
                model_request_parameters,
                function_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.function_tools],
                output_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.output_tools],
            )
            if output_object := model_request_parameters.output_object:
                model_request_parameters = replace(
                    model_request_parameters,
                    output_object=_customize_output_object(transformer, output_object),
                )

        return model_request_parameters

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name."""
        raise NotImplementedError()

    @cached_property
    def profile(self) -> ModelProfile:
        """The model profile."""
        _profile = self._profile
        if callable(_profile):
            _profile = _profile(self.model_name)

        if _profile is None:
            return DEFAULT_PROFILE

        return _profile

    @property
    @abstractmethod
    def system(self) -> str:
        """The system / model provider, ex: openai.

        Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute,
        so should use well-known values listed in
        https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system
        when applicable.
        """
        raise NotImplementedError()

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @staticmethod
    def _get_instructions(messages: list[ModelMessage]) -> str | None:
        """Get instructions from the first ModelRequest found when iterating messages in reverse.

        In the case that a "mock" request was generated to include a tool-return part for a result tool,
        we want to use the instructions from the second-to-most-recent request (which should correspond to the
        original request that generated the response that resulted in the tool-return part).
        """
        last_two_requests: list[ModelRequest] = []
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                last_two_requests.append(message)
                if len(last_two_requests) == 2:
                    break
                if message.instructions is not None:
                    return message.instructions

        # If we don't have two requests, and we didn't already return instructions, there are definitely not any:
        if len(last_two_requests) != 2:
            return None

        most_recent_request = last_two_requests[0]
        second_most_recent_request = last_two_requests[1]

        # If we've gotten this far and the most recent request consists of only tool-return parts or retry-prompt parts,
        # we use the instructions from the second-to-most-recent request. This is necessary because when handling
        # result tools, we generate a "mock" ModelRequest with a tool-return part for it, and that ModelRequest will not
        # have the relevant instructions from the agent.

        # While it's possible that you could have a message history where the most recent request has only tool returns,
        # I believe there is no way to achieve that would _change_ the instructions without manually crafting the most
        # recent message. That might make sense in principle for some usage pattern, but it's enough of an edge case
        # that I think it's not worth worrying about, since you can work around this by inserting another ModelRequest
        # with no parts at all immediately before the request that has the tool calls (that works because we only look
        # at the two most recent ModelRequests here).

        # If you have a use case where this causes pain, please open a GitHub issue and we can discuss alternatives.

        if all(p.part_kind == 'tool-return' or p.part_kind == 'retry-prompt' for p in most_recent_request.parts):
            return second_most_recent_request.instructions

        return None


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
            parts=self._parts_manager.get_parts(),
            model_name=self.model_name,
            timestamp=self.timestamp,
            usage=self.usage(),
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


def infer_model(model: Model | KnownModelName | str) -> Model:  # noqa: C901
    """Infer the model from the name."""
    if isinstance(model, Model):
        return model
    elif model == 'test':
        from .test import TestModel

        return TestModel()

    try:
        provider, model_name = model.split(':', maxsplit=1)
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
        provider = 'google-vertex'  # pragma: no cover

    if provider == 'cohere':
        from .cohere import CohereModel

        return CohereModel(model_name, provider=provider)
    elif provider in (
        'openai',
        'deepseek',
        'azure',
        'openrouter',
        'vercel',
        'grok',
        'moonshotai',
        'fireworks',
        'together',
        'heroku',
        'github',
    ):
        from .openai import OpenAIModel

        return OpenAIModel(model_name, provider=provider)
    elif provider in ('google-gla', 'google-vertex'):
        from .google import GoogleModel

        return GoogleModel(model_name, provider=provider)
    elif provider == 'groq':
        from .groq import GroqModel

        return GroqModel(model_name, provider=provider)
    elif provider == 'mistral':
        from .mistral import MistralModel

        return MistralModel(model_name, provider=provider)
    elif provider == 'anthropic':
        from .anthropic import AnthropicModel

        return AnthropicModel(model_name, provider=provider)
    elif provider == 'bedrock':
        from .bedrock import BedrockConverseModel

        return BedrockConverseModel(model_name, provider=provider)
    elif provider == 'huggingface':
        from .huggingface import HuggingFaceModel

        return HuggingFaceModel(model_name, provider=provider)
    else:
        raise UserError(f'Unknown model: {model}')  # pragma: no cover


def cached_async_http_client(*, provider: str | None = None, timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    """Cached HTTPX async client that creates a separate client for each provider.

    The client is cached based on the provider parameter. If provider is None, it's used for non-provider specific
    requests (like downloading images). Multiple agents and calls can share the same client when they use the same provider.

    There are good reasons why in production you should use a `httpx.AsyncClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful not to.

    The default timeouts match those of OpenAI,
    see <https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9>.
    """
    client = _cached_async_http_client(provider=provider, timeout=timeout, connect=connect)
    if client.is_closed:
        # This happens if the context manager is used, so we need to create a new client.
        _cached_async_http_client.cache_clear()
        client = _cached_async_http_client(provider=provider, timeout=timeout, connect=connect)
    return client


@cache
def _cached_async_http_client(provider: str | None, timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_cached_async_http_transport(),
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


@cache
def _cached_async_http_transport() -> httpx.AsyncHTTPTransport:
    return httpx.AsyncHTTPTransport()


DataT = TypeVar('DataT', str, bytes)


class DownloadedItem(TypedDict, Generic[DataT]):
    """The downloaded data and its type."""

    data: DataT
    """The downloaded data."""

    data_type: str
    """The type of data that was downloaded.

    Extracted from header "content-type", but defaults to the media type inferred from the file URL if content-type is "application/octet-stream".
    """


@overload
async def download_item(
    item: FileUrl,
    data_format: Literal['bytes'],
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[bytes]: ...


@overload
async def download_item(
    item: FileUrl,
    data_format: Literal['base64', 'base64_uri', 'text'],
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[str]: ...


async def download_item(
    item: FileUrl,
    data_format: Literal['bytes', 'base64', 'base64_uri', 'text'] = 'bytes',
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[str] | DownloadedItem[bytes]:
    """Download an item by URL and return the content as a bytes object or a (base64-encoded) string.

    Args:
        item: The item to download.
        data_format: The format to return the content in:
            - `bytes`: The raw bytes of the content.
            - `base64`: The base64-encoded content.
            - `base64_uri`: The base64-encoded content as a data URI.
            - `text`: The content as a string.
        type_format: The format to return the media type in:
            - `mime`: The media type as a MIME type.
            - `extension`: The media type as an extension.

    Raises:
        UserError: If the URL points to a YouTube video or its protocol is gs://.
    """
    if item.url.startswith('gs://'):
        raise UserError('Downloading from protocol "gs://" is not supported.')
    elif isinstance(item, VideoUrl) and item.is_youtube:
        raise UserError('Downloading YouTube videos is not supported.')

    client = cached_async_http_client()
    response = await client.get(item.url, follow_redirects=True)
    response.raise_for_status()

    if content_type := response.headers.get('content-type'):
        content_type = content_type.split(';')[0]
        if content_type == 'application/octet-stream':
            content_type = None

    media_type = content_type or item.media_type

    data_type = media_type
    if type_format == 'extension':
        data_type = item.format

    data = response.content
    if data_format in ('base64', 'base64_uri'):
        data = base64.b64encode(data).decode('utf-8')
        if data_format == 'base64_uri':
            data = f'data:{media_type};base64,{data}'
        return DownloadedItem[str](data=data, data_type=data_type)
    elif data_format == 'text':
        return DownloadedItem[str](data=data.decode('utf-8'), data_type=data_type)
    else:
        return DownloadedItem[bytes](data=data, data_type=data_type)


@cache
def get_user_agent() -> str:
    """Get the user agent string for the HTTP client."""
    from .. import __version__

    return f'pydantic-ai/{__version__}'


def _customize_tool_def(transformer: type[JsonSchemaTransformer], t: ToolDefinition):
    schema_transformer = transformer(t.parameters_json_schema, strict=t.strict)
    parameters_json_schema = schema_transformer.walk()
    if t.strict is None:
        t = replace(t, strict=schema_transformer.is_strict_compatible)
    return replace(t, parameters_json_schema=parameters_json_schema)


def _customize_output_object(transformer: type[JsonSchemaTransformer], o: OutputObjectDefinition):
    schema_transformer = transformer(o.json_schema, strict=True)
    son_schema = schema_transformer.walk()
    return replace(o, json_schema=son_schema)
