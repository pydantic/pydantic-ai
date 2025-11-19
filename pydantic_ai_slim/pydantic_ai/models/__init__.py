"""Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the
specific LLM being used.
"""

from __future__ import annotations as _annotations

import base64
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cache, cached_property
from typing import Any, Generic, Literal, TypeVar, overload

import httpx
from typing_extensions import TypeAliasType, TypedDict

from .. import _utils
from .._json_schema import JsonSchemaTransformer
from .._output import OutputObjectDefinition, PromptedOutputSchema
from .._parts_manager import ModelResponsePartsManager
from .._run_context import RunContext
from ..builtin_tools import AbstractBuiltinTool
from ..exceptions import UserError
from ..messages import (
    BaseToolCallPart,
    BinaryImage,
    FilePart,
    FileUrl,
    FinalResultEvent,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    VideoUrl,
)
from ..output import OutputMode
from ..profiles import DEFAULT_PROFILE, ModelProfile, ModelProfileSpec
from ..providers import Provider, infer_provider
from ..settings import ModelSettings, merge_model_settings
from ..tools import ToolDefinition
from ..usage import RequestUsage

KnownModelName = TypeAliasType(
    'KnownModelName',
    Literal[
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
        'anthropic:claude-4-opus-20250514',
        'anthropic:claude-4-sonnet-20250514',
        'anthropic:claude-haiku-4-5',
        'anthropic:claude-haiku-4-5-20251001',
        'anthropic:claude-opus-4-0',
        'anthropic:claude-opus-4-1-20250805',
        'anthropic:claude-opus-4-20250514',
        'anthropic:claude-sonnet-4-0',
        'anthropic:claude-sonnet-4-20250514',
        'anthropic:claude-sonnet-4-5',
        'anthropic:claude-sonnet-4-5-20250929',
        'bedrock:amazon.titan-text-express-v1',
        'bedrock:amazon.titan-text-lite-v1',
        'bedrock:amazon.titan-tg1-large',
        'bedrock:anthropic.claude-3-5-haiku-20241022-v1:0',
        'bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0',
        'bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0',
        'bedrock:anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock:anthropic.claude-3-opus-20240229-v1:0',
        'bedrock:anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock:anthropic.claude-haiku-4-5-20251001-v1:0',
        'bedrock:anthropic.claude-instant-v1',
        'bedrock:anthropic.claude-opus-4-20250514-v1:0',
        'bedrock:anthropic.claude-sonnet-4-20250514-v1:0',
        'bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0',
        'bedrock:anthropic.claude-v2',
        'bedrock:anthropic.claude-v2:1',
        'bedrock:cohere.command-light-text-v14',
        'bedrock:cohere.command-r-plus-v1:0',
        'bedrock:cohere.command-r-v1:0',
        'bedrock:cohere.command-text-v14',
        'bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0',
        'bedrock:eu.anthropic.claude-sonnet-4-20250514-v1:0',
        'bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'bedrock:meta.llama3-1-405b-instruct-v1:0',
        'bedrock:meta.llama3-1-70b-instruct-v1:0',
        'bedrock:meta.llama3-1-8b-instruct-v1:0',
        'bedrock:meta.llama3-70b-instruct-v1:0',
        'bedrock:meta.llama3-8b-instruct-v1:0',
        'bedrock:mistral.mistral-7b-instruct-v0:2',
        'bedrock:mistral.mistral-large-2402-v1:0',
        'bedrock:mistral.mistral-large-2407-v1:0',
        'bedrock:mistral.mixtral-8x7b-instruct-v0:1',
        'bedrock:us.amazon.nova-lite-v1:0',
        'bedrock:us.amazon.nova-micro-v1:0',
        'bedrock:us.amazon.nova-pro-v1:0',
        'bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'bedrock:us.anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock:us.anthropic.claude-3-opus-20240229-v1:0',
        'bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'bedrock:us.anthropic.claude-opus-4-20250514-v1:0',
        'bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0',
        'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'bedrock:us.meta.llama3-1-70b-instruct-v1:0',
        'bedrock:us.meta.llama3-1-8b-instruct-v1:0',
        'bedrock:us.meta.llama3-2-11b-instruct-v1:0',
        'bedrock:us.meta.llama3-2-1b-instruct-v1:0',
        'bedrock:us.meta.llama3-2-3b-instruct-v1:0',
        'bedrock:us.meta.llama3-2-90b-instruct-v1:0',
        'bedrock:us.meta.llama3-3-70b-instruct-v1:0',
        'cerebras:gpt-oss-120b',
        'cerebras:llama-3.3-70b',
        'cerebras:llama3.1-8b',
        'cerebras:qwen-3-235b-a22b-instruct-2507',
        'cerebras:qwen-3-235b-a22b-thinking-2507',
        'cerebras:qwen-3-32b',
        'cerebras:zai-glm-4.6',
        'cohere:c4ai-aya-expanse-32b',
        'cohere:c4ai-aya-expanse-8b',
        'cohere:command-nightly',
        'cohere:command-r-08-2024',
        'cohere:command-r-plus-08-2024',
        'cohere:command-r7b-12-2024',
        'deepseek:deepseek-chat',
        'deepseek:deepseek-reasoner',
        'google-gla:gemini-flash-latest',
        'google-gla:gemini-flash-lite-latest',
        'google-gla:gemini-2.0-flash',
        'google-gla:gemini-2.0-flash-lite',
        'google-gla:gemini-2.5-flash',
        'google-gla:gemini-2.5-flash-preview-09-2025',
        'google-gla:gemini-2.5-flash-image',
        'google-gla:gemini-2.5-flash-lite',
        'google-gla:gemini-2.5-flash-lite-preview-09-2025',
        'google-gla:gemini-2.5-pro',
        'google-gla:gemini-3-pro-preview',
        'google-vertex:gemini-flash-latest',
        'google-vertex:gemini-flash-lite-latest',
        'google-vertex:gemini-2.0-flash',
        'google-vertex:gemini-2.0-flash-lite',
        'google-vertex:gemini-2.5-flash',
        'google-vertex:gemini-2.5-flash-preview-09-2025',
        'google-vertex:gemini-2.5-flash-image',
        'google-vertex:gemini-2.5-flash-lite',
        'google-vertex:gemini-2.5-flash-lite-preview-09-2025',
        'google-vertex:gemini-2.5-pro',
        'google-vertex:gemini-3-pro-preview',
        'grok:grok-2-image-1212',
        'grok:grok-2-vision-1212',
        'grok:grok-3',
        'grok:grok-3-fast',
        'grok:grok-3-mini',
        'grok:grok-3-mini-fast',
        'grok:grok-4',
        'grok:grok-4-0709',
        'groq:deepseek-r1-distill-llama-70b',
        'groq:deepseek-r1-distill-qwen-32b',
        'groq:distil-whisper-large-v3-en',
        'groq:gemma2-9b-it',
        'groq:llama-3.1-8b-instant',
        'groq:llama-3.2-11b-vision-preview',
        'groq:llama-3.2-1b-preview',
        'groq:llama-3.2-3b-preview',
        'groq:llama-3.2-90b-vision-preview',
        'groq:llama-3.3-70b-specdec',
        'groq:llama-3.3-70b-versatile',
        'groq:llama-guard-3-8b',
        'groq:llama3-70b-8192',
        'groq:llama3-8b-8192',
        'groq:mistral-saba-24b',
        'groq:moonshotai/kimi-k2-instruct',
        'groq:playai-tts',
        'groq:playai-tts-arabic',
        'groq:qwen-2.5-32b',
        'groq:qwen-2.5-coder-32b',
        'groq:qwen-qwq-32b',
        'groq:whisper-large-v3',
        'groq:whisper-large-v3-turbo',
        'heroku:amazon-rerank-1-0',
        'heroku:claude-3-5-haiku',
        'heroku:claude-3-5-sonnet-latest',
        'heroku:claude-3-7-sonnet',
        'heroku:claude-3-haiku',
        'heroku:claude-4-5-haiku',
        'heroku:claude-4-5-sonnet',
        'heroku:claude-4-sonnet',
        'heroku:cohere-rerank-3-5',
        'heroku:gpt-oss-120b',
        'heroku:nova-lite',
        'heroku:nova-pro',
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
        'moonshotai:kimi-k2-0711-preview',
        'moonshotai:kimi-latest',
        'moonshotai:kimi-thinking-preview',
        'moonshotai:moonshot-v1-128k',
        'moonshotai:moonshot-v1-128k-vision-preview',
        'moonshotai:moonshot-v1-32k',
        'moonshotai:moonshot-v1-32k-vision-preview',
        'moonshotai:moonshot-v1-8k',
        'moonshotai:moonshot-v1-8k-vision-preview',
        'openai:chatgpt-4o-latest',
        'openai:codex-mini-latest',
        'openai:computer-use-preview',
        'openai:computer-use-preview-2025-03-11',
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
        'openai:gpt-5',
        'openai:gpt-5-2025-08-07',
        'openai:gpt-5-chat-latest',
        'openai:gpt-5-codex',
        'openai:gpt-5-mini',
        'openai:gpt-5-mini-2025-08-07',
        'openai:gpt-5-nano',
        'openai:gpt-5-nano-2025-08-07',
        'openai:gpt-5-pro',
        'openai:gpt-5-pro-2025-10-06',
        'openai:gpt-5.1',
        'openai:gpt-5.1-2025-11-13',
        'openai:gpt-5.1-chat-latest',
        'openai:gpt-5.1-codex',
        'openai:gpt-5.1-mini',
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
        'openai:o3-pro',
        'openai:o3-pro-2025-06-10',
        'openai:o4-mini',
        'openai:o4-mini-2025-04-16',
        'openai:o4-mini-deep-research',
        'openai:o4-mini-deep-research-2025-06-26',
        'test',
    ],
)
"""Known model names that can be used with the `model` parameter of [`Agent`][pydantic_ai.Agent].

`KnownModelName` is provided as a concise way to specify a model.
"""


@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and output handling."""

    function_tools: list[ToolDefinition] = field(default_factory=list)
    builtin_tools: list[AbstractBuiltinTool] = field(default_factory=list)

    output_mode: OutputMode = 'text'
    output_object: OutputObjectDefinition | None = None
    output_tools: list[ToolDefinition] = field(default_factory=list)
    prompted_output_template: str | None = None
    allow_text_output: bool = True
    allow_image_output: bool = False

    @cached_property
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {tool_def.name: tool_def for tool_def in [*self.function_tools, *self.output_tools]}

    @cached_property
    def prompted_output_instructions(self) -> str | None:
        if self.output_mode == 'prompted' and self.prompted_output_template and self.output_object:
            return PromptedOutputSchema.build_instructions(self.prompted_output_template, self.output_object)
        return None

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

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """Make a request to the model for counting tokens."""
        # This method is not required, but you need to implement it if you want to support `UsageLimits.count_tokens_before_request`.
        raise NotImplementedError(f'Token counting ahead of the request is not supported by {self.__class__.__name__}')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
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

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        """Prepare request inputs before they are passed to the provider.

        This merges the given `model_settings` with the model's own `settings` attribute and ensures
        `customize_request_parameters` is applied to the resolved
        [`ModelRequestParameters`][pydantic_ai.models.ModelRequestParameters]. Subclasses can override this method if
        they need to customize the preparation flow further, but most implementations should simply call
        `self.prepare_request(...)` at the start of their `request` (and related) methods.
        """
        model_settings = merge_model_settings(self.settings, model_settings)

        params = self.customize_request_parameters(model_request_parameters)

        if builtin_tools := params.builtin_tools:
            # Deduplicate builtin tools
            params = replace(
                params,
                builtin_tools=list({tool.unique_id: tool for tool in builtin_tools}.values()),
            )

        if params.output_mode == 'auto':
            output_mode = self.profile.default_structured_output_mode
            params = replace(
                params,
                output_mode=output_mode,
                allow_text_output=output_mode in ('native', 'prompted'),
            )

        # Reset irrelevant fields
        if params.output_tools and params.output_mode != 'tool':
            params = replace(params, output_tools=[])
        if params.output_object and params.output_mode not in ('native', 'prompted'):
            params = replace(params, output_object=None)
        if params.prompted_output_template and params.output_mode != 'prompted':
            params = replace(params, prompted_output_template=None)  # pragma: no cover

        # Set default prompted output template
        if params.output_mode == 'prompted' and not params.prompted_output_template:
            params = replace(params, prompted_output_template=self.profile.prompted_output_template)

        # Check if output mode is supported
        if params.output_mode == 'native' and not self.profile.supports_json_schema_output:
            raise UserError('Native structured output is not supported by this model.')
        if params.output_mode == 'tool' and not self.profile.supports_tools:
            raise UserError('Tool output is not supported by this model.')
        if params.allow_image_output and not self.profile.supports_image_output:
            raise UserError('Image output is not supported by this model.')

        return model_settings, params

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
        """The model provider, ex: openai.

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
    def _get_instructions(
        messages: list[ModelMessage], model_request_parameters: ModelRequestParameters | None = None
    ) -> str | None:
        """Get instructions from the first ModelRequest found when iterating messages in reverse.

        In the case that a "mock" request was generated to include a tool-return part for a result tool,
        we want to use the instructions from the second-to-most-recent request (which should correspond to the
        original request that generated the response that resulted in the tool-return part).
        """
        instructions = None

        last_two_requests: list[ModelRequest] = []
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                last_two_requests.append(message)
                if len(last_two_requests) == 2:
                    break
                if message.instructions is not None:
                    instructions = message.instructions
                    break

        # If we don't have two requests, and we didn't already return instructions, there are definitely not any:
        if instructions is None and len(last_two_requests) == 2:
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
                instructions = second_most_recent_request.instructions

        if model_request_parameters and (output_instructions := model_request_parameters.prompted_output_instructions):
            if instructions:
                instructions = '\n\n'.join([instructions, output_instructions])
            else:
                instructions = output_instructions

        return instructions


@dataclass
class StreamedResponse(ABC):
    """Streamed response from an LLM when calling a tool."""

    model_request_parameters: ModelRequestParameters

    final_result_event: FinalResultEvent | None = field(default=None, init=False)

    provider_response_id: str | None = field(default=None, init=False)
    provider_details: dict[str, Any] | None = field(default=None, init=False)
    finish_reason: FinishReason | None = field(default=None, init=False)

    _parts_manager: ModelResponsePartsManager = field(default_factory=ModelResponsePartsManager, init=False)
    _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
    _usage: RequestUsage = field(default_factory=RequestUsage, init=False)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream the response as an async iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This proxies the `_event_iterator()` and emits all events, while also checking for matches
        on the result schema and emitting a [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] if/when the
        first match is found.
        """
        if self._event_iterator is None:

            async def iterator_with_final_event(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                async for event in iterator:
                    yield event
                    if (
                        final_result_event := _get_final_result_event(event, self.model_request_parameters)
                    ) is not None:
                        self.final_result_event = final_result_event
                        yield final_result_event
                        break

                # If we broke out of the above loop, we need to yield the rest of the events
                # If we didn't, this will just be a no-op
                async for event in iterator:
                    yield event

            async def iterator_with_part_end(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                last_start_event: PartStartEvent | None = None

                def part_end_event(next_part: ModelResponsePart | None = None) -> PartEndEvent | None:
                    if not last_start_event:
                        return None

                    index = last_start_event.index
                    part = self._parts_manager.get_parts()[index]
                    if not isinstance(part, TextPart | ThinkingPart | BaseToolCallPart):
                        # Parts other than these 3 don't have deltas, so don't need an end part.
                        return None

                    return PartEndEvent(
                        index=index,
                        part=part,
                        next_part_kind=next_part.part_kind if next_part else None,
                    )

                async for event in iterator:
                    if isinstance(event, PartStartEvent):
                        if last_start_event:
                            end_event = part_end_event(event.part)
                            if end_event:
                                yield end_event

                            event.previous_part_kind = last_start_event.part.part_kind
                        last_start_event = event

                    yield event

                end_event = part_end_event()
                if end_event:
                    yield end_event

            self._event_iterator = iterator_with_part_end(iterator_with_final_event(self._get_event_iterator()))
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
            provider_name=self.provider_name,
            provider_response_id=self.provider_response_id,
            provider_details=self.provider_details,
            finish_reason=self.finish_reason,
        )

    # TODO (v2): Make this a property
    def usage(self) -> RequestUsage:
        """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
        return self._usage

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name of the response."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def provider_name(self) -> str | None:
        """Get the provider name."""
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


def infer_model(  # noqa: C901
    model: Model | KnownModelName | str, provider_factory: Callable[[str], Provider[Any]] = infer_provider
) -> Model:
    """Infer the model from the name.

    Args:
        model:
            Model name to instantiate, in the format of `provider:model`. Use the string "test" to instantiate TestModel.
        provider_factory:
            Function that instantiates a provider object. The provider name is passed into the function parameter. Defaults to `provider.infer_provider`.
    """
    if isinstance(model, Model):
        return model
    elif model == 'test':
        from .test import TestModel

        return TestModel()

    try:
        provider_name, model_name = model.split(':', maxsplit=1)
    except ValueError:
        provider_name = None
        model_name = model
        if model_name.startswith(('gpt', 'o1', 'o3')):
            provider_name = 'openai'
        elif model_name.startswith('claude'):
            provider_name = 'anthropic'
        elif model_name.startswith('gemini'):
            provider_name = 'google-gla'

        if provider_name is not None:
            warnings.warn(
                f"Specifying a model name without a provider prefix is deprecated. Instead of {model_name!r}, use '{provider_name}:{model_name}'.",
                DeprecationWarning,
            )
        else:
            raise UserError(f'Unknown model: {model}')

    if provider_name == 'vertexai':  # pragma: no cover
        warnings.warn(
            "The 'vertexai' provider name is deprecated. Use 'google-vertex' instead.",
            DeprecationWarning,
        )
        provider_name = 'google-vertex'

    provider: Provider[Any] = provider_factory(provider_name)

    model_kind = provider_name
    if model_kind.startswith('gateway/'):
        from ..providers.gateway import normalize_gateway_provider

        model_kind = provider_name.removeprefix('gateway/')
        model_kind = normalize_gateway_provider(model_kind)
    if model_kind in (
        'openai',
        'azure',
        'deepseek',
        'cerebras',
        'fireworks',
        'github',
        'grok',
        'heroku',
        'moonshotai',
        'ollama',
        'openrouter',
        'together',
        'vercel',
        'litellm',
        'nebius',
        'ovhcloud',
    ):
        model_kind = 'openai-chat'
    elif model_kind in ('google-gla', 'google-vertex'):
        model_kind = 'google'

    if model_kind == 'openai-chat':
        from .openai import OpenAIChatModel

        return OpenAIChatModel(model_name, provider=provider)
    elif model_kind == 'openai-responses':
        from .openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name, provider=provider)
    elif model_kind == 'google':
        from .google import GoogleModel

        return GoogleModel(model_name, provider=provider)
    elif model_kind == 'groq':
        from .groq import GroqModel

        return GroqModel(model_name, provider=provider)
    elif model_kind == 'cohere':
        from .cohere import CohereModel

        return CohereModel(model_name, provider=provider)
    elif model_kind == 'mistral':
        from .mistral import MistralModel

        return MistralModel(model_name, provider=provider)
    elif model_kind == 'anthropic':
        from .anthropic import AnthropicModel

        return AnthropicModel(model_name, provider=provider)
    elif model_kind == 'bedrock':
        from .bedrock import BedrockConverseModel

        return BedrockConverseModel(model_name, provider=provider)
    elif model_kind == 'huggingface':
        from .huggingface import HuggingFaceModel

        return HuggingFaceModel(model_name, provider=provider)
    else:
        raise UserError(f'Unknown model: {model}')  # pragma: no cover


def cached_async_http_client(*, provider: str | None = None, timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    """Cached HTTPX async client that creates a separate client for each provider.

    The client is cached based on the provider parameter. If provider is None, it's used for non-provider specific
    requests (like downloading images). Multiple agents and calls can share the same client when they use the same provider.

    Each client will get its own transport with its own connection pool. The default pool size is defined by `httpx.DEFAULT_LIMITS`.

    There are good reasons why in production you should use a `httpx.AsyncClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful not to.

    The default timeouts match those of OpenAI,
    see <https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9>.
    """
    client = _cached_async_http_client(provider=provider, timeout=timeout, connect=connect)
    if client.is_closed:
        # This happens if the context manager is used, so we need to create a new client.
        # Since there is no API from `functools.cache` to clear the cache for a specific
        #  key, clear the entire cache here as a workaround.
        _cached_async_http_client.cache_clear()
        client = _cached_async_http_client(provider=provider, timeout=timeout, connect=connect)
    return client


@cache
def _cached_async_http_client(provider: str | None, timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


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
    return replace(
        t,
        parameters_json_schema=parameters_json_schema,
        strict=schema_transformer.is_strict_compatible if t.strict is None else t.strict,
    )


def _customize_output_object(transformer: type[JsonSchemaTransformer], o: OutputObjectDefinition):
    schema_transformer = transformer(o.json_schema, strict=o.strict)
    json_schema = schema_transformer.walk()
    return replace(
        o,
        json_schema=json_schema,
        strict=schema_transformer.is_strict_compatible if o.strict is None else o.strict,
    )


def _get_final_result_event(e: ModelResponseStreamEvent, params: ModelRequestParameters) -> FinalResultEvent | None:
    """Return an appropriate FinalResultEvent if `e` corresponds to a part that will produce a final result."""
    if isinstance(e, PartStartEvent):
        new_part = e.part
        if (isinstance(new_part, TextPart) and params.allow_text_output) or (
            isinstance(new_part, FilePart) and params.allow_image_output and isinstance(new_part.content, BinaryImage)
        ):
            return FinalResultEvent(tool_name=None, tool_call_id=None)
        elif isinstance(new_part, ToolCallPart) and (tool_def := params.tool_defs.get(new_part.tool_name)):
            if tool_def.kind == 'output':
                return FinalResultEvent(tool_name=new_part.tool_name, tool_call_id=new_part.tool_call_id)
            elif tool_def.defer:
                return FinalResultEvent(tool_name=None, tool_call_id=None)
