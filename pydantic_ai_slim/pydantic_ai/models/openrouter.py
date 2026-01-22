from __future__ import annotations as _annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import BaseModel, Discriminator
from typing_extensions import TypedDict, assert_never, override

from ..exceptions import ModelHTTPError
from ..messages import (
    FinishReason,
    ModelResponseStreamEvent,
    ThinkingPart,
)
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..providers.openrouter import OpenRouterProvider
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import APIError, AsyncOpenAI
    from openai.types import chat, completion_usage
    from openai.types.chat import chat_completion, chat_completion_chunk, chat_completion_message_function_tool_call
    from openai.types.chat.chat_completion_message import Annotation as _OpenAIAnnotation

    from .openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIStreamedResponse,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenRouter model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

_CHAT_FINISH_REASON_MAP: dict[Literal['stop', 'length', 'tool_calls', 'content_filter', 'error'], FinishReason] = {
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_call',
    'content_filter': 'content_filter',
    'error': 'error',
}


class _OpenRouterMaxPrice(TypedDict, total=False):
    """The object specifying the maximum price you want to pay for this request. USD price per million tokens, for prompt and completion."""

    prompt: int
    completion: int
    image: int
    audio: int
    request: int


KnownOpenRouterProviders = Literal[
    'z-ai',
    'cerebras',
    'venice',
    'moonshotai',
    'morph',
    'stealth',
    'wandb',
    'klusterai',
    'openai',
    'sambanova',
    'amazon-bedrock',
    'mistral',
    'nextbit',
    'atoma',
    'ai21',
    'minimax',
    'baseten',
    'anthropic',
    'featherless',
    'groq',
    'lambda',
    'azure',
    'ncompass',
    'deepseek',
    'hyperbolic',
    'crusoe',
    'cohere',
    'mancer',
    'avian',
    'perplexity',
    'novita',
    'siliconflow',
    'switchpoint',
    'xai',
    'inflection',
    'fireworks',
    'deepinfra',
    'inference-net',
    'inception',
    'atlas-cloud',
    'nvidia',
    'alibaba',
    'friendli',
    'infermatic',
    'targon',
    'ubicloud',
    'aion-labs',
    'liquid',
    'nineteen',
    'cloudflare',
    'nebius',
    'chutes',
    'enfer',
    'crofai',
    'open-inference',
    'phala',
    'gmicloud',
    'meta',
    'relace',
    'parasail',
    'together',
    'google-ai-studio',
    'google-vertex',
]
"""Known providers in the OpenRouter marketplace"""

OpenRouterProviderName = str | KnownOpenRouterProviders
"""Possible OpenRouter provider names.

Since OpenRouter is constantly updating their list of providers, we explicitly list some known providers but
allow any name in the type hints.
See [the OpenRouter API](https://openrouter.ai/docs/api-reference/list-available-providers) for a full list.
"""

OpenRouterTransforms = Literal['middle-out']
"""Available messages transforms for OpenRouter models with limited token windows.

Currently only supports 'middle-out', but is expected to grow in the future.
"""


class OpenRouterProviderConfig(TypedDict, total=False):
    """Represents the 'Provider' object from the OpenRouter API."""

    order: list[OpenRouterProviderName]
    """List of provider slugs to try in order (e.g. ["anthropic", "openai"]). [See details](https://openrouter.ai/docs/features/provider-routing#ordering-specific-providers)"""

    allow_fallbacks: bool
    """Whether to allow backup providers when the primary is unavailable. [See details](https://openrouter.ai/docs/features/provider-routing#disabling-fallbacks)"""

    require_parameters: bool
    """Only use providers that support all parameters in your request."""

    data_collection: Literal['allow', 'deny']
    """Control whether to use providers that may store data. [See details](https://openrouter.ai/docs/features/provider-routing#requiring-providers-to-comply-with-data-policies)"""

    zdr: bool
    """Restrict routing to only ZDR (Zero Data Retention) endpoints. [See details](https://openrouter.ai/docs/features/provider-routing#zero-data-retention-enforcement)"""

    only: list[OpenRouterProviderName]
    """List of provider slugs to allow for this request. [See details](https://openrouter.ai/docs/features/provider-routing#allowing-only-specific-providers)"""

    ignore: list[str]
    """List of provider slugs to skip for this request. [See details](https://openrouter.ai/docs/features/provider-routing#ignoring-providers)"""

    quantizations: list[Literal['int4', 'int8', 'fp4', 'fp6', 'fp8', 'fp16', 'bf16', 'fp32', 'unknown']]
    """List of quantization levels to filter by (e.g. ["int4", "int8"]). [See details](https://openrouter.ai/docs/features/provider-routing#quantization)"""

    sort: Literal['price', 'throughput', 'latency']
    """Sort providers by price or throughput. (e.g. "price" or "throughput"). [See details](https://openrouter.ai/docs/features/provider-routing#provider-sorting)"""

    max_price: _OpenRouterMaxPrice
    """The maximum pricing you want to pay for this request. [See details](https://openrouter.ai/docs/features/provider-routing#max-price)"""


class OpenRouterReasoning(TypedDict, total=False):
    """Configuration for reasoning tokens in OpenRouter requests.

    Reasoning tokens allow models to show their step-by-step thinking process.
    You can configure this using either OpenAI-style effort levels or Anthropic-style
    token limits, but not both simultaneously.
    """

    effort: Literal['high', 'medium', 'low']
    """OpenAI-style reasoning effort level. Cannot be used with max_tokens."""

    max_tokens: int
    """Anthropic-style specific token limit for reasoning. Cannot be used with effort."""

    exclude: bool
    """Whether to exclude reasoning tokens from the response. Default is False. All models support this."""

    enabled: bool
    """Whether to enable reasoning with default parameters. Default is inferred from effort or max_tokens."""


class OpenRouterUsageConfig(TypedDict, total=False):
    """Configuration for OpenRouter usage."""

    include: bool


class OpenRouterModelSettings(ModelSettings, total=False):
    """Settings used for an OpenRouter model request."""

    # ALL FIELDS MUST BE `openrouter_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openrouter_models: list[str]
    """A list of fallback models.

    These models will be tried, in order, if the main model returns an error. [See details](https://openrouter.ai/docs/features/model-routing#the-models-parameter)
    """

    openrouter_provider: OpenRouterProviderConfig
    """OpenRouter routes requests to the best available providers for your model. By default, requests are load balanced across the top providers to maximize uptime.

    You can customize how your requests are routed using the provider object. [See more](https://openrouter.ai/docs/features/provider-routing)"""

    openrouter_preset: str
    """Presets allow you to separate your LLM configuration from your code.

    Create and manage presets through the OpenRouter web application to control provider routing, model selection, system prompts, and other parameters, then reference them in OpenRouter API requests. [See more](https://openrouter.ai/docs/features/presets)"""

    openrouter_transforms: list[OpenRouterTransforms]
    """To help with prompts that exceed the maximum context size of a model.

    Transforms work by removing or truncating messages from the middle of the prompt, until the prompt fits within the model's context window. [See more](https://openrouter.ai/docs/features/message-transforms)
    """

    openrouter_reasoning: OpenRouterReasoning
    """To control the reasoning tokens in the request.

    The reasoning config object consolidates settings for controlling reasoning strength across different models. [See more](https://openrouter.ai/docs/use-cases/reasoning-tokens)
    """

    openrouter_usage: OpenRouterUsageConfig
    """To control the usage of the model.

    The usage config object consolidates settings for enabling detailed usage information. [See more](https://openrouter.ai/docs/use-cases/usage-accounting)
    """


class _OpenRouterError(BaseModel):
    """Utility class to validate error messages from OpenRouter."""

    code: int
    message: str


class _BaseReasoningDetail(BaseModel, frozen=True):
    """Common fields shared across all reasoning detail types."""

    id: str | None = None
    format: (
        Literal['unknown', 'openai-responses-v1', 'anthropic-claude-v1', 'xai-responses-v1', 'google-gemini-v1']
        | str
        | None
    )
    index: int | None
    type: Literal['reasoning.text', 'reasoning.summary', 'reasoning.encrypted']


class _ReasoningSummary(_BaseReasoningDetail, frozen=True):
    """Represents a high-level summary of the reasoning process."""

    type: Literal['reasoning.summary']
    summary: str = ''


class _ReasoningEncrypted(_BaseReasoningDetail, frozen=True):
    """Represents encrypted reasoning data."""

    type: Literal['reasoning.encrypted']
    data: str = ''


class _ReasoningText(_BaseReasoningDetail, frozen=True):
    """Represents raw text reasoning."""

    type: Literal['reasoning.text']
    text: str = ''
    signature: str | None = None


_OpenRouterReasoningDetail = _ReasoningSummary | _ReasoningEncrypted | _ReasoningText


def _from_reasoning_detail(reasoning: _OpenRouterReasoningDetail) -> ThinkingPart:
    provider_name = 'openrouter'
    provider_details = reasoning.model_dump(include={'format', 'index', 'type'})
    if isinstance(reasoning, _ReasoningText):
        return ThinkingPart(
            id=reasoning.id,
            content=reasoning.text,
            signature=reasoning.signature,
            provider_name=provider_name,
            provider_details=provider_details,
        )
    elif isinstance(reasoning, _ReasoningSummary):
        return ThinkingPart(
            id=reasoning.id, content=reasoning.summary, provider_name=provider_name, provider_details=provider_details
        )
    elif isinstance(reasoning, _ReasoningEncrypted):
        return ThinkingPart(
            id=reasoning.id,
            content='',
            signature=reasoning.data,
            provider_name=provider_name,
            provider_details=provider_details,
        )
    else:
        assert_never(reasoning)


def _into_reasoning_detail(thinking_part: ThinkingPart) -> _OpenRouterReasoningDetail | None:
    if thinking_part.provider_details is None:  # pragma: lax no cover
        return None

    data = _BaseReasoningDetail.model_validate(thinking_part.provider_details)

    if data.type == 'reasoning.text':
        return _ReasoningText(
            type=data.type,
            id=thinking_part.id,
            format=data.format,
            index=data.index,
            text=thinking_part.content,
            signature=thinking_part.signature,
        )
    elif data.type == 'reasoning.summary':
        return _ReasoningSummary(
            type=data.type,
            id=thinking_part.id,
            format=data.format,
            index=data.index,
            summary=thinking_part.content,
        )
    elif data.type == 'reasoning.encrypted':
        assert thinking_part.signature is not None
        return _ReasoningEncrypted(
            type=data.type,
            id=thinking_part.id,
            format=data.format,
            index=data.index,
            data=thinking_part.signature,
        )
    else:
        assert_never(data.type)


class _OpenRouterFileAnnotation(BaseModel, frozen=True):
    """File annotation from OpenRouter.

    OpenRouter can return file annotations when processing uploaded files like PDFs.
    The schema is flexible since OpenRouter doesn't document the exact fields.
    """

    type: Literal['file']
    file: dict[str, Any] | None = None


_OpenRouterAnnotation: TypeAlias = _OpenAIAnnotation | _OpenRouterFileAnnotation


class _OpenRouterFunction(chat_completion_message_function_tool_call.Function):
    arguments: str | None  # type: ignore[reportIncompatibleVariableOverride]
    """
    The arguments to call the function with, as generated by the model in JSON
    format. Note that the model does not always generate valid JSON, and may
    hallucinate parameters not defined by your function schema. Validate the
    arguments in your code before calling your function.
    """


class _OpenRouterChatCompletionMessageFunctionToolCall(chat.ChatCompletionMessageFunctionToolCall):
    function: _OpenRouterFunction  # type: ignore[reportIncompatibleVariableOverride]
    """The function that the model called."""


_OpenRouterChatCompletionMessageToolCallUnion: TypeAlias = Annotated[
    _OpenRouterChatCompletionMessageFunctionToolCall | chat.ChatCompletionMessageCustomToolCall,
    Discriminator(discriminator='type'),
]


class _OpenRouterCompletionMessage(chat.ChatCompletionMessage):
    """Wrapped chat completion message with OpenRouter specific attributes."""

    reasoning: str | None = None
    """The reasoning text associated with the message, if any."""

    reasoning_details: list[_OpenRouterReasoningDetail] | None = None
    """The reasoning details associated with the message, if any."""

    tool_calls: list[_OpenRouterChatCompletionMessageToolCallUnion] | None = None  # type: ignore[reportIncompatibleVariableOverride]
    """The tool calls generated by the model, such as function calls."""

    annotations: list[_OpenRouterAnnotation] | None = None  # type: ignore[reportIncompatibleVariableOverride]
    """Annotations associated with the message, supporting both url_citation and file types."""


class _OpenRouterChoice(chat_completion.Choice):
    """Wraps OpenAI chat completion choice with OpenRouter specific attributes."""

    native_finish_reason: str | None
    """The provided finish reason by the downstream provider from OpenRouter."""

    finish_reason: Literal['stop', 'length', 'tool_calls', 'content_filter', 'error']  # type: ignore[reportIncompatibleVariableOverride]
    """OpenRouter specific finish reasons.

    Notably, removes 'function_call' and adds 'error'  finish reasons.
    """

    message: _OpenRouterCompletionMessage  # type: ignore[reportIncompatibleVariableOverride]
    """A wrapped chat completion message with OpenRouter specific attributes."""


@dataclass
class _OpenRouterCostDetails:
    """OpenRouter specific cost details."""

    upstream_inference_cost: float | None = None

    # TODO rework fields, tests/models/cassettes/test_openrouter/test_openrouter_google_nested_schema.yaml
    # shows an `upstream_inference_completions_cost` field as well


class _OpenRouterPromptTokenDetails(completion_usage.PromptTokensDetails):
    """Wraps OpenAI completion token details with OpenRouter specific attributes."""

    video_tokens: int | None = None


class _OpenRouterCompletionTokenDetails(completion_usage.CompletionTokensDetails):
    """Wraps OpenAI completion token details with OpenRouter specific attributes."""

    image_tokens: int | None = None


class _OpenRouterUsage(completion_usage.CompletionUsage):
    """Wraps OpenAI completion usage with OpenRouter specific attributes."""

    cost: float | None = None

    cost_details: _OpenRouterCostDetails | None = None

    is_byok: bool | None = None

    prompt_tokens_details: _OpenRouterPromptTokenDetails | None = None  # type: ignore[reportIncompatibleVariableOverride]

    completion_tokens_details: _OpenRouterCompletionTokenDetails | None = None  # type: ignore[reportIncompatibleVariableOverride]


class _OpenRouterChatCompletion(chat.ChatCompletion):
    """Wraps OpenAI chat completion with OpenRouter specific attributes."""

    provider: str
    """The downstream provider that was used by OpenRouter."""

    choices: list[_OpenRouterChoice]  # type: ignore[reportIncompatibleVariableOverride]
    """A list of chat completion choices modified with OpenRouter specific attributes."""

    error: _OpenRouterError | None = None
    """OpenRouter specific error attribute."""

    usage: _OpenRouterUsage | None = None  # type: ignore[reportIncompatibleVariableOverride]
    """OpenRouter specific usage attribute."""


def _map_openrouter_provider_details(
    response: _OpenRouterChatCompletion | _OpenRouterChatCompletionChunk,
) -> dict[str, Any]:
    provider_details: dict[str, Any] = {}

    provider_details['downstream_provider'] = response.provider
    if native_finish_reason := response.choices[0].native_finish_reason:
        provider_details['finish_reason'] = native_finish_reason

    if usage := response.usage:
        if cost := usage.cost:
            provider_details['cost'] = cost

        if cost_details := usage.cost_details:
            provider_details['upstream_inference_cost'] = cost_details.upstream_inference_cost

        if (is_byok := usage.is_byok) is not None:
            provider_details['is_byok'] = is_byok

    return provider_details


def _openrouter_settings_to_openai_settings(model_settings: OpenRouterModelSettings) -> OpenAIChatModelSettings:
    """Transforms a 'OpenRouterModelSettings' object into an 'OpenAIChatModelSettings' object.

    Args:
        model_settings: The 'OpenRouterModelSettings' object to transform.

    Returns:
        An 'OpenAIChatModelSettings' object with equivalent settings.
    """
    extra_body = cast(dict[str, Any], model_settings.get('extra_body', {}))

    if models := model_settings.pop('openrouter_models', None):
        extra_body['models'] = models
    if provider := model_settings.pop('openrouter_provider', None):
        extra_body['provider'] = provider
    if preset := model_settings.pop('openrouter_preset', None):
        extra_body['preset'] = preset
    if transforms := model_settings.pop('openrouter_transforms', None):
        extra_body['transforms'] = transforms
    if reasoning := model_settings.pop('openrouter_reasoning', None):
        extra_body['reasoning'] = reasoning
    if usage := model_settings.pop('openrouter_usage', None):
        extra_body['usage'] = usage

    model_settings['extra_body'] = extra_body

    return OpenAIChatModelSettings(**model_settings)  # type: ignore[reportCallIssue]


class OpenRouterModel(OpenAIChatModel):
    """Extends OpenAIModel to capture extra metadata for Openrouter."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['openrouter'] | Provider[AsyncOpenAI] = 'openrouter',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenRouter model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use for authentication and API access. If not provided, a new provider will be created with the default settings.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider or OpenRouterProvider(), profile=profile, settings=settings)

    @override
    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        merged_settings, customized_parameters = super().prepare_request(model_settings, model_request_parameters)
        new_settings = _openrouter_settings_to_openai_settings(cast(OpenRouterModelSettings, merged_settings or {}))
        return new_settings, customized_parameters

    @override
    def _validate_completion(self, response: chat.ChatCompletion) -> _OpenRouterChatCompletion:
        response = _OpenRouterChatCompletion.model_validate(response.model_dump())

        if error := response.error:
            raise ModelHTTPError(status_code=error.code, model_name=response.model, body=error.message)

        return response

    @override
    def _process_thinking(self, message: chat.ChatCompletionMessage) -> list[ThinkingPart] | None:
        assert isinstance(message, _OpenRouterCompletionMessage)

        if reasoning_details := message.reasoning_details:
            return [_from_reasoning_detail(detail) for detail in reasoning_details]
        else:
            return super()._process_thinking(message)

    @override
    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any] | None:
        assert isinstance(response, _OpenRouterChatCompletion)

        provider_details = super()._process_provider_details(response) or {}
        provider_details.update(_map_openrouter_provider_details(response))
        return provider_details or None

    @dataclass
    class _MapModelResponseContext(OpenAIChatModel._MapModelResponseContext):  # type: ignore[reportPrivateUsage]
        reasoning_details: list[dict[str, Any]] = field(default_factory=list)

        def _into_message_param(self) -> chat.ChatCompletionAssistantMessageParam:
            message_param = super()._into_message_param()
            if self.reasoning_details:
                message_param['reasoning_details'] = self.reasoning_details  # type: ignore[reportGeneralTypeIssues]
            return message_param

        @override
        def _map_response_thinking_part(self, item: ThinkingPart) -> None:
            assert isinstance(self._model, OpenRouterModel)
            if item.provider_name == self._model.system:
                if reasoning_detail := _into_reasoning_detail(item):  # pragma: lax no cover
                    self.reasoning_details.append(reasoning_detail.model_dump())
            else:  # pragma: lax no cover
                super()._map_response_thinking_part(item)

    @property
    @override
    def _streamed_response_cls(self):
        return OpenRouterStreamedResponse

    @override
    def _map_finish_reason(  # type: ignore[reportIncompatibleMethodOverride]
        self, key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'error']
    ) -> FinishReason | None:
        return _CHAT_FINISH_REASON_MAP.get(key)


class _OpenRouterChoiceDelta(chat_completion_chunk.ChoiceDelta):
    """Wrapped chat completion message with OpenRouter specific attributes."""

    reasoning: str | None = None
    """The reasoning text associated with the message, if any."""

    reasoning_details: list[_OpenRouterReasoningDetail] | None = None
    """The reasoning details associated with the message, if any."""

    annotations: list[_OpenRouterAnnotation] | None = None
    """Annotations associated with the message, supporting both url_citation and file types."""


class _OpenRouterChunkChoice(chat_completion_chunk.Choice):
    """Wraps OpenAI chat completion chunk choice with OpenRouter specific attributes."""

    native_finish_reason: str | None
    """The provided finish reason by the downstream provider from OpenRouter."""

    finish_reason: Literal['stop', 'length', 'tool_calls', 'content_filter', 'error'] | None  # type: ignore[reportIncompatibleVariableOverride]
    """OpenRouter specific finish reasons for streaming chunks.

    Notably, removes 'function_call' and adds 'error' finish reasons.
    """

    delta: _OpenRouterChoiceDelta  # type: ignore[reportIncompatibleVariableOverride]
    """A wrapped chat completion delta with OpenRouter specific attributes."""


class _OpenRouterChatCompletionChunk(chat.ChatCompletionChunk):
    """Wraps OpenAI chat completion with OpenRouter specific attributes."""

    provider: str
    """The downstream provider that was used by OpenRouter."""

    choices: list[_OpenRouterChunkChoice]  # type: ignore[reportIncompatibleVariableOverride]
    """A list of chat completion chunk choices modified with OpenRouter specific attributes."""

    usage: _OpenRouterUsage | None = None  # type: ignore[reportIncompatibleVariableOverride]
    """Usage statistics for the completion request."""


@dataclass
class OpenRouterStreamedResponse(OpenAIStreamedResponse):
    """Implementation of `StreamedResponse` for OpenRouter models."""

    @override
    async def _validate_response(self):
        try:
            async for chunk in self._response:
                yield _OpenRouterChatCompletionChunk.model_validate(chunk.model_dump())
        except APIError as e:
            error = _OpenRouterError.model_validate(e.body)
            raise ModelHTTPError(status_code=error.code, model_name=self._model_name, body=error.message)

    @override
    def _map_thinking_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        assert isinstance(choice, _OpenRouterChunkChoice)

        if reasoning_details := choice.delta.reasoning_details:
            for i, detail in enumerate(reasoning_details):
                thinking_part = _from_reasoning_detail(detail)
                # Use unique vendor_part_id for each reasoning detail type to prevent
                # different detail types (e.g., reasoning.text, reasoning.encrypted)
                # from being incorrectly merged into a single ThinkingPart.
                # This is required for Gemini 3 Pro which returns multiple reasoning
                # detail types that must be preserved separately for thought_signature handling.
                vendor_id = f'reasoning_detail_{detail.type}_{i}'
                yield from self._parts_manager.handle_thinking_delta(
                    vendor_part_id=vendor_id,
                    id=thinking_part.id,
                    content=thinking_part.content,
                    signature=thinking_part.signature,
                    provider_name=self._provider_name,
                    provider_details=thinking_part.provider_details,
                )
        else:
            return super()._map_thinking_delta(choice)

    @override
    def _map_provider_details(self, chunk: chat.ChatCompletionChunk) -> dict[str, Any] | None:
        assert isinstance(chunk, _OpenRouterChatCompletionChunk)

        provider_details = super()._map_provider_details(chunk) or {}
        provider_details.update(_map_openrouter_provider_details(chunk))
        return provider_details or None

    @override
    def _map_finish_reason(  # type: ignore[reportIncompatibleMethodOverride]
        self, key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'error']
    ) -> FinishReason | None:
        return _CHAT_FINISH_REASON_MAP.get(key)
