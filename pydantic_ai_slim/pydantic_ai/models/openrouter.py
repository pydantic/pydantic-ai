import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from pydantic import AliasChoices, BaseModel, Field, TypeAdapter
from typing_extensions import TypedDict, assert_never, override

from ..exceptions import ModelHTTPError, UnexpectedModelBehavior
from ..messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FinishReason,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from ..profiles import ModelProfileSpec
from ..providers.openrouter import OpenRouterProvider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import ModelRequestParameters

try:
    from openai import APIError
    from openai.types import chat, completion_usage
    from openai.types.chat import chat_completion, chat_completion_chunk

    from .openai import OpenAIChatModel, OpenAIChatModelSettings, OpenAIStreamedResponse
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


class OpenRouterMaxPrice(TypedDict, total=False):
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

Transforms = Literal['middle-out']
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

    max_price: OpenRouterMaxPrice
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


class WebPlugin(TypedDict, total=False):
    """You can incorporate relevant web search results for any model on OpenRouter by activating and customizing the web plugin.

    The web search plugin is powered by native search for Anthropic and OpenAI natively and by Exa for other models. For Exa, it uses their "auto" method (a combination of keyword search and embeddings-based web search) to find the most relevant results and augment/ground your prompt.
    """

    id: Literal['web']

    engine: Literal['native', 'exa', 'undefined']
    """The web search plugin supports the following options for the engine parameter:

    `native`: Always uses the model provider's built-in web search capabilities
    `exa`: Uses Exa's search API for web results
    `undefined` (not specified): Uses native search if available for the provider, otherwise falls back to Exa

    Native search is used by default for OpenAI and Anthropic models that support it
    Exa search is used for all other models or when native search is not supported.

    When you explicitly specify "engine": "native", it will always attempt to use the provider's native search, even if the model doesn't support it (which may result in an error)."""

    max_results: int
    """The maximum results allowed by the web plugin."""

    search_prompt: str
    """The prompt used to attach results to your message."""


OpenRouterPlugin = WebPlugin


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

    openrouter_transforms: list[Transforms]
    """To help with prompts that exceed the maximum context size of a model.

    Transforms work by removing or truncating messages from the middle of the prompt, until the prompt fits within the model's context window. [See more](https://openrouter.ai/docs/features/message-transforms)
    """

    openrouter_reasoning: OpenRouterReasoning
    """To control the reasoning tokens in the request.

    The reasoning config object consolidates settings for controlling reasoning strength across different models. [See more](https://openrouter.ai/docs/use-cases/reasoning-tokens)
    """

    openrouter_plugins: list[OpenRouterPlugin]
    """To enable plugins in the request.

    Plugins are tools that can be used to extend the functionality of the model. [See more](https://openrouter.ai/docs/features/web-search)
    """

    openrouter_usage: OpenRouterUsageConfig
    """To control the usage of the model.

    The usage config object consolidates settings for enabling detailed usage information. [See more](https://openrouter.ai/docs/use-cases/usage-accounting)
    """


class OpenRouterError(BaseModel):
    """Utility class to validate error messages from OpenRouter."""

    code: int
    message: str


class _BaseReasoningDetail(BaseModel):
    """Common fields shared across all reasoning detail types."""

    id: str | None = None
    format: Literal['unknown', 'openai-responses-v1', 'anthropic-claude-v1', 'xai-responses-v1'] | None
    index: int | None


class _ReasoningSummary(_BaseReasoningDetail):
    """Represents a high-level summary of the reasoning process."""

    type: Literal['reasoning.summary']
    summary: str = Field(validation_alias=AliasChoices('summary', 'content'))


class _ReasoningEncrypted(_BaseReasoningDetail):
    """Represents encrypted reasoning data."""

    type: Literal['reasoning.encrypted']
    data: str = Field(validation_alias=AliasChoices('data', 'signature'))


class _ReasoningText(_BaseReasoningDetail):
    """Represents raw text reasoning."""

    type: Literal['reasoning.text']
    text: str = Field(validation_alias=AliasChoices('text', 'content'))
    signature: str | None = None


_OpenRouterReasoningDetail = _ReasoningSummary | _ReasoningEncrypted | _ReasoningText
_reasoning_detail_adapter: TypeAdapter[_OpenRouterReasoningDetail] = TypeAdapter(_OpenRouterReasoningDetail)


def _from_reasoning_detail(reasoning: _OpenRouterReasoningDetail) -> ThinkingPart:
    provider_name = 'openrouter'
    reasoning_id = reasoning.model_dump_json(include={'id', 'format', 'index', 'type'})
    if isinstance(reasoning, _ReasoningText):
        return ThinkingPart(
            id=reasoning_id,
            content=reasoning.text,
            signature=reasoning.signature,
            provider_name=provider_name,
        )
    elif isinstance(reasoning, _ReasoningSummary):
        return ThinkingPart(
            id=reasoning_id,
            content=reasoning.summary,
            provider_name=provider_name,
        )
    else:
        return ThinkingPart(
            id=reasoning_id,
            content='',
            signature=reasoning.data,
            provider_name=provider_name,
        )


def _into_reasoning_detail(thinking_part: ThinkingPart) -> _OpenRouterReasoningDetail:
    if thinking_part.id is None:  # pragma: lax no cover
        raise UnexpectedModelBehavior('OpenRouter thinking part has no ID')

    data = asdict(thinking_part)
    data.update(json.loads(thinking_part.id))
    return _reasoning_detail_adapter.validate_python(data)


class OpenRouterCompletionMessage(chat.ChatCompletionMessage):
    """Wrapped chat completion message with OpenRouter specific attributes."""

    reasoning: str | None = None
    """The reasoning text associated with the message, if any."""

    reasoning_details: list[_OpenRouterReasoningDetail] | None = None
    """The reasoning details associated with the message, if any."""


class OpenRouterChoice(chat_completion.Choice):
    """Wraps OpenAI chat completion choice with OpenRouter specific attributes."""

    native_finish_reason: str
    """The provided finish reason by the downstream provider from OpenRouter."""

    finish_reason: Literal['stop', 'length', 'tool_calls', 'content_filter', 'error']  # type: ignore[reportIncompatibleVariableOverride]
    """OpenRouter specific finish reasons.

    Notably, removes 'function_call' and adds 'error'  finish reasons.
    """

    message: OpenRouterCompletionMessage  # type: ignore[reportIncompatibleVariableOverride]
    """A wrapped chat completion message with OpenRouter specific attributes."""


class OpenRouterCostDetails(BaseModel):
    """OpenRouter specific cost details."""

    upstream_inference_cost: int | None = None


class OpenRouterCompletionTokenDetails(completion_usage.CompletionTokensDetails):
    """Wraps OpenAI completion token details with OpenRouter specific attributes."""

    image_tokens: int | None = None


class OpenRouterUsage(completion_usage.CompletionUsage):
    """Wraps OpenAI completion usage with OpenRouter specific attributes."""

    cost: float | None = None

    cost_details: OpenRouterCostDetails | None = None

    is_byok: bool | None = None

    completion_tokens_details: OpenRouterCompletionTokenDetails | None = None  # type: ignore[reportIncompatibleVariableOverride]


class OpenRouterChatCompletion(chat.ChatCompletion):
    """Wraps OpenAI chat completion with OpenRouter specific attributes."""

    provider: str
    """The downstream provider that was used by OpenRouter."""

    choices: list[OpenRouterChoice]  # type: ignore[reportIncompatibleVariableOverride]
    """A list of chat completion choices modified with OpenRouter specific attributes."""

    error: OpenRouterError | None = None
    """OpenRouter specific error attribute."""

    usage: OpenRouterUsage | None = None  # type: ignore[reportIncompatibleVariableOverride]
    """OpenRouter specific usage attribute."""


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
    if usage := model_settings.pop('openrouter_usage', None):
        extra_body['usage'] = usage

    model_settings['extra_body'] = extra_body

    return OpenAIChatModelSettings(**model_settings)  # type: ignore[reportCallIssue]


def _map_usage(
    response: chat.ChatCompletion | chat.ChatCompletionChunk,
    provider: str,
    provider_url: str,
    model: str,
) -> RequestUsage:
    assert isinstance(response, OpenRouterChatCompletion) or isinstance(response, OpenRouterChatCompletionChunk)
    builder = RequestUsage()

    response_usage = response.usage
    if response_usage is None:
        return builder

    builder.input_tokens = response_usage.prompt_tokens
    builder.output_tokens = response_usage.completion_tokens

    if prompt_token_details := response_usage.prompt_tokens_details:
        if cached_tokens := prompt_token_details.cached_tokens:
            builder.cache_read_tokens = cached_tokens

        if audio_tokens := prompt_token_details.audio_tokens:  # pragma: lax no cover
            builder.input_audio_tokens = audio_tokens

        if video_tokens := prompt_token_details.video_tokens:  # pragma: lax no cover
            builder.details['input_video_tokens'] = video_tokens

    if completion_token_details := response_usage.completion_tokens_details:
        if reasoning_tokens := completion_token_details.reasoning_tokens:
            builder.details['reasoning_tokens'] = reasoning_tokens

        if image_tokens := completion_token_details.image_tokens:  # pragma: lax no cover
            builder.details['output_image_tokens'] = image_tokens

    if (is_byok := response_usage.is_byok) is not None:
        builder.details['is_byok'] = is_byok

    if cost := response_usage.cost:
        builder.details['cost'] = int(cost * 1000000)  # convert to microcost

    return builder


class OpenRouterModel(OpenAIChatModel):
    """Extends OpenAIModel to capture extra metadata for Openrouter."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenRouterProvider | None = None,
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
    def _validate_completion(self, response: chat.ChatCompletion) -> OpenRouterChatCompletion:
        response = OpenRouterChatCompletion.model_validate(response.model_dump())

        if error := response.error:
            raise ModelHTTPError(status_code=error.code, model_name=response.model, body=error.message)

        return response

    @override
    def _process_thinking(self, message: chat.ChatCompletionMessage) -> list[ThinkingPart] | None:
        assert isinstance(message, OpenRouterCompletionMessage)

        if reasoning_details := message.reasoning_details:
            return [_from_reasoning_detail(detail) for detail in reasoning_details]
        else:
            return super()._process_thinking(message)

    @override
    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any]:
        assert isinstance(response, OpenRouterChatCompletion)

        provider_details = super()._process_provider_details(response)

        provider_details['downstream_provider'] = response.provider
        provider_details['native_finish_reason'] = response.choices[0].native_finish_reason

        return provider_details

    @override
    def _map_model_response(self, message: ModelResponse) -> chat.ChatCompletionMessageParam:
        texts: list[str] = []
        tool_calls: list[chat.ChatCompletionMessageFunctionToolCallParam] = []
        reasoning_details: list[dict[str, Any]] = []
        for item in message.parts:
            if isinstance(item, TextPart):
                texts.append(item.content)
            elif isinstance(item, ThinkingPart):
                if item.provider_name == self.system:
                    reasoning_details.append(_into_reasoning_detail(item).model_dump())
                else:  # pragma: no cover
                    pass
            elif isinstance(item, ToolCallPart):
                tool_calls.append(self._map_tool_call(item))
            elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                pass
            elif isinstance(item, FilePart):  # pragma: no cover
                pass
            else:
                assert_never(item)
        message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
        if texts:
            message_param['content'] = '\n\n'.join(texts)
        else:
            message_param['content'] = None
        if tool_calls:
            message_param['tool_calls'] = tool_calls
        if reasoning_details:
            message_param['reasoning_details'] = reasoning_details  # type: ignore[reportGeneralTypeIssues]
        return message_param

    @property
    @override
    def _streamed_response_cls(self):
        return OpenRouterStreamedResponse

    @override
    def _map_usage(self, response: chat.ChatCompletion) -> RequestUsage:
        return _map_usage(response, self._provider.name, self._provider.base_url, self._model_name)

    @override
    def _map_finish_reason(  # type: ignore[reportIncompatibleMethodOverride]
        self, key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'error']
    ) -> FinishReason | None:
        return _CHAT_FINISH_REASON_MAP.get(key)


class OpenRouterChoiceDelta(chat_completion_chunk.ChoiceDelta):
    """Wrapped chat completion message with OpenRouter specific attributes."""

    reasoning: str | None = None
    """The reasoning text associated with the message, if any."""

    reasoning_details: list[_OpenRouterReasoningDetail] | None = None
    """The reasoning details associated with the message, if any."""


class OpenRouterChunkChoice(chat_completion_chunk.Choice):
    """Wraps OpenAI chat completion chunk choice with OpenRouter specific attributes."""

    native_finish_reason: str | None
    """The provided finish reason by the downstream provider from OpenRouter."""

    finish_reason: Literal['stop', 'length', 'tool_calls', 'content_filter', 'error'] | None  # type: ignore[reportIncompatibleVariableOverride]
    """OpenRouter specific finish reasons for streaming chunks.

    Notably, removes 'function_call' and adds 'error' finish reasons.
    """

    delta: OpenRouterChoiceDelta  # type: ignore[reportIncompatibleVariableOverride]
    """A wrapped chat completion delta with OpenRouter specific attributes."""


class OpenRouterChatCompletionChunk(chat.ChatCompletionChunk):
    """Wraps OpenAI chat completion with OpenRouter specific attributes."""

    provider: str
    """The downstream provider that was used by OpenRouter."""

    choices: list[OpenRouterChunkChoice]  # type: ignore[reportIncompatibleVariableOverride]
    """A list of chat completion chunk choices modified with OpenRouter specific attributes."""

    usage: OpenRouterUsage | None = None  # type: ignore[reportIncompatibleVariableOverride]
    """Usage statistics for the completion request."""


@dataclass
class OpenRouterStreamedResponse(OpenAIStreamedResponse):
    """Implementation of `StreamedResponse` for OpenRouter models."""

    @override
    async def _validate_response(self):
        try:
            async for chunk in self._response:
                yield OpenRouterChatCompletionChunk.model_validate(chunk.model_dump())
        except APIError as e:
            error = OpenRouterError.model_validate(e.body)
            raise ModelHTTPError(status_code=error.code, model_name=self._model_name, body=error.message)

    @override
    def _map_thinking_delta(self, choice: chat_completion_chunk.Choice):
        assert isinstance(choice, OpenRouterChunkChoice)

        if reasoning_details := choice.delta.reasoning_details:
            for detail in reasoning_details:
                thinking_part = _from_reasoning_detail(detail)
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id='reasoning_detail',
                    id=thinking_part.id,
                    content=thinking_part.content,
                    provider_name=self._provider_name,
                )

    @override
    def _map_provider_details(self, chunk: chat.ChatCompletionChunk) -> dict[str, str] | None:
        assert isinstance(chunk, OpenRouterChatCompletionChunk)

        if provider_details := super()._map_provider_details(chunk):
            if provider := chunk.provider:  # pragma: lax no cover
                provider_details['downstream_provider'] = provider

            if native_finish_reason := chunk.choices[0].native_finish_reason:  # pragma: lax no cover
                provider_details['native_finish_reason'] = native_finish_reason

            return provider_details

    @override
    def _map_usage(self, response: chat.ChatCompletionChunk):
        return _map_usage(response, self._provider_name, self._provider_url, self._model_name)

    @override
    def _map_finish_reason(  # type: ignore[reportIncompatibleMethodOverride]
        self, key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'error']
    ) -> FinishReason | None:
        return _CHAT_FINISH_REASON_MAP.get(key)
