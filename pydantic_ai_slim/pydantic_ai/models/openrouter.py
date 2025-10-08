from typing import Any, Literal, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from typing_extensions import TypedDict

from ..exceptions import ModelHTTPError, UnexpectedModelBehavior
from ..messages import ModelResponse
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
from . import ModelRequestParameters
from .openai import OpenAIChatModel, OpenAIChatModelSettings


class OpenRouterMaxPrice(TypedDict, total=False):
    """The object specifying the maximum price you want to pay for this request. USD price per million tokens, for prompt and completion."""

    prompt: int
    completion: int
    image: int
    audio: int
    request: int


LatestOpenRouterSlugs = Literal[
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

OpenRouterSlug = str | LatestOpenRouterSlugs
"""Possible OpenRouter provider slugs.

Since OpenRouter is constantly updating their list of providers, we explicitly list some known providers but
allow any name in the type hints.
See [the OpenRouter API](https://openrouter.ai/docs/api-reference/list-available-providers) for a full list.
"""

Transforms = Literal['middle-out']
"""Available messages transforms for OpenRouter models with limited token windows.

Currently only supports 'middle-out', but is expected to grow in the future.
"""


class OpenRouterProvider(TypedDict, total=False):
    """Represents the 'Provider' object from the OpenRouter API."""

    order: list[OpenRouterSlug]
    """List of provider slugs to try in order (e.g. ["anthropic", "openai"]). [See details](https://openrouter.ai/docs/features/provider-routing#ordering-specific-providers)"""

    allow_fallbacks: bool
    """Whether to allow backup providers when the primary is unavailable. [See details](https://openrouter.ai/docs/features/provider-routing#disabling-fallbacks)"""

    require_parameters: bool
    """Only use providers that support all parameters in your request."""

    data_collection: Literal['allow', 'deny']
    """Control whether to use providers that may store data. [See details](https://openrouter.ai/docs/features/provider-routing#requiring-providers-to-comply-with-data-policies)"""

    zdr: bool
    """Restrict routing to only ZDR (Zero Data Retention) endpoints. [See details](https://openrouter.ai/docs/features/provider-routing#zero-data-retention-enforcement)"""

    only: list[OpenRouterSlug]
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


class OpenRouterModelSettings(ModelSettings, total=False):
    """Settings used for an OpenRouter model request."""

    # ALL FIELDS MUST BE `openrouter_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openrouter_models: list[str]
    """A list of fallback models.

    These models will be tried, in order, if the main model returns an error. [See details](https://openrouter.ai/docs/features/model-routing#the-models-parameter)
    """

    openrouter_provider: OpenRouterProvider
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


class OpenRouterError(BaseModel):
    """Utility class to validate error messages from OpenRouter."""

    code: int
    message: str


def _openrouter_settings_to_openai_settings(model_settings: OpenRouterModelSettings) -> OpenAIChatModelSettings:
    """Transforms a 'OpenRouterModelSettings' object into an 'OpenAIChatModelSettings' object.

    Args:
        model_settings: The 'OpenRouterModelSettings' object to transform.

    Returns:
        An 'OpenAIChatModelSettings' object with equivalent settings.
    """
    extra_body: dict[str, Any] = {}

    if models := model_settings.get('openrouter_models'):
        extra_body['models'] = models
    if provider := model_settings.get('openrouter_provider'):
        extra_body['provider'] = provider
    if preset := model_settings.get('openrouter_preset'):
        extra_body['preset'] = preset
    if transforms := model_settings.get('openrouter_transforms'):
        extra_body['transforms'] = transforms

    base_keys = ModelSettings.__annotations__.keys()
    base_data: dict[str, Any] = {k: model_settings[k] for k in base_keys if k in model_settings}

    new_settings = OpenAIChatModelSettings(**base_data, extra_body=extra_body)

    return new_settings


def _verify_response_is_not_error(response: ChatCompletion) -> ChatCompletion:
    """Checks a pre-validation 'ChatCompletion' object for the error attribute.

    Args:
        response: The 'ChatCompletion' object to validate.

    Returns:
        The same 'ChatCompletion' object.

    Raises:
        ModelHTTPError: If the response contains an error attribute.
        UnexpectedModelBehavior: If the response does not contain an error attribute but contains an 'error' finish_reason.
    """
    if openrouter_error := getattr(response, 'error', None):
        error = OpenRouterError.model_validate(openrouter_error)
        raise ModelHTTPError(status_code=error.code, model_name=response.model, body=error.message)
    else:
        choice = response.choices[0]

        if choice.finish_reason == 'error':  # type: ignore[reportUnnecessaryComparison]
            raise UnexpectedModelBehavior(
                'Invalid response from OpenRouter chat completions endpoint, error finish_reason without error data'
            )

        return response


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
            provider: The provider to use for authentication and API access. Currently, uses OpenAI as the internal client. Can be either the string
                'openrouter' or an instance of `Provider[AsyncOpenAI]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        merged_settings, customized_parameters = super().prepare_request(model_settings, model_request_parameters)
        new_settings = _openrouter_settings_to_openai_settings(cast(OpenRouterModelSettings, merged_settings or {}))
        return new_settings, customized_parameters

    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        if not isinstance(response, ChatCompletion):
            raise UnexpectedModelBehavior(
                'Invalid response from OpenRouter chat completions endpoint, expected JSON data'
            )

        response = _verify_response_is_not_error(response)

        model_response = super()._process_response(response=response)

        provider_details: dict[str, Any] = {}

        if openrouter_provider := getattr(response, 'provider', None):  # pragma: lax no cover
            provider_details['downstream_provider'] = openrouter_provider

        choice = response.choices[0]

        if native_finish_reason := getattr(choice, 'native_finish_reason', None):  # pragma: lax no cover
            provider_details['native_finish_reason'] = native_finish_reason

        if reasoning_details := getattr(choice.message, 'reasoning_details', None):
            provider_details['reasoning_details'] = reasoning_details

        model_response.provider_details = provider_details

        return model_response
