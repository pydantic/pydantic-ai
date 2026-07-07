"""Snowflake Cortex model implementation using Snowflake's OpenAI-compatible Chat Completions API."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import field_validator
from typing_extensions import TypedDict, override

from ..messages import ModelResponseStreamEvent, ThinkingPart
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings, ThinkingLevel
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI, omit
    from openai.types import chat
    from openai.types.chat import chat_completion, chat_completion_chunk

    from ..providers.snowflake import SnowflakeModelProfile
    from .openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIStreamedResponse,
        _ChatCompletion,  # pyright: ignore[reportPrivateUsage]
        _ChatCompletionChunk,  # pyright: ignore[reportPrivateUsage]
    )

    # Cortex adopted OpenRouter's `reasoning_details` format for Claude reasoning, so we reuse
    # the OpenRouter reasoning detail codec.
    from .openrouter import (
        _from_reasoning_detail,  # pyright: ignore[reportPrivateUsage]
        _into_reasoning_detail,  # pyright: ignore[reportPrivateUsage]
        _OpenRouterReasoningDetail,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Snowflake model, '
        'you can use the `snowflake` optional group — `pip install "pydantic-ai-slim[snowflake]"`'
    ) from _import_error

__all__ = ('SnowflakeModel', 'SnowflakeModelName', 'SnowflakeModelSettings', 'SnowflakeReasoning')

LatestSnowflakeModelNames = Literal[
    'claude-4-sonnet',
    'claude-haiku-4-5',
    'claude-opus-4-5',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'claude-sonnet-4-5',
    'claude-sonnet-4-6',
    'deepseek-r1',
    'llama3.1-405b',
    'llama3.1-70b',
    'llama3.1-8b',
    'llama4-maverick',
    'mistral-7b',
    'mistral-large',
    'mistral-large2',
    'openai-gpt-4.1',
    'openai-gpt-5',
    'openai-gpt-5-chat',
    'openai-gpt-5-mini',
    'openai-gpt-5-nano',
    'openai-gpt-5.1',
    'openai-gpt-5.2',
    'openai-gpt-5.4',
    'snowflake-llama-3.3-70b',
]

SnowflakeModelName = str | LatestSnowflakeModelNames
"""Possible Snowflake Cortex model names.

Since Snowflake Cortex serves a variety of models and the list changes frequently, we explicitly
list known models but allow any name in the type hints. Fine-tuned models can be referenced as
`database.schema.model`.

See <https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api> for an up to date list of models.
"""

_REASONING_EFFORT_MAP: dict[ThinkingLevel, Literal['low', 'medium', 'high']] = {
    True: 'medium',
    'minimal': 'low',
    'low': 'low',
    'medium': 'medium',
    'high': 'high',
    'xhigh': 'high',
}


class SnowflakeReasoning(TypedDict, total=False):
    """Configuration for reasoning tokens in Snowflake Cortex requests to Claude models.

    See <https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api> for details.
    """

    effort: Literal['high', 'medium', 'low']
    """Reasoning effort level. Converted to a reasoning token budget by Cortex. Cannot be used with `max_tokens`."""

    max_tokens: int
    """Specific token limit for reasoning. Cannot be used with `effort`."""


class SnowflakeModelSettings(ModelSettings, total=False):
    """Settings used for a Snowflake Cortex model request.

    ALL FIELDS MUST BE `snowflake_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    snowflake_reasoning: SnowflakeReasoning
    """Configure reasoning tokens for Claude models.

    Defaults to an effort level based on the unified `thinking` setting.
    """


class _SnowflakeCompletionMessage(chat.ChatCompletionMessage):
    """Chat completion message with the OpenRouter-format `reasoning_details` Cortex returns for Claude models."""

    reasoning_details: list[_OpenRouterReasoningDetail] | None = None
    """The reasoning details associated with the message, if any."""


class _SnowflakeChoice(chat_completion.Choice):
    message: _SnowflakeCompletionMessage  # pyright: ignore[reportIncompatibleVariableOverride]


class _SnowflakeChatCompletion(_ChatCompletion):
    choices: list[_SnowflakeChoice]  # pyright: ignore[reportIncompatibleVariableOverride]


class _SnowflakeChoiceDelta(chat_completion_chunk.ChoiceDelta):
    reasoning_details: list[_OpenRouterReasoningDetail] | None = None
    """The reasoning details associated with the delta, if any."""


class _SnowflakeChunkChoice(chat_completion_chunk.Choice):
    delta: _SnowflakeChoiceDelta  # pyright: ignore[reportIncompatibleVariableOverride]

    @field_validator('finish_reason', mode='before')
    @classmethod
    def _coerce_empty_finish_reason(cls, value: Any) -> Any:
        # Cortex returns an empty `finish_reason` for Claude models.
        return value or None


class _SnowflakeChatCompletionChunk(_ChatCompletionChunk):
    choices: list[_SnowflakeChunkChoice]  # pyright: ignore[reportIncompatibleVariableOverride]


@dataclass(init=False)
class SnowflakeModel(OpenAIChatModel):
    """A model that uses Snowflake Cortex's OpenAI-compatible Chat Completions API.

    Snowflake Cortex serves Claude, GPT, Llama, Mistral, DeepSeek, and Snowflake's own models,
    with all inference running inside the customer's Snowflake account.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: SnowflakeModelName,
        *,
        provider: Literal['snowflake'] | Provider[AsyncOpenAI] = 'snowflake',
        profile: ModelProfileSpec | None = None,
        settings: SnowflakeModelSettings | None = None,
    ):
        """Initialize a Snowflake Cortex model.

        Args:
            model_name: The name of the Snowflake Cortex model to use.
            provider: The provider to use. Defaults to 'snowflake'.
            profile: The model profile to use. Defaults to a profile based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @property
    def _resolved_profile(self) -> SnowflakeModelProfile:
        return cast(SnowflakeModelProfile, self.profile)

    @override
    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        merged_settings, customized_parameters = super().prepare_request(model_settings, model_request_parameters)
        new_settings = _snowflake_settings_to_openai_settings(
            cast(SnowflakeModelSettings, merged_settings or {}), customized_parameters, profile=self._resolved_profile
        )
        return new_settings, customized_parameters

    @override
    def _translate_thinking(
        self,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        """Pass through an explicit `openai_reasoning_effort`, but don't derive one from unified thinking for Claude.

        Cortex ignores `reasoning_effort` for Claude models, which take the `reasoning` object
        injected in `prepare_request` instead.
        """
        if self._resolved_profile.get('snowflake_supports_reasoning', False):
            if effort := model_settings.get('openai_reasoning_effort'):
                return effort
            return omit
        return super()._translate_thinking(model_settings, model_request_parameters)

    @override
    def _validate_completion(self, response: chat.ChatCompletion) -> _SnowflakeChatCompletion:
        # Cortex returns an empty `finish_reason` for Claude models, which would fail validation.
        for choice in response.choices:
            if not choice.finish_reason:
                choice.finish_reason = 'tool_calls' if choice.message.tool_calls else 'stop'
        return _SnowflakeChatCompletion.model_validate(response.model_dump())

    @override
    def _process_thinking(self, message: chat.ChatCompletionMessage) -> list[ThinkingPart] | None:
        assert isinstance(message, _SnowflakeCompletionMessage)

        if reasoning_details := message.reasoning_details:
            return [_from_reasoning_detail(detail, self.system) for detail in reasoning_details]
        else:
            return super()._process_thinking(message)

    @dataclass
    class _MapModelResponseContext(OpenAIChatModel._MapModelResponseContext):  # pyright: ignore[reportPrivateUsage]
        reasoning_details: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

        def _into_message_param(self) -> chat.ChatCompletionAssistantMessageParam | None:
            message_param = super()._into_message_param()
            if self.reasoning_details:
                if message_param is None:  # pragma: no cover
                    message_param = chat.ChatCompletionAssistantMessageParam(role='assistant', content=None)
                message_param['reasoning_details'] = self.reasoning_details  # pyright: ignore[reportGeneralTypeIssues]
            return message_param

        @override
        def _map_response_thinking_part(self, item: ThinkingPart) -> None:
            if item.provider_name == self._model.system and (reasoning_detail := _into_reasoning_detail(item)):
                self.reasoning_details.append(reasoning_detail.model_dump())
            else:
                super()._map_response_thinking_part(item)

    @property
    @override
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return SnowflakeStreamedResponse


@dataclass
class SnowflakeStreamedResponse(OpenAIStreamedResponse):
    """Implementation of `StreamedResponse` for Snowflake Cortex models."""

    @override
    async def _validate_response(self) -> AsyncIterable[chat.ChatCompletionChunk]:
        async for chunk in self._response:
            yield _SnowflakeChatCompletionChunk.model_validate(chunk.model_dump())

    @override
    def _map_thinking_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        assert isinstance(choice, _SnowflakeChunkChoice)

        if reasoning_details := choice.delta.reasoning_details:
            for i, detail in enumerate(reasoning_details):
                thinking_part = _from_reasoning_detail(detail, self._provider_name)
                # Use unique vendor_part_id for each reasoning detail type to prevent
                # different detail types from being incorrectly merged into a single ThinkingPart.
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
            yield from super()._map_thinking_delta(choice)


def _snowflake_settings_to_openai_settings(
    model_settings: SnowflakeModelSettings,
    model_request_parameters: ModelRequestParameters,
    *,
    profile: SnowflakeModelProfile,
) -> OpenAIChatModelSettings:
    """Transforms a `SnowflakeModelSettings` object into an `OpenAIChatModelSettings` object.

    Args:
        model_settings: The `SnowflakeModelSettings` object to transform.
        model_request_parameters: The `ModelRequestParameters` object to use for the transformation.
        profile: The model's profile, which determines whether the model takes the `reasoning`
            object and whether reasoning requires `temperature` to be 1.

    Returns:
        An `OpenAIChatModelSettings` object with equivalent settings.
    """
    # Copy so the `snowflake_` pops never mutate the caller's dict: `merge_model_settings` can return the
    # model's own `settings` by identity, so popping in place would drop the keys on the next request.
    settings = model_settings.copy()
    extra_body = dict(cast(dict[str, Any], settings.get('extra_body', {})))

    # Fall back to unified thinking when snowflake_reasoning is not set
    if 'snowflake_reasoning' not in settings and profile.get('snowflake_supports_reasoning', False):
        thinking = model_request_parameters.thinking
        if thinking is not None and thinking is not False:
            settings['snowflake_reasoning'] = SnowflakeReasoning(effort=_REASONING_EFFORT_MAP[thinking])

    if reasoning := settings.pop('snowflake_reasoning', None):
        extra_body['reasoning'] = reasoning
        if profile.get('snowflake_reasoning_requires_temperature_1', False):
            settings.setdefault('temperature', 1)

    if extra_body:
        settings['extra_body'] = extra_body

    return OpenAIChatModelSettings(**settings)  # pyright: ignore[reportCallIssue]
