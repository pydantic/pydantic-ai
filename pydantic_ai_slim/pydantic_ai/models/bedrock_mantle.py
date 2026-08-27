from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast

from pydantic import ValidationError
from typing_extensions import override

from .. import UnexpectedModelBehavior
from ..exceptions import UserError
from ..messages import FinishReason
from ..profiles import ModelProfileSpec
from ..providers.bedrock_mantle import BedrockMantleModelProfile, BedrockMantleProvider
from .openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)

try:
    from openai.types.chat import chat_completion, chat_completion_chunk

    from .openai import (
        _CHAT_FINISH_REASON_MAP,  # pyright: ignore[reportPrivateUsage]
        OpenAIStreamedResponse,
        _ChatCompletion,  # pyright: ignore[reportPrivateUsage]
        _ChatCompletionChunk,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError('Please install the `openai` package to use Bedrock Mantle models.') from _import_error

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types import chat

LatestBedrockMantleModelNames = Literal[
    'openai.gpt-5.4',
    'openai.gpt-5.4-2026-03-05',
    'openai.gpt-5.5',
    'openai.gpt-5.5-2026-04-23',
    'openai.gpt-5.6-luna',
    'openai.gpt-5.6-sol',
    'openai.gpt-5.6-terra',
    'openai.gpt-oss-20b',
    'openai.gpt-oss-120b',
    'openai.gpt-oss-safeguard-20b',
    'openai.gpt-oss-safeguard-120b',
]
"""Latest OpenAI models served through Amazon Bedrock Mantle."""

BedrockMantleModelName = str | LatestBedrockMantleModelNames
"""Possible Amazon Bedrock Mantle model names.

Since Bedrock Mantle supports a variety of OpenAI models and the list changes frequently, we explicitly
list the latest models but allow any name in the type hints.
"""

# Amazon Bedrock can terminate generations with platform-native finish reasons that the OpenAI
# SDK's strict `Choice.finish_reason` Literal rejects — most notably `guardrail_intervened`,
# documented by AWS as the outcome when a guardrail intervenes. litellm treats the same value as
# Bedrock-specific in its cross-provider finish-reason mapping (see BerriAI/litellm#22138).
# Such values currently fail pydantic-ai's re-validation and abort the run with
# `UnexpectedModelBehavior` (#7816); widening the literal lets them flow through to
# `_map_finish_reason` instead.
_BEDROCK_MANTLE_EXTRA_FINISH_REASONS = Literal['guardrail_intervened']
_BEDROCK_MANTLE_FINISH_REASON = Literal[
    'stop', 'length', 'tool_calls', 'content_filter', 'function_call', 'guardrail_intervened'
]


class _BedrockMantleChoice(chat_completion.Choice):
    """Bedrock Mantle's choice type widens `finish_reason` to accept Bedrock-native causes."""

    finish_reason: _BEDROCK_MANTLE_FINISH_REASON | None  # type: ignore[reportIncompatibleVariableOverride]


class _BedrockMantleChatCompletion(_ChatCompletion):
    """Bedrock Mantle's chat-completion type widens the choice list to `_BedrockMantleChoice`."""

    choices: list[_BedrockMantleChoice]  # type: ignore[reportIncompatibleVariableOverride]


class _BedrockMantleChunkChoice(chat_completion_chunk.Choice):
    """Bedrock Mantle's chunk-choice type widens `finish_reason` the same way."""

    finish_reason: _BEDROCK_MANTLE_FINISH_REASON | None  # type: ignore[reportIncompatibleVariableOverride]


class _BedrockMantleChatCompletionChunk(_ChatCompletionChunk):
    """Bedrock Mantle's chunk type widens the choice list to `_BedrockMantleChunkChoice`."""

    choices: list[_BedrockMantleChunkChoice]  # type: ignore[reportIncompatibleVariableOverride]


# Map Bedrock-native terminations onto the standard pydantic-ai `FinishReason` enum. The raw
# string is preserved in `provider_details['finish_reason']` by the base `_map_provider_details`.
# A guardrail intervention is semantically identical to content moderation: the generation was
# filtered rather than completed or truncated.
_BEDROCK_MANTLE_FINISH_REASON_MAP: dict[_BEDROCK_MANTLE_FINISH_REASON, FinishReason] = {
    **_CHAT_FINISH_REASON_MAP,
    'guardrail_intervened': 'content_filter',
}


@dataclass(init=False)
class BedrockMantleResponsesModel(OpenAIResponsesModel):
    """An OpenAI Responses model served by Amazon Bedrock Mantle.

    Serves GPT-5.4+ (on the `/openai/v1` endpoint) and GPT-OSS (on the `/v1` endpoint); the endpoint is
    chosen from the model profile.
    """

    _mantle_client: AsyncOpenAI = field(repr=False)

    def __init__(
        self,
        model_name: BedrockMantleModelName,
        *,
        provider: Literal['bedrock-mantle'] | BedrockMantleProvider = 'bedrock-mantle',
        profile: ModelProfileSpec | None = None,
        settings: OpenAIResponsesModelSettings | None = None,
    ) -> None:
        """Initialize a Bedrock Mantle Responses model.

        Args:
            model_name: The name of the model, e.g. `openai.gpt-5.6-luna`.
            provider: The provider to use. Defaults to the `bedrock-mantle` provider.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the
                model name.
            settings: The model settings to use. Defaults to `None`.
        """
        provider = BedrockMantleProvider() if isinstance(provider, str) else provider
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)
        interface = self.profile.get('bedrock_mantle_interface', 'openai-responses')
        if interface == 'chat':
            raise UserError(
                f'Model {model_name!r} is served on the Bedrock Mantle Chat Completions API; '
                'construct it with `BedrockMantleChatModel` instead.'
            )
        self._mantle_client = provider._client_for_interface(interface)  # pyright: ignore[reportPrivateUsage]

    @cached_property
    def profile(self) -> BedrockMantleModelProfile:
        return cast(BedrockMantleModelProfile, super().profile)

    @property
    def client(self) -> AsyncOpenAI:
        return self._mantle_client


@dataclass(init=False)
class BedrockMantleChatModel(OpenAIChatModel):
    """An OpenAI Chat Completions model served by Amazon Bedrock Mantle (GPT-OSS Safeguard).

    The response-scoped tool-call-ID normalization added for #6536 is Responses-only: Mantle's Chat
    Completions API returns globally-unique `chatcmpl-tool-*` IDs across separate responses (verified
    live), unlike the `/openai/v1/responses` endpoint's per-response `call_0` counter, so the Chat path
    needs no normalization.
    """

    _mantle_client: AsyncOpenAI = field(repr=False)

    def __init__(
        self,
        model_name: BedrockMantleModelName,
        *,
        provider: Literal['bedrock-mantle'] | BedrockMantleProvider = 'bedrock-mantle',
        profile: ModelProfileSpec | None = None,
        settings: OpenAIChatModelSettings | None = None,
    ) -> None:
        """Initialize a Bedrock Mantle Chat Completions model.

        Args:
            model_name: The name of the model, e.g. `openai.gpt-oss-safeguard-20b`.
            provider: The provider to use. Defaults to the `bedrock-mantle` provider.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the
                model name.
            settings: The model settings to use. Defaults to `None`.
        """
        provider = BedrockMantleProvider() if isinstance(provider, str) else provider
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)
        interface = self.profile.get('bedrock_mantle_interface', 'chat')
        if interface != 'chat':
            raise UserError(
                f'Model {model_name!r} is served on the Bedrock Mantle Responses API; '
                'construct it with `BedrockMantleResponsesModel` instead.'
            )
        self._mantle_client = provider._client_for_interface(interface)  # pyright: ignore[reportPrivateUsage]

    @cached_property
    def profile(self) -> BedrockMantleModelProfile:
        return cast(BedrockMantleModelProfile, super().profile)

    @property
    def client(self) -> AsyncOpenAI:
        return self._mantle_client

    @override
    def _validate_completion(self, response: chat.ChatCompletion) -> _BedrockMantleChatCompletion:
        """Re-validate the SDK completion dict through `_BedrockMantleChatCompletion`.

        The base OpenAI re-validation path uses the strict 5-value `finish_reason` Literal.
        Amazon Bedrock can terminate a generation with platform-native causes such as
        `guardrail_intervened` (litellm documents that value as Bedrock-specific), which fail
        that strict Literal and abort the run with `UnexpectedModelBehavior`. This override
        widens the literal so the same shape passes, then `_map_finish_reason` below normalises
        the non-standard values onto the standard pydantic-ai `FinishReason` enum (raw string
        kept in `provider_details['finish_reason']`).
        """
        try:
            return _BedrockMantleChatCompletion.model_validate(response.model_dump())
        except ValidationError as exc:
            # Preserve the same exception semantics as the base method: unknown validation
            # failures still surface as `UnexpectedModelBehavior`.
            raise UnexpectedModelBehavior(
                f'Invalid response from {self.system} chat completions endpoint: {exc}'
            ) from exc

    @override
    def _map_finish_reason(
        self,
        key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call', 'guardrail_intervened'],
    ) -> FinishReason | None:
        return _BEDROCK_MANTLE_FINISH_REASON_MAP.get(key)

    @property
    @override
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return BedrockMantleStreamedResponse


@dataclass
class BedrockMantleStreamedResponse(OpenAIStreamedResponse):
    """Streamed response handler for Bedrock Mantle.

    Re-validates each chunk through `_BedrockMantleChatCompletionChunk` so Bedrock-native
    finish reasons on terminal chunks are accepted rather than raising `ValidationError`
    when pydantic-ai consumes the streamed finish, keeping streaming symmetric with the
    non-streaming path while still failing loudly on genuinely malformed chunks.
    """

    @override
    async def _validate_response(self):
        """Pass each SDK chunk through `_BedrockMantleChatCompletionChunk` validation.

        The OpenAI SDK constructs `ChatCompletionChunk` leniently on the wire, so
        Bedrock-native finish reasons used to arrive without crashing even before this
        override; validating each chunk keeps both paths symmetric and guarantees that a
        malformed choice fails loudly instead of being silently type-incorrect downstream.
        """
        async for chunk in self._response:
            try:
                yield _BedrockMantleChatCompletionChunk.model_validate(chunk.model_dump())
            except ValidationError as exc:
                raise UnexpectedModelBehavior(
                    f'Invalid response from {self._model_name} chat completions stream: {exc}'
                ) from exc

    @override
    def _map_finish_reason(
        self,
        key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call', 'guardrail_intervened'],
    ) -> FinishReason | None:
        return _BEDROCK_MANTLE_FINISH_REASON_MAP.get(key)
