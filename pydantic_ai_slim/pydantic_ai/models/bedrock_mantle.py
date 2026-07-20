from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..exceptions import UserError
from ..profiles import ModelProfileSpec
from ..providers.bedrock_mantle import BedrockMantleProvider
from .openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

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
        self._mantle_client = provider._openai_client(  # pyright: ignore[reportPrivateUsage]
            'openai-responses' if interface == 'openai-responses' else 'responses'
        )

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
        self._mantle_client = provider._openai_client('chat')  # pyright: ignore[reportPrivateUsage]

    @property
    def client(self) -> AsyncOpenAI:
        return self._mantle_client
