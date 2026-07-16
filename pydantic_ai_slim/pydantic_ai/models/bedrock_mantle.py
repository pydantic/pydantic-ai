from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.providers.bedrock import BedrockProvider

LatestBedrockMantleModelNames = Literal[
    'anthropic.claude-fable-5',
    'anthropic.claude-haiku-4-5',
    'anthropic.claude-opus-4-7',
    'anthropic.claude-opus-4-8',
    'anthropic.claude-sonnet-5',
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
BedrockMantleModelName = str | LatestBedrockMantleModelNames


@dataclass(init=False)
class BedrockMantleResponsesModel(OpenAIResponsesModel):
    """An OpenAI Responses model served by Amazon Bedrock Mantle."""

    def __init__(
        self,
        model_name: BedrockMantleModelName,
        *,
        provider: Literal['bedrock', 'bedrock-mantle'] | BedrockProvider = 'bedrock',
        profile: ModelProfileSpec | None = None,
        settings: OpenAIResponsesModelSettings | None = None,
    ) -> None:
        bedrock_provider = BedrockProvider() if isinstance(provider, str) else provider
        super().__init__(
            model_name,
            provider=bedrock_provider.mantle_provider('mantle-openai-responses', model_name),
            profile=profile,
            settings=settings,
        )


@dataclass(init=False)
class BedrockMantleChatModel(OpenAIChatModel):
    """An OpenAI Chat Completions model served by Amazon Bedrock Mantle."""

    def __init__(
        self,
        model_name: BedrockMantleModelName,
        *,
        provider: Literal['bedrock', 'bedrock-mantle'] | BedrockProvider = 'bedrock',
        profile: ModelProfileSpec | None = None,
        settings: OpenAIChatModelSettings | None = None,
    ) -> None:
        bedrock_provider = BedrockProvider() if isinstance(provider, str) else provider
        super().__init__(
            model_name,
            provider=bedrock_provider.mantle_provider('mantle-openai-chat', model_name),
            profile=profile,
            settings=settings,
        )


@dataclass(init=False)
class BedrockMantleMessagesModel(AnthropicModel):
    """An Anthropic Messages model served by Amazon Bedrock Mantle."""

    def __init__(
        self,
        model_name: BedrockMantleModelName,
        *,
        provider: Literal['bedrock', 'bedrock-mantle'] | BedrockProvider = 'bedrock',
        profile: ModelProfileSpec | None = None,
        settings: AnthropicModelSettings | None = None,
    ) -> None:
        bedrock_provider = BedrockProvider() if isinstance(provider, str) else provider
        super().__init__(
            model_name,
            provider=bedrock_provider.mantle_provider('mantle-anthropic-messages', model_name),
            profile=profile,
            settings=settings,
        )
