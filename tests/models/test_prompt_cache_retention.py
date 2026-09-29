from __future__ import annotations

from datetime import timedelta
from typing import Literal

import pytest

from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIResponsesModel,
    )
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='provider extras not installed')


def test_model_resolve_prompt_cache_retention_defaults_to_none() -> None:
    model: Model = TestModel()

    assert model.resolve_prompt_cache_retention(None) is None


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize(
    ('setting', 'expected'),
    [
        (None, None),
        ('in_memory', None),
        ('24h', timedelta(hours=24)),
    ],
)
def test_openai_resolve_prompt_cache_retention(
    api: Literal['chat', 'responses'],
    setting: Literal['in_memory', '24h'] | None,
    expected: timedelta | None,
) -> None:
    model_type = OpenAIChatModel if api == 'chat' else OpenAIResponsesModel
    model = model_type('gpt-5.6', provider=OpenAIProvider(api_key='test-key'))
    settings = OpenAIChatModelSettings(openai_prompt_cache_retention=setting) if setting is not None else None

    assert model.resolve_prompt_cache_retention(settings) == expected


@pytest.mark.parametrize(
    ('settings', 'expected'),
    [
        ({'anthropic_cache': True}, timedelta(minutes=5)),
        ({'anthropic_cache': '5m'}, timedelta(minutes=5)),
        ({'anthropic_cache': '1h'}, timedelta(hours=1)),
        ({'anthropic_cache_instructions': True}, timedelta(minutes=5)),
        ({'anthropic_cache_instructions': '1h'}, timedelta(hours=1)),
        ({'anthropic_cache_tool_definitions': True}, timedelta(minutes=5)),
        ({'anthropic_cache_tool_definitions': '1h'}, timedelta(hours=1)),
        ({'anthropic_cache_messages': True}, timedelta(minutes=5)),
        ({'anthropic_cache_messages': '1h'}, timedelta(hours=1)),
    ],
)
def test_anthropic_resolve_prompt_cache_retention(settings: AnthropicModelSettings, expected: timedelta) -> None:
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test-key'))

    assert model.resolve_prompt_cache_retention(settings) == expected


def test_anthropic_resolve_prompt_cache_retention_biases_high() -> None:
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test-key'))
    settings = AnthropicModelSettings(
        anthropic_cache_instructions=True,
        anthropic_cache_tool_definitions='1h',
        anthropic_cache_messages='5m',
    )

    assert model.resolve_prompt_cache_retention(settings) == timedelta(hours=1)
    assert model.resolve_prompt_cache_retention(None) is None


@pytest.mark.parametrize(
    ('settings', 'profile', 'expected'),
    [
        (
            {'bedrock_cache_instructions': True},
            {'bedrock_supports_prompt_caching': True},
            timedelta(minutes=5),
        ),
        (
            {'bedrock_cache_messages': '1h'},
            {'bedrock_supports_prompt_caching': True},
            timedelta(hours=1),
        ),
        (
            {'bedrock_cache_tool_definitions': '5m'},
            {'bedrock_supports_tool_caching': True},
            timedelta(minutes=5),
        ),
        (
            {'bedrock_cache_instructions': '1h'},
            {'bedrock_supports_prompt_caching': False},
            None,
        ),
        (
            {'bedrock_cache_tool_definitions': '1h'},
            {'bedrock_supports_tool_caching': False},
            None,
        ),
        (None, {'bedrock_supports_prompt_caching': True, 'bedrock_supports_tool_caching': True}, None),
    ],
)
def test_bedrock_resolve_prompt_cache_retention(
    bedrock_provider: BedrockProvider,
    settings: BedrockModelSettings | None,
    profile: BedrockModelProfile,
    expected: timedelta | None,
) -> None:
    model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider, profile=profile
    )

    assert model.resolve_prompt_cache_retention(settings) == expected


def test_bedrock_resolve_prompt_cache_retention_biases_high(bedrock_provider: BedrockProvider) -> None:
    model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-20250514-v1:0',
        provider=bedrock_provider,
        profile=BedrockModelProfile(
            bedrock_supports_prompt_caching=True,
            bedrock_supports_tool_caching=True,
        ),
    )
    settings = BedrockModelSettings(
        bedrock_cache_instructions=True,
        bedrock_cache_messages='1h',
        bedrock_cache_tool_definitions='5m',
    )

    assert model.resolve_prompt_cache_retention(settings) == timedelta(hours=1)


def test_openrouter_resolve_prompt_cache_retention() -> None:
    model = OpenRouterModel(
        'anthropic/claude-sonnet-4.6',
        provider=OpenRouterProvider(api_key='test-key'),
        profile=OpenRouterModelProfile(
            openrouter_supports_cache_control=True,
            openrouter_supports_cache_ttl=True,
            openrouter_supports_tool_cache=True,
        ),
    )

    assert model.resolve_prompt_cache_retention(
        OpenRouterModelSettings(openrouter_cache_instructions=True)
    ) == timedelta(minutes=5)
    assert model.resolve_prompt_cache_retention(OpenRouterModelSettings(openrouter_cache_messages='5m')) == timedelta(
        minutes=5
    )
    assert model.resolve_prompt_cache_retention(
        OpenRouterModelSettings(openrouter_cache_tool_definitions='1h')
    ) == timedelta(hours=1)
    assert model.resolve_prompt_cache_retention(None) is None


def test_openrouter_resolve_prompt_cache_retention_biases_high() -> None:
    model = OpenRouterModel(
        'anthropic/claude-sonnet-4.6',
        provider=OpenRouterProvider(api_key='test-key'),
        profile=OpenRouterModelProfile(
            openrouter_supports_cache_control=True,
            openrouter_supports_cache_ttl=True,
            openrouter_supports_tool_cache=True,
        ),
    )
    settings = OpenRouterModelSettings(
        openrouter_cache_instructions=True,
        openrouter_cache_messages='1h',
        openrouter_cache_tool_definitions='5m',
    )

    assert model.resolve_prompt_cache_retention(settings) == timedelta(hours=1)


@pytest.mark.parametrize(
    'profile',
    [
        {'openrouter_supports_cache_control': True, 'openrouter_supports_tool_cache': True},
        {
            'openrouter_supports_cache_ttl': True,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_tool_cache': False,
        },
    ],
)
def test_openrouter_resolve_prompt_cache_retention_ignores_unsupported_settings(
    profile: OpenRouterModelProfile,
) -> None:
    model = OpenRouterModel(
        'google/gemini-3.1-pro-preview',
        provider=OpenRouterProvider(api_key='test-key'),
        profile=profile,
    )
    settings = OpenRouterModelSettings(
        openrouter_cache_instructions='1h',
        openrouter_cache_messages='1h',
        openrouter_cache_tool_definitions='1h',
    )

    assert model.resolve_prompt_cache_retention(settings) is None
