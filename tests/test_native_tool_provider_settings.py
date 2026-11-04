"""Every option in a native tool's `provider_settings` has to reach its provider's request."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from typing_extensions import get_type_hints

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import (
    AbstractNativeTool,
    AdvisorProviderSettings,
    AdvisorTool,
    WebFetchProviderSettings,
    WebFetchTool,
    WebSearchProviderSettings,
    WebSearchTool,
)

from .conftest import try_import

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from .models import test_anthropic as anthropic_mocks
    from .models.mock_openai import MockOpenAI, get_mock_chat_completion_kwargs
    from .models.test_openrouter import _openrouter_completion  # pyright: ignore[reportPrivateUsage]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic or openai not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]


def _anthropic_request_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    text = BetaTextBlock(text='ok', type='text')
    client = anthropic_mocks.MockAnthropic.create_mock(
        anthropic_mocks.completion_message([text], BetaUsage(input_tokens=1, output_tokens=1))
    )
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return anthropic_mocks.get_mock_chat_completion_kwargs(client)[0]['tools'][0]


def _openrouter_request_parameters(tool: AbstractNativeTool) -> dict[str, Any]:
    client = MockOpenAI.create_mock(_openrouter_completion('ok'))
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(openai_client=client))
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return get_mock_chat_completion_kwargs(client)[0]['tools'][0]['parameters']


ANTHROPIC_OPTIONS: list[tuple[AbstractNativeTool, str, Any]] = [
    (
        WebSearchTool(provider_settings={'anthropic': {'response_inclusion': 'excluded'}}),
        'response_inclusion',
        'excluded',
    ),
    (
        WebFetchTool(provider_settings={'anthropic': {'response_inclusion': 'excluded'}}),
        'response_inclusion',
        'excluded',
    ),
    (WebFetchTool(provider_settings={'anthropic': {'use_cache': False}}), 'use_cache', False),
]
OPENROUTER_OPTIONS: list[tuple[AbstractNativeTool, str, Any]] = [
    (WebSearchTool(provider_settings={'openrouter': {'engine': 'exa'}}), 'engine', 'exa'),
    (WebSearchTool(provider_settings={'openrouter': {'mode': 'fast'}}), 'mode', 'fast'),
    (WebSearchTool(provider_settings={'openrouter': {'max_results': 3}}), 'max_results', 3),
    (WebSearchTool(provider_settings={'openrouter': {'max_total_results': 9}}), 'max_total_results', 9),
    (WebSearchTool(provider_settings={'openrouter': {'max_characters': 100}}), 'max_characters', 100),
    (
        AdvisorTool(model='advisor', provider_settings={'openrouter': {'forward_transcript': True}}),
        'forward_transcript',
        True,
    ),
]


@pytest.mark.parametrize(('tool', 'option', 'value'), ANTHROPIC_OPTIONS)
def test_anthropic_option_reaches_the_request(tool: AbstractNativeTool, option: str, value: Any) -> None:
    assert _anthropic_request_tool(tool)[option] == value


@pytest.mark.parametrize(('tool', 'option', 'value'), OPENROUTER_OPTIONS)
def test_openrouter_option_reaches_the_request(tool: AbstractNativeTool, option: str, value: Any) -> None:
    assert _openrouter_request_parameters(tool)[option] == value


def test_every_declared_option_is_covered() -> None:
    """Adding an option without covering it above fails here, rather than silently doing nothing for users."""
    settings_types: dict[type[AbstractNativeTool], Any] = {
        WebSearchTool: WebSearchProviderSettings,
        WebFetchTool: WebFetchProviderSettings,
        AdvisorTool: AdvisorProviderSettings,
    }
    declared = {
        (tool_type, provider, option)
        for tool_type, settings_type in settings_types.items()
        for provider, options in get_type_hints(settings_type).items()
        for option in get_type_hints(options)
    }
    covered = {(type(tool), 'anthropic', option) for tool, option, _ in ANTHROPIC_OPTIONS} | {
        (type(tool), 'openrouter', option) for tool, option, _ in OPENROUTER_OPTIONS
    }
    assert declared == covered


@pytest.mark.parametrize('option', ['engine', 'mode'])
def test_value_openrouter_adds_later_is_forwarded(option: str) -> None:
    tool = WebSearchTool(provider_settings={'openrouter': {option: 'added-later'}})  # pyright: ignore[reportArgumentType]

    assert _openrouter_request_parameters(tool)[option] == 'added-later'


@pytest.mark.parametrize(
    'settings',
    [
        pytest.param({'madeup': {}}, id='unknown-key'),
        pytest.param({'openrouter': {'typo': 1}}, id='unknown-option'),
        pytest.param({'openrouter': {'response_inclusion': 'full'}}, id='another-providers-option'),
    ],
)
def test_unknown_keys_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        WebSearchTool(provider_settings=settings)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    'settings',
    [
        pytest.param({'anthropic': {'response_inclusion': 'nonsense'}}, id='bad-literal'),
        pytest.param({'openrouter': {'engine': 1}}, id='wrong-type'),
        pytest.param({'openrouter': {'max_results': '3'}}, id='not-coerced'),
        pytest.param({'openrouter': {'engine': None}}, id='none-is-not-unset'),
    ],
)
def test_invalid_values_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        WebSearchTool(provider_settings=settings)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ('tool_type', 'kwargs'),
    [(WebFetchTool, {}), (AdvisorTool, {'model': 'advisor'})],
)
def test_every_tool_rejects_unknown_keys(tool_type: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        tool_type(**kwargs, provider_settings={'madeup': {}})
