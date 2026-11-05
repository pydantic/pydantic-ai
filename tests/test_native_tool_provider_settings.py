"""Every option in a native tool's `provider_settings` has to reach its provider's request."""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest
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


def test_each_model_reads_only_its_own_settings() -> None:
    tool = WebSearchTool(
        provider_settings={'anthropic': {'response_inclusion': 'excluded'}, 'openrouter': {'engine': 'exa'}}
    )

    anthropic_tool = _anthropic_request_tool(tool)
    assert anthropic_tool['response_inclusion'] == 'excluded'
    assert 'engine' not in anthropic_tool
    openrouter_parameters = _openrouter_request_parameters(tool)
    assert openrouter_parameters['engine'] == 'exa'
    assert 'response_inclusion' not in openrouter_parameters


@pytest.mark.parametrize('option', ['engine', 'mode'])
def test_value_openrouter_adds_later_is_forwarded(option: str) -> None:
    tool = WebSearchTool(provider_settings={'openrouter': {option: 'added-later'}})  # pyright: ignore[reportArgumentType]

    assert _openrouter_request_parameters(tool)[option] == 'added-later'


UNKNOWN_SETTINGS: Any = {
    'madeup': {},
    'anthropic': {'typo': 1},
    'openrouter': {'typo': 1, 'allowed_domains': ['example.com']},
}


@pytest.mark.parametrize(
    ('read_request', 'make_tool'),
    [
        (_anthropic_request_tool, WebSearchTool),
        (_anthropic_request_tool, WebFetchTool),
        (_openrouter_request_parameters, WebSearchTool),
        (_openrouter_request_parameters, partial(AdvisorTool, model='advisor')),
    ],
)
def test_unknown_settings_are_ignored(read_request: Any, make_tool: Any) -> None:
    """Like model settings, an unknown option is never sent, so it can't override a field like `allowed_domains`."""
    assert read_request(make_tool(provider_settings=UNKNOWN_SETTINGS)) == read_request(make_tool())
