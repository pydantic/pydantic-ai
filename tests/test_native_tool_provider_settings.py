"""Every option declared in a native tool's `provider_settings` has to reach its provider's request."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from typing_extensions import get_type_hints

from pydantic_ai.models import ModelRequestParameters
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
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.openai import OpenAIChatModelSettings
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic or openai not installed')


def _anthropic_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test'))
    tools, *_ = model._add_native_tools(  # pyright: ignore[reportPrivateUsage]
        [], ModelRequestParameters(native_tools=[tool]), AnthropicModelSettings()
    )
    return tools[0]


def _openrouter_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(api_key='test'))
    tools, _ = model._get_tool_choice(  # pyright: ignore[reportPrivateUsage]
        OpenAIChatModelSettings(), ModelRequestParameters(native_tools=[tool])
    )
    return tools[0]['parameters']  # pyright: ignore[reportTypedDictNotRequiredAccess, reportGeneralTypeIssues]


@dataclass(frozen=True)
class Case:
    tool: Callable[..., AbstractNativeTool]
    system: str
    option: str
    value: Any
    request: Callable[[AbstractNativeTool], dict[str, Any]]


def _advisor(**kwargs: Any) -> AbstractNativeTool:
    return AdvisorTool(model='advisor-model', **kwargs)


CASES = [
    Case(WebSearchTool, 'anthropic', 'response_inclusion', 'excluded', _anthropic_tool),
    Case(WebSearchTool, 'openrouter', 'engine', 'exa', _openrouter_tool),
    Case(WebSearchTool, 'openrouter', 'mode', 'fast', _openrouter_tool),
    Case(WebSearchTool, 'openrouter', 'max_results', 3, _openrouter_tool),
    Case(WebSearchTool, 'openrouter', 'max_total_results', 9, _openrouter_tool),
    Case(WebSearchTool, 'openrouter', 'max_characters', 100, _openrouter_tool),
    Case(WebFetchTool, 'anthropic', 'response_inclusion', 'excluded', _anthropic_tool),
    Case(WebFetchTool, 'anthropic', 'use_cache', False, _anthropic_tool),
    Case(_advisor, 'openrouter', 'forward_transcript', True, _openrouter_tool),
]

SETTINGS_TYPES = {
    WebSearchTool: WebSearchProviderSettings,
    WebFetchTool: WebFetchProviderSettings,
    _advisor: AdvisorProviderSettings,
}


@pytest.mark.parametrize('case', CASES, ids=lambda c: f'{c.system}-{c.option}')
def test_option_reaches_the_request(case: Case) -> None:
    tool = case.tool(provider_settings={case.system: {case.option: case.value}})
    assert case.request(tool)[case.option] == case.value


def test_every_declared_option_has_a_case() -> None:
    """Adding an option without mapping it fails here, rather than silently doing nothing for users."""
    declared = {
        (tool, system, option)
        for tool, settings_type in SETTINGS_TYPES.items()
        for system, system_settings in get_type_hints(settings_type).items()
        for option in get_type_hints(system_settings)
    }
    assert declared == {(case.tool, case.system, case.option) for case in CASES}


@pytest.mark.parametrize('tool', SETTINGS_TYPES)
@pytest.mark.parametrize(
    'settings',
    [
        pytest.param({'madeup': {}}, id='unknown-system'),
        pytest.param({'openrouter': {'typo': 1}}, id='unknown-option'),
    ],
)
def test_unknown_keys_are_rejected(tool: Callable[..., AbstractNativeTool], settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        tool(provider_settings=settings)


def test_another_systems_option_is_rejected() -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        WebSearchTool(provider_settings={'openrouter': {'response_inclusion': 'full'}})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    'settings',
    [
        pytest.param({'openrouter': {'engine': 'nonsense'}}, id='bad-literal'),
        pytest.param({'openrouter': {'max_results': '3'}}, id='wrong-type-not-coerced'),
        pytest.param({'openrouter': {'engine': None}}, id='none-is-not-unset'),
    ],
)
def test_invalid_values_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        WebSearchTool(provider_settings=settings)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('case', CASES, ids=lambda c: f'{c.system}-{c.option}')
def test_settings_survive_a_serialization_round_trip(case: Case) -> None:
    """Durable execution sends native tools through this round trip between activities."""
    tool = case.tool(provider_settings={case.system: {case.option: case.value}})
    adapter = TypeAdapter(ModelRequestParameters)

    payload = adapter.dump_json(ModelRequestParameters(native_tools=[tool]))

    assert adapter.validate_json(payload).native_tools == [tool]


def test_payload_with_an_unknown_option_is_rejected() -> None:
    """A payload written by a newer version fails loudly here, not by silently dropping the option."""
    payload = {'native_tools': [{'kind': 'web_search', 'provider_settings': {'openrouter': {'future': 1}}}]}

    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        TypeAdapter(ModelRequestParameters).validate_python(payload)
