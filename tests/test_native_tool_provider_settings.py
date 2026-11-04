"""Every option declared in a native tool's `provider_settings` has to reach its provider's request."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2
import pytest
from pydantic import TypeAdapter, ValidationError
from typing_extensions import get_type_hints

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models import Model, ModelRequestParameters
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
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic or openai not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]

ANTHROPIC_RESPONSE: dict[str, Any] = {
    'id': 'msg',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-4-6',
    'content': [{'type': 'text', 'text': 'ok'}],
    'stop_reason': 'end_turn',
    'stop_sequence': None,
    'usage': {'input_tokens': 1, 'output_tokens': 1},
}
# Anthropic's `20260318` web tools are always streamed.
ANTHROPIC_STREAM = ''.join(
    f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'
    for event in [
        {'type': 'message_start', 'message': {**ANTHROPIC_RESPONSE, 'content': [], 'stop_reason': None}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'ok'}},
        {'type': 'content_block_stop', 'index': 0},
        {
            'type': 'message_delta',
            'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
            'usage': {'output_tokens': 1},
        },
        {'type': 'message_stop'},
    ]
)
OPENROUTER_RESPONSE: dict[str, Any] = {
    'id': 'chat',
    'object': 'chat.completion',
    'created': 1,
    'model': 'openai/gpt-5',
    'provider': 'OpenAI',
    'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}],
}


def _client(response: dict[str, Any], bodies: list[dict[str, Any]]) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get('stream'):
            return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=ANTHROPIC_STREAM)
        return httpx2.Response(200, json=response)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def _run(model: Model, tool: AbstractNativeTool, bodies: list[dict[str, Any]]) -> dict[str, Any]:
    Agent(model, capabilities=[NativeTool(tool)]).run_sync('hi')
    return bodies[-1]['tools'][0]


def _anthropic_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []
    provider = AnthropicProvider(api_key='test', http_client=_client(ANTHROPIC_RESPONSE, bodies))
    return _run(AnthropicModel('claude-sonnet-4-6', provider=provider), tool, bodies)


def _openrouter_tool(tool: AbstractNativeTool) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []
    provider = OpenRouterProvider(api_key='test', http_client=_client(OPENROUTER_RESPONSE, bodies))
    return _run(OpenRouterModel('openai/gpt-5', provider=provider), tool, bodies)['parameters']


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
        pytest.param({'anthropic': {'response_inclusion': 'nonsense'}}, id='bad-literal'),
        pytest.param({'openrouter': {'engine': 1}}, id='wrong-type'),
        pytest.param({'openrouter': {'max_results': '3'}}, id='wrong-type-not-coerced'),
        pytest.param({'openrouter': {'engine': None}}, id='none-is-not-unset'),
    ],
)
def test_invalid_values_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        WebSearchTool(provider_settings=settings)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('option', ['engine', 'mode'])
def test_value_openrouter_adds_later_is_forwarded(option: str) -> None:
    tool = WebSearchTool(provider_settings={'openrouter': {option: 'added-later'}})  # pyright: ignore[reportArgumentType]

    assert _openrouter_tool(tool)[option] == 'added-later'


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
