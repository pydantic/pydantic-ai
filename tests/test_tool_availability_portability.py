from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnContent,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from .cassette_utils import single_request_body

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]

Origin = Literal['R1', 'R2', 'R3', 'R4', 'R5']
Target = Literal['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
Rendering = Literal['native-search', 'local-search', 'tool-addition', 'additional-tools']


@dataclass(frozen=True)
class Case:
    origin: Origin
    target: Target
    rendering: Rendering

    @property
    def id(self) -> str:
        return f'{self.origin}-{self.target}-{self.rendering}'


_TARGET_RENDERINGS: dict[Target, tuple[Rendering, Rendering]] = {
    'T1': ('native-search', 'tool-addition'),
    'T2': ('native-search', 'native-search'),
    'T3': ('native-search', 'additional-tools'),
    'T4': ('local-search', 'local-search'),
    'T5': ('local-search', 'local-search'),
    'T6': ('local-search', 'local-search'),
}
CASES = [
    Case(origin, target, _TARGET_RENDERINGS[target][1 if origin == 'R4' else 0])
    for origin in ('R1', 'R2', 'R3', 'R4', 'R5')
    for target in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')
]

_TOOL_NAME = 'lookup_exchange_rate'
_SEARCH_CALL_ID = 'search_call_1'


def _history(origin: Origin) -> list[ModelMessage]:
    prompt = ModelRequest(parts=[UserPromptPart(content='Find the exchange-rate tool.')])
    discovered: ToolSearchReturnContent = {'discovered_tools': [{'name': _TOOL_NAME}]}
    search_call_id = 'srvtoolu_search_call_1' if origin == 'R2' else _SEARCH_CALL_ID
    if origin == 'R1':
        return [
            prompt,
            ModelResponse(parts=[ToolSearchCallPart(args={'queries': ['exchange rate']}, tool_call_id=search_call_id)]),
            ModelRequest(parts=[ToolSearchReturnPart(content=discovered, tool_call_id=search_call_id)]),
        ]
    if origin in ('R2', 'R3'):
        provider_name = 'anthropic' if origin == 'R2' else 'openai'
        return [
            prompt,
            ModelResponse(
                parts=[
                    NativeToolSearchCallPart(
                        args={'queries': ['exchange rate']},
                        tool_call_id=search_call_id,
                        provider_name=provider_name,
                    ),
                    NativeToolSearchReturnPart(
                        content=discovered,
                        tool_call_id=search_call_id,
                        provider_name=provider_name,
                        provider_details=(
                            {'id': 'tso_search_output_1', 'call_id': search_call_id, 'status': 'completed'}
                            if origin == 'R3'
                            else None
                        ),
                    ),
                ],
                provider_name=provider_name,
            ),
        ]
    if origin == 'R4':
        return [
            prompt,
            ModelResponse(parts=[]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(added=[_TOOL_NAME], tool_call_id=_SEARCH_CALL_ID)]),
        ]
    return [
        prompt,
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='search_tools',
                    args={'queries': ['exchange rate']},
                    tool_call_id=_SEARCH_CALL_ID,
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='search_tools',
                    content='Found one matching tool.',
                    tool_call_id=_SEARCH_CALL_ID,
                    metadata={'discovered_tools': [_TOOL_NAME]},
                )
            ]
        ),
    ]


def _target_model(
    target: Target,
    *,
    anthropic_api_key: str,
    openai_api_key: str,
    gemini_api_key: str,
) -> Model:
    if target == 'T1':
        return AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key))
    if target == 'T2':
        return AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    if target == 'T3':
        return OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key=openai_api_key))
    if target == 'T4':
        return OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    if target == 'T5':
        return GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key=gemini_api_key))
    return OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))


def _walk(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        node = cast(dict[str, Any], value)
        yield node
        for child in node.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in cast(list[Any], value):
            yield from _walk(child)


def _wire_facts(body: dict[str, Any]) -> dict[str, Any]:
    nodes = list(_walk(body))
    serialized_nodes = [json.dumps(node, sort_keys=True) for node in nodes]
    search_call_nodes = [
        node
        for node, serialized in zip(nodes, serialized_nodes)
        if (
            node.get('type') in {'tool_search_call', 'server_tool_use', 'tool_use', 'function_call'}
            or 'tool_calls' in node
            or 'functionCall' in node
        )
        and ('tool_search' in serialized or 'search_tools' in serialized)
    ]
    search_return_nodes = [
        node
        for node, serialized in zip(nodes, serialized_nodes)
        if (
            node.get('type') in {'tool_search_output', 'tool_search_tool_result', 'tool_result', 'function_call_output'}
            or node.get('role') == 'tool'
            or 'functionResponse' in node
        )
        and (_SEARCH_CALL_ID in serialized or _TOOL_NAME in serialized or 'search_tools' in serialized)
    ]
    tool_additions = [node for node in nodes if node.get('type') == 'tool_addition']
    additional_tools = [node for node in nodes if node.get('type') == 'additional_tools']
    top_level_tools = body.get('tools', [])
    tool_definition_nodes = list(_walk(top_level_tools))
    search_tool_nodes = [
        node
        for node in tool_definition_nodes
        if str(node.get('type', '')).startswith('tool_search')
        or node.get('name') == 'search_tools'
        or node.get('name') == 'tool_search_tool_bm25'
    ]
    revealed_tool_nodes = [
        node
        for node in nodes
        if node.get('name') == _TOOL_NAME
        and (
            'input_schema' in node
            or 'parameters' in node
            or 'parametersJsonSchema' in node
            or 'parameters_json_schema' in node
            or 'defer_loading' in node
            or node.get('type') == 'tool_reference'
        )
    ]
    return {
        'search_calls': len(search_call_nodes),
        'search_returns': len(search_return_nodes),
        'tool_additions': len(tool_additions),
        'additional_tools': len(additional_tools),
        'search_tools': len(search_tool_nodes),
        'revealed_tools': len(revealed_tool_nodes),
        'revealed_defer_loading': sorted(
            {node['defer_loading'] for node in revealed_tool_nodes if isinstance(node.get('defer_loading'), bool)}
        ),
    }


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_tool_availability_portability_matrix(
    case: Case,
    allow_model_requests: None,
    anthropic_api_key: str,
    openai_api_key: str,
    gemini_api_key: str,
    vcr: Cassette,
) -> None:
    """Every stored availability representation remains callable, explicable, and well-formed."""
    model = _target_model(
        case.target,
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
    )
    agent = Agent(model, capabilities=[ToolSearch()])

    @agent.tool_plain(defer_loading=True)
    def lookup_exchange_rate(currency: str) -> str:
        """Look up an exchange rate."""
        return f'1 {currency} = 1 test unit'

    @agent.tool_plain
    def always_ready() -> str:
        """Provide an always-available tool so provider tool lists remain valid."""
        return 'ready'

    await agent.run(
        'Acknowledge the available exchange-rate tool without calling it.',
        message_history=_history(case.origin),
    )

    body = single_request_body(vcr)
    facts = _wire_facts(body)

    if case.rendering in ('native-search', 'local-search'):
        assert facts['search_calls'] >= 1
        assert facts['search_returns'] >= 1
        assert facts['search_tools'] >= 1
        assert facts['tool_additions'] == facts['additional_tools'] == 0
    else:
        assert facts['search_calls'] == facts['search_returns'] == facts['search_tools'] == 0
        assert facts['tool_additions'] == (1 if case.rendering == 'tool-addition' else 0)
        assert facts['additional_tools'] == (1 if case.rendering == 'additional-tools' else 0)

    assert facts['revealed_tools'] >= 1
    if case.rendering == 'local-search':
        assert facts['revealed_defer_loading'] in ([], [False])
    elif case.rendering == 'native-search':
        assert facts['revealed_defer_loading'] == [True]


@pytest.mark.parametrize('origin', ['R1', 'R2', 'R3', 'R4', 'R5'])
def test_tool_availability_history_is_stable_across_a_b_a(origin: Origin) -> None:
    """Preparing the same stored history for Anthropic → Gemini → Anthropic never mutates it."""
    history = _history(origin)
    original = deepcopy(history)
    anthropic = AnthropicModel(
        'claude-opus-4-8',
        provider=AnthropicProvider(api_key='test-key'),
    )
    google = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='test-key'))

    first = anthropic.prepare_messages(history)
    google.prepare_messages(history)
    second = anthropic.prepare_messages(history)

    assert first == second
    assert history == original
