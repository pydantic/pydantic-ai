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
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnContent,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.tools import ToolDefinition

from .cassette_utils import single_request_body
from .conftest import try_import

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message
    from .models.test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic, google or openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

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
    # T4 keeps local search for the search-shaped origins — `gpt-5` has no native `tool_search` — but a
    # stored delta reaches it as `additional_tools` like any other first-party model now does.
    'T4': ('local-search', 'additional-tools'),
    'T5': ('local-search', 'local-search'),
    'T6': ('local-search', 'local-search'),
}
CASES = [
    Case(origin, target, _TARGET_RENDERINGS[target][1 if origin == 'R4' else 0])
    for origin in ('R1', 'R2', 'R3', 'R4', 'R5')
    for target in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')
]

_NATIVE_TOOL_SEARCH_TARGETS: frozenset[Target] = frozenset({'T1', 'T2', 'T3'})
"""Targets whose model exposes a provider-hosted tool-search surface, and so can declare a corpus."""

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

    # Both bodies are unreachable by design: the prompt asks the model to acknowledge the tool without
    # calling it, because what's under test is the wire shape the availability change renders as, not what
    # the tool returns.
    @agent.tool_plain(defer_loading=True)
    def lookup_exchange_rate(currency: str) -> str:  # pragma: no cover
        """Look up an exchange rate."""
        return f'1 {currency} = 1 test unit'

    @agent.tool_plain
    def always_ready() -> str:  # pragma: no cover
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
        # No search happened — a delta is control, not discovery — but the search *surface* stays on the
        # wire, and so does the revealed tool's own deferred declaration. `tools` is the first cache
        # section, so a delta turn has to send it exactly as the turn before did; both adapters used to
        # rewrite it here, which had the feature busting the very prefix it exists to protect. See
        # `test_tool_availability_delta_and_the_tools_cache_section`, which measures that directly — these
        # assertions only notice the symptom.
        assert facts['search_calls'] == facts['search_returns'] == 0
        assert facts['search_tools'] >= (1 if case.target in _NATIVE_TOOL_SEARCH_TARGETS else 0)
        assert facts['tool_additions'] == (1 if case.rendering == 'tool-addition' else 0)
        assert facts['additional_tools'] == (1 if case.rendering == 'additional-tools' else 0)

    assert facts['revealed_tools'] >= 1
    # A deferred declaration in `tools` is what a delta *reveals*, so it's there exactly when the target
    # has a native tool-search surface to have declared it. Where there isn't one — `gpt-5` on Responses,
    # Gemini, OpenAI Chat — the tool was never on the wire, so the item introduces it instead and there's
    # no `defer_loading` to find. Both are prefix-stable; they differ in what there was to preserve.
    if case.target in _NATIVE_TOOL_SEARCH_TARGETS and case.rendering != 'local-search':
        assert facts['revealed_defer_loading'] == [True]
    else:
        assert facts['revealed_defer_loading'] in ([], [False])


def _empty_responses_message() -> Any:
    """A minimal Responses reply, so the two requests under test are the only thing that differs."""
    return response_message(
        [
            ResponseOutputMessage(
                id='output-1',
                content=[ResponseOutputText(text='ok', type='output_text', annotations=[])],
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )


@pytest.mark.parametrize('provider', ['anthropic', 'openai-responses'])
async def test_tool_availability_delta_and_the_tools_cache_section(allow_model_requests: None, provider: str) -> None:
    """A delta leaves `tools` byte-for-byte alone — the first cache section, so it decides the whole prefix.

    This is the property the feature exists for, and the matrix above cannot see it. VCR matches on
    method, path and host, so `single_request_body` reads what the *cassette* holds: a rendering that
    rewrote `tools` replayed its recording and passed anyway. And a cassette records one request, where
    the question is about two. So: two requests differing only in the trailing delta, mocked, compared as
    bytes.

    Both adapters used to rewrite it, on the one turn that was supposed to be free, deepest into the
    conversation where the cache is worth most. Anthropic dropped `tool_search_tool_bm25` as soon as any
    delta appeared in history. OpenAI Responses was worse: it promoted the revealed definition out of
    `tools` into the `additional_tools` item and dropped `tool_search` behind it, taking two of three
    entries with it.

    Neither bought anything. The API accepts the stable shape on both: a `tool_addition` block alongside
    `tool_search_tool_bm25`, and on OpenAI a still-deferred entry plus `tool_search` plus an item naming
    the same tool — which is also *cheaper*, because the model then calls the tool directly instead of
    burning a `tool_search_call` round-trip first.

    "Was this already declared?" is `defer_loading` on the resolved request: `prepare_request` leaves
    it set for exactly the tools this model can declare while withholding their schema, and the
    authored value stays put through a reveal, so the answer doesn't change when one lands. A
    capability-gated corpus keeps `tools` stable the other way, by never declaring the tool on
    OpenAI at all — `test_openai_capability_only_corpus_keeps_tools_byte_identical` measures that half.
    """
    tool = ToolDefinition(
        name=_TOOL_NAME,
        description='Look up an exchange rate.',
        parameters_json_schema={'type': 'object', 'properties': {'currency': {'type': 'string'}}},
        defer_loading=True,
        with_native=ToolSearchTool.kind,
    )
    always_ready = ToolDefinition(
        name='always_ready', description='Always available.', parameters_json_schema={'type': 'object'}
    )
    parameters = ModelRequestParameters(function_tools=[always_ready, tool], native_tools=[ToolSearchTool()])
    before: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Find the exchange-rate tool.')]),
        ModelResponse(parts=[TextPart(content='Looking.')]),
    ]
    after: list[ModelMessage] = [*before, ModelRequest(parts=[ToolAvailabilityDeltaPart(added=[_TOOL_NAME])])]

    if provider == 'anthropic':
        anthropic_client = MockAnthropic.create_mock(
            [
                completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1)),
                completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1)),
            ]
        )
        model: Model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=anthropic_client))
        await model.request(before, None, parameters)
        await model.request(after, None, parameters)
        sent = [kwargs['tools'] for kwargs in get_mock_chat_completion_kwargs(anthropic_client)]
    else:
        openai_client = MockOpenAIResponses.create_mock(
            [_empty_responses_message(), _empty_responses_message()],
        )
        model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(openai_client=openai_client))
        await model.request(before, None, parameters)
        await model.request(after, None, parameters)
        sent = [kwargs['tools'] for kwargs in get_mock_responses_kwargs(openai_client)]

    before_tools, after_tools = sent
    assert json.dumps(after_tools, sort_keys=True) == json.dumps(before_tools, sort_keys=True)
    # And the deferred declaration is genuinely still there, rather than both turns sending nothing.
    assert any(node.get('name') == _TOOL_NAME for node in _walk(after_tools))


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


def test_two_deltas_with_the_same_tools_get_distinct_synthesized_ids() -> None:
    """A history can legitimately carry the same tool names twice, and the ids must still differ.

    The synthesized id is a digest of the tool names, so a tool withdrawn and re-added — or a UI
    adapter replaying the same frontend tool set — used to produce one id for both exchanges.
    Providers that require call ids to be unique reject that, and anything pairing a call to its
    return by id binds the second return to the first call.
    """
    model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='start')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(added=['lookup'])]),
        ModelResponse(parts=[TextPart(content='ok')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(added=['lookup'])]),
    ]

    call_ids = [
        part.tool_call_id
        for message in model.prepare_messages(history)
        for part in message.parts
        if isinstance(part, ToolSearchCallPart)
    ]

    assert len(call_ids) == 2
    assert call_ids[0] != call_ids[1]

    # Each return pairs with its own call, not with the other one's.
    return_ids = [
        part.tool_call_id
        for message in model.prepare_messages(history)
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart)
    ]
    assert return_ids == call_ids

    # And the ids are stable across turns, or they would move the prefix they exist to protect.
    assert [
        part.tool_call_id
        for message in model.prepare_messages([*history, ModelResponse(parts=[TextPart(content='more')])])
        for part in message.parts
        if isinstance(part, ToolSearchCallPart)
    ] == call_ids


async def test_unrenderable_delta_raises_user_error_not_assertion(allow_model_requests: None) -> None:
    """`Model.request` is public and doesn't run `prepare_messages`, so a caller can reach this.

    The history here is perfectly valid; the only thing missing is the projection step the agent
    normally runs. That makes it a caller-fixable mistake — a `UserError` naming the step — rather
    than an assertion about an invariant the caller was never told about.
    """
    model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='start')]),
        ModelResponse(parts=[TextPart(content='ok')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(added=['lookup'])]),
    ]

    with pytest.raises(UserError, match=r'prepare_messages'):
        await model.request(history, None, ModelRequestParameters())
