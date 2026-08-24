from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai._utils import TOOL_CALL_ID_PREFIX
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    SystemPromptPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnContent,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    TOOL_AVAILABILITY_ANNOUNCEMENT,
    Model,
    ModelProfile,
    ModelRequestParameters,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._tool_search import parse_discovered_tools

from .cassette_utils import single_request_body
from .conftest import try_import

with try_import() as imports_successful:
    from anthropic.types.beta import BetaMessage, BetaRawMessageStartEvent, BetaTextBlock, BetaUsage
    from google.genai.types import Candidate, Content, GenerateContentResponse, Part
    from openai.types.chat import ChatCompletionMessage
    from openai.types.responses import ResponseCreatedEvent, ResponseOutputMessage, ResponseOutputText

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.profiles import merge_profile
    from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from .models.mock_openai import (
        MockOpenAI,
        MockOpenAIResponses,
        completion_message as openai_completion_message,
        get_mock_chat_completion_kwargs as get_mock_openai_chat_completion_kwargs,
        get_mock_responses_kwargs,
        response_message,
    )
    from .models.test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic, google or openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

Origin = Literal['R1', 'R2', 'R3', 'R4', 'R5']
Target = Literal['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
Rendering = Literal[
    'native-search',
    'local-search',
    'local-search-additional-tools',
    'tool-addition',
    'additional-tools',
    'announcement',
]


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
    'T5': ('local-search', 'announcement'),
    'T6': ('local-search', 'announcement'),
    'T7': ('local-search', 'announcement'),
}
CASES = [
    Case(
        origin,
        target,
        (
            'local-search-additional-tools'
            if target == 'T4' and origin in ('R1', 'R2', 'R3')
            else _TARGET_RENDERINGS[target][1 if origin == 'R4' else 0]
        ),
    )
    for origin in ('R1', 'R2', 'R3', 'R4', 'R5')
    # T7 has no first-party provider to record against, so it belongs only to the unit projection matrix.
    for target in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')
]

_NATIVE_TOOL_SEARCH_TARGETS: frozenset[Target] = frozenset({'T1', 'T2', 'T3'})
"""Targets whose model exposes a provider-hosted tool-search surface, and so can declare a corpus."""

_TWO_REVEALED_TOOL_NODES: frozenset[str] = frozenset(
    {
        'R1-T3-native-search',
        'R2-T3-native-search',
        'R3-T3-native-search',
        'R4-T1-tool-addition',
        'R4-T3-additional-tools',
    }
)
"""Cases where the tool's name reaches the wire attached to a schema twice rather than once.

Measured, not derived. Two things can carry it: the deferred declaration in `tools`, and whatever
performs the reveal. They coincide when the reveal has a payload of its own — Anthropic's
`tool_addition` block and Responses' `additional_tools` item both name the tool again — and when a
stored *structured* search return replays onto Responses, which R5's plain-text return doesn't do.
Pinning the exact count is the point: `>= 1` passed just as happily when a reveal was emitted twice.
"""

_TOOL_NAME = 'lookup_exchange_rate'
_SEARCH_CALL_ID = 'search_call_1'

ToolSearchRecordKind = Literal['anthropic-native', 'openai-native', 'openai-client', 'local-fallback']
ToolSearchReplayTarget = Literal['anthropic', 'openai', 'google']


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
        # A standalone `ModelRequest` carrying only the delta, with no assistant turn ahead of it, is
        # the shape the framework itself appends (`ToolSearch.before_model_request`) and the shape a
        # UI adapter rebuilds from an `ActivityMessage`. Two `ModelRequest`s in a row is the point:
        # a delta is control, so nothing generated between the turn that caused it and the turn it
        # announces.
        return [
            prompt,
            ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[_TOOL_NAME], tool_call_id=_SEARCH_CALL_ID)]),
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


def _legacy_fabricated_history(origin: Literal['R1', 'R5']) -> list[ModelMessage]:
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='load_capability',
                    args={'id': 'finance'},
                    tool_call_id='load_1',
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='load_capability',
                    content={'instructions': 'Use the finance tools.'},
                    tool_call_id='load_1',
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolSearchCallPart(
                    args={'queries': ['exchange rate']},
                    tool_call_id='pyd_ai_legacy_search',
                )
                if origin == 'R1'
                else ToolCallPart(
                    tool_name='search_tools',
                    args={'queries': ['exchange rate']},
                    tool_call_id='pyd_ai_legacy_search',
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': _TOOL_NAME}]},
                    tool_call_id='pyd_ai_legacy_search',
                )
                if origin == 'R1'
                else ToolReturnPart(
                    tool_name='search_tools',
                    content='Found one matching tool.',
                    tool_call_id='pyd_ai_legacy_search',
                    metadata={'discovered_tools': [_TOOL_NAME]},
                )
            ]
        ),
    ]
    return history


def _portability_parameters(*, capability_owned: bool = False) -> ModelRequestParameters:
    return ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='always_ready'),
            ToolDefinition(name='load_capability'),
            ToolDefinition(
                name=_TOOL_NAME,
                defer_loading=True,
                with_native=None if capability_owned else ToolSearchTool.kind,
                capability_id='finance' if capability_owned else None,
            ),
            ToolDefinition(name='search_tools', unless_native=ToolSearchTool.kind),
        ],
        native_tools=[ToolSearchTool(optional=True)],
        revealed_tool_names={_TOOL_NAME},
    )


def _projected_reveal_shape(messages: list[ModelMessage]) -> str:
    parts = [part for message in messages for part in message.parts]
    if any(isinstance(part, ToolAvailabilityDeltaPart) for part in parts):
        return 'delta'
    if any(isinstance(part, SystemPromptPart) for part in parts):
        return 'announcement'
    if any(isinstance(part, UserPromptPart) and 'tool(s) are now available' in str(part.content) for part in parts):
        return 'announcement'
    if any(isinstance(part, NativeToolSearchReturnPart) for part in parts):
        return 'native-search'
    if any(isinstance(part, ToolReturnPart) and part.tool_name == 'search_tools' for part in parts):
        return 'local-search'
    raise AssertionError(f'No reveal rendering in {messages!r}')  # pragma: no cover


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
    if target == 'T6':
        return OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    return OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        profile=merge_profile(
            openai_model_profile('gpt-5'),
            OpenAIModelProfile(tool_deferral_mode=None, tool_addition_mode=None),
        ),
    )


_PROJECTED_MATRIX: dict[tuple[Origin, Target], str] = {
    ('R1', 'T1'): 'local-search',
    ('R1', 'T2'): 'local-search',
    ('R1', 'T3'): 'local-search',
    ('R1', 'T4'): 'local-search',
    ('R1', 'T5'): 'local-search',
    ('R1', 'T6'): 'local-search',
    ('R1', 'T7'): 'local-search',
    ('R2', 'T1'): 'native-search',
    ('R2', 'T2'): 'native-search',
    ('R2', 'T3'): 'local-search',
    ('R2', 'T4'): 'local-search',
    ('R2', 'T5'): 'local-search',
    ('R2', 'T6'): 'local-search',
    ('R2', 'T7'): 'local-search',
    ('R3', 'T1'): 'local-search',
    ('R3', 'T2'): 'local-search',
    ('R3', 'T3'): 'native-search',
    ('R3', 'T4'): 'local-search',
    ('R3', 'T5'): 'local-search',
    ('R3', 'T6'): 'local-search',
    ('R3', 'T7'): 'local-search',
    ('R4', 'T1'): 'delta',
    ('R4', 'T2'): 'local-search',
    ('R4', 'T3'): 'delta',
    ('R4', 'T4'): 'delta',
    ('R4', 'T5'): 'announcement',
    ('R4', 'T6'): 'announcement',
    ('R4', 'T7'): 'announcement',
    ('R5', 'T1'): 'local-search',
    ('R5', 'T2'): 'local-search',
    ('R5', 'T3'): 'local-search',
    ('R5', 'T4'): 'local-search',
    ('R5', 'T5'): 'local-search',
    ('R5', 'T6'): 'local-search',
    ('R5', 'T7'): 'local-search',
}


@pytest.mark.parametrize(('origin', 'target'), _PROJECTED_MATRIX)
def test_tool_availability_portability_projection_matrix(origin: Origin, target: Target) -> None:
    """Every origin reaches Stage 4 in the target's callable reveal shape, without cassette indirection."""
    model = _target_model(target, anthropic_api_key='test', openai_api_key='test', gemini_api_key='test')
    _, parameters = model.prepare_request(None, _portability_parameters())
    prepared = model.prepare_messages(_history(origin), parameters)

    assert _projected_reveal_shape(prepared) == _PROJECTED_MATRIX[(origin, target)]


def _tool_search_transition_history(record_kind: ToolSearchRecordKind) -> list[ModelMessage]:
    prompt = ModelRequest(parts=[UserPromptPart(content='Find the exchange-rate tool.')])
    discovered: ToolSearchReturnContent = {'discovered_tools': [{'name': _TOOL_NAME}]}
    if record_kind in ('anthropic-native', 'openai-native'):
        provider_name = 'anthropic' if record_kind == 'anthropic-native' else 'openai'
        return [
            prompt,
            ModelResponse(
                parts=[
                    NativeToolSearchCallPart(
                        args={'queries': ['exchange rate']},
                        tool_call_id=_SEARCH_CALL_ID,
                        provider_name=provider_name,
                    ),
                    NativeToolSearchReturnPart(
                        content=discovered,
                        tool_call_id=_SEARCH_CALL_ID,
                        provider_name=provider_name,
                        provider_details=(
                            {'id': 'tso_1', 'call_id': _SEARCH_CALL_ID, 'status': 'completed'}
                            if provider_name == 'openai'
                            else None
                        ),
                    ),
                ],
                provider_name=provider_name,
            ),
            ModelRequest(parts=[UserPromptPart(content='Use the revealed tool.')]),
        ]

    return [
        prompt,
        ModelResponse(
            parts=[
                ToolSearchCallPart(
                    args={'queries': ['exchange rate']},
                    tool_call_id=_SEARCH_CALL_ID,
                    provider_name='openai' if record_kind == 'openai-client' else None,
                )
            ],
            provider_name='openai' if record_kind == 'openai-client' else None,
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content=discovered,
                    tool_call_id=_SEARCH_CALL_ID,
                ),
                *(
                    [ToolAvailabilityDeltaPart(tools_added=[_TOOL_NAME], tool_call_id=_SEARCH_CALL_ID)]
                    if record_kind == 'local-fallback'
                    else []
                ),
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content='Use the revealed tool.')]),
    ]


@pytest.mark.parametrize(
    ('record_kind', 'target'),
    [
        pytest.param(record_kind, target, id=f'{record_kind}-to-{target}')
        for record_kind in ('anthropic-native', 'openai-native', 'openai-client', 'local-fallback')
        for target in ('anthropic', 'openai', 'google')
    ],
)
async def test_tool_search_record_replay_transition_matrix(
    record_kind: ToolSearchRecordKind,
    target: ToolSearchReplayTarget,
    allow_model_requests: None,
    mocker: MockerFixture,
) -> None:
    """Every recorded search kind projects its reveal onto each target's legal wire channel."""
    history = _tool_search_transition_history(record_kind)
    tool = ToolDefinition(
        name=_TOOL_NAME,
        description='Look up an exchange rate.',
        parameters_json_schema={'type': 'object', 'properties': {'currency': {'type': 'string'}}},
        defer_loading=True,
        with_native=ToolSearchTool.kind,
    )
    authored_parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='always_ready'), tool],
        native_tools=[ToolSearchTool(optional=True)],
        revealed_tool_names=parse_discovered_tools(history),
    )
    assert authored_parameters.revealed_tool_names == {_TOOL_NAME}

    anthropic_client = None
    openai_client = None
    generate = None
    if target == 'anthropic':
        anthropic_client = MockAnthropic.create_mock(
            completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1))
        )
        model: Model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=anthropic_client))
    elif target == 'openai':
        openai_client = MockOpenAIResponses.create_mock(response_message([]))
        model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(openai_client=openai_client))
    else:
        model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='test'))
        response = GenerateContentResponse(
            candidates=[Candidate(content=Content(parts=[Part(text='ok')], role='model'))],
            response_id='response-1',
            model_version='gemini-3-flash-preview',
        )
        generate = mocker.patch.object(model.client.aio.models, 'generate_content', return_value=response)

    settings, parameters = model.prepare_request(None, authored_parameters)
    # `prepare_messages` is part of the request pipeline (`_agent_graph._make_request` calls it
    # before the adapter's message prep): it is what translates a foreign provider's native
    # search exchange into the local shape each adapter knows how to replay. Skipping it here
    # would test a path production never takes — and make every foreign-native cell look broken.
    request_history = model.prepare_messages(history, parameters)
    await model.request(request_history, settings, parameters)

    if target == 'anthropic':
        assert anthropic_client is not None
        request = get_mock_chat_completion_kwargs(anthropic_client)[0]
        messages = request['messages']
        roles = [message['role'] for message in messages]
        assert all(current != 'system' or previous == 'user' for previous, current in zip(roles, roles[1:]))
        tool_additions = [
            block
            for message in messages
            for block in message['content']
            if block.get('type') == 'tool_addition' and block.get('tool', {}).get('name') == _TOOL_NAME
        ]
        declared = [wire_tool for wire_tool in request['tools'] if wire_tool.get('name') == _TOOL_NAME]
        native_replay = any(
            block.get('type') == 'tool_search_tool_result' for message in messages for block in message['content']
        )
        # A foreign or local exchange replays as `tool_reference` blocks inside the search
        # `tool_result` — the mechanism `test_anthropic_defer_loading_needs_a_reveal_mechanism`
        # pins as "the reveal and the flag travel together".
        replayed_references = [
            content_block
            for message in messages
            for block in message['content']
            if block.get('type') == 'tool_result' and isinstance(block.get('content'), list)
            for content_block in block['content']
            if content_block.get('type') == 'tool_reference' and content_block.get('tool_name') == _TOOL_NAME
        ]
        assert declared and (native_replay or tool_additions or replayed_references)
        if record_kind == 'anthropic-native':
            assert native_replay and not tool_additions
    elif target == 'openai':
        assert openai_client is not None
        request = get_mock_responses_kwargs(openai_client)[0]
        declared = [wire_tool for wire_tool in request.get('tools', []) if wire_tool.get('name') == _TOOL_NAME]
        additional = [
            item
            for item in request['input']
            if item.get('type') == 'additional_tools'
            and any(tool.get('name') == _TOOL_NAME for tool in item.get('tools', []))
        ]
        search_outputs = [
            item
            for item in request['input']
            if item.get('type') == 'tool_search_output'
            and any(tool.get('name') == _TOOL_NAME for tool in item.get('tools', []))
        ]
        # A bare deferred declaration is not enough for a *revealed* tool: its schema is withheld,
        # so without a schema-carrying item the discovery is silently lost and the model would have
        # to search again. Require the schema to reach the wire somewhere.
        declared_with_schema = [wire_tool for wire_tool in declared if not wire_tool.get('defer_loading')]
        assert declared_with_schema or additional or search_outputs
        if record_kind in ('openai-native', 'openai-client'):
            assert search_outputs and not additional
    else:
        assert generate is not None
        config = generate.call_args.kwargs['config']
        declarations = [
            declaration
            for tools in config['tools']
            for declaration in tools.get('function_declarations', [])
            if declaration.get('name') == _TOOL_NAME
        ]
        assert len(declarations) == 1
        assert parameters.visibility_of(_TOOL_NAME) == 'visible'


@pytest.mark.parametrize(
    ('origin', 'target', 'expected'),
    [
        ('R1', 'T1', 'delta'),
        ('R1', 'T2', 'local-search'),
        ('R1', 'T3', 'delta'),
        ('R1', 'T4', 'delta'),
        ('R1', 'T5', 'local-search'),
        ('R1', 'T6', 'local-search'),
        ('R5', 'T1', 'delta'),
        ('R5', 'T2', 'local-search'),
        ('R5', 'T3', 'delta'),
        ('R5', 'T4', 'delta'),
        ('R5', 'T5', 'local-search'),
        ('R5', 'T6', 'local-search'),
    ],
)
def test_legacy_fabricated_search_translation_matrix(
    origin: Literal['R1', 'R5'], target: Target, expected: str
) -> None:
    """Confident legacy fabrications upgrade only toward a native reveal channel.

    On channel-less targets the fabricated exchange replays as-is: the revealed tool reaches the
    wire regardless, and rewriting the exchange would change the replayed prefix for no gain.
    """
    model = _target_model(target, anthropic_api_key='test', openai_api_key='test', gemini_api_key='test')
    _, parameters = model.prepare_request(None, _portability_parameters(capability_owned=True))
    history = _legacy_fabricated_history(origin)
    stored = deepcopy(history)

    prepared = model.prepare_messages(history, parameters)

    assert _projected_reveal_shape(prepared) == expected
    assert history == stored


@pytest.mark.parametrize('origin', ['R1', 'R5'])
def test_legacy_fabricated_search_identity_path_is_byte_identical(origin: Literal['R1', 'R5']) -> None:
    """Channel-less Anthropic replays its legacy synthesized exchange without a cache-busting rewrite."""
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test'))
    _, parameters = model.prepare_request(None, _portability_parameters(capability_owned=True))
    history = _legacy_fabricated_history(origin)
    before = ModelMessagesTypeAdapter.dump_json(history)

    prepared = model.prepare_messages(history, parameters)

    assert ModelMessagesTypeAdapter.dump_json(prepared) == before


@pytest.mark.parametrize('origin', ['R1', 'R5'])
@pytest.mark.parametrize('missing_condition', ['framework-prefix', 'adjacency', 'capability-subset'])
def test_legacy_search_is_left_genuine_without_all_recognizer_conditions(
    origin: Literal['R1', 'R5'], missing_condition: str
) -> None:
    """A near-match remains a model-authored search when any one confidence signal is absent."""
    history = _legacy_fabricated_history(origin)
    parameters = _portability_parameters(capability_owned=True)
    if missing_condition == 'framework-prefix':
        call = cast(ModelResponse, history[2]).parts[0]
        search_return = cast(ModelRequest, history[3]).parts[0]
        cast(ToolCallPart, call).tool_call_id = 'model_search_1'
        cast(ToolReturnPart, search_return).tool_call_id = 'model_search_1'
    elif missing_condition == 'adjacency':
        history.insert(2, ModelResponse(parts=[TextPart(content='Capability loaded.')]))
    else:
        parameters = replace(
            parameters,
            function_tools=[
                replace(tool, capability_id='other') if tool.name == _TOOL_NAME else tool
                for tool in parameters.function_tools
            ],
        )
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='test'))
    _, parameters = model.prepare_request(None, parameters)

    prepared = model.prepare_messages(history, parameters)

    assert _projected_reveal_shape(prepared) == 'local-search'


@pytest.mark.parametrize(
    'defect',
    [
        'not-after-load',
        'malformed-load-args',
        'non-string-capability-id',
        'metadata-not-a-list',
        'metadata-non-string-names',
        'extra-discovery',
    ],
)
def test_legacy_search_is_left_genuine_on_malformed_shapes(defect: str) -> None:
    """Malformed or overreaching near-matches also remain model-authored searches."""
    history = _legacy_fabricated_history('R5')
    parameters = _portability_parameters(capability_owned=True)
    if defect == 'not-after-load':
        cast(ModelRequest, history[1]).parts = [
            ToolReturnPart(tool_name='always_ready', content='ok', tool_call_id='load_1')
        ]
    elif defect == 'malformed-load-args':
        cast(ToolCallPart, cast(ModelResponse, history[0]).parts[0]).args = 'not json'
    elif defect == 'non-string-capability-id':
        cast(ToolCallPart, cast(ModelResponse, history[0]).parts[0]).args = {'id': 123}
    elif defect == 'metadata-not-a-list':
        cast(ToolReturnPart, cast(ModelRequest, history[3]).parts[0]).metadata = {'discovered_tools': 'oops'}
    elif defect == 'metadata-non-string-names':
        cast(ToolReturnPart, cast(ModelRequest, history[3]).parts[0]).metadata = {'discovered_tools': [_TOOL_NAME, 1]}
    else:
        cast(ToolReturnPart, cast(ModelRequest, history[3]).parts[0]).metadata = {
            'discovered_tools': [_TOOL_NAME, 'not_owned_by_the_capability']
        }
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='test'))
    _, prepared_parameters = model.prepare_request(None, parameters)

    prepared = model.prepare_messages(history, prepared_parameters)

    assert _projected_reveal_shape(prepared) == 'local-search'


@pytest.mark.parametrize(
    ('model_name', 'expects_tool_addition'),
    [('claude-opus-4-8', True), ('claude-sonnet-4-6', False)],
)
async def test_legacy_translation_anthropic_beta_matches_rendered_tool_addition(
    allow_model_requests: None, model_name: str, expects_tool_addition: bool
) -> None:
    """The beta header follows the post-Stage-3 list exactly: upgraded on, identity replay off."""
    anthropic_client = MockAnthropic.create_mock(
        [completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1))]
    )
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=anthropic_client))
    model_settings, parameters = model.prepare_request(None, _portability_parameters(capability_owned=True))
    prepared = model.prepare_messages(_legacy_fabricated_history('R1'), parameters)

    await model.request(prepared, model_settings, parameters)

    request = get_mock_chat_completion_kwargs(anthropic_client)[0]
    betas = request['betas']
    has_beta = isinstance(betas, list) and 'mid-conversation-tool-changes-2026-07-01' in betas
    has_tool_addition = '"type": "tool_addition"' in json.dumps(request['messages'], sort_keys=True)
    assert has_beta is expects_tool_addition
    assert has_tool_addition is expects_tool_addition


async def test_anthropic_beta_is_absent_when_delta_renders_no_tool_addition(allow_model_requests: None) -> None:
    """A no-op reveal of an already-visible tool emits neither the block nor its enabling beta."""
    anthropic_client = MockAnthropic.create_mock(
        [completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1))]
    )
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=anthropic_client))
    model_settings, parameters = model.prepare_request(
        None,
        ModelRequestParameters(function_tools=[ToolDefinition(name='always_ready')]),
    )
    messages = model.prepare_messages(
        [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['always_ready'])])], parameters
    )

    await model.request(messages, model_settings, parameters)

    request = get_mock_chat_completion_kwargs(anthropic_client)[0]
    assert not isinstance(request['betas'], list)
    assert '"type": "tool_addition"' not in json.dumps(request['messages'], sort_keys=True)


async def test_anthropic_beta_is_absent_when_delta_names_an_output_tool(allow_model_requests: None) -> None:
    """A delta naming an output tool renders nothing, so it must not enable the beta either.

    Output tools are never revealed: the `tool_addition` renderer only ever references function
    tools, and the header predicate must stay co-extensive with it even for forged or replayed
    history that names an output tool.
    """
    anthropic_client = MockAnthropic.create_mock(
        [completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1))]
    )
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=anthropic_client))
    model_settings, parameters = model.prepare_request(
        None,
        ModelRequestParameters(
            output_mode='tool',
            allow_text_output=False,
            function_tools=[ToolDefinition(name='always_ready')],
            output_tools=[ToolDefinition(name='final_result')],
        ),
    )
    messages = model.prepare_messages(
        [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['final_result'])])], parameters
    )

    await model.request(messages, model_settings, parameters)

    request = get_mock_chat_completion_kwargs(anthropic_client)[0]
    assert not isinstance(request['betas'], list)
    assert '"type": "tool_addition"' not in json.dumps(request['messages'], sort_keys=True)


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
    conversation = next(body[key] for key in ('messages', 'input', 'contents') if key in body)
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
    tool_addition_blocks = [node for node in nodes if node.get('type') == 'tool_addition']
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
    serialized_conversation = json.dumps(conversation)
    return {
        'announcements': serialized_conversation.count(TOOL_AVAILABILITY_ANNOUNCEMENT.format(names=f'`{_TOOL_NAME}`')),
        'search_calls': len(search_call_nodes),
        'search_returns': len(search_return_nodes),
        'tool_addition_blocks': len(tool_addition_blocks),
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

    if case.rendering in ('native-search', 'local-search', 'local-search-additional-tools'):
        assert facts['search_calls'] == facts['search_returns'] == 1
        assert facts['tool_addition_blocks'] == 0
        assert facts['additional_tools'] == (1 if case.rendering == 'local-search-additional-tools' else 0)
        assert facts['announcements'] == 0
    elif case.rendering == 'announcement':
        assert facts['search_calls'] == facts['search_returns'] == 0
        assert facts['tool_addition_blocks'] == facts['additional_tools'] == 0
        assert facts['announcements'] == 1
    else:
        # No search happened — a delta is control, not discovery — but the search *surface* stays on the
        # wire, and so does the revealed tool's own deferred declaration. `tools` is the first cache
        # section, so a delta turn has to send it exactly as the turn before did; both adapters used to
        # rewrite it here, which had the feature busting the very prefix it exists to protect. See
        # `test_tool_availability_delta_and_the_tools_cache_section`, which measures that directly — these
        # assertions only notice the symptom.
        assert facts['search_calls'] == facts['search_returns'] == 0
        assert facts['tool_addition_blocks'] == (1 if case.rendering == 'tool-addition' else 0)
        assert facts['additional_tools'] == (1 if case.rendering == 'additional-tools' else 0)
        assert facts['announcements'] == 0

    # The search *surface* is on the wire in every case, delta turns included — a target without a
    # native one still declares the local `search_tools`. Exactly one of it, in both worlds: the two
    # surfaces are alternatives, and a case that sent both would be sending the model two ways to ask
    # the same question.
    assert facts['search_tools'] == 1
    assert facts['revealed_tools'] == (2 if case.id in _TWO_REVEALED_TOOL_NODES else 1)
    # A deferred declaration in `tools` is what a delta *reveals*, so it's there exactly when the target
    # has a native tool-search surface to have declared it. Where there isn't one — `gpt-5` on Responses,
    # Gemini, OpenAI Chat — the tool was never on the wire, so the item introduces it instead and there's
    # no `defer_loading` to find. Both are prefix-stable; they differ in what there was to preserve.
    if case.target in _NATIVE_TOOL_SEARCH_TARGETS and case.rendering != 'local-search':
        assert facts['revealed_defer_loading'] == [True]
    else:
        assert facts['revealed_defer_loading'] == []


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


@pytest.mark.parametrize(
    ('provider', 'model_name'),
    [
        ('anthropic', 'claude-opus-4-8'),
        ('anthropic', 'claude-sonnet-4-6'),
        ('openai-responses', 'gpt-5.6'),
        ('openai-responses', 'gpt-5'),
    ],
)
async def test_tool_availability_delta_and_the_tools_cache_section(
    allow_model_requests: None, provider: str, model_name: str
) -> None:
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
    local_search = ToolDefinition(
        name='search_tools',
        description='Search tools.',
        parameters_json_schema={'type': 'object'},
        unless_native=ToolSearchTool.kind,
    )
    parameters = ModelRequestParameters(
        function_tools=[always_ready, local_search, tool], native_tools=[ToolSearchTool()]
    )
    before: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Find the exchange-rate tool.')]),
        ModelResponse(parts=[TextPart(content='Looking.')]),
    ]
    after: list[ModelMessage] = [*before, ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[_TOOL_NAME])])]

    if provider == 'anthropic':
        anthropic_client = MockAnthropic.create_mock(
            [
                completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1)),
                completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1)),
            ]
        )
        model: Model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=anthropic_client))
        model_settings, parameters = model.prepare_request(None, parameters)
        await model.request(model.prepare_messages(before, parameters), model_settings, parameters)
        await model.request(model.prepare_messages(after, parameters), model_settings, parameters)
        anthropic_requests = get_mock_chat_completion_kwargs(anthropic_client)
        sent = [kwargs['tools'] for kwargs in anthropic_requests]
        if model_name == 'claude-sonnet-4-6':
            serialized_messages = json.dumps(anthropic_requests[1]['messages'], sort_keys=True)
            assert '"type": "tool_use"' in serialized_messages
            assert '"type": "tool_result"' in serialized_messages
            assert 'tool_reference' in serialized_messages
    else:
        openai_client = MockOpenAIResponses.create_mock(
            [_empty_responses_message(), _empty_responses_message()],
        )
        model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=openai_client))
        model_settings, before_parameters = model.prepare_request(None, parameters)
        _, after_parameters = model.prepare_request(None, replace(parameters, revealed_tool_names={_TOOL_NAME}))
        await model.request(before, model_settings, before_parameters)
        await model.request(after, model_settings, after_parameters)
        openai_requests = get_mock_responses_kwargs(openai_client)
        sent = [kwargs['tools'] for kwargs in openai_requests]
        assert any(node.get('type') == 'additional_tools' for node in _walk(openai_requests[1]))

    before_tools, after_tools = sent
    assert json.dumps(after_tools, sort_keys=True) == json.dumps(before_tools, sort_keys=True)
    # And the deferred declaration is genuinely still there, rather than both turns sending nothing.
    if model_name != 'gpt-5':
        assert any(node.get('name') == _TOOL_NAME for node in _walk(after_tools))


@pytest.mark.parametrize('provider', ['openai-chat', 'openai-responses'])
async def test_no_delta_channel_deliberately_moves_the_cache_prefix(allow_model_requests: None, provider: str) -> None:
    """Level C deliberately degrades: revealing a tool appends one schema and moves the cache prefix.

    Both announcement-path adapters receive the one-time system text after the schema is appended.
    """
    tool = ToolDefinition(
        name='lookup_refund_policy',
        description='Look up the refund policy for an order.',
        parameters_json_schema={'type': 'object'},
        defer_loading=True,
        capability_id='refunds',
    )
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='load_capability', parameters_json_schema={'type': 'object'}), tool],
        revealed_tool_names=set(),
    )
    before: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Can I get a refund?')])]
    after = [*before, ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[tool.name])])]
    revealed_parameters = ModelRequestParameters(
        function_tools=parameters.function_tools,
        revealed_tool_names={tool.name},
    )

    if provider == 'openai-chat':
        client = MockOpenAI.create_mock(
            [
                openai_completion_message(ChatCompletionMessage(role='assistant', content='ok')),
                openai_completion_message(ChatCompletionMessage(role='assistant', content='ok')),
            ]
        )
        model: Model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(openai_client=client))
        _, before_parameters = model.prepare_request(None, parameters)
        _, after_parameters = model.prepare_request(None, revealed_parameters)
        await model.request(model.prepare_messages(before, before_parameters), None, before_parameters)
        await model.request(model.prepare_messages(after, after_parameters), None, after_parameters)
        requests = get_mock_openai_chat_completion_kwargs(client)
        assert requests[1]['messages'][-1] == {
            'role': 'system',
            'content': 'The following tool(s) are now available: `lookup_refund_policy`',
        }
    else:
        client = MockOpenAIResponses.create_mock([_empty_responses_message(), _empty_responses_message()])
        model = OpenAIResponsesModel(
            'gpt-5',
            provider=OpenAIProvider(openai_client=client),
            profile=merge_profile(
                openai_model_profile('gpt-5'),
                OpenAIModelProfile(tool_addition_mode=None),
            ),
        )
        _, before_parameters = model.prepare_request(None, parameters)
        _, after_parameters = model.prepare_request(None, revealed_parameters)
        await model.request(model.prepare_messages(before, before_parameters), None, before_parameters)
        await model.request(model.prepare_messages(after, after_parameters), None, after_parameters)
        requests = get_mock_responses_kwargs(client)
        assert requests[1]['input'][-1] == {
            'role': 'system',
            'content': 'The following tool(s) are now available: `lookup_refund_policy`',
        }

    before_tools, after_tools = (request['tools'] for request in requests)

    def names(tools: list[dict[str, Any]]) -> list[str]:
        return [cast(str, tool.get('name') or tool['function']['name']) for tool in tools]

    assert names(before_tools) == ['load_capability']
    assert names(after_tools) == ['load_capability', 'lookup_refund_policy']
    assert after_tools[:-1] == before_tools


async def test_google_delta_announcement_is_appended_once_and_stays_put(
    allow_model_requests: None, mocker: MockerFixture
) -> None:
    """Gemini appends one announcement, keeps it fixed thereafter, and adds one wire tool schema."""
    model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='test-key'))
    response = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text='ok')], role='model'))],
        response_id='response-1',
        model_version='gemini-3-flash-preview',
    )
    generate = mocker.patch.object(model.client.aio.models, 'generate_content', return_value=response)
    tool = ToolDefinition(
        name=_TOOL_NAME, description='Look up an exchange rate.', parameters_json_schema={}, defer_loading=True
    )
    before_parameters = ModelRequestParameters(function_tools=[])
    after_parameters = ModelRequestParameters(function_tools=[tool], revealed_tool_names={tool.name})
    before: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Find the exchange-rate tool.')])]
    after = [*before, ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[tool.name])])]

    await model.request(model.prepare_messages(before, before_parameters), None, before_parameters)
    await model.request(model.prepare_messages(after, after_parameters), None, after_parameters)
    await model.request(model.prepare_messages(after, after_parameters), None, after_parameters)

    configs = [call.kwargs['config'] for call in generate.call_args_list]
    contents = [call.kwargs['contents'] for call in generate.call_args_list]
    assert configs[0].get('tools') is None
    assert configs[1]['tools'] == configs[2]['tools']
    assert len(configs[1]['tools'][0]['function_declarations']) == 1
    announcement = 'The following tool(s) are now available: `lookup_exchange_rate`'
    assert json.dumps(contents[0], default=str).count(announcement) == 0
    assert json.dumps(contents[1], default=str).count(announcement) == 1
    assert json.dumps(contents[2], default=str).count(announcement) == 1
    assert contents[1] == contents[2]


async def test_anthropic_live_delta_preserves_the_warmed_cache_prefix(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
) -> None:
    """Two live Opus requests let the cassette prefix checker guard a delta after a warmed turn."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key))
    tool = ToolDefinition(
        name=_TOOL_NAME,
        description='Look up an exchange rate.',
        parameters_json_schema={'type': 'object'},
        defer_loading=True,
        with_native=ToolSearchTool.kind,
    )
    parameters = ModelRequestParameters(function_tools=[tool], native_tools=[ToolSearchTool()])
    before: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Reply only with: ready')])]

    warm_response = await model.request(before, None, parameters)
    after = [*before, warm_response, ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[tool.name])])]
    await model.request(after, None, parameters)

    assert len(cast(Any, vcr).requests) == 2


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


def test_a_revealed_tool_is_announced_once_and_the_text_never_moves() -> None:
    """One reveal, one announcement, byte-identical on every later turn.

    `ToolSearch.before_model_request` records a reveal only for tools absent from
    `ctx.discovered_tool_names`, so using a tool repeatedly doesn't re-announce it. That matters twice
    over: a second announcement would be noise the model has to reconcile, and because the projection
    reruns over the whole history each turn, any announcement whose text moved would invalidate the
    cached prefix it sits in front of.
    """
    model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='start')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['lookup'])]),
        ModelResponse(parts=[TextPart(content='ok')]),
    ]

    def announcements_of(messages: list[ModelMessage]) -> list[Any]:
        prepared = model.prepare_messages(messages)
        # Nothing tool-search-shaped: the history never claims a search the model didn't run.
        assert not [part for message in prepared for part in message.parts if isinstance(part, ToolSearchCallPart)]
        return [
            part.content
            for message in prepared
            for part in message.parts
            if isinstance(part, UserPromptPart) and 'tool(s) are now available' in str(part.content)
        ]

    # Google has no inline system-prompt support, so the announcement arrives `<system>`-tagged.
    assert announcements_of(history) == snapshot(['<system>The following tool(s) are now available: `lookup`</system>'])

    # Three turns later, still exactly one, still the same bytes.
    assert announcements_of(
        [
            *history,
            ModelRequest(parts=[UserPromptPart(content='again')]),
            ModelResponse(parts=[TextPart(content='sure')]),
            ModelRequest(parts=[UserPromptPart(content='and again')]),
        ]
    ) == snapshot(['<system>The following tool(s) are now available: `lookup`</system>'])


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
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['lookup'])]),
    ]

    with pytest.raises(UserError, match=r'prepare_messages'):
        await model.request(history, None, ModelRequestParameters())


def test_a_mixed_corpus_reveal_gets_the_mechanism_not_just_the_news() -> None:
    """The same model answers differently per request, which is why the parameters are needed.

    On an OpenAI-compatible endpoint without `additional_tools`, a capability-only corpus has nothing
    searchable, so no `tool_search` tool survives, so nothing can be sent wire-deferred — the revealed
    tool is plainly in `tools` and the change is only news. Add one standalone searchable tool and the
    search surface returns, `defer_loading` is sent, and the reveal has to be the tool-search exchange
    or the capability's tool stays locked behind a flag prose can't lift.

    Deciding this from the profile alone gets the mixed case wrong, which is the whole reason
    `prepare_messages` takes `ModelRequestParameters`.
    """
    model = OpenAIResponsesModel(
        'gpt-5.6',
        provider=OpenAIProvider(api_key='test-key'),
        profile=merge_profile(
            openai_model_profile('gpt-5.6'),
            OpenAIModelProfile(tool_addition_mode=None),
        ),
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')]),
        ModelResponse(parts=[TextPart(content='ok')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'])]),
    ]
    gated = ToolDefinition(name='lookup_refund_policy', defer_loading=True, capability_id='refunds')
    searchable = ToolDefinition(name='get_weather', defer_loading=True, with_native=ToolSearchTool.kind)

    def rendering(function_tools: list[ToolDefinition]) -> list[str]:
        _, parameters = model.prepare_request(
            None,
            ModelRequestParameters(
                function_tools=function_tools,
                native_tools=[ToolSearchTool(optional=True)],
                revealed_tool_names={'lookup_refund_policy'},
            ),
        )
        prepared = model.prepare_messages(
            history,
            parameters,
        )
        return [type(part).__name__ for message in prepared for part in message.parts]

    assert rendering([gated]) == snapshot(['UserPromptPart', 'TextPart', 'SystemPromptPart'])
    assert rendering([searchable, gated]) == snapshot(
        ['UserPromptPart', 'TextPart', 'ToolSearchCallPart', 'ToolSearchReturnPart']
    )


def _project_synthesized_reveal(history: list[ModelMessage]) -> list[ModelMessage]:
    """Project `history` on a model whose delta rendering is the synthesized search exchange."""
    model = OpenAIResponsesModel(
        'gpt-5.6',
        provider=OpenAIProvider(api_key='test-key'),
        profile=merge_profile(
            openai_model_profile('gpt-5.6'),
            OpenAIModelProfile(tool_addition_mode=None),
        ),
    )
    gated = ToolDefinition(name='lookup_refund_policy', defer_loading=True, capability_id='refunds')
    searchable = ToolDefinition(name='get_weather', defer_loading=True, with_native=ToolSearchTool.kind)
    _, parameters = model.prepare_request(
        None,
        ModelRequestParameters(
            function_tools=[searchable, gated],
            native_tools=[ToolSearchTool(optional=True)],
            revealed_tool_names={'lookup_refund_policy'},
        ),
    )
    return model.prepare_messages(history, parameters)


def _reveal_digest_id(ordinal: int, *names: str) -> str:
    digest = hashlib.blake2s(
        '\x00'.join([str(ordinal), *names]).encode(), digest_size=8, usedforsecurity=False
    ).hexdigest()
    return f'{TOOL_CALL_ID_PREFIX}{digest}'


def test_delta_reusing_a_live_call_id_gets_a_fresh_synthesized_id() -> None:
    """A client-authored delta naming a call that is still in the history must not clone its id.

    The UI adapters round-trip client-authored deltas verbatim, and the field's docstring invites
    setting `tool_call_id` to the call that triggered the load — which is typically still present.
    Passing it through would emit two assistant call parts with the same id, which providers
    requiring globally unique call ids reject or mis-pair (https://github.com/pydantic/pydantic-ai/issues/7187).
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')]),
        ModelResponse(parts=[ToolCallPart(tool_name='load_refunds', args={}, tool_call_id='call_x')]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='load_refunds', content='ok', tool_call_id='call_x'),
                ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'], tool_call_id='call_x'),
            ]
        ),
    ]

    prepared = _project_synthesized_reveal(history)

    [search_call] = [part for message in prepared for part in message.parts if isinstance(part, ToolSearchCallPart)]
    [search_return] = [part for message in prepared for part in message.parts if isinstance(part, ToolSearchReturnPart)]
    assert search_call.tool_call_id == _reveal_digest_id(0, 'lookup_refund_policy')
    assert search_return.tool_call_id == search_call.tool_call_id
    # The original exchange keeps its id untouched: exactly one call and one return carry `call_x`.
    original_parts = [
        part
        for message in prepared
        for part in message.parts
        if isinstance(part, ToolCallPart | ToolReturnPart) and part.tool_call_id == 'call_x'
    ]
    assert [type(part).__name__ for part in original_parts] == ['ToolCallPart', 'ToolReturnPart']


def test_delta_reusing_a_collapsed_exchange_id_passes_it_through() -> None:
    """When the id's original exchange is absent — the projection collapsed it — reuse is the point:
    the synthesized exchange stands in for the one that used to carry that id, as in the R4 fixtures."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')]),
        ModelResponse(parts=[TextPart(content='ok')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'], tool_call_id='call_y')]),
    ]

    prepared = _project_synthesized_reveal(history)

    [search_call] = [part for message in prepared for part in message.parts if isinstance(part, ToolSearchCallPart)]
    assert search_call.tool_call_id == 'call_y'


def test_fabricated_id_skips_a_client_authored_lookalike() -> None:
    """A history part carrying the exact id fabrication would produce forces the next ordinal."""
    lookalike = _reveal_digest_id(0, 'lookup_refund_policy')
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')]),
        ModelResponse(parts=[ToolCallPart(tool_name='load_refunds', args={}, tool_call_id=lookalike)]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='load_refunds', content='ok', tool_call_id=lookalike),
                ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy']),
            ]
        ),
    ]

    prepared = _project_synthesized_reveal(history)

    [search_call] = [part for message in prepared for part in message.parts if isinstance(part, ToolSearchCallPart)]
    assert search_call.tool_call_id == _reveal_digest_id(1, 'lookup_refund_policy')


def test_openai_models_enforce_inherited_reveal_channel_claims() -> None:
    """Each OpenAI API shape honors only the reveal channels its adapter declares.

    Pass-through providers (e.g. OpenRouter) serve Anthropic-family profiles whose
    `tool_deferral_mode='standalone'` claim only the Anthropic API can honor. `OpenAIChatModel`
    renders no channel at all; `OpenAIResponsesModel` implements only `with_tool_search` deferral
    and `with_definitions` additions. The profile keeps the vendor facts, while the effective model
    properties reject foreign modes so a hidden tool resolves `withheld` instead of a
    `defer_loading` wire shape the endpoint would reject.
    """
    provider = OpenRouterProvider(api_key='x')
    vendor_profile = provider.model_profile('anthropic/claude-sonnet-4-6')
    assert vendor_profile is not None and vendor_profile.get('tool_deferral_mode') == 'standalone'

    hidden = ToolDefinition(
        name='hidden_tool', parameters_json_schema={'type': 'object'}, defer_loading=True, capability_id='refunds'
    )
    visible = ToolDefinition(name='visible_tool', parameters_json_schema={'type': 'object'})

    responses_model = OpenAIResponsesModel('anthropic/claude-sonnet-4-6', provider=provider)
    assert responses_model.profile.get('tool_deferral_mode') == 'standalone'
    assert responses_model.tool_deferral_mode is None
    _, prepared = responses_model.prepare_request(None, ModelRequestParameters(function_tools=[hidden, visible]))
    assert prepared.tool_visibility == {'hidden_tool': 'withheld', 'visible_tool': 'visible'}

    chat_model = OpenAIChatModel('anthropic/claude-sonnet-4-6', provider=OpenRouterProvider(api_key='x'))
    assert chat_model.profile.get('tool_deferral_mode') == 'standalone'
    assert chat_model.tool_deferral_mode is None

    # The first-party claims are the shapes the Responses renderer implements — they survive.
    first_party = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='x'))
    assert first_party.tool_deferral_mode == 'with_tool_search'
    assert first_party.tool_addition_mode == 'with_definitions'

    # Explicit foreign profile claims remain profile facts but are behaviorally ignored.
    foreign = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key='x'),
        profile=merge_profile(
            openai_model_profile('gpt-5'),
            OpenAIModelProfile(tool_deferral_mode='standalone', tool_addition_mode='by_reference'),
        ),
    )
    assert foreign.profile.get('tool_deferral_mode') == 'standalone'
    assert foreign.profile.get('tool_addition_mode') == 'by_reference'
    assert foreign.tool_deferral_mode is None
    assert foreign.tool_addition_mode is None


def test_base_model_requires_structural_reveal_channel_declarations() -> None:
    """A pass-through profile cannot enable wire behavior an adapter did not declare.

    The base default is empty, so a `Model` subclass that declares nothing is safe by
    construction. The test doubles (`TestModel`, `FunctionModel`) declare every mode — they
    have no wire, so the profile handed to them is the whole simulation — which is why the
    undeclared case below explicitly clears the sets to model a real channel-less adapter.
    """
    assert Model.supported_tool_deferral_modes == frozenset()
    assert Model.supported_tool_addition_modes == frozenset()

    class UndeclaredModel(TestModel):
        supported_tool_deferral_modes = frozenset()
        supported_tool_addition_modes = frozenset()

    profile = ModelProfile(tool_deferral_mode='standalone', tool_addition_mode='by_reference')
    hidden = ToolDefinition(
        name='hidden_tool', parameters_json_schema={'type': 'object'}, defer_loading=True, capability_id='refunds'
    )
    params = ModelRequestParameters(function_tools=[hidden])

    undeclared = UndeclaredModel(profile=profile)
    _, undeclared_params = undeclared.prepare_request(None, params)
    assert undeclared.tool_deferral_mode is None
    assert undeclared.tool_addition_mode is None
    assert undeclared_params.tool_visibility == {'hidden_tool': 'withheld'}
    assert undeclared_params.declared_tool_defs == {}

    history: list[ModelMessage] = [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['hidden_tool'])])]
    prepared_messages = undeclared.prepare_messages(history, undeclared_params)
    assert TOOL_AVAILABILITY_ANNOUNCEMENT.format(names='`hidden_tool`') in json.dumps(
        ModelMessagesTypeAdapter.dump_python(prepared_messages, mode='json')
    )

    # The double itself declares everything, so the same profile is honored without a subclass.
    declared = TestModel(profile=profile)
    _, declared_params = declared.prepare_request(None, params)
    assert declared.tool_deferral_mode == 'standalone'
    assert declared.tool_addition_mode == 'by_reference'
    assert declared_params.tool_visibility == {'hidden_tool': 'deferred'}
    assert set(declared_params.declared_tool_defs) == {'hidden_tool'}


async def test_openai_responses_deduplicates_additional_tools_across_parts(allow_model_requests: None) -> None:
    """Several history parts naming the same revealed tool render one `additional_tools` item.

    A UI round-trip can legitimately duplicate a delta; one declaration per request is enough,
    mirroring the Anthropic renderer's per-request `tool_addition` dedupe.
    """
    tool = ToolDefinition(
        name=_TOOL_NAME,
        description='Look up an exchange rate.',
        parameters_json_schema={'type': 'object'},
        defer_loading=True,
        capability_id='refunds',
    )
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='load_capability', parameters_json_schema={'type': 'object'}), tool],
        revealed_tool_names={_TOOL_NAME},
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[_TOOL_NAME])]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[_TOOL_NAME])]),
    ]
    client = MockOpenAIResponses.create_mock([_empty_responses_message()])
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=client))
    _, prepared = model.prepare_request(None, parameters)

    await model.request(model.prepare_messages(history, prepared), None, prepared)

    request = get_mock_responses_kwargs(client)[0]
    additional = [node for node in _walk(request) if node.get('type') == 'additional_tools']
    assert len(additional) == 1
    assert [tool_param['name'] for tool_param in additional[0]['tools']] == [_TOOL_NAME]


@pytest.mark.parametrize('origin', ['R1', 'R5'])
@pytest.mark.parametrize('model_name', ['gpt-5.6', 'gpt-5'])
async def test_legacy_fabricated_search_upgrades_to_responses_additional_tools(
    allow_model_requests: None, origin: Literal['R1', 'R5'], model_name: str
) -> None:
    """A recognized legacy fabrication becomes one complete Responses reveal item on the wire."""
    client = MockOpenAIResponses.create_mock(_empty_responses_message())
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=client))
    tool = ToolDefinition(
        name=_TOOL_NAME,
        description='Look up an exchange rate.',
        parameters_json_schema={'type': 'object', 'properties': {'currency': {'type': 'string'}}},
        defer_loading=True,
        capability_id='finance',
    )
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='load_capability'), tool],
        revealed_tool_names={tool.name},
    )
    settings, parameters = model.prepare_request(None, parameters)

    await model.request(model.prepare_messages(_legacy_fabricated_history(origin), parameters), settings, parameters)

    request = get_mock_responses_kwargs(client)[0]
    additional = [item for item in request['input'] if item.get('type') == 'additional_tools']
    assert additional == [
        {
            'type': 'additional_tools',
            'role': 'developer',
            'tools': [
                {
                    'name': _TOOL_NAME,
                    'parameters': {
                        'type': 'object',
                        'properties': {'currency': {'type': 'string'}},
                        'additionalProperties': False,
                    },
                    'strict': False,
                    'type': 'function',
                    'description': 'Look up an exchange rate.',
                }
            ],
        }
    ]
    assert not any(item.get('name') == 'search_tools' for item in _walk(request['input']))


async def test_delta_wire_shape_survives_anthropic_google_anthropic_hops(
    allow_model_requests: None, mocker: MockerFixture
) -> None:
    """A channel-less detour cannot consume a stored delta or reorder its definitions."""
    tools = [
        ToolDefinition(name='always_ready'),
        ToolDefinition(name='a_tool', description='A.', defer_loading=True),
        ToolDefinition(name='b_tool', description='B.', defer_loading=True),
    ]
    parameters = ModelRequestParameters(function_tools=tools, revealed_tool_names={'a_tool', 'b_tool'})
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='start')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['b_tool', 'a_tool'])]),
    ]

    anthropic_client = MockAnthropic.create_mock(
        [
            completion_message([BetaTextBlock(text='first', type='text')], BetaUsage(input_tokens=1, output_tokens=1)),
            completion_message([BetaTextBlock(text='third', type='text')], BetaUsage(input_tokens=1, output_tokens=1)),
        ]
    )
    anthropic = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=anthropic_client))
    settings, anthropic_parameters = anthropic.prepare_request(None, parameters)
    await anthropic.request(anthropic.prepare_messages(history, anthropic_parameters), settings, anthropic_parameters)

    google = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='test'))
    google_response = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text='google')], role='model'))],
        response_id='response-1',
        model_version='gemini-3-flash-preview',
    )
    generate = mocker.patch.object(google.client.aio.models, 'generate_content', return_value=google_response)
    _, google_parameters = google.prepare_request(None, parameters)
    response = await google.request(google.prepare_messages(history, google_parameters), None, google_parameters)

    await anthropic.request(
        anthropic.prepare_messages([*history, response], anthropic_parameters), settings, anthropic_parameters
    )

    first, third = get_mock_chat_completion_kwargs(anthropic_client)
    for request in (first, third):
        assert 'mid-conversation-tool-changes-2026-07-01' in request['betas']
        assert json.dumps(request['messages'], sort_keys=True).count('"type": "tool_addition"') == 2
    assert [tool.get('name') for tool in third['tools']][-2:] == ['b_tool', 'a_tool']
    google_contents = generate.call_args.kwargs['contents']
    announcement = TOOL_AVAILABILITY_ANNOUNCEMENT.format(names='`b_tool`, `a_tool`')
    assert json.dumps(google_contents, default=str).count(f'<system>{announcement}</system>') == 1


async def test_truncated_reveal_omits_anthropic_channel(allow_model_requests: None) -> None:
    """A deferred non-corpus tool without surviving reveal evidence stays completely off the wire."""
    client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=1, output_tokens=1))
    )
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=client))
    parameters = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='always_ready'),
            ToolDefinition(name='hidden_tool', defer_loading=True, capability_id='hidden'),
        ]
    )
    settings, parameters = model.prepare_request(None, parameters)
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='continue')])]

    await model.request(model.prepare_messages(history, parameters), settings, parameters)

    request = get_mock_chat_completion_kwargs(client)[0]
    assert not isinstance(request['betas'], list)
    assert 'tool_addition' not in json.dumps(request['messages'], sort_keys=True)
    assert request['tools'] == [
        {'name': 'always_ready', 'description': '', 'input_schema': {'type': 'object', 'properties': {}}},
        {
            'name': 'hidden_tool',
            'description': '',
            'input_schema': {'type': 'object', 'properties': {}},
            'defer_loading': True,
        },
    ]


@pytest.mark.parametrize('supports_inline_system_prompts', [True, False])
def test_announcement_part_respects_inline_system_prompt_support(supports_inline_system_prompts: bool) -> None:
    """The fallback channel uses the target profile's supported message role."""
    model = FunctionModel(
        lambda _messages, _info: ModelResponse(parts=[TextPart(content='unused')]),
        profile=ModelProfile(supports_inline_system_prompts=supports_inline_system_prompts),
    )
    tool = ToolDefinition(name='revealed_tool', defer_loading=True)
    prepared = model.prepare_messages(
        [
            ModelRequest(parts=[UserPromptPart(content='start')]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[tool.name])]),
        ],
        ModelRequestParameters(function_tools=[tool], revealed_tool_names={tool.name}),
    )

    [part] = prepared[1].parts
    expected = TOOL_AVAILABILITY_ANNOUNCEMENT.format(names='`revealed_tool`')
    if supports_inline_system_prompts:
        assert isinstance(part, SystemPromptPart)
        assert part.content == expected
    else:
        assert isinstance(part, UserPromptPart)
        assert part.content == f'<system>{expected}</system>'


def test_announcements_render_after_sibling_tool_returns() -> None:
    """Announcements for a turn that reveals several tools land after the request's tool returns."""
    model = FunctionModel(
        lambda _messages, _info: ModelResponse(parts=[TextPart(content='unused')]),
        profile=ModelProfile(supports_inline_system_prompts=True),
    )
    tools = [
        ToolDefinition(name='first_tool', defer_loading=True),
        ToolDefinition(name='second_tool', defer_loading=True),
    ]
    prepared = model.prepare_messages(
        [
            ModelRequest(parts=[UserPromptPart(content='start')]),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='load_capability', content='loaded', tool_call_id='call_1'),
                    ToolAvailabilityDeltaPart(tools_added=['first_tool']),
                    ToolReturnPart(tool_name='load_capability', content='loaded', tool_call_id='call_2'),
                    ToolAvailabilityDeltaPart(tools_added=['second_tool']),
                ]
            ),
        ],
        ModelRequestParameters(function_tools=tools, revealed_tool_names={tool.name for tool in tools}),
    )

    parts = prepared[1].parts
    assert [type(part) for part in parts] == [ToolReturnPart, ToolReturnPart, SystemPromptPart, SystemPromptPart]
    assert [part.tool_call_id for part in parts if isinstance(part, ToolReturnPart)] == ['call_1', 'call_2']
    assert [part.content for part in parts if isinstance(part, SystemPromptPart)] == [
        TOOL_AVAILABILITY_ANNOUNCEMENT.format(names='`first_tool`'),
        TOOL_AVAILABILITY_ANNOUNCEMENT.format(names='`second_tool`'),
    ]


def test_standing_system_prompt_stays_ahead_of_sorted_tool_returns() -> None:
    """The first request's opening system prompt keeps its place when tool returns sort forward."""
    model = FunctionModel(
        lambda _messages, _info: ModelResponse(parts=[TextPart(content='unused')]),
        profile=ModelProfile(supports_inline_system_prompts=True),
    )
    tool = ToolDefinition(name='first_tool', defer_loading=True)
    prepared = model.prepare_messages(
        [
            # Drops out entirely (its only delta names a tool that is no longer served), passing
            # the leading role to the next request.
            ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['withdrawn_tool'])]),
            ModelRequest(
                parts=[
                    SystemPromptPart(content='standing'),
                    ToolReturnPart(tool_name='load_capability', content='loaded', tool_call_id='call_1'),
                    ToolAvailabilityDeltaPart(tools_added=['first_tool']),
                ]
            ),
        ],
        ModelRequestParameters(function_tools=[tool], revealed_tool_names={tool.name}),
    )

    parts = prepared[0].parts
    assert [type(part) for part in parts] == [SystemPromptPart, ToolReturnPart, SystemPromptPart]
    assert [part.content for part in parts if isinstance(part, SystemPromptPart)] == [
        'standing',
        TOOL_AVAILABILITY_ANNOUNCEMENT.format(names='`first_tool`'),
    ]


async def test_responses_output_tool_stays_forceable_alongside_reveal(allow_model_requests: None) -> None:
    """Output-tool forcing remains independent from a revealed function in `additional_tools`."""
    client = MockOpenAIResponses.create_mock(_empty_responses_message())
    model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(openai_client=client))
    revealed = ToolDefinition(name='revealed_tool', description='Revealed.', defer_loading=True)
    output = ToolDefinition(name='final_result', description='Return the result.')
    settings, parameters = model.prepare_request(
        OpenAIResponsesModelSettings(tool_choice=['final_result']),
        ModelRequestParameters(
            function_tools=[revealed],
            output_tools=[output],
            output_mode='tool',
            allow_text_output=False,
            revealed_tool_names={revealed.name},
        ),
    )

    await model.request(
        [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[revealed.name])])], settings, parameters
    )

    request = get_mock_responses_kwargs(client)[0]
    assert [tool['name'] for tool in request['tools']] == ['final_result']
    [additional] = [item for item in request['input'] if item.get('type') == 'additional_tools']
    assert [tool['name'] for tool in additional['tools']] == ['revealed_tool']
    assert request['tool_choice'] == {'type': 'function', 'name': 'final_result'}


async def test_anthropic_streaming_request_carries_tool_addition(allow_model_requests: None) -> None:
    """Streaming uses the same Anthropic reveal request assembly as non-streaming."""
    stream = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='message-1',
                content=[],
                model='claude-opus-4-8',
                role='assistant',
                stop_reason=None,
                type='message',
                usage=BetaUsage(input_tokens=1, output_tokens=0),
            ),
        )
    ]
    client = MockAnthropic.create_stream_mock(stream)
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=client))
    tool = ToolDefinition(name='revealed_tool', defer_loading=True)
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='always_ready'), tool], revealed_tool_names={tool.name}
    )
    settings, parameters = model.prepare_request(None, parameters)

    async with model.request_stream(
        [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[tool.name])])], settings, parameters
    ):
        pass

    [request] = get_mock_chat_completion_kwargs(client)
    assert 'mid-conversation-tool-changes-2026-07-01' in request['betas']
    assert '"type": "tool_addition"' in json.dumps(request['messages'], sort_keys=True)


async def test_responses_streaming_request_carries_additional_tools(allow_model_requests: None) -> None:
    """Streaming uses the same Responses reveal request assembly as non-streaming."""
    response = _empty_responses_message()
    client = MockOpenAIResponses.create_mock_stream(
        [ResponseCreatedEvent(response=response, sequence_number=0, type='response.created')]
    )
    model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(openai_client=client))
    tool = ToolDefinition(name='revealed_tool', defer_loading=True)
    parameters = ModelRequestParameters(function_tools=[tool], revealed_tool_names={tool.name})
    settings, parameters = model.prepare_request(None, parameters)

    async with model.request_stream(
        [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[tool.name])])], settings, parameters
    ):
        pass

    [request] = get_mock_responses_kwargs(client)
    [additional] = [item for item in request['input'] if item.get('type') == 'additional_tools']
    assert [wire_tool['name'] for wire_tool in additional['tools']] == ['revealed_tool']
