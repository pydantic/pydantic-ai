"""Anthropic's computer use tools, sent in place of a tool of the `computer` kind.

These aren't recorded (VCR) tests. They check the exact request and stream shapes, which the cassettes' body matching
can miss. They also cover cases no live model produces, such as stored histories. The recorded
runs are in `tests/harness/computer_use/`.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import Any, Literal

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import Hooks, NativeTool
from pydantic_ai.computer_use import COMPUTER_KIND, Click, ComputerCallPart, ComputerToolAction, PressKeys
from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.native_tools import AdvisorTool, ComputerTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from ...computer_fakes import FAKE_DISPLAY_SIZE, FakeComputer
from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic import Omit
    from anthropic.types.beta import (
        BetaContentBlock,
        BetaInputJSONDelta,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaTextBlock,
        BetaToolUseBlock,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    # Hypothesis warns when another thread uses `random` mid-draw (see `tests/test_history_properties.py`), which
    # `filterwarnings = error` makes a flaky failure. No strategy here uses `random`.
    pytest.mark.filterwarnings(
        'ignore:Do not use the `random` module inside strategies:hypothesis.errors.HypothesisDeprecationWarning'
    ),
]


def _agent(model: AnthropicModel, computer: FakeComputer) -> Agent[None, str]:
    return Agent(model, capabilities=[computer, InMemoryProvider().capability()])


def _tool_use(id: str, name: str, input: dict[str, Any]) -> BetaToolUseBlock:
    return BetaToolUseBlock.model_validate(
        {'id': id, 'type': 'tool_use', 'name': name, 'input': input, 'toolset_name': 'computer'}
    )


def _turn(*blocks: BetaContentBlock) -> BetaMessage:
    return completion_message(list(blocks), BetaUsage(input_tokens=1, output_tokens=1))


async def _mapped(name: str, input: dict[str, Any]) -> list[dict[str, Any]]:
    """The `computer` actions that a response with one toolset call becomes."""
    client = MockAnthropic.create_mock(_turn(_tool_use('toolu_1', name, input)))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    params = ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True)
    response = await model.request([ModelRequest.user_text_prompt('Go.')], None, params)
    (call,) = [part for part in response.parts if isinstance(part, ComputerCallPart)]
    return call.args_as_dict()['actions']


async def test_each_toolset_call_is_one_computer_call_with_its_own_result(allow_model_requests: None) -> None:
    computer = FakeComputer()
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6], 'text': 'shift'}),
                # One toolset call can become several actions, such as two key presses.
                _tool_use('toolu_2', 'key', {'text': 'ctrl+a BackSpace'}),
                _tool_use('toolu_3', 'hold_key', {'text': 'ctrl', 'duration': 1}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    result = await _agent(model, computer).run('Go.')

    calls = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)]
    assert [(call.tool_call_id, call.args_as_dict()) for call in calls] == [
        (
            'toolu_1',
            {'actions': [{'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': ['shift']}]},
        ),
        (
            'toolu_2',
            {'actions': [{'type': 'press_keys', 'keys': ['ctrl', 'a']}, {'type': 'press_keys', 'keys': ['backspace']}]},
        ),
        ('toolu_3', {'actions': [{'type': 'hold_keys', 'keys': ['ctrl'], 'seconds': 1}]}),
    ]
    assert computer.actions == [
        Click(x=5, y=6, modifiers=('shift',)),
        PressKeys(keys=('ctrl', 'a')),
        PressKeys(keys=('backspace',)),
    ]

    first, second = get_mock_chat_completion_kwargs(client)
    assert first['tools'] == [{'type': 'computer_toolset_20260801'}]
    # The toolset calls go back unchanged, each answered by its own call's result, which names the toolset.
    calls = second['messages'][-2]['content']
    assert [(c['id'], c['name'], c['toolset_name']) for c in calls] == [
        ('toolu_1', 'left_click', 'computer'),
        ('toolu_2', 'key', 'computer'),
        ('toolu_3', 'hold_key', 'computer'),
    ]
    results = second['messages'][-1]['content']
    # A failed call is an error result with only the error's text, as the toolset asks.
    assert [(r['tool_use_id'], r['toolset_name'], r['is_error'], r['content']) for r in results] == [
        ('toolu_1', 'computer', False, results[0]['content']),
        ('toolu_2', 'computer', False, results[1]['content']),
        ('toolu_3', 'computer', True, [{'type': 'text', 'text': "the fake display cannot perform 'hold_keys'"}]),
    ]
    assert [r['content'][0]['text'] for r in results[:2]] == ['Ran 1 of 1 actions.', 'Ran 2 of 2 actions.']


async def test_a_screenshot_of_a_region_runs_as_a_zoom(allow_model_requests: None) -> None:
    """Claude Sonnet 5.5 sends `region` on `screenshot`, which Anthropic documents only on `zoom`."""
    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'screenshot', {'region': [0, 0, 1280, 200]})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    result = await _agent(model, FakeComputer()).run('Go.')

    (call,) = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)]
    assert call.args_as_dict()['actions'] == [
        {'type': 'zoom', 'region': {'left': 0, 'top': 0, 'right': 1280, 'bottom': 200}}
    ]
    _, second = get_mock_chat_completion_kwargs(client)
    # The call goes back as Claude sent it, and the zoom image is in its result.
    assert second['messages'][-2]['content'][-1]['input'] == {'region': [0, 0, 1280, 200]}
    (sent,) = second['messages'][-1]['content']
    assert [block['type'] for block in sent['content']] == ['text', 'image', 'image']


async def test_a_thread_continue_request_names_the_toolset_on_each_result(allow_model_requests: None) -> None:
    """The calls aren't in the request, which sends only the new messages, so the results are matched to them by ID."""
    client = MockAnthropic.create_mock(
        [
            _turn(
                BetaTextBlock(text='Clicking the field.', type='text'),
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
                _tool_use('toolu_2', 'type', {'text': 'hi'}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    settings: AnthropicModelSettings = {'anthropic_previous_message_id': 'auto'}
    await _agent(model, FakeComputer()).run('Go.', model_settings=settings)

    _, second = get_mock_chat_completion_kwargs(client)
    assert second['extra_body']['thread'] == {'type': 'continue', 'previous_message_id': '123'}
    (results,) = [m['content'] for m in second['messages']]
    assert [(r['tool_use_id'], r['toolset_name']) for r in results] == [
        ('toolu_1', 'computer'),
        ('toolu_2', 'computer'),
    ]


@pytest.mark.parametrize('change', ['actions', 'name'])
async def test_a_call_a_hook_changed_goes_back_as_a_plain_call(allow_model_requests: None, change: str) -> None:
    """If it were sent back as Claude's original calls, Claude would read that its click ran, but something else ran."""
    hooks = Hooks[object]()

    @hooks.on.after_model_request
    async def look_instead(
        ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        for part in response.parts:
            if isinstance(part, ComputerCallPart) and change == 'actions':
                part.args = {'actions': [{'type': 'screenshot'}]}
            elif isinstance(part, ComputerCallPart):
                part.tool_name = 'look'
        return response

    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))

    def look(actions: list[dict[str, Any]]) -> str:
        return 'Looked.'

    agent = Agent(model, tools=[look], capabilities=[FakeComputer(), InMemoryProvider().capability(), hooks])
    await agent.run('Click.')

    _, second = get_mock_chat_completion_kwargs(client)
    (call,) = second['messages'][-2]['content']
    expected: tuple[str, dict[str, Any]] = (
        ('computer', {'actions': [{'type': 'screenshot'}]})
        if change == 'actions'
        else ('look', {'actions': [{'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': []}]})
    )
    assert (call['name'], call['input'], 'toolset_name' in call) == (*expected, False)
    (result,) = second['messages'][-1]['content']
    assert (result['tool_use_id'], 'toolset_name' in result) == ('toolu_1', False)


@pytest.mark.parametrize('call', ['request', 'count_tokens'])
async def test_a_forced_computer_drops_the_advisor_and_its_history(allow_model_requests: None, call: str) -> None:
    """Requiring the computer drops the other native tools and their history. Otherwise the API refuses the request."""
    client = MockAnthropic.create_mock(_turn(_tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]})))
    model = AnthropicModel(
        'claude-sonnet-4-6',
        provider=AnthropicProvider(anthropic_client=client),
        profile=AnthropicModelProfile(anthropic_count_tokens_accepts_server_tools=True),
    )
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('What is 2 + 2?'),
        ModelResponse(
            parts=[
                NativeToolCallPart(provider_name='anthropic', tool_name=AdvisorTool.kind, tool_call_id='adv_1'),
                NativeToolReturnPart(
                    provider_name='anthropic',
                    tool_name=AdvisorTool.kind,
                    content={'stop_reason': 'end_turn', 'text': 'It is 4.', 'type': 'advisor_result'},
                    tool_call_id='adv_1',
                ),
                TextPart('4.'),
            ],
            provider_name='anthropic',
        ),
        ModelRequest.user_text_prompt('Now click.'),
    ]
    params = ModelRequestParameters(
        function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL, AdvisorTool(model='claude-opus-4-8')]
    )
    settings: ModelSettings = {'tool_choice': 'required'}
    if call == 'request':
        await model.request(history, settings, params)
    else:
        await model.count_tokens(history, settings, params)

    (sent,) = get_mock_chat_completion_kwargs(client)
    assert [tool['name'] for tool in sent['tools']] == ['computer']
    assert [block['type'] for block in sent['messages'][1]['content']] == ['text']


async def test_a_failed_calls_text_goes_back_on_its_own_result(allow_model_requests: None) -> None:
    async def computer(actions: list[ComputerToolAction]) -> ToolReturn:
        raise ToolFailed('Taking the screenshot failed: scrot: not found')

    client = MockAnthropic.create_mock(
        [_turn(_tool_use('toolu_1', 'screenshot', {})), _turn(BetaTextBlock(text='done', type='text'))]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await Agent(model, tools=[Tool(computer, tool_kind=COMPUTER_KIND)], capabilities=[NativeTool(COMPUTER_TOOL)]).run(
        'Go.'
    )

    _, second = get_mock_chat_completion_kwargs(client)
    (sent,) = second['messages'][-1]['content']
    assert (sent['is_error'], sent['toolset_name'], [block['text'] for block in sent['content']]) == (
        True,
        'computer',
        ['Taking the screenshot failed: scrot: not found'],
    )


async def test_an_invalid_member_is_retried_and_does_not_run(allow_model_requests: None) -> None:
    computer = FakeComputer()
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
                _tool_use('toolu_2', 'left_click', {'coordinate': [-1, 6]}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await _agent(model, computer).run('Go.')

    assert computer.actions == [Click(x=5, y=6)]
    _, second = get_mock_chat_completion_kwargs(client)
    results = second['messages'][-1]['content']
    assert [(r['tool_use_id'], r['is_error']) for r in results] == [('toolu_1', False), ('toolu_2', True)]
    assert 'greater than or equal to 0' in results[1]['content'][0]['text']


async def test_a_member_with_a_field_that_does_not_map_runs_nothing(allow_model_requests: None) -> None:
    # Otherwise the press would run without the held keys, and the keys would still be stored in the history.
    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'left_mouse_down', {'text': 'shift'})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    computer = FakeComputer()
    result = await _agent(model, computer).run('Go.')

    assert computer.actions == []
    call = next(part for part in result.all_messages()[1].parts if isinstance(part, ToolCallPart))
    # The action keeps its type, so the retry message names the field instead of refusing the whole action.
    assert call.args_as_dict() == {'actions': [{'type': 'mouse_down', 'unsupported_fields': ['text']}]}
    (retry,) = [part for part in result.all_messages()[2].parts if isinstance(part, ToolReturnPart)]
    assert retry.outcome == 'retried'
    assert '"loc":["actions",0,"mouse_down","unsupported_fields"]' in retry.model_response_str()


async def test_a_legacy_call_whose_action_is_not_a_string_runs_nothing_and_is_retried(
    allow_model_requests: None,
) -> None:
    def call(input: dict[str, Any]) -> BetaToolUseBlock:
        return BetaToolUseBlock.model_validate(
            {'id': 'toolu_1', 'type': 'tool_use', 'name': 'computer', 'input': input}
        )

    client = MockAnthropic.create_mock(
        [_turn(call({'action': [], 'coordinate': [5, 6]})), _turn(BetaTextBlock(text='done', type='text'))]
    )
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    computer = FakeComputer()
    result = await _agent(model, computer).run('Go.')

    assert computer.actions == []
    call_part = next(part for part in result.all_messages()[1].parts if isinstance(part, ToolCallPart))
    assert call_part.args_as_dict() == {'actions': [{'type': '', 'unsupported_fields': ['action', 'coordinate']}]}
    (retry,) = [part for part in result.all_messages()[2].parts if isinstance(part, ToolReturnPart)]
    assert retry.outcome == 'retried'


async def test_a_deferred_computer_tool_stays_a_deferred_function(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    computer = ToolDefinition(
        name='computer', parameters_json_schema={'type': 'object'}, tool_kind=COMPUTER_KIND.id, defer_loading=True
    )
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(
        function_tools=[computer, note], native_tools=[COMPUTER_TOOL], allow_text_output=True
    )
    await model.request([ModelRequest.user_text_prompt('Look.')], None, params)

    (request,) = get_mock_chat_completion_kwargs(client)
    assert [(tool.get('type'), tool.get('name'), tool.get('defer_loading')) for tool in request['tools']] == [
        (None, 'computer', True),
        (None, 'note', None),
    ]


@pytest.mark.parametrize(
    'model_name, native, profile',
    [
        pytest.param('claude-sonnet-4-5', True, None, id='a-model-without-a-computer-use-tool'),
        pytest.param(
            'claude-sonnet-5-5',
            True,
            AnthropicModelProfile(supported_native_tools=frozenset()),
            id='a-profile-whose-native-tools-leave-it-out',
        ),
        # This is what `ComputerUse(native=False)` sets, so Claude gets the `computer` function tool's own actions.
        pytest.param('claude-sonnet-5-5', False, None, id='a-tool-that-opts-out'),
    ],
)
async def test_the_computer_tool_stays_a_function_on_a_model_without_one_or_when_it_opts_out(
    allow_model_requests: None, model_name: str, native: bool, profile: ModelProfile | None
) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client), profile=profile)
    capabilities = [FakeComputer(native=native), InMemoryProvider().capability()]
    await Agent(model, capabilities=capabilities).run('Go.')

    (request,) = get_mock_chat_completion_kwargs(client)
    assert [(tool['name'], 'input_schema' in tool) for tool in request['tools']] == [('computer', True)]


@pytest.mark.parametrize(
    'profile, sent',
    [
        pytest.param(
            AnthropicModelProfile(
                anthropic_computer_tool_version='computer_20251124',
                supported_native_tools=frozenset({ComputerTool}),
            ),
            'computer_20251124',
            id='with-a-version',
        ),
        pytest.param(
            AnthropicModelProfile(supported_native_tools=frozenset({ComputerTool})),
            'computer_20250124',
            id='with-the-default-version',
        ),
    ],
)
async def test_a_profile_that_lists_it_gets_the_computer_use_tool(
    allow_model_requests: None, profile: AnthropicModelProfile, sent: str
) -> None:
    """The native tools set through `profile=` decide what is sent, even on a model Anthropic's table doesn't list."""
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client), profile=profile)
    await Agent(model, capabilities=[FakeComputer(), InMemoryProvider().capability()]).run('Go.')

    (request,) = get_mock_chat_completion_kwargs(client)
    assert [tool['type'] for tool in request['tools']] == [sent]


def test_the_computer_use_tool_is_sent_by_default_only_on_the_clients_it_is_recorded_on() -> None:
    from anthropic import (
        AsyncAnthropicAWS,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    supported: dict[str, bool] = {}
    for name, client in (
        ('direct', None),
        ('bedrock', AsyncAnthropicBedrock(aws_access_key='x', aws_secret_key='y', aws_region='us-east-1')),
        ('mantle', AsyncAnthropicBedrockMantle(aws_region='us-east-1', aws_access_key='x', aws_secret_key='y')),
        ('vertex', AsyncAnthropicVertex(project_id='p', region='us-central1', access_token='x')),
        ('foundry', AsyncAnthropicFoundry(api_key='x', base_url='https://example.com')),
        ('aws', AsyncAnthropicAWS(api_key='x', aws_region='us-east-1', workspace_id='w')),
    ):
        provider = AnthropicProvider(api_key='x') if client is None else AnthropicProvider(anthropic_client=client)
        model = AnthropicModel('claude-opus-5-5', provider=provider)
        supported[name] = ComputerTool in model.profile.get('supported_native_tools', ())
    assert supported == {
        'direct': True,
        'bedrock': True,
        'mantle': True,
        'vertex': False,
        'foundry': False,
        'aws': False,
    }
    # Leaving the tool out is only a default. A user who has checked the tool on one of these clients can turn it on.
    vertex = AsyncAnthropicVertex(project_id='p', region='us-central1', access_token='x')
    opted_in = AnthropicModel(
        'claude-opus-5-5',
        provider=AnthropicProvider(anthropic_client=vertex),
        profile={'supported_native_tools': frozenset({ComputerTool})},
    )
    assert ComputerTool in opted_in.profile.get('supported_native_tools', ())


async def test_a_required_computer_tool_on_a_model_without_one_is_refused(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(
        model,
        capabilities=[
            FakeComputer(native=False),
            NativeTool(ComputerTool(display_size=FAKE_DISPLAY_SIZE)),
            InMemoryProvider().capability(),
        ],
    )
    with pytest.raises(UserError, match=r"Native tool\(s\) \['ComputerTool'\] not supported by this model"):
        await agent.run('Go.')


async def test_computer_calls_from_another_provider_replay_as_a_plain_tool_call(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    history = [
        ModelRequest(parts=[UserPromptPart('Look.')]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    'computer',
                    {'actions': [{'type': 'screenshot'}]},
                    tool_call_id='call_1',
                    provider_name='openai',
                    # Details shaped like Anthropic's, so only the provider name keeps the call a plain call.
                    provider_details={
                        'computer_block': {'id': 'call_1', 'name': 'screenshot', 'input': {}},
                        'computer_tool_version': 'computer_toolset_20260801',
                    },
                )
            ],
            provider_name='openai',
        ),
        ModelRequest(parts=[ToolReturnPart('computer', ['Ran 1 action.'], tool_call_id='call_1')]),
    ]
    await _agent(model, FakeComputer()).run('Go on.', message_history=history)

    (request,) = get_mock_chat_completion_kwargs(client)
    tool_use = request['messages'][1]['content'][0]
    assert (tool_use['name'], 'toolset_name' in tool_use) == ('computer', False)


@pytest.mark.parametrize(
    'name, input, action',
    [
        (
            'left_click',
            {'coordinate': [1, 2]},
            {'type': 'click', 'x': 1, 'y': 2, 'button': 'left', 'count': 1, 'modifiers': []},
        ),
        ('right_click', {}, {'type': 'click', 'button': 'right', 'count': 1, 'modifiers': []}),
        (
            'triple_click',
            {'coordinate': [1, 2], 'text': 'ctrl+shift'},
            {'type': 'click', 'x': 1, 'y': 2, 'button': 'left', 'count': 3, 'modifiers': ['ctrl', 'shift']},
        ),
        ('middle_click', {}, {'type': 'click', 'button': 'middle', 'count': 1, 'modifiers': []}),
        ('double_click', {}, {'type': 'click', 'button': 'left', 'count': 2, 'modifiers': []}),
        (
            'mouse_move',
            {'coordinate': [3, 4]},
            {'type': 'mouse_move', 'x': 3, 'y': 4},
        ),
        (
            'left_click_drag',
            {'start_coordinate': [1, 2], 'coordinate': [3, 4]},
            {'type': 'drag', 'path': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}], 'modifiers': []},
        ),
        ('left_click_drag', {'coordinate': [3, 4]}, {'type': 'drag', 'unsupported_fields': ['start_coordinate']}),
        # A required field that is missing, or a field with the wrong type, is refused, not filled in.
        ('left_click', {'coordinate': [5]}, {'type': 'click', 'unsupported_fields': ['coordinate']}),
        ('left_click', {'coordinate': ['1', '2']}, {'type': 'click', 'unsupported_fields': ['coordinate']}),
        ('triple_click', {'text': 7}, {'type': 'click', 'unsupported_fields': ['text']}),
        ('key', {'text': 5}, {'type': 'press_keys', 'unsupported_fields': ['text']}),
        ('type', {}, {'type': 'type_text', 'unsupported_fields': ['text']}),
        ('mouse_move', {}, {'type': 'mouse_move', 'unsupported_fields': ['coordinate']}),
        ('wait', {}, {'type': 'wait', 'unsupported_fields': ['duration']}),
        ('left_mouse_down', {}, {'type': 'mouse_down'}),
        ('left_mouse_up', {}, {'type': 'mouse_up'}),
        # Claude's earlier tools press or release the button at a point, so the pointer moves there first.
        ('left_mouse_down', {'coordinate': [5, 6]}, [{'type': 'mouse_move', 'x': 5, 'y': 6}, {'type': 'mouse_down'}]),
        ('left_mouse_up', {'coordinate': [7, 8]}, [{'type': 'mouse_move', 'x': 7, 'y': 8}, {'type': 'mouse_up'}]),
        (
            'scroll',
            {'scroll_direction': 'down', 'scroll_amount': 80, 'coordinate': [1, 2]},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'down', 'amount': 80, 'modifiers': []},
        ),
        ('type', {'text': 'hi'}, {'type': 'type_text', 'text': 'hi'}),
        ('key', {'text': 'ctrl+s', 'repeat': 2}, {'type': 'press_keys', 'keys': ['ctrl', 's'], 'repeat': 2}),
        # xdotool syntax: key combinations separated by spaces are pressed one after another.
        (
            'key',
            {'text': 'ctrl+a BackSpace'},
            [
                {'type': 'press_keys', 'keys': ['ctrl', 'a'], 'repeat': None},
                {'type': 'press_keys', 'keys': ['backspace'], 'repeat': None},
            ],
        ),
        # X11 key names, as Claude sends them, in the tool's spelling. A `+` at the end of a combination is the plus
        # key. A key the tool doesn't have, such as volume up, passes through for the tool to refuse.
        (
            'key',
            {'text': 'super+space Page_Down KP_Add ctrl+plus ctrl++ XF86AudioRaiseVolume'},
            [
                {'type': 'press_keys', 'keys': ['super', 'space'], 'repeat': None},
                {'type': 'press_keys', 'keys': ['pagedown'], 'repeat': None},
                {'type': 'press_keys', 'keys': ['+'], 'repeat': None},
                {'type': 'press_keys', 'keys': ['ctrl', '+'], 'repeat': None},
                {'type': 'press_keys', 'keys': ['ctrl', '+'], 'repeat': None},
                {'type': 'press_keys', 'keys': ['xf86audioraisevolume'], 'repeat': None},
            ],
        ),
        ('hold_key', {'text': 'shift', 'duration': 30}, {'type': 'hold_keys', 'keys': ['shift'], 'seconds': 30}),
        ('scroll', {'scroll_direction': 'up'}, {'type': 'scroll', 'unsupported_fields': ['scroll_amount']}),
        ('wait', {'duration': 2}, {'type': 'wait', 'seconds': 2}),
        ('screenshot', {}, {'type': 'screenshot'}),
        (
            'screenshot',
            {'region': [0, 0, 1280, 200]},
            {'type': 'zoom', 'region': {'left': 0, 'top': 0, 'right': 1280, 'bottom': 200}},
        ),
        ('screenshot', {'region': [0, 0]}, {'type': 'screenshot', 'unsupported_fields': ['region']}),
        (
            'zoom',
            {'region': [1, 2, 30, 40]},
            {'type': 'zoom', 'region': {'left': 1, 'top': 2, 'right': 30, 'bottom': 40}},
        ),
        ('zoom', {}, {'type': 'zoom', 'unsupported_fields': ['region']}),
        ('cursor_position', {}, {'type': 'cursor_position'}),
        ('teleport', {'coordinate': [1, 2]}, {'type': 'teleport', 'unsupported_fields': ['coordinate']}),
        ('type', {'text': 'hi', 'delay': 5}, {'type': 'type_text', 'unsupported_fields': ['delay']}),
    ],
)
async def test_toolset_members_map_onto_the_computer_tool(
    allow_model_requests: None, name: str, input: dict[str, Any], action: dict[str, Any] | list[dict[str, Any]]
) -> None:
    """Every action Anthropic documents, including ones no recording has. Recordings cover only what Claude sent."""
    expected = action if isinstance(action, list) else [action]
    # The call's arguments leave out any field the conversion sets to `None`.
    assert await _mapped(name, input) == [{k: v for k, v in a.items() if v is not None} for a in expected]


@pytest.mark.parametrize('model_name', ['claude-haiku-5-5', 'claude-opus-4-8'])
def test_the_newest_models_get_the_computer_toolset(model_name: str) -> None:
    """`tests/profiles/test_resolution_matrix.py` checks the earlier tools and the models that get none."""
    profile = AnthropicProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('anthropic_computer_tool_version') == 'computer_toolset_20260801'


def _stream(*blocks: BetaContentBlock) -> list[BetaRawMessageStreamEvent]:
    """The events Anthropic streams for a message made of these blocks."""
    events: list[BetaRawMessageStreamEvent] = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_1',
                model='claude-sonnet-5-5',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=1, output_tokens=0),
            ),
        )
    ]
    for index, block in enumerate(blocks):
        if isinstance(block, BetaToolUseBlock):
            start = block.model_copy(update={'input': {}})
            events.append(BetaRawContentBlockStartEvent(type='content_block_start', index=index, content_block=start))
            events.append(
                BetaRawContentBlockDeltaEvent(
                    type='content_block_delta',
                    index=index,
                    delta=BetaInputJSONDelta(type='input_json_delta', partial_json=json.dumps(block.input)),
                )
            )
        else:
            events.append(BetaRawContentBlockStartEvent(type='content_block_start', index=index, content_block=block))
        events.append(BetaRawContentBlockStopEvent(type='content_block_stop', index=index))
    events.append(
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='tool_use'),
            usage=BetaMessageDeltaUsage(output_tokens=1),
        )
    )
    events.append(BetaRawMessageStopEvent(type='message_stop'))
    return events


COMPUTER_TOOL = ComputerTool(display_size=FAKE_DISPLAY_SIZE, optional=True)

COMPUTER = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'}, tool_kind=COMPUTER_KIND.id)


async def _streamed(events: list[BetaRawMessageStreamEvent], *function_tools: ToolDefinition) -> ModelResponse:
    model = AnthropicModel(
        'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_stream_mock(events))
    )
    params = ModelRequestParameters(
        function_tools=[COMPUTER, *function_tools], native_tools=[COMPUTER_TOOL], allow_text_output=True
    )
    async with model.request_stream([ModelRequest.user_text_prompt('Go.')], None, params) as stream:
        async for _ in stream:
            pass
    return stream.get()


async def test_a_streamed_member_call_whose_input_came_with_its_start_keeps_it(allow_model_requests: None) -> None:
    block = _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]})
    events = _stream(block)
    # No input deltas: the whole input comes with the block's start event.
    events[1] = BetaRawContentBlockStartEvent(type='content_block_start', index=0, content_block=block)
    del events[2]
    (call,) = (await _streamed(events)).parts
    assert isinstance(call, ComputerCallPart)
    assert call.args_as_dict() == {
        'actions': [{'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': []}]
    }


@pytest.mark.parametrize('partial_json', ['{"coordinate": [5', '[]'], ids=['cut-off', 'not-an-object'])
async def test_a_streamed_member_call_with_unusable_input_keeps_it_raw(
    allow_model_requests: None, partial_json: str
) -> None:
    events = _stream(_tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}))
    events[2] = BetaRawContentBlockDeltaEvent(
        type='content_block_delta',
        index=0,
        delta=BetaInputJSONDelta(type='input_json_delta', partial_json=partial_json),
    )
    (call,) = (await _streamed(events)).parts
    # Kept as a string, so the agent treats it like any other incomplete or invalid tool call.
    assert isinstance(call, ComputerCallPart)
    assert (call.tool_name, call.args, call.tool_call_id, call.provider_details) == (
        'computer',
        partial_json,
        'toolu_1',
        None,
    )


async def test_member_calls_keep_their_order_streamed_or_not(allow_model_requests: None) -> None:
    blocks = [
        _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
        BetaTextBlock(text='Now the field.', type='text'),
        _tool_use('toolu_2', 'type', {'text': 'hi'}),
    ]
    params = ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True)
    prompt: list[ModelMessage] = [ModelRequest.user_text_prompt('Go.')]

    whole_client = MockAnthropic.create_mock(_turn(*blocks))
    whole = await AnthropicModel(
        'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=whole_client)
    ).request(prompt, None, params)

    stream_client = MockAnthropic.create_stream_mock(_stream(*blocks))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=stream_client))
    async with model.request_stream(prompt, None, params) as stream:
        events = [event async for event in stream]
    streamed = stream.get()

    def parts(response: ModelResponse) -> list[tuple[type, Any]]:
        return [
            (type(part), part.args_as_dict() if isinstance(part, ToolCallPart) else None) for part in response.parts
        ]

    assert (
        parts(streamed)
        == parts(whole)
        == [
            (
                ComputerCallPart,
                {'actions': [{'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': []}]},
            ),
            (TextPart, None),
            (ComputerCallPart, {'actions': [{'type': 'type_text', 'text': 'hi'}]}),
        ]
    )
    starts = [(event.index, type(event.part)) for event in events if isinstance(event, PartStartEvent)]
    assert starts == [(0, ComputerCallPart), (1, TextPart), (2, ComputerCallPart)]


async def test_each_computer_20251124_call_is_its_own_call_streamed_or_not(allow_model_requests: None) -> None:
    def call(id: str, input: dict[str, Any]) -> BetaToolUseBlock:
        return BetaToolUseBlock.model_validate({'id': id, 'type': 'tool_use', 'name': 'computer', 'input': input})

    blocks = [
        call('toolu_1', {'action': 'left_click', 'coordinate': [5, 6]}),
        call('toolu_2', {'action': 'screenshot'}),
    ]
    params = ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True)
    prompt: list[ModelMessage] = [ModelRequest.user_text_prompt('Go.')]

    whole_client = MockAnthropic.create_mock(_turn(*blocks))
    whole = await AnthropicModel(
        'claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=whole_client)
    ).request(prompt, None, params)
    stream_client = MockAnthropic.create_stream_mock(_stream(*blocks))
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=stream_client))
    async with model.request_stream(prompt, None, params) as stream:
        async for _ in stream:
            pass

    def calls(response: ModelResponse) -> list[tuple[str, Any]]:
        return [(part.tool_call_id, part.args_as_dict()) for part in response.parts if isinstance(part, ToolCallPart)]

    assert (
        calls(stream.get())
        == calls(whole)
        == [
            (
                'toolu_1',
                {'actions': [{'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': []}]},
            ),
            ('toolu_2', {'actions': [{'type': 'screenshot'}]}),
        ]
    )


async def test_a_call_named_computer_is_a_function_tools_when_tool_choice_leaves_out_the_computer(
    allow_model_requests: None,
) -> None:
    client = MockAnthropic.create_mock(
        _turn(
            BetaToolUseBlock(
                id='toolu_1', name='computer', input={'action': 'left_click', 'coordinate': [5, 6]}, type='tool_use'
            )
        )
    )
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    function = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'})
    desktop = ToolDefinition(name='desktop', parameters_json_schema={'type': 'object'}, tool_kind=COMPUTER_KIND.id)
    params = ModelRequestParameters(
        function_tools=[function, desktop], native_tools=[COMPUTER_TOOL], allow_text_output=True
    )

    response = await model.request(
        [ModelRequest.user_text_prompt('Go.')], {'tool_choice': ToolOrOutput(function_tools=['computer'])}, params
    )

    (sent,) = get_mock_chat_completion_kwargs(client)
    assert [tool['name'] for tool in sent['tools']] == ['computer']
    assert isinstance(sent['betas'], Omit)  # nor the computer use beta
    # The computer use tool wasn't sent, so the call is a function tool call, not a `desktop` computer call.
    (call,) = response.parts
    assert isinstance(call, ToolCallPart) and not isinstance(call, ComputerCallPart)
    assert (call.tool_name, call.args) == ('computer', {'action': 'left_click', 'coordinate': [5, 6]})


async def test_a_tool_added_mid_run_is_declared_when_tool_choice_leaves_out_the_computer(
    allow_model_requests: None,
) -> None:
    toolset = FunctionToolset[None]()

    def new_tool() -> str:
        return 'new tool ran'  # pragma: no cover

    @toolset.tool_plain
    def add_new_tool() -> str:
        toolset.add_function(new_tool)
        return 'new_tool is now available'

    def settings(ctx: RunContext[None]) -> ModelSettings:
        return {'tool_choice': ToolOrOutput(['new_tool'])} if 'new_tool' in toolset.tools else {}

    client = MockAnthropic.create_mock(
        [
            _turn(BetaToolUseBlock(id='toolu_1', name='add_new_tool', input={}, type='tool_use')),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(
        model, deps_type=type(None), toolsets=[toolset], capabilities=[FakeComputer()], model_settings=settings
    )

    # `tool_choice` leaves out the computer use tool, so the newly added tool is the request's only tool. It can't
    # stay deferred, because Anthropic rejects a request whose tools are all deferred.
    assert (await agent.run('Add the new tool.')).output == 'done'
    _, sent = get_mock_chat_completion_kwargs(client)
    assert [(tool['name'], tool.get('defer_loading', False)) for tool in sent['tools']] == [('new_tool', False)]


async def test_a_stored_call_whose_actions_do_not_validate_still_replays_as_its_member_calls(
    allow_model_requests: None,
) -> None:
    """A click off the screen is sent back for a retry. Once stored, its call is a plain `ToolCallPart`."""
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    block = {'id': 'toolu_1', 'name': 'left_click', 'input': {'coordinate': [-1, 6]}}
    call = ToolCallPart(
        'computer',
        {'actions': [{'type': 'click', 'x': -1, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': []}]},
        tool_call_id='toolu_1',
        tool_kind=COMPUTER_KIND.id,
        provider_name='anthropic',
        provider_details={'computer_block': block, 'computer_tool_version': 'computer_toolset_20260801'},
    )
    history = ModelMessagesTypeAdapter.validate_json(
        ModelMessagesTypeAdapter.dump_json(
            [
                ModelRequest.user_text_prompt('Click.'),
                ModelResponse(parts=[call], provider_name='anthropic'),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            'computer',
                            'Positions are at least 0.',
                            tool_call_id='toolu_1',
                            tool_kind=COMPUTER_KIND.id,
                            outcome='retried',
                        )
                    ]
                ),
            ]
        )
    )
    assert not isinstance(history[1].parts[0], ComputerCallPart)
    await model.request(
        history,
        None,
        ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True),
    )

    (body,) = get_mock_chat_completion_kwargs(client)
    (tool_use,) = body['messages'][1]['content']
    assert (tool_use['name'], tool_use.get('toolset_name')) == ('left_click', 'computer')
    (result,) = body['messages'][2]['content']
    assert (result['tool_use_id'], result['is_error']) == ('toolu_1', True)


async def test_without_the_toolset_past_member_calls_replay_as_plain_calls(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=client))
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    other = ToolDefinition(name='other', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(
        function_tools=[COMPUTER, note, other], native_tools=[COMPUTER_TOOL], allow_text_output=True
    )
    block: dict[str, Any] = {'id': 'toolu_1', 'name': 'screenshot', 'input': {}}
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Look.'),
        ModelResponse(
            parts=[
                ComputerCallPart(
                    'computer',
                    {'actions': [{'type': 'screenshot'}]},
                    tool_call_id='toolu_1',
                    provider_name='anthropic',
                    provider_details={'computer_block': block, 'computer_tool_version': 'computer_toolset_20260801'},
                )
            ],
            provider_name='anthropic',
        ),
        ModelRequest(parts=[ToolReturnPart('computer', 'Ran 1 action.', tool_call_id='toolu_1')]),
    ]

    # `tool_choice` leaves `computer` out, so the toolset isn't sent, and its past calls can't name it.
    await model.request(history, {'tool_choice': ['note', 'other']}, params)

    (body,) = get_mock_chat_completion_kwargs(client)
    assert [tool.get('name') for tool in body['tools']] == ['note', 'other']
    assert 'toolset_name' not in str(body['messages'])
    (call,) = [
        b for m in body['messages'] if isinstance(m['content'], list) for b in m['content'] if b['type'] == 'tool_use'
    ]
    assert (call['name'], call['input']) == ('computer', {'actions': [{'type': 'screenshot'}]})


_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8),
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=3),
    max_leaves=8,
)
_MEMBER_NAMES = [
    *('left_click', 'right_click', 'middle_click', 'double_click', 'triple_click'),
    *('mouse_move', 'left_click_drag', 'left_mouse_down', 'left_mouse_up', 'scroll', 'type', 'key'),
    *('hold_key', 'wait', 'zoom', 'screenshot', 'cursor_position', 'teleport'),
]
_POINT = st.lists(st.integers(0, 4000), min_size=2, max_size=2)
# Inputs close to a real action's input: each documented field has either its documented type or any value.
_NEAR_INPUT = st.fixed_dictionaries(
    {},
    optional={
        'coordinate': _POINT | _JSON,
        'start_coordinate': _POINT | _JSON,
        'text': st.text(max_size=8) | _JSON,
        'scroll_direction': st.sampled_from(['up', 'down', 'left', 'right']) | _JSON,
        'scroll_amount': st.integers(0, 20) | _JSON,
        'repeat': st.integers(0, 5) | _JSON,
        'duration': st.floats(0, 10) | _JSON,
        'region': st.lists(st.integers(0, 4000), min_size=4, max_size=4) | _JSON,
        'extra': _JSON,
    },
)
_TOOL_ACTION: TypeAdapter[ComputerToolAction] = TypeAdapter(ComputerToolAction)


# The fixture only allows model requests, so sharing it across examples is safe.
@settings(max_examples=50, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.sampled_from(_MEMBER_NAMES), _NEAR_INPUT | st.dictionaries(st.text(max_size=8), _JSON, max_size=3))
async def test_any_member_input_maps_without_raising_and_a_refusal_is_rejected(
    allow_model_requests: None, name: str, input: dict[str, Any]
) -> None:
    """The tool checks every action before running any. If it rejects one, none of the call runs."""
    for action in await _mapped(name, input):
        if 'unsupported_fields' in action:
            with pytest.raises(ValidationError):
                _TOOL_ACTION.validate_python(action)


@pytest.mark.parametrize(
    'model_name, build',
    [
        (
            'claude-sonnet-4-6',
            lambda: [
                BetaToolUseBlock(id='toolu_long_1', type='tool_use', name='computer', input={'action': 'screenshot'})
            ],
        ),
        (
            'claude-sonnet-5-5',
            lambda: [_tool_use('toolu_long_1', 'screenshot', {}), _tool_use('toolu_long_2', 'screenshot', {})],
        ),
    ],
    ids=['computer-tool', 'toolset'],
)
async def test_a_replayed_call_and_its_results_pair_up_when_ids_are_fitted(
    allow_model_requests: None, model_name: str, build: Callable[[], list[BetaToolUseBlock]]
) -> None:
    # Built inside the test, so collecting the file doesn't need `anthropic` installed.
    blocks = build()
    client = MockAnthropic.create_mock([_turn(*blocks), _turn(BetaTextBlock(text='done', type='text'))])
    model = AnthropicModel(
        model_name, provider=AnthropicProvider(anthropic_client=client), profile={'tool_call_id_length': 9}
    )
    await Agent(model, capabilities=[FakeComputer()]).run('Look.')
    sent = get_mock_chat_completion_kwargs(client)[1]['messages']
    calls = [block['id'] for message in sent for block in message['content'] if block['type'] == 'tool_use']
    results = [
        block['tool_use_id'] for message in sent for block in message['content'] if block['type'] == 'tool_result'
    ]
    assert calls == results
    assert len(calls[0]) == 9


@pytest.mark.parametrize('outcome', ['denied', 'interrupted'])
async def test_a_denied_or_interrupted_toolset_call_is_not_a_tool_error(
    allow_model_requests: None, outcome: Literal['denied', 'interrupted']
) -> None:
    """A failed call is a tool error: see `test_a_failed_calls_text_goes_back_on_its_own_result`."""
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    call = ComputerCallPart(
        'computer',
        {'actions': [{'type': 'click', 'x': 1, 'y': 2, 'button': 'left', 'count': 1, 'modifiers': []}]},
        tool_call_id='toolu_1',
        provider_name='anthropic',
        provider_details={
            'computer_block': {'id': 'toolu_1', 'name': 'left_click', 'input': {'coordinate': [1, 2]}},
            'computer_tool_version': 'computer_toolset_20260801',
        },
    )
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Click.'),
        ModelResponse(parts=[call], provider_name='anthropic'),
        ModelRequest(
            parts=[
                ToolReturnPart('computer', 'No.', tool_call_id='toolu_1', tool_kind=COMPUTER_KIND.id, outcome=outcome)
            ]
        ),
    ]
    await model.request(
        history,
        None,
        ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True),
    )
    results = get_mock_chat_completion_kwargs(client)[0]['messages'][-1]['content']
    assert [(result['tool_use_id'], result['is_error']) for result in results] == [('toolu_1', False)]


@pytest.mark.parametrize(
    'provider_details',
    [
        pytest.param({'computer_block': {'id': 'toolu_1'}}, id='details-that-do-not-validate'),
        pytest.param(
            {
                'computer_block': {'id': 'toolu_1', 'name': 'computer', 'input': {'action': 'screenshot'}},
                'computer_tool_version': 'computer_20251124',
            },
            id='another-tool-version',
        ),
    ],
)
async def test_a_stored_call_claude_cannot_have_made_goes_back_plain(
    allow_model_requests: None, provider_details: dict[str, Any]
) -> None:
    """Details changed in storage, or a call from a model with another computer use tool, would misstate Claude's call."""
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    call = ComputerCallPart(
        'computer',
        {'actions': [{'type': 'screenshot'}]},
        tool_call_id='toolu_1',
        provider_name='anthropic',
        provider_details=provider_details,
    )
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Look.'),
        ModelResponse(parts=[call], provider_name='anthropic'),
        ModelRequest(parts=[ToolReturnPart('computer', 'Ran 1 action.', tool_call_id='toolu_1')]),
    ]
    await model.request(
        history,
        None,
        ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True),
    )

    (body,) = get_mock_chat_completion_kwargs(client)
    (tool_use,) = body['messages'][1]['content']
    assert (tool_use['name'], tool_use['input'], 'toolset_name' in tool_use) == (
        'computer',
        {'actions': [{'type': 'screenshot'}]},
        False,
    )
    (result,) = body['messages'][2]['content']
    assert 'toolset_name' not in result


async def test_another_tool_forced_beside_the_computer_is_forced_by_name(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(
        function_tools=[COMPUTER, note], native_tools=[COMPUTER_TOOL], allow_text_output=True
    )
    await model.request([ModelRequest.user_text_prompt('Note it.')], {'tool_choice': ['note']}, params)

    (body,) = get_mock_chat_completion_kwargs(client)
    assert body['tool_choice'] == {'type': 'tool', 'name': 'note'}
