"""Anthropic's computer use tools, sent for a tool of the `computer` kind.

Not VCR tests: they pin the request and stream shapes the cassettes' body matching can miss, and cases no live model
produces, such as stored histories and cut-off streams; the recorded runs are in `tests/harness/computer_use/`.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterable, Callable
from typing import Any, Literal

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import Agent, AgentStreamEvent, Tool, capture_run_messages
from pydantic_ai.capabilities import Hooks, NativeTool, RawToolArgs, ValidatedToolArgs
from pydantic_ai.computer_use import COMPUTER_KIND, Click, ComputerCallPart, ComputerToolAction, PressKeys, TypeText
from pydantic_ai.exceptions import IncompleteToolCall, ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import (
    BinaryImage,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import ComputerTool
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from ...computer_fakes import FAKE_DISPLAY_SIZE, FakeComputer
from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic, Omit
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

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.anthropic._computer import map_computer_member
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile, anthropic_model_profile
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    # Another thread touching `random` during a draw trips Hypothesis's check, as `tests/test_history_properties.py`
    # explains, and `filterwarnings = error` makes it a flaky error. No strategy here uses `random`.
    pytest.mark.filterwarnings(
        'ignore:Do not use the `random` module inside strategies:hypothesis.errors.HypothesisDeprecationWarning'
    ),
]

NOT_EXECUTED = 'Not executed: an earlier computer action in this turn failed.'


def _agent(model: AnthropicModel, computer: FakeComputer) -> Agent[None, str]:
    return Agent(model, capabilities=[computer, InMemoryProvider().capability()])


def _tool_use(id: str, name: str, input: dict[str, Any]) -> BetaToolUseBlock:
    return BetaToolUseBlock.model_validate(
        {'id': id, 'type': 'tool_use', 'name': name, 'input': input, 'toolset_name': 'computer'}
    )


def _turn(*blocks: BetaContentBlock) -> BetaMessage:
    return completion_message(list(blocks), BetaUsage(input_tokens=1, output_tokens=1))


async def test_a_batch_is_one_call_and_its_result_is_split_per_member(allow_model_requests: None) -> None:
    computer = FakeComputer()
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6], 'text': 'shift'}),
                # One member, two presses, one result.
                _tool_use('toolu_2', 'key', {'text': 'ctrl+a BackSpace'}),
                _tool_use('toolu_3', 'hold_key', {'text': 'ctrl', 'duration': 1}),
                _tool_use('toolu_4', 'type', {'text': 'never typed'}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    result = await _agent(model, computer).run('Go.')

    (call,) = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)]
    assert call.tool_call_id == 'toolu_1'
    assert call.args_as_dict() == {
        'actions': [
            {'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': ['shift']},
            {'type': 'keypress', 'keys': ['ctrl+a']},
            {'type': 'keypress', 'keys': ['BackSpace']},
            {'type': 'hold_key', 'keys': ['ctrl'], 'seconds': 1},
            {'type': 'type', 'text': 'never typed'},
        ]
    }
    assert computer.actions == [
        Click(x=5, y=6, modifiers=('shift',)),
        PressKeys(keys=('ctrl', 'a')),
        PressKeys(keys=('backspace',)),
    ]

    first, second = get_mock_chat_completion_kwargs(client)
    assert first['tools'] == [{'type': 'computer_toolset_20260801'}]
    # The member calls go back as they came, each answered by one result; both name the toolset.
    calls = second['messages'][-2]['content']
    assert [(c['id'], c['name'], c['toolset_name']) for c in calls] == [
        ('toolu_1', 'left_click', 'computer'),
        ('toolu_2', 'key', 'computer'),
        ('toolu_3', 'hold_key', 'computer'),
        ('toolu_4', 'type', 'computer'),
    ]
    results = second['messages'][-1]['content']
    assert [(r['tool_use_id'], r['is_error'], [block['type'] for block in r['content']]) for r in results] == [
        ('toolu_1', False, ['text']),
        ('toolu_2', False, ['text', 'image']),
        ('toolu_3', True, ['text']),
        ('toolu_4', True, ['text']),
    ]
    assert [r['content'][0]['text'] for r in results] == [
        'OK',
        'OK',
        "the fake display cannot perform 'hold_key'",
        NOT_EXECUTED,
    ]
    assert all(r['toolset_name'] == 'computer' for r in results)


async def test_a_failed_call_gives_every_member_its_text(allow_model_requests: None) -> None:
    async def computer(actions: list[ComputerToolAction]) -> ToolReturn:
        raise ToolFailed('Taking the screenshot failed: scrot: not found')

    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}), _tool_use('toolu_2', 'screenshot', {})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await Agent(model, tools=[Tool(computer, tool_kind=COMPUTER_KIND)], capabilities=[NativeTool(COMPUTER_TOOL)]).run(
        'Go.'
    )

    _, second = get_mock_chat_completion_kwargs(client)
    # The call may have acted before it failed, so no member may claim it didn't: each carries the text.
    results = second['messages'][-1]['content']
    assert [(r['is_error'], [block['text'] for block in r['content']]) for r in results] == [
        (True, ['Taking the screenshot failed: scrot: not found']),
        (True, ['Taking the screenshot failed: scrot: not found']),
    ]


async def test_an_invalid_member_is_refused_and_nothing_runs(allow_model_requests: None) -> None:
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

    assert computer.actions == []
    _, second = get_mock_chat_completion_kwargs(client)
    results = second['messages'][-1]['content']
    assert [(r['tool_use_id'], r['is_error']) for r in results] == [('toolu_1', True), ('toolu_2', True)]
    assert all('greater than or equal to 0' in r['content'][0]['text'] for r in results)


async def test_a_member_with_a_field_that_does_not_map_runs_nothing(allow_model_requests: None) -> None:
    # A press with held keys would otherwise run without them, and the keys would be stored in the history.
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
    # Under the tool's own type, so the retry names the field rather than refusing the member.
    assert call.args_as_dict() == {'actions': [{'type': 'mouse_down', 'unsupported_fields': ['text']}]}
    (retry,) = [part for part in result.all_messages()[2].parts if isinstance(part, ToolReturnPart)]
    assert retry.outcome == 'retried'
    assert '"loc":["actions",0,"mouse_down","unsupported_fields"]' in retry.model_response_str()


async def test_an_earlier_tool_result_stored_as_a_json_string_keeps_its_screenshot(allow_model_requests: None) -> None:
    call = ComputerCallPart(
        'computer',
        {'actions': [{'type': 'screenshot'}]},
        tool_call_id='toolu_1',
        provider_name='anthropic',
        provider_details={
            'computer_blocks': [{'id': 'toolu_1', 'name': 'computer', 'input': {'action': 'screenshot'}}],
            'computer_tool': 'computer_20251124',
            'computer_action_counts': [1],
        },
    )
    content = json.dumps(['Ran 1 action.', {'kind': 'binary', 'data': 'cG5n', 'media_type': 'image/png'}])
    result = ToolReturnPart(
        'computer', content, tool_call_id='toolu_1', metadata={'performed': 1}, tool_kind=COMPUTER_KIND.id
    )
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    await model.request(
        [
            ModelRequest.user_text_prompt('Look.'),
            ModelResponse(parts=[call], provider_name='anthropic'),
            ModelRequest(parts=[result]),
        ],
        None,
        ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True),
    )

    (body,) = get_mock_chat_completion_kwargs(client)
    (sent,) = body['messages'][-1]['content']
    assert [block['type'] for block in sent['content']] == ['text', 'image']


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


def _without_computer_tool(profile: ModelProfile) -> ModelProfile:
    """A model without a computer use tool, whose profile, like one that doesn't list its native tools, lists them all."""
    trimmed = profile.copy()
    trimmed.pop('supported_native_tools', None)
    return merge_profile(trimmed, AnthropicModelProfile(anthropic_computer_tool_version=None))


@pytest.mark.parametrize(
    'native, profile',
    [
        pytest.param(True, _without_computer_tool, id='a-model-without-a-computer-use-tool'),
        pytest.param(True, ModelProfile(supported_native_tools=frozenset()), id='a-profile-without-native-tools'),
        # As `ComputerUse(native=False)` asks, so Claude gets the tool's own actions.
        pytest.param(False, None, id='a-tool-that-opts-out'),
    ],
)
async def test_the_computer_tool_stays_a_function_on_a_model_without_one_or_when_it_opts_out(
    allow_model_requests: None, native: bool, profile: ModelProfile | Callable[[ModelProfile], ModelProfile] | None
) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client), profile=profile)
    capabilities = [FakeComputer(native=native), InMemoryProvider().capability()]
    await Agent(model, capabilities=capabilities).run('Go.')

    (request,) = get_mock_chat_completion_kwargs(client)
    assert [(tool['name'], 'input_schema' in tool) for tool in request['tools']] == [('computer', True)]


def test_the_computer_use_tool_is_sent_only_on_the_clients_it_is_recorded_on() -> None:
    from anthropic import (
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
    ):
        provider = AnthropicProvider(api_key='x') if client is None else AnthropicProvider(anthropic_client=client)
        model = AnthropicModel('claude-opus-5-5', provider=provider)
        supported[name] = ComputerTool in model.profile.get('supported_native_tools', frozenset[Any]())
    assert supported == {'direct': True, 'bedrock': True, 'mantle': True, 'vertex': False, 'foundry': False}


async def test_a_required_computer_tool_on_a_model_without_one_is_refused(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))

    model = AnthropicModel(
        'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client), profile=_without_computer_tool
    )
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
                    # Details shaped like Anthropic's, so only the provider keeps the call plain.
                    provider_details={
                        'computer_blocks': [{'id': 'call_1', 'name': 'screenshot', 'input': {}}],
                        'computer_tool': 'computer_toolset_20260801',
                        'computer_action_counts': [1],
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
            {'coordinate': [1, 2], 'text': 'ctrl'},
            {'type': 'click', 'x': 1, 'y': 2, 'button': 'left', 'count': 3, 'modifiers': ['ctrl']},
        ),
        ('middle_click', {}, {'type': 'click', 'button': 'middle', 'count': 1, 'modifiers': []}),
        ('double_click', {}, {'type': 'click', 'button': 'left', 'count': 2, 'modifiers': []}),
        (
            'mouse_move',
            {'coordinate': [3, 4]},
            {'type': 'move', 'x': 3, 'y': 4},
        ),
        (
            'left_click_drag',
            {'start_coordinate': [1, 2], 'coordinate': [3, 4]},
            {'type': 'drag', 'path': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}], 'modifiers': []},
        ),
        ('left_click_drag', {'coordinate': [3, 4]}, {'type': 'drag', 'unsupported_fields': ['start_coordinate']}),
        # A field that is missing though required, or has the wrong type, is refused rather than filled in.
        ('left_click', {'coordinate': [5]}, {'type': 'click', 'unsupported_fields': ['coordinate']}),
        ('left_click', {'coordinate': ['1', '2']}, {'type': 'click', 'unsupported_fields': ['coordinate']}),
        ('triple_click', {'text': 7}, {'type': 'click', 'unsupported_fields': ['text']}),
        ('key', {'text': 5}, {'type': 'keypress', 'unsupported_fields': ['text']}),
        ('type', {}, {'type': 'type', 'unsupported_fields': ['text']}),
        ('mouse_move', {}, {'type': 'move', 'unsupported_fields': ['coordinate']}),
        ('wait', {}, {'type': 'wait', 'unsupported_fields': ['duration']}),
        ('left_mouse_down', {}, {'type': 'mouse_down'}),
        ('left_mouse_up', {}, {'type': 'mouse_up'}),
        # Claude's earlier tools press or release at a point, so the pointer moves there first.
        ('left_mouse_down', {'coordinate': [5, 6]}, [{'type': 'move', 'x': 5, 'y': 6}, {'type': 'mouse_down'}]),
        ('left_mouse_up', {'coordinate': [7, 8]}, [{'type': 'move', 'x': 7, 'y': 8}, {'type': 'mouse_up'}]),
        (
            'scroll',
            {'scroll_direction': 'down', 'scroll_amount': 80, 'coordinate': [1, 2]},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'down', 'amount': 80, 'modifiers': []},
        ),
        ('type', {'text': 'hi'}, {'type': 'type', 'text': 'hi'}),
        ('key', {'text': 'ctrl+s', 'repeat': 2}, {'type': 'keypress', 'keys': ['ctrl+s'], 'repeat': 2}),
        # xdotool syntax: chords separated by spaces are pressed one after another.
        (
            'key',
            {'text': 'ctrl+a BackSpace'},
            [
                {'type': 'keypress', 'keys': ['ctrl+a'], 'repeat': None},
                {'type': 'keypress', 'keys': ['BackSpace'], 'repeat': None},
            ],
        ),
        ('hold_key', {'text': 'shift', 'duration': 30}, {'type': 'hold_key', 'keys': ['shift'], 'seconds': 30}),
        ('scroll', {'scroll_direction': 'up'}, {'type': 'scroll', 'unsupported_fields': ['scroll_amount']}),
        ('wait', {'duration': 2}, {'type': 'wait', 'seconds': 2}),
        ('screenshot', {}, {'type': 'screenshot'}),
        (
            'zoom',
            {'region': [1, 2, 30, 40]},
            {'type': 'zoom', 'region': {'left': 1, 'top': 2, 'right': 30, 'bottom': 40}},
        ),
        ('zoom', {}, {'type': 'zoom', 'unsupported_fields': ['region']}),
        ('cursor_position', {}, {'type': 'cursor_position'}),
        ('teleport', {'coordinate': [1, 2]}, {'type': 'teleport', 'unsupported_fields': ['coordinate']}),
        ('type', {'text': 'hi', 'delay': 5}, {'type': 'type', 'unsupported_fields': ['delay']}),
    ],
)
def test_toolset_members_map_onto_the_computer_tool(name: str, input: dict[str, Any], action: Any) -> None:
    """Every member Anthropic documents, including ones no recording has: a VCR test would pin only what Claude sent."""
    assert map_computer_member(name, input) == (action if isinstance(action, list) else [action])


@pytest.mark.parametrize('model_name', ['claude-haiku-5-5', 'claude-opus-4-8'])
def test_the_newest_models_get_the_computer_toolset(model_name: str) -> None:
    """The earlier tools, and models without one, are pinned by `tests/profiles/test_resolution_matrix.py`."""
    profile = anthropic_model_profile(model_name)
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


async def test_member_calls_in_one_response_are_one_call_streamed_or_not(allow_model_requests: None) -> None:
    # Anthropic runs a response's member calls as one batch, stopped at the first failure, so text between them
    # doesn't split it.
    blocks = [
        _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
        _tool_use('toolu_2', 'type', {'text': 'hi'}),
        BetaTextBlock(text='Now the other field.', type='text'),
        _tool_use('toolu_3', 'key', {'text': 'Return'}),
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

    def calls(response: ModelResponse) -> list[tuple[str, Any]]:
        return [(part.tool_call_id, part.args_as_dict()) for part in response.parts if isinstance(part, ToolCallPart)]

    assert (
        calls(streamed)
        == calls(whole)
        == [
            (
                'toolu_1',
                {
                    'actions': [
                        {'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': []},
                        {'type': 'type', 'text': 'hi'},
                        {'type': 'keypress', 'keys': ['Return']},
                    ]
                },
            ),
        ]
    )
    # The call stays where the first member was, so it runs before any call that came after it.
    assert (
        [type(part) for part in streamed.parts] == [type(part) for part in whole.parts] == [ComputerCallPart, TextPart]
    )
    starts = [(event.index, type(event.part)) for event in events if isinstance(event, PartStartEvent)]
    # Started before the text that follows it, so events replay in the response's order, and again with the member
    # after the text, at the same index, so a UI sees one call.
    assert starts == [(0, ComputerCallPart), (1, TextPart), (0, ComputerCallPart)]


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


def _cut_off(blocks: list[BetaContentBlock], partial_json: str) -> list[BetaRawMessageStreamEvent]:
    """The events Anthropic streams when `max_tokens` cuts off the last block, a computer call, mid-input.

    As recorded live for `computer_20251124` and `computer_20250124`: the call's input stops part way, no block stop
    follows, and the message stops for `max_tokens`. The toolset sends no member it cut off, so its case is a guard.
    """
    events = _stream(*blocks)
    last = len(blocks) - 1
    (delta,) = [event for event in events if isinstance(event, BetaRawContentBlockDeltaEvent) and event.index == last]
    delta.delta = BetaInputJSONDelta(type='input_json_delta', partial_json=partial_json)
    message_delta = next(event for event in events if isinstance(event, BetaRawMessageDeltaEvent))
    message_delta.delta = Delta(stop_reason='max_tokens')
    return [event for event in events if not (isinstance(event, BetaRawContentBlockStopEvent) and event.index == last)]


@pytest.mark.parametrize(
    'model_name, blocks',
    [
        pytest.param(
            'claude-sonnet-4-6',
            [
                BetaTextBlock(text='Typing it.', type='text'),
                BetaToolUseBlock.model_validate({'id': 'toolu_2', 'type': 'tool_use', 'name': 'computer', 'input': {}}),
            ],
            id='computer_20251124',
        ),
        # The batch's part sits at its first call, so the cut-off call must still come last.
        pytest.param(
            'claude-sonnet-5-5',
            [
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
                _tool_use('toolu_2', 'type', {}),
            ],
            id='toolset-after-a-whole-member',
        ),
    ],
)
async def test_a_streamed_computer_call_cut_off_mid_input_stops_the_run(
    allow_model_requests: None, model_name: str, blocks: list[BetaContentBlock]
) -> None:
    events = _cut_off(blocks, '{"text": "word0 wo')
    model = AnthropicModel(
        model_name, provider=AnthropicProvider(anthropic_client=MockAnthropic.create_stream_mock(events))
    )

    async def drain(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:
            pass

    with capture_run_messages() as messages:
        with pytest.raises(IncompleteToolCall, match=r'Model token limit \(60\) exceeded while generating a tool call'):
            await _agent(model, FakeComputer()).run(
                'Go.', model_settings={'max_tokens': 60}, event_stream_handler=drain
            )
    # The cut-off call keeps its partial input and is the response's last part, which is what the run checks.
    *_, response = [message for message in messages if isinstance(message, ModelResponse)]
    assert response.finish_reason == 'length'
    last = response.parts[-1]
    assert isinstance(last, ComputerCallPart)
    assert (last.tool_call_id, last.args) == ('toolu_2', '{"text": "word0 wo')


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
    # The computer use tool wasn't sent, so the call is the function tool's, not the `desktop` tool's.
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

    # `tool_choice` leaves out the computer use tool, so the newcomer is the request's only tool and can't stay
    # deferred: Anthropic rejects a request whose tools are all deferred.
    assert (await agent.run('Add the new tool.')).output == 'done'
    _, sent = get_mock_chat_completion_kwargs(client)
    assert [(tool['name'], tool.get('defer_loading', False)) for tool in sent['tools']] == [('new_tool', False)]


_CLICK_TYPE: list[dict[str, Any]] = [
    {'id': 'toolu_1', 'name': 'left_click', 'input': {}},
    {'id': 'toolu_2', 'name': 'type', 'input': {'text': 'hi'}},
]
_PNG = BinaryImage(data=b'png', media_type='image/png')
_FAILED = 'Action 2 of 2 (type) failed: no keyboard. Screenshot attached (1280x800).'
_CHANGED = 'Before it ran, this call was changed to: [...]. Ran 1 action.'


@pytest.mark.parametrize(
    'blocks,counts,content,metadata,results',
    [
        # As a capability that drops old screenshots leaves it: the summary, then the note in place of the screenshot.
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            ['Ran 2 actions.', '[Older screenshot removed.]'],
            {'performed': 2},
            [(False, ['OK']), (False, ['Ran 2 actions. [Older screenshot removed.]'])],
            id='screenshot-dropped',
        ),
        # As a history processor that removes old screenshots outright leaves it: the summary still reaches the model.
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            ['Ran 2 actions.'],
            {'performed': 2},
            [(False, ['OK']), (False, ['Ran 2 actions.'])],
            id='screenshot-removed',
        ),
        # A zoom's image is its call's output, not the screen, which was removed.
        pytest.param(
            [{'id': 'toolu_1', 'name': 'zoom', 'input': {'region': [0, 0, 4, 4]}}, _CLICK_TYPE[0]],
            [1, 1],
            ['Ran 2 actions.', _PNG],
            {'performed': 2},
            [(False, ['OK', 'image']), (False, ['Ran 2 actions.'])],
            id='zoom-without-the-screen',
        ),
        # Not these calls' outcome: each call gets the text rather than `OK`, which would say its own call ran.
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            [_CHANGED, _PNG],
            {'performed': 1, 'changed': True},
            [(False, [_CHANGED]), (False, [_CHANGED, 'image'])],
            id='changed-before-it-ran',
        ),
        # As a UI rebuilds it, or a guardrail answers: nothing says which calls ran, so no call is said to have run.
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            [_FAILED, _PNG],
            None,
            [(False, [_FAILED]), (False, [_FAILED, 'image'])],
            id='no-metadata',
        ),
        # Content stored as a JSON string reads as the list it encodes, screenshot and all.
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            json.dumps(['Ran 2 actions.', {'kind': 'binary', 'data': 'cG5n', 'media_type': 'image/png'}]),
            {'performed': 2},
            [(False, ['OK']), (False, ['OK', 'image'])],
            id='json-string-content',
        ),
        # The counts recorded with the call decide which calls ran, even where mapping again would split differently.
        pytest.param(
            [{'id': 'toolu_1', 'name': 'key', 'input': {'text': 'a b'}}, _CLICK_TYPE[1]],
            [1, 1],
            ['Action 2 of 2 (type) failed: no keyboard.', _PNG],
            {'performed': 1, 'error': 'no keyboard'},
            [(False, ['OK', 'image']), (True, ['no keyboard'])],
            id='recorded-counts',
        ),
    ],
)
async def test_a_result_answers_each_member_call(
    allow_model_requests: None,
    blocks: list[dict[str, Any]],
    counts: list[int],
    content: list[Any] | str,
    metadata: dict[str, Any] | None,
    results: list[tuple[bool, list[str]]],
) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    details: dict[str, Any] = {
        'computer_blocks': blocks,
        'computer_tool': 'computer_toolset_20260801',
        'computer_action_counts': counts,
    }
    # One action per count, as the call was mapped; what they are doesn't matter to how the result is split.
    actions = [{'type': 'screenshot'}] * sum(counts)
    call = ComputerCallPart(
        'computer', {'actions': actions}, tool_call_id='toolu_1', provider_name='anthropic', provider_details=details
    )
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Go.'),
        ModelResponse(parts=[call], provider_name='anthropic'),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    'computer', content, tool_call_id='toolu_1', metadata=metadata, tool_kind=COMPUTER_KIND.id
                )
            ]
        ),
    ]
    await model.request(
        history,
        None,
        ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True),
    )

    (body,) = get_mock_chat_completion_kwargs(client)
    sent = body['messages'][-1]['content']
    assert [r['tool_use_id'] for r in sent] == [block['id'] for block in blocks]
    assert [
        (r['is_error'], [b['text'] if b['type'] == 'text' else b['type'] for b in r['content']]) for r in sent
    ] == results


@pytest.mark.parametrize(
    'counts, actions',
    [
        # A call that failed on its first action would read as having run its first member call.
        pytest.param([0, 2], 2, id='a-zero-count'),
        pytest.param([1, 1], 3, id='counts-short-of-the-actions'),
        pytest.param([2, 2], 3, id='counts-past-the-actions'),
    ],
)
async def test_kept_counts_that_do_not_match_the_actions_send_the_call_back_plain(
    allow_model_requests: None, counts: list[int], actions: int
) -> None:
    """Each member call becomes at least one action, and the result counts actions, so other counts came from elsewhere."""
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    blocks = [
        {'id': 'toolu_1', 'name': 'left_click', 'input': {'coordinate': [1, 2]}},
        {'id': 'toolu_2', 'name': 'type', 'input': {'text': 'hi'}},
    ]
    details: dict[str, Any] = {
        'computer_blocks': blocks,
        'computer_tool': 'computer_toolset_20260801',
        'computer_action_counts': counts,
    }
    call = ComputerCallPart(
        'computer',
        {'actions': [{'type': 'screenshot'}] * actions},
        tool_call_id='toolu_1',
        provider_name='anthropic',
        provider_details=details,
    )
    result = ToolReturnPart(
        'computer',
        ['Action 1 of 2 (click) failed: busy.', _PNG],
        tool_call_id='toolu_1',
        metadata={'performed': 0, 'error': 'busy'},
        tool_kind=COMPUTER_KIND.id,
    )
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Go.'),
        ModelResponse(parts=[call], provider_name='anthropic'),
        ModelRequest(parts=[result]),
    ]
    await model.request(
        history,
        None,
        ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True),
    )

    (body,) = get_mock_chat_completion_kwargs(client)
    sent = body['messages'][-1]['content']
    # The call goes back as the `computer` call it is, answered by one plain result: no member call is told `OK`.
    (tool_use,) = body['messages'][1]['content']
    assert (tool_use['id'], tool_use['name'], tool_use['input'], 'toolset_name' in tool_use) == (
        'toolu_1',
        'computer',
        {'actions': [{'type': 'screenshot'}] * actions},
        False,
    )
    assert [
        (r['tool_use_id'], r['is_error'], [b['text'] if b['type'] == 'text' else b['type'] for b in r['content']])
        for r in sent
    ] == [('toolu_1', False, ['Action 1 of 2 (click) failed: busy.', 'image'])]


async def test_a_stored_call_whose_actions_do_not_validate_still_replays_as_its_member_calls(
    allow_model_requests: None,
) -> None:
    """A click off the screen is sent back for a retry; once stored, its call is no longer typed."""
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    block = {'id': 'toolu_1', 'name': 'left_click', 'input': {'coordinate': [-1, 6]}}
    call = ToolCallPart(
        'computer',
        {'actions': [{'type': 'click', 'x': -1, 'y': 6}]},
        tool_call_id='toolu_1',
        tool_kind=COMPUTER_KIND.id,
        provider_name='anthropic',
        provider_details={
            'computer_blocks': [block],
            'computer_tool': 'computer_toolset_20260801',
            'computer_action_counts': [1],
        },
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
                    provider_details={
                        'computer_blocks': [block],
                        'computer_tool': 'computer_toolset_20260801',
                        'computer_action_counts': [1],
                    },
                )
            ],
            provider_name='anthropic',
        ),
        ModelRequest(parts=[ToolReturnPart('computer', 'Ran 1 action.', tool_call_id='toolu_1')]),
    ]

    # Tool choice leaves `computer` out, so the toolset isn't sent and its past calls can't name it.
    await model.request(history, {'tool_choice': ['note', 'other']}, params)

    (body,) = get_mock_chat_completion_kwargs(client)
    assert [tool.get('name') for tool in body['tools']] == ['note', 'other']
    assert 'toolset_name' not in str(body['messages'])
    (call,) = [
        b for m in body['messages'] if isinstance(m['content'], list) for b in m['content'] if b['type'] == 'tool_use'
    ]
    assert (call['name'], call['input']) == ('computer', {'actions': [{'type': 'screenshot'}]})


@pytest.mark.parametrize('stream', [False, True], ids=['whole', 'streamed'])
async def test_a_batch_split_by_another_tool_call_is_refused(allow_model_requests: None, stream: bool) -> None:
    """Typing would otherwise run in the window the click chose, before the tool between them focused another."""
    focus = BetaToolUseBlock.model_validate({'id': 'toolu_f', 'type': 'tool_use', 'name': 'focus', 'input': {}})
    blocks = [
        _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
        focus,
        _tool_use('toolu_2', 'type', {'text': 'hi'}),
    ]
    focus_tool = ToolDefinition(name='focus', parameters_json_schema={'type': 'object'})
    if stream:
        response = await _streamed(_stream(*blocks), focus_tool)
    else:
        params = ModelRequestParameters(
            function_tools=[COMPUTER, focus_tool], native_tools=[COMPUTER_TOOL], allow_text_output=True
        )
        model = AnthropicModel(
            'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(_turn(*blocks)))
        )
        response = await model.request([ModelRequest.user_text_prompt('Go.')], None, params)

    calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
    assert [call.tool_name for call in calls] == ['computer', 'focus']
    assert calls[0].args_as_dict()['actions'][1] == {'type': 'type', 'after_tool_calls': ['focus']}
    assert isinstance(calls[0], ComputerCallPart) and calls[0].actions is None


def _split_batch_client() -> AsyncAnthropic:
    focus = BetaToolUseBlock.model_validate({'id': 'toolu_f', 'type': 'tool_use', 'name': 'focus', 'input': {}})
    return MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
                focus,
                _tool_use('toolu_2', 'type', {'text': 'hi'}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )


def _split_batch_results(client: AsyncAnthropic) -> list[tuple[bool, list[dict[str, Any]]]]:
    _, second = get_mock_chat_completion_kwargs(client)
    results = {r['tool_use_id']: r for r in second['messages'][-1]['content']}
    return [(results[id]['is_error'], results[id]['content']) for id in ('toolu_1', 'toolu_2')]


async def test_claude_is_told_in_a_sentence_why_a_split_batch_did_not_run(allow_model_requests: None) -> None:
    client = _split_batch_client()
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    computer = FakeComputer()
    agent = _agent(model, computer)
    agent.tool_plain(name='focus')(lambda: 'Focused.')
    await agent.run('Go.')

    assert computer.actions == []
    # Fixed text: nothing in it comes from the history, which may come from storage or a UI.
    refusal = (
        'No computer action ran: they run together as one batch, and another tool call came between them. Make the '
        "computer calls before or after the other tools' calls, not between them."
    )
    assert _split_batch_results(client) == [
        (True, [{'type': 'text', 'text': refusal}]),
        (True, [{'type': 'text', 'text': refusal}]),
    ]


async def test_a_split_batch_a_hook_let_run_gets_its_real_retry_text(allow_model_requests: None) -> None:
    hooks = Hooks[None]()

    @hooks.on.before_tool_validate(tools=['computer'])
    async def run_anyway(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs
    ) -> RawToolArgs:
        return {'actions': [{'type': 'click', 'x': 5, 'y': 6}, {'type': 'type', 'text': 'hi'}]}

    @hooks.on.after_tool_execute(tools=['computer'])
    async def retry(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, result: object
    ) -> object:
        raise ModelRetry('Check the window first.')

    client = _split_batch_client()
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    computer = FakeComputer()
    agent = Agent(model, deps_type=type(None), capabilities=[computer, InMemoryProvider().capability(), hooks])
    agent.tool_plain(name='focus')(lambda: 'Focused.')
    await agent.run('Go.')

    assert computer.actions == [Click(x=5, y=6), TypeText(text='hi')]
    # The calls ran, so the refusal would be false: Claude gets the retry's own text.
    retry_text = [{'type': 'text', 'text': 'Check the window first.'}]
    assert _split_batch_results(client) == [(True, retry_text), (True, retry_text)]


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
# Close to a member's input: each documented field either of its documented type or anything.
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


@settings(max_examples=50, derandomize=True, deadline=None)
@given(st.sampled_from(_MEMBER_NAMES), _NEAR_INPUT | st.dictionaries(st.text(max_size=8), _JSON, max_size=3))
def test_any_member_input_maps_without_raising_and_a_refusal_is_rejected(name: str, input: dict[str, Any]) -> None:
    """The tool checks every action before running any, so one it rejects means none of the call runs."""
    for action in map_computer_member(name, input):
        if 'unsupported_fields' in action:
            with pytest.raises(ValidationError):
                _TOOL_ACTION.validate_python(action)


async def test_a_hook_emptying_a_result_answers_every_call(allow_model_requests: None) -> None:
    hooks = Hooks[None]()

    @hooks.on.after_tool_execute(tools=['computer'])
    async def redact(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, result: object
    ) -> list[str]:
        return []  # such as a hook taking the screen out of the history

    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'mouse_move', {'coordinate': [1, 2]}), _tool_use('toolu_2', 'screenshot', {})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    result = await Agent(model, deps_type=type(None), capabilities=[FakeComputer(), hooks]).run('Look.')
    assert result.output == 'done'
    sent = get_mock_chat_completion_kwargs(client)[1]['messages']
    results = [block for message in sent for block in message['content'] if block['type'] == 'tool_result']
    assert [block['tool_use_id'] for block in results] == ['toolu_1', 'toolu_2']


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
    # Built in the test, so collecting the file needs no `anthropic` install.
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
    """A failed call is a tool error: see `test_a_failed_call_gives_every_member_its_text`."""
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    call = ComputerCallPart(
        'computer',
        {'actions': [{'type': 'click', 'x': 1, 'y': 2}, {'type': 'screenshot'}]},
        tool_call_id='toolu_1',
        provider_name='anthropic',
        provider_details={
            'computer_blocks': [
                {'id': 'toolu_1', 'name': 'left_click', 'input': {'coordinate': [1, 2]}},
                {'id': 'toolu_2', 'name': 'screenshot', 'input': {}},
            ],
            'computer_tool': 'computer_toolset_20260801',
            'computer_action_counts': [1, 1],
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
    assert [(result['tool_use_id'], result['is_error']) for result in results] == [
        ('toolu_1', False),
        ('toolu_2', False),
    ]
