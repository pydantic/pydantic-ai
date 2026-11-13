"""Anthropic's computer use tools, sent for a tool of the `computer` kind."""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import Any, Literal

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.computer_use import COMPUTER_KIND, Click, ComputerCallPart, ComputerToolAction, PressKeys
from pydantic_ai.exceptions import ToolFailed, UnexpectedModelBehavior, UserError
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
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
        Omit,
    )
    from anthropic.types.beta import (
        BetaInputJSONDelta,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaTextBlock,
        BetaToolUseBlock,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.anthropic._computer import map_computer_member
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile, anthropic_model_profile
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')

NOT_EXECUTED = 'Not executed: an earlier computer action in this turn failed.'


def _agent(model: AnthropicModel, computer: FakeComputer) -> Agent[None, str]:
    return Agent(model, capabilities=[computer, InMemoryProvider().capability()])


def _tool_use(id: str, name: str, input: dict[str, Any]) -> BetaToolUseBlock:
    return BetaToolUseBlock.model_validate(
        {'id': id, 'type': 'tool_use', 'name': name, 'input': input, 'toolset_name': 'computer'}
    )


def _turn(*blocks: Any) -> Any:
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


async def test_zoom_and_cursor_position_answer_their_own_member_calls(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'zoom', {'region': [0, 0, 40, 20]}),
                _tool_use('toolu_2', 'cursor_position', {}),
                _tool_use('toolu_3', 'left_click', {'coordinate': [5, 6]}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await _agent(model, FakeComputer()).run('Go.')

    _, second = get_mock_chat_completion_kwargs(client)
    results = second['messages'][-1]['content']
    assert [[block['type'] for block in r['content']] for r in results] == [
        ['text', 'image'],
        ['text', 'text'],
        ['text', 'image'],
    ]
    assert results[1]['content'][1]['text'] == 'The pointer is at (640, 400).'


@pytest.mark.parametrize(
    'model_name, tool, beta, zoom',
    [
        ('claude-sonnet-4-6', 'computer_20251124', 'computer-use-2025-11-24', {'enable_zoom': True}),
        ('claude-haiku-4-5', 'computer_20250124', 'computer-use-2025-01-24', {}),
    ],
)
async def test_an_earlier_model_gets_its_tool_with_the_display_size(
    allow_model_requests: None, model_name: str, tool: str, beta: str, zoom: dict[str, Any]
) -> None:
    provider, computer = InMemoryProvider(), FakeComputer(display_size=(1280, 800))
    computer_call = BetaToolUseBlock.model_validate(
        {
            'id': 'toolu_1',
            'type': 'tool_use',
            'name': 'computer',
            # The earlier tools name a click's modifier keys `key`, not `text`.
            'input': {'action': 'left_click', 'coordinate': [5, 6], 'key': 'shift'},
        }
    )
    client = MockAnthropic.create_mock([_turn(computer_call), _turn(BetaTextBlock(text='done', type='text'))])
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(model, capabilities=[computer, provider.capability()])
    await agent.run('Go.')

    assert computer.actions == [Click(x=5, y=6, modifiers=('shift',))]
    first, second = get_mock_chat_completion_kwargs(client)
    assert first['tools'] == [
        {'type': tool, 'name': 'computer', 'display_width_px': 1280, 'display_height_px': 800, **zoom}
    ]
    assert beta in first['betas']
    # The call goes back as it came; its result is an ordinary tool result.
    (call,) = second['messages'][-2]['content']
    assert (call['name'], call['input'], 'toolset_name' in call) == (
        'computer',
        {'action': 'left_click', 'coordinate': [5, 6], 'key': 'shift'},
        False,
    )
    (result,) = second['messages'][-1]['content']
    assert [block['type'] for block in result['content']] == ['text', 'image']


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


async def test_a_screenshot_before_other_calls_is_refused(allow_model_requests: None) -> None:
    """The screen is captured once, after the batch, so an earlier screenshot call would get no image of its own."""
    computer = FakeComputer()
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'screenshot', {}),
                _tool_use('toolu_2', 'left_click', {'coordinate': [5, 6]}),
                _tool_use('toolu_3', 'screenshot', {}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await _agent(model, computer).run('Go.')

    assert computer.actions == []
    _, second = get_mock_chat_completion_kwargs(client)
    results = second['messages'][-1]['content']
    assert [r['is_error'] for r in results] == [True, True, True]
    assert all('not_the_last_call' in r['content'][0]['text'] for r in results)


async def test_a_member_with_a_field_that_does_not_map_runs_nothing(allow_model_requests: None) -> None:
    # A press at a point would otherwise run at the pointer, and the point would be stored in the history.
    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'left_mouse_down', {'coordinate': [5, 6]})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    computer = FakeComputer()
    result = await _agent(model, computer).run('Go.')

    assert computer.actions == []
    call = next(part for part in result.all_messages()[1].parts if isinstance(part, ToolCallPart))
    # Under the tool's own type, so the retry names the field rather than refusing the member.
    assert call.args_as_dict() == {'actions': [{'type': 'mouse_down', 'unsupported_fields': ['coordinate']}]}
    (retry,) = [part for part in result.all_messages()[2].parts if isinstance(part, ToolReturnPart)]
    assert retry.outcome == 'retried'
    assert '"loc":["actions",0,"mouse_down","unsupported_fields"]' in retry.model_response_str()


_CLICK_TEXT_REFUSED = (
    '"loc":["actions",0,"click","unsupported_fields"],"msg":"Unexpected keyword argument","input":["text"]'
)


@pytest.mark.parametrize(
    'tool_input, retry_names',
    [
        # The SDK takes any JSON as the input, so a malformed `action` gets a retry rather than ending the run.
        # As for a call without an `action`, the retry shows the fields that didn't map.
        ({'action': ['left_click'], 'coordinate': [5, 6]}, '"unsupported_fields":["action","coordinate"]'),
        # The earlier tools' clicks take modifier keys as `key` only, so `text` is refused rather than either winning.
        ({'action': 'left_click', 'coordinate': [5, 6], 'key': 'ctrl', 'text': 'shift'}, _CLICK_TEXT_REFUSED),
        ({'action': 'left_click', 'coordinate': [5, 6], 'text': 'shift', 'key': 'ctrl'}, _CLICK_TEXT_REFUSED),
    ],
)
async def test_an_earlier_tool_call_with_input_that_does_not_map_runs_nothing(
    allow_model_requests: None, tool_input: dict[str, Any], retry_names: str
) -> None:
    block = BetaToolUseBlock.model_validate(
        {'id': 'toolu_1', 'type': 'tool_use', 'name': 'computer', 'input': tool_input}
    )
    client = MockAnthropic.create_mock([_turn(block), _turn(BetaTextBlock(text='done', type='text'))])
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    computer = FakeComputer()
    result = await _agent(model, computer).run('Go.')

    assert result.output == 'done'
    assert computer.actions == []
    (retry,) = [part for part in result.all_messages()[2].parts if isinstance(part, ToolReturnPart)]
    assert retry.outcome == 'retried'
    assert retry_names in retry.model_response_str()


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


def test_bedrock_gets_the_earlier_beta_tool() -> None:
    pytest.importorskip('botocore')
    native: dict[str, Any] = {}
    for name, client in (
        ('direct', AsyncAnthropic(api_key='x')),
        ('vertex', AsyncAnthropicVertex(project_id='p', region='us-central1', access_token='x')),
        ('bedrock', AsyncAnthropicBedrock(aws_access_key='x', aws_secret_key='y', aws_region='us-east-1')),
        ('mantle', AsyncAnthropicBedrockMantle(aws_region='us-east-1', aws_access_key='x', aws_secret_key='y')),
        ('foundry', AsyncAnthropicFoundry(api_key='x', base_url='https://example.com')),
    ):
        model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
        native[name] = model.profile.get('anthropic_computer_tool_version')
    assert native == {
        'direct': 'computer_toolset_20260801',
        'vertex': 'computer_toolset_20260801',
        'bedrock': 'computer_20251124',
        'mantle': 'computer_20251124',
        'foundry': 'computer_toolset_20260801',
    }


async def test_computer_calls_from_another_provider_replay_as_a_plain_tool_call(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    history = [
        ModelRequest(parts=[UserPromptPart('Look.')]),
        ModelResponse(
            parts=[ToolCallPart('computer', {'actions': [{'type': 'screenshot'}]}, tool_call_id='call_1')],
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
            {'coordinate': [3, 4], 'text': 'shift'},
            {'type': 'move', 'x': 3, 'y': 4, 'modifiers': ['shift']},
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
        ('left_mouse_down', {}, {'type': 'mouse_down'}),
        ('left_mouse_up', {}, {'type': 'mouse_up'}),
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


@pytest.mark.parametrize(
    'model_name, version',
    [
        ('claude-haiku-5-5', 'computer_toolset_20260801'),
        ('claude-opus-4-8', 'computer_toolset_20260801'),
        ('claude-mythos-preview', 'computer_toolset_20260801'),
        ('claude-opus-4-7', 'computer_20251124'),
        ('claude-haiku-4-5', 'computer_20250124'),
        ('claude-3-5-haiku-latest', None),
    ],
)
def test_each_model_gets_its_computer_use_tool(model_name: str, version: str | None) -> None:
    profile = anthropic_model_profile(model_name)
    assert profile is not None
    assert profile.get('anthropic_computer_tool_version') == version


def _stream(*blocks: Any) -> list[Any]:
    """The events Anthropic streams for a message made of these blocks."""
    events: list[Any] = [
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


# Input that isn't an object has nothing to map either.
@pytest.mark.parametrize('streamed', ['', '{"text": "unfinished', 'null', '[]', '"hello"', 'n'])
async def test_a_streamed_member_cut_off_mid_input_runs_nothing(allow_model_requests: None, streamed: str) -> None:
    events = _stream(_tool_use('toolu_1', 'type', {'text': 'hello'}))
    (delta,) = [event for event in events if isinstance(event, BetaRawContentBlockDeltaEvent)]
    delta.delta = BetaInputJSONDelta(type='input_json_delta', partial_json=streamed)
    model = AnthropicModel(
        'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_stream_mock(events))
    )
    params = ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True)
    async with model.request_stream([ModelRequest.user_text_prompt('Go.')], None, params) as stream:
        async for _ in stream:
            pass

    # The tool refuses the call instead of typing part of it.
    (call,) = [part for part in stream.get().parts if isinstance(part, ToolCallPart)]
    assert call.args_as_dict() == {'actions': [{'type': 'type', 'incomplete_input': True}]}


async def test_a_streamed_member_that_takes_no_input_runs(allow_model_requests: None) -> None:
    # Anthropic streams no input for it, as for any tool without arguments.
    events = _stream(_tool_use('toolu_1', 'screenshot', {}))
    (delta,) = [event for event in events if isinstance(event, BetaRawContentBlockDeltaEvent)]
    delta.delta = BetaInputJSONDelta(type='input_json_delta', partial_json='')
    model = AnthropicModel(
        'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_stream_mock(events))
    )
    params = ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True)
    async with model.request_stream([ModelRequest.user_text_prompt('Go.')], None, params) as stream:
        async for _ in stream:
            pass

    (call,) = [part for part in stream.get().parts if isinstance(part, ToolCallPart)]
    assert call.args_as_dict() == {'actions': [{'type': 'screenshot'}]}


@pytest.mark.parametrize('received', [None, [], 'hello'])
async def test_a_member_whose_input_is_not_an_object_runs_nothing(allow_model_requests: None, received: Any) -> None:
    block = _tool_use('toolu_1', 'type', {'text': 'hello'})
    block.input = received  # as the SDK hands over whatever the API sent
    model = AnthropicModel(
        'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(_turn(block)))
    )
    params = ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True)

    response = await model.request([ModelRequest.user_text_prompt('Go.')], None, params)

    (call,) = [part for part in response.parts if isinstance(part, ToolCallPart)]
    assert call.args_as_dict() == {'actions': [{'type': 'type', 'incomplete_input': True}]}


@pytest.mark.parametrize(
    'model_name, settings',
    [
        pytest.param('claude-sonnet-4-6', None, id='auto'),
        pytest.param('claude-sonnet-4-6', {'tool_choice': ['computer']}, id='forcing-the-function'),
        # The toolset is named `computer` too, and Anthropic refuses a tool of that name beside it.
        pytest.param('claude-sonnet-5-5', None, id='toolset'),
    ],
)
async def test_a_function_tool_named_computer_beside_a_computer_tool_of_another_name_is_refused(
    allow_model_requests: None, model_name: str, settings: ModelSettings | None
) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    function = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'})
    desktop = ToolDefinition(name='desktop', parameters_json_schema={'type': 'object'}, tool_kind=COMPUTER_KIND.id)
    params = ModelRequestParameters(
        function_tools=[function, desktop], native_tools=[COMPUTER_TOOL], allow_text_output=True
    )

    # Both would be named `computer` on the wire, and a call named `computer` would run on the desktop.
    with pytest.raises(UserError, match='cannot be sent beside a function tool named `computer`'):
        await model.request([ModelRequest.user_text_prompt('Go.')], settings, params)
    assert get_mock_chat_completion_kwargs(client) == []


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


async def test_forcing_the_computer_offers_the_toolset_alone(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=client))
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(
        function_tools=[COMPUTER, note], native_tools=[COMPUTER_TOOL], allow_text_output=True
    )

    await model.request([ModelRequest.user_text_prompt('Go.')], {'tool_choice': ['computer']}, params)

    (forced,) = get_mock_chat_completion_kwargs(client)
    # The toolset can't be named, so forcing the computer offers it alone and requires a tool.
    assert [tool['type'] for tool in forced['tools']] == ['computer_toolset_20260801']
    assert forced['tool_choice'] == {'type': 'any'}


_CLICK_TYPE: list[dict[str, Any]] = [
    {'id': 'toolu_1', 'name': 'left_click', 'input': {}},
    {'id': 'toolu_2', 'name': 'type', 'input': {'text': 'hi'}},
]
_PNG = BinaryImage(data=b'png', media_type='image/png')
_FAILED = 'Action 2 of 2 (type) failed: no keyboard. Screenshot attached (1280x800).'
_REFUSED = 'Typing is not allowed here; nothing ran.'
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
            [['OK'], ['Ran 2 actions. [Older screenshot removed.]']],
            id='screenshot-dropped',
        ),
        # As a history processor that removes old screenshots outright leaves it: the summary still reaches the model.
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            ['Ran 2 actions.'],
            {'performed': 2},
            [['OK'], ['Ran 2 actions.']],
            id='screenshot-removed',
        ),
        # Not these calls' outcome: each call gets the text rather than `OK`, which would say its own call ran.
        pytest.param(_CLICK_TYPE, [1, 1], [_REFUSED], None, [[_REFUSED], [_REFUSED]], id='a-guardrail-refusal'),
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            [_CHANGED, _PNG],
            {'performed': 1, 'changed': True},
            [[_CHANGED], [_CHANGED, 'image']],
            id='changed-before-it-ran',
        ),
        # As a UI rebuilds it: the metadata that says which calls ran is gone, so no call is said to have run.
        pytest.param(_CLICK_TYPE, [1, 1], [_FAILED, _PNG], None, [[_FAILED], [_FAILED, 'image']], id='no-metadata'),
        # Nothing ran before the failed call to carry the screen, so it does, saying in its text that it failed.
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            ['Action 1 of 2 (click) failed: busy.', _PNG],
            {'performed': 0, 'error': 'busy'},
            [['Failed: busy', 'image'], [NOT_EXECUTED]],
            id='first-call-failed',
        ),
        # Content stored as a JSON string reads as the list it encodes, screenshot and all.
        pytest.param(
            _CLICK_TYPE,
            [1, 1],
            json.dumps(['Ran 2 actions.', {'kind': 'binary', 'data': 'cG5n', 'media_type': 'image/png'}]),
            {'performed': 2},
            [['OK'], ['OK', 'image']],
            id='json-string-content',
        ),
        # What a zoom returned answers its own call; the screen goes with the last.
        pytest.param(
            [{'id': 'toolu_1', 'name': 'zoom', 'input': {'region': [0, 0, 4, 4]}}, _CLICK_TYPE[0] | {'id': 'toolu_2'}],
            [1, 1],
            ['Ran 2 actions.', _PNG, _PNG],
            {'performed': 2},
            [['OK', 'image'], ['OK', 'image']],
            id='zoom-output',
        ),
        # The counts recorded with the call decide which calls ran, even where mapping again would split differently.
        pytest.param(
            [{'id': 'toolu_1', 'name': 'key', 'input': {'text': 'a b'}}, _CLICK_TYPE[1]],
            [1, 1],
            ['Action 2 of 2 (type) failed: no keyboard.', _PNG],
            {'performed': 1, 'error': 'no keyboard'},
            [['OK', 'image'], ['no keyboard']],
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
    results: list[list[str]],
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
    assert [[b['text'] if b['type'] == 'text' else b['type'] for b in r['content']] for r in sent] == results


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
    # One plain result for the one plain call: no member call is told `OK`.
    assert [r['tool_use_id'] for r in sent] == ['toolu_1']
    assert all(b.get('text') != 'OK' for r in sent for b in r['content'])


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


async def _streamed(events: list[Any], *function_tools: ToolDefinition) -> ModelResponse:
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


async def test_a_streamed_member_cut_off_in_an_optional_field_runs_nothing(allow_model_requests: None) -> None:
    """Keeping what parsed would drop the held key and leave a valid plain click."""
    events = _stream(_tool_use('toolu_1', 'left_click', {'coordinate': [100, 200], 'text': 'ctrl'}))
    (delta,) = [event for event in events if isinstance(event, BetaRawContentBlockDeltaEvent)]
    delta.delta = BetaInputJSONDelta(type='input_json_delta', partial_json='{"coordinate": [100, 200], "text": "ctrl')

    (call,) = [part for part in (await _streamed(events)).parts if isinstance(part, ToolCallPart)]
    assert call.args_as_dict() == {'actions': [{'type': 'click', 'incomplete_input': True}]}


async def test_a_stream_that_ends_inside_the_batch_runs_none_of_it(allow_model_requests: None) -> None:
    """The click arrived whole, but the batch it belongs to didn't, and it runs only as a whole."""
    events = _stream(
        _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
        _tool_use('toolu_2', 'type', {'text': 'hello'}),
    )
    # The body ends after the second call's input, without its stop or the message's.
    cut = [index for index, event in enumerate(events) if isinstance(event, BetaRawContentBlockDeltaEvent)][-1]
    with pytest.raises(UnexpectedModelBehavior, match='ended before Claude finished its computer use calls'):
        await _streamed(events[: cut + 1])


@pytest.mark.parametrize('stop_reason', ['refusal', 'max_tokens'])
@pytest.mark.parametrize('stream', [False, True], ids=['whole', 'streamed'])
async def test_a_call_in_a_response_that_did_not_finish_normally_runs_nothing(
    allow_model_requests: None, stop_reason: Literal['refusal', 'max_tokens'], stream: bool
) -> None:
    """Anthropic says not to run the calls of a response that stopped for a refusal or the token limit."""
    click = _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]})
    done = BetaTextBlock(text='done', type='text')
    if stream:
        first = _stream(click)
        (end,) = [event for event in first if isinstance(event, BetaRawMessageDeltaEvent)]
        end.delta = Delta(stop_reason=stop_reason)
        client = MockAnthropic.create_stream_mock([first, _stream(done)])
    else:
        client = MockAnthropic.create_mock([_turn(click).model_copy(update={'stop_reason': stop_reason}), _turn(done)])
    computer = FakeComputer()
    agent = _agent(AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client)), computer)
    if stream:
        async with agent.run_stream_events('Go.') as events:
            async for _ in events:
                pass
    else:
        await agent.run('Go.')
    assert computer.actions == []


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


@settings(max_examples=500, deadline=None)
@given(st.sampled_from(_MEMBER_NAMES), _NEAR_INPUT | st.dictionaries(st.text(max_size=8), _JSON, max_size=3))
def test_any_member_input_maps_without_raising_and_a_refusal_is_rejected(name: str, input: dict[str, Any]) -> None:
    """The tool checks every action before running any, so one it rejects means none of the call runs."""
    for action in map_computer_member(name, input):
        if 'unsupported_fields' in action:
            with pytest.raises(ValidationError):
                _TOOL_ACTION.validate_python(action)


async def test_a_hook_emptying_a_result_answers_every_call(allow_model_requests: None) -> None:
    from pydantic_ai.capabilities import Hooks

    hooks = Hooks()

    @hooks.on.after_tool_execute(tools=['computer'])
    async def redact(
        ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], result: Any
    ) -> Any:
        return []  # such as a hook taking the screen out of the history

    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'screenshot', {}), _tool_use('toolu_2', 'mouse_move', {'coordinate': [1, 2]})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    result = await Agent(model, capabilities=[FakeComputer(), hooks]).run('Look.')
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


async def test_a_toolset_call_whose_name_isnt_text_is_refused(allow_model_requests: None) -> None:
    # The SDK builds responses without validating them.
    block = BetaToolUseBlock.model_construct(
        id='toolu_1', type='tool_use', name=['left_click'], input={'coordinate': [5, 6]}, toolset_name='computer'
    )
    client = MockAnthropic.create_mock(completion_message([block], BetaUsage(input_tokens=1, output_tokens=1)))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    response = await model.request(
        [ModelRequest.user_text_prompt('Go.')],
        None,
        ModelRequestParameters(function_tools=[COMPUTER], native_tools=[COMPUTER_TOOL], allow_text_output=True),
    )
    (call,) = response.parts
    assert isinstance(call, ComputerCallPart)
    assert call.actions is None
    assert call.tool_call_id == 'toolu_1'


@pytest.mark.parametrize('outcome, is_error', [('denied', False), ('interrupted', False), ('failed', True)])
async def test_only_a_failed_toolset_call_is_a_tool_error(
    allow_model_requests: None, outcome: Literal['denied', 'interrupted', 'failed'], is_error: bool
) -> None:
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
        ('toolu_1', is_error),
        ('toolu_2', is_error),
    ]
