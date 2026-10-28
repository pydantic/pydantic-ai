"""Anthropic's `computer_toolset_20260801`, sent for a tool of the `computer` kind."""

from __future__ import annotations as _annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from pydantic_ai import Agent, Tool
from pydantic_ai.computer_use import COMPUTER_KIND, DISPLAY_SIZE, ComputerToolAction
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import (
    BinaryImage,
    ModelMessage,
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
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import Click, PressKeys, TypeText

from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider, computer_toolset
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
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
    from pydantic_ai.profiles.anthropic import anthropic_model_profile
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'), pytest.mark.anyio]

TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'
NOT_EXECUTED = 'Not executed: an earlier computer action in this turn failed.'


def _agent(model: AnthropicModel, provider: InMemoryProvider, **settings: Any) -> Agent[None, str]:
    return Agent(model, toolsets=[computer_toolset()], capabilities=[provider.capability(**settings)])


@pytest.mark.vcr
async def test_claude_member_calls_run_on_the_workspace(allow_model_requests: None, anthropic_api_key: str) -> None:
    provider = InMemoryProvider()
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    await _agent(model, provider).run(TASK)

    # Claude's real member calls became `computer` calls that ran.
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]


@pytest.mark.vcr
async def test_claude_computer_20251124_calls_run_on_the_workspace(
    allow_model_requests: None, anthropic_api_key: str
) -> None:
    provider = InMemoryProvider()
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model, toolsets=[computer_toolset(display_size=(1280, 800))], capabilities=[provider.capability()])
    await agent.run(TASK)

    # Claude Sonnet 4.6's real `computer_20251124` calls became `computer` calls that ran.
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]


def _tool_use(id: str, name: str, input: dict[str, Any]) -> BetaToolUseBlock:
    return BetaToolUseBlock.model_validate(
        {'id': id, 'type': 'tool_use', 'name': name, 'input': input, 'toolset_name': 'computer'}
    )


def _turn(*blocks: Any) -> Any:
    return completion_message(list(blocks), BetaUsage(input_tokens=1, output_tokens=1))


async def test_a_batch_is_one_call_and_its_result_is_split_per_member(allow_model_requests: None) -> None:
    provider = InMemoryProvider()
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6], 'text': 'shift'}),
                _tool_use('toolu_2', 'hold_key', {'text': 'ctrl', 'duration': 1}),
                _tool_use('toolu_3', 'type', {'text': 'never typed'}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    result = await _agent(model, provider).run('Go.')

    (call,) = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)]
    assert call.tool_call_id == 'toolu_1'
    assert call.args_as_dict() == {
        'actions': [
            {'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'count': 1, 'modifiers': ['shift']},
            {'type': 'hold_key', 'keys': ['ctrl'], 'seconds': 1},
            {'type': 'type', 'text': 'never typed'},
        ]
    }
    assert provider.actions == [Click(x=5, y=6, modifiers=('shift',))]

    first, second = get_mock_chat_completion_kwargs(client)
    assert first['tools'] == [
        {
            'type': 'computer_toolset_20260801',
        }
    ]
    # The member calls go back as they came, each answered by one result; both name the toolset.
    calls = second['messages'][-2]['content']
    assert [(c['id'], c['name'], c['toolset_name']) for c in calls] == [
        ('toolu_1', 'left_click', 'computer'),
        ('toolu_2', 'hold_key', 'computer'),
        ('toolu_3', 'type', 'computer'),
    ]
    results = second['messages'][-1]['content']
    assert [(r['tool_use_id'], r['is_error'], [block['type'] for block in r['content']]) for r in results] == [
        ('toolu_1', False, ['text', 'image']),
        ('toolu_2', True, ['text']),
        ('toolu_3', True, ['text']),
    ]
    assert [r['content'][0]['text'] for r in results] == [
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
    await Agent(model, tools=[Tool(computer, tool_kind=COMPUTER_KIND)]).run('Go.')

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
    await _agent(model, InMemoryProvider()).run('Go.')

    _, second = get_mock_chat_completion_kwargs(client)
    results = second['messages'][-1]['content']
    assert [[block['type'] for block in r['content']] for r in results] == [
        ['text', 'image'],
        ['text', 'text'],
        ['text', 'image'],
    ]
    assert results[1]['content'][1]['text'] == 'The pointer is at (640, 400).'


async def test_a_key_sequence_is_one_member_however_many_presses(allow_model_requests: None) -> None:
    provider = InMemoryProvider()
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'key', {'text': 'ctrl+a BackSpace'}),
                _tool_use('toolu_2', 'hold_key', {'text': 'shift', 'duration': 1}),
                _tool_use('toolu_3', 'type', {'text': 'never typed'}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await _agent(model, provider).run('Go.')

    assert provider.actions == [PressKeys(keys=('ctrl', 'a')), PressKeys(keys=('backspace',))]
    _, second = get_mock_chat_completion_kwargs(client)
    results = second['messages'][-1]['content']
    assert [(r['tool_use_id'], r['is_error']) for r in results] == [
        ('toolu_1', False),
        ('toolu_2', True),
        ('toolu_3', True),
    ]
    assert results[2]['content'][0]['text'] == NOT_EXECUTED


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
    provider = InMemoryProvider()
    computer_call = BetaToolUseBlock.model_validate(
        {
            'id': 'toolu_1',
            'type': 'tool_use',
            'name': 'computer',
            'input': {'action': 'left_click', 'coordinate': [5, 6]},
        }
    )
    client = MockAnthropic.create_mock([_turn(computer_call), _turn(BetaTextBlock(text='done', type='text'))])
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(model, toolsets=[computer_toolset(display_size=(1280, 800))], capabilities=[provider.capability()])
    await agent.run('Go.')

    assert provider.actions == [Click(x=5, y=6)]
    first, second = get_mock_chat_completion_kwargs(client)
    assert first['tools'] == [
        {'type': tool, 'name': 'computer', 'display_width_px': 1280, 'display_height_px': 800, **zoom}
    ]
    assert beta in first['betas']
    # The call goes back as it came; its result is an ordinary tool result.
    (call,) = second['messages'][-2]['content']
    assert (call['name'], call['input'], 'toolset_name' in call) == (
        'computer',
        {'action': 'left_click', 'coordinate': [5, 6]},
        False,
    )
    (result,) = second['messages'][-1]['content']
    assert [block['type'] for block in result['content']] == ['text', 'image']


async def test_an_invalid_member_is_refused_and_nothing_runs(allow_model_requests: None) -> None:
    provider = InMemoryProvider()
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
                _tool_use('toolu_2', 'wait', {'duration': 60}),
            ),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await _agent(model, provider).run('Go.')

    assert provider.actions == []
    _, second = get_mock_chat_completion_kwargs(client)
    results = second['messages'][-1]['content']
    assert [(r['tool_use_id'], r['is_error']) for r in results] == [('toolu_1', True), ('toolu_2', True)]
    assert all('less than or equal to 30' in r['content'][0]['text'] for r in results)


async def test_a_member_with_a_field_that_does_not_map_runs_nothing(allow_model_requests: None) -> None:
    # A press at a point would otherwise run at the pointer, and the point would be stored in the history.
    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'left_mouse_down', {'coordinate': [5, 6]})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    provider = InMemoryProvider()
    result = await _agent(model, provider).run('Go.')

    assert provider.actions == []
    call = next(part for part in result.all_messages()[1].parts if isinstance(part, ToolCallPart))
    # Under the tool's own type, so the retry names the field rather than refusing the member.
    assert call.args_as_dict() == {'actions': [{'type': 'mouse_down', 'unsupported_fields': ['coordinate']}]}
    (retry,) = [part for part in result.all_messages()[2].parts if isinstance(part, ToolReturnPart)]
    assert retry.outcome == 'retried'
    assert '"loc":["actions",0,"mouse_down","unsupported_fields"]' in retry.model_response_str()


async def test_a_deferred_computer_tool_stays_a_deferred_function(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    computer = ToolDefinition(
        name='computer', parameters_json_schema={'type': 'object'}, tool_kind='computer', defer_loading=True
    )
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(function_tools=[computer, note], allow_text_output=True)
    await model.request([ModelRequest.user_text_prompt('Look.')], None, params)

    (request,) = get_mock_chat_completion_kwargs(client)
    assert [(tool.get('type'), tool.get('name'), tool.get('defer_loading')) for tool in request['tools']] == [
        (None, 'computer', True),
        (None, 'note', None),
    ]


async def test_an_earlier_tool_without_the_display_size_stays_a_function(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    await _agent(model, InMemoryProvider()).run('Go.')

    (request,) = get_mock_chat_completion_kwargs(client)
    assert [(tool['name'], 'input_schema' in tool) for tool in request['tools']] == [('computer', True)]


def test_bedrock_and_foundry_get_the_earlier_beta_tool() -> None:
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
    # Foundry doesn't list the Claude 5.5 models for the earlier tool, so they get a function tool there.
    assert native == {
        'direct': 'computer_toolset_20260801',
        'vertex': 'computer_toolset_20260801',
        'bedrock': 'computer_20251124',
        'mantle': 'computer_20251124',
        'foundry': None,
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
    await _agent(model, InMemoryProvider()).run('Go on.', message_history=history)

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
        ('left_click_drag', {'coordinate': [3, 4]}, {'type': 'drag', 'path': [{'x': 3, 'y': 4}], 'modifiers': []}),
        ('left_click', {'coordinate': [5]}, {'type': 'click', 'x': [5], 'button': 'left', 'count': 1, 'modifiers': []}),
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
        ('scroll', {'scroll_direction': 'up'}, {'type': 'scroll', 'direction': 'up', 'amount': None, 'modifiers': []}),
        ('wait', {'duration': 2}, {'type': 'wait', 'seconds': 2}),
        ('screenshot', {}, {'type': 'screenshot'}),
        (
            'zoom',
            {'region': [1, 2, 30, 40]},
            {'type': 'zoom', 'region': {'left': 1, 'top': 2, 'right': 30, 'bottom': 40}},
        ),
        ('zoom', {}, {'type': 'zoom', 'region': None}),
        ('cursor_position', {}, {'type': 'cursor_position'}),
        ('teleport', {'coordinate': [1, 2]}, {'type': 'teleport', 'unsupported_fields': ['coordinate']}),
        ('type', {'text': 'hi', 'delay': 5}, {'type': 'type', 'unsupported_fields': ['delay']}),
    ],
)
def test_toolset_members_map_onto_the_computer_tool(name: str, input: dict[str, Any], action: Any) -> None:
    assert map_computer_member(name, input) == (action if isinstance(action, list) else [action])


@pytest.mark.parametrize(
    'model_name, version',
    [
        ('claude-haiku-5-5', 'computer_toolset_20260801'),
        ('claude-opus-4-8', 'computer_toolset_20260801'),
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


COMPUTER = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'}, tool_kind='computer')


async def test_member_calls_in_one_response_are_one_call_streamed_or_not(allow_model_requests: None) -> None:
    # Anthropic runs a response's member calls as one batch, stopped at the first failure, so text between them
    # doesn't split it.
    blocks = [
        _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
        _tool_use('toolu_2', 'type', {'text': 'hi'}),
        BetaTextBlock(text='Now the other field.', type='text'),
        _tool_use('toolu_3', 'key', {'text': 'Return'}),
    ]
    params = ModelRequestParameters(function_tools=[COMPUTER], allow_text_output=True)
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
    assert [type(part) for part in streamed.parts] == [type(part) for part in whole.parts] == [ToolCallPart, TextPart]
    starts = [
        event.part for event in events if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart)
    ]
    # Started once, with the whole batch, so a UI sees one call.
    assert [len(part.args_as_dict()['actions']) for part in starts] == [3]


async def test_each_computer_20251124_call_is_its_own_call_streamed_or_not(allow_model_requests: None) -> None:
    def call(id: str, input: dict[str, Any]) -> BetaToolUseBlock:
        return BetaToolUseBlock.model_validate({'id': id, 'type': 'tool_use', 'name': 'computer', 'input': input})

    blocks = [
        call('toolu_1', {'action': 'left_click', 'coordinate': [5, 6]}),
        call('toolu_2', {'action': 'screenshot'}),
    ]
    computer = replace(COMPUTER, metadata={DISPLAY_SIZE: (1280, 800)})
    params = ModelRequestParameters(function_tools=[computer], allow_text_output=True)
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


async def test_a_streamed_member_cut_off_mid_input_keeps_only_what_parsed(allow_model_requests: None) -> None:
    events = _stream(_tool_use('toolu_1', 'type', {'text': 'hello'}))
    (delta,) = [event for event in events if isinstance(event, BetaRawContentBlockDeltaEvent)]
    delta.delta = BetaInputJSONDelta(type='input_json_delta', partial_json='{"text": "unfinished')
    model = AnthropicModel(
        'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_stream_mock(events))
    )
    params = ModelRequestParameters(function_tools=[COMPUTER], allow_text_output=True)
    async with model.request_stream([ModelRequest.user_text_prompt('Go.')], None, params) as stream:
        async for _ in stream:
            pass

    # The cut-off text is dropped, so the tool refuses the empty `type` instead of typing part of it.
    (call,) = [part for part in stream.get().parts if isinstance(part, ToolCallPart)]
    assert call.args_as_dict() == {'actions': [{'type': 'type', 'text': ''}]}


async def test_forcing_the_computer_offers_the_toolset_alone(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=client))
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(function_tools=[COMPUTER, note], allow_text_output=True)

    await model.request([ModelRequest.user_text_prompt('Go.')], {'tool_choice': ['computer']}, params)

    (forced,) = get_mock_chat_completion_kwargs(client)
    # The toolset can't be named, so forcing the computer offers it alone and requires a tool.
    assert [tool['type'] for tool in forced['tools']] == ['computer_toolset_20260801']
    assert forced['tool_choice'] == {'type': 'any'}


async def test_a_result_without_metadata_or_screenshot_runs_every_call_and_ends_with_its_text() -> None:
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='unused'))
    blocks: list[dict[str, Any]] = [
        {'id': 'toolu_1', 'name': 'left_click', 'input': {}},
        {'id': 'toolu_2', 'name': 'type', 'input': {'text': 'hi'}},
    ]
    # As `keep_screenshots` leaves it: the summary, then the note in place of the screenshot.
    part = ToolReturnPart('computer', ['Ran 2 actions.', '[Older screenshot removed.]'], tool_call_id='toolu_1')

    results = await model._map_computer_results(part, blocks)  # pyright: ignore[reportPrivateUsage]

    assert [(r['tool_use_id'], r.get('is_error'), r.get('content')) for r in results] == [
        ('toolu_1', False, [{'text': 'OK', 'type': 'text'}]),
        ('toolu_2', False, [{'text': 'Ran 2 actions. [Older screenshot removed.]', 'type': 'text'}]),
    ]


@pytest.mark.parametrize(
    'content,metadata',
    [
        pytest.param(['Typing is not allowed here; nothing ran.'], None, id='a-guardrail-refusal'),
        pytest.param(
            [
                'Before it ran, this call was changed to: [...]. Ran 1 action.',
                BinaryImage(data=b'png', media_type='image/png'),
            ],
            {'performed': 1, 'changed': True},
            id='changed-before-it-ran',
        ),
    ],
)
async def test_a_result_that_is_not_these_calls_outcome_gives_each_call_its_text(
    content: list[Any], metadata: dict[str, Any] | None
) -> None:
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='unused'))
    blocks: list[dict[str, Any]] = [
        {'id': 'toolu_1', 'name': 'left_click', 'input': {}},
        {'id': 'toolu_2', 'name': 'type', 'input': {'text': 'hi'}},
    ]
    part = ToolReturnPart('computer', content, tool_call_id='toolu_1', metadata=metadata)

    results = await model._map_computer_results(part, blocks)  # pyright: ignore[reportPrivateUsage]

    # Rather than `OK`, which would tell Claude its own calls ran.
    assert [(r['tool_use_id'], list(r.get('content', []))[0]) for r in results] == [
        ('toolu_1', {'text': content[0], 'type': 'text'}),
        ('toolu_2', {'text': content[0], 'type': 'text'}),
    ]


async def test_what_any_action_returned_answers_its_own_member_call() -> None:
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='unused'))
    blocks: list[dict[str, Any]] = [
        {'id': 'toolu_1', 'name': 'read_clipboard', 'input': {}},  # an action that returns something, added later
        {'id': 'toolu_2', 'name': 'left_click', 'input': {}},
    ]
    part = ToolReturnPart(
        'computer',
        ['Ran 2 actions.', 'copied text', BinaryImage(data=b'png', media_type='image/png')],
        tool_call_id='toolu_1',
        metadata={'performed': 2},
    )

    results = await model._map_computer_results(part, blocks)  # pyright: ignore[reportPrivateUsage]

    assert [list(r.get('content', []))[:2] for r in results] == [
        [{'text': 'OK', 'type': 'text'}, {'text': 'copied text', 'type': 'text'}],
        [
            {'text': 'OK', 'type': 'text'},
            {'source': {'data': 'cG5n', 'media_type': 'image/png', 'type': 'base64'}, 'type': 'image'},
        ],
    ]


async def test_a_result_without_metadata_tells_the_last_call_what_happened() -> None:
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='unused'))
    blocks: list[dict[str, Any]] = [
        {'id': 'toolu_1', 'name': 'left_click', 'input': {}},
        {'id': 'toolu_2', 'name': 'type', 'input': {'text': 'hi'}},
    ]
    # As a UI rebuilds it: the metadata that says which calls ran is gone, the summary still says.
    summary = 'Action 2 of 2 (type) failed: no keyboard. Screenshot attached (1280x800).'
    part = ToolReturnPart(
        'computer', [summary, BinaryImage(data=b'png', media_type='image/png')], tool_call_id='toolu_1'
    )

    results = await model._map_computer_results(part, blocks)  # pyright: ignore[reportPrivateUsage]

    assert [(r['tool_use_id'], r.get('is_error'), list(r.get('content', []))[0]) for r in results] == [
        ('toolu_1', False, {'text': 'OK', 'type': 'text'}),
        ('toolu_2', False, {'text': summary, 'type': 'text'}),
    ]


async def test_without_the_toolset_past_member_calls_replay_as_plain_calls(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=client))
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    other = ToolDefinition(name='other', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(function_tools=[COMPUTER, note, other], allow_text_output=True)
    block: dict[str, Any] = {'id': 'toolu_1', 'name': 'screenshot', 'input': {}}
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Look.'),
        ModelResponse(
            parts=[
                ToolCallPart(
                    'computer',
                    {'actions': [{'type': 'screenshot'}]},
                    tool_call_id='toolu_1',
                    provider_name='anthropic',
                    provider_details={'computer_blocks': [block]},
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
