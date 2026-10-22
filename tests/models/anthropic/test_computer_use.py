"""Anthropic's `computer_toolset_20260801` behind the `ComputerUse` capability."""

from __future__ import annotations as _annotations

import json
from typing import Any

import pytest
from cassetter import Cassette

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, ComputerUse
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import Click, TypeText, WorkspaceBackend, WorkspaceError, WorkspaceRef

from ...cassette_utils import request_json
from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider, ProviderBackend
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic import (
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

    from pydantic_ai.models.anthropic import AnthropicModel, _map_computer_member  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.profiles.anthropic import anthropic_model_profile
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'), pytest.mark.anyio]

TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'
NOT_EXECUTED = 'Not executed: an earlier computer action in this turn failed.'


def _members(body: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [
        block
        for message in body['messages']
        if isinstance(message['content'], list)
        for block in message['content']
        if block.get('type') == kind and block.get('toolset_name') == 'computer'
    ]


def _agent(model: AnthropicModel, provider: InMemoryProvider, **settings: Any) -> Agent[None, str]:
    return Agent(model, capabilities=[provider.capability(**settings), ComputerUse(settle_seconds=0)])


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
async def test_the_toolset_is_sent_and_member_calls_are_replayed(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, stream: bool
) -> None:
    provider = InMemoryProvider()
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    async def drain(ctx: Any, events: Any) -> None:
        async for _ in events:
            pass

    await _agent(model, provider).run(TASK, event_stream_handler=drain if stream else None)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, *_, last = [request_json(request) for request in vcr.requests]
    assert first['tools'] == [
        {
            'type': 'computer_toolset_20260801',
            'configs': {'zoom': {'enabled': False}, 'cursor_position': {'enabled': False}},
        }
    ]
    # Every member call comes back as it was sent, with one result each, both naming the toolset.
    calls, results = _members(last, 'tool_use'), _members(last, 'tool_result')
    assert [call['id'] for call in calls] == [result['tool_use_id'] for result in results]
    assert {call['name'] for call in calls} >= {'screenshot', 'left_click', 'type'}


@pytest.mark.vcr
async def test_a_refused_action_gets_an_error_result_and_the_rest_are_not_executed(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
) -> None:
    provider = InMemoryProvider()
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    await _agent(model, provider, read_only=True).run(
        'Without taking a screenshot first, click at x=100 y=200 and then type "hello", in one batch. Then reply '
        '"done" whatever happens.'
    )

    results = _members(request_json(vcr.requests[-1]), 'tool_result')
    # Nothing ran, so every member is an error: the click with its refusal and the rest as not executed.
    assert results and all(result['is_error'] for result in results)
    assert all(block['type'] == 'text' for result in results for block in result['content'])
    assert {result['content'][0]['text'] for result in results[1:]} <= {NOT_EXECUTED}
    assert provider.actions == []


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

    _, second = get_mock_chat_completion_kwargs(client)
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


async def test_a_lone_failed_call_gets_text_only(allow_model_requests: None) -> None:
    provider = InMemoryProvider()
    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'hold_key', {'text': 'ctrl', 'duration': 1})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await _agent(model, provider).run('Go.')

    _, second = get_mock_chat_completion_kwargs(client)
    (only,) = second['messages'][-1]['content']
    assert only['is_error'] is True
    assert [block['type'] for block in only['content']] == ['text']


async def test_a_failed_call_gives_every_member_its_text(allow_model_requests: None) -> None:
    provider = InMemoryProvider()

    class NoScreenshots(ProviderBackend):
        async def screenshot(self) -> bytes:
            raise WorkspaceError('scrot: not found')

    class Supplier(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return NoScreenshots(provider, ref)

    client = MockAnthropic.create_mock(
        [
            _turn(_tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}), _tool_use('toolu_2', 'screenshot', {})),
            _turn(BetaTextBlock(text='done', type='text')),
        ]
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=client))
    await Agent(model, capabilities=[Supplier(), ComputerUse(settle_seconds=0)]).run('Go.')

    _, second = get_mock_chat_completion_kwargs(client)
    # The click ran, so no member may claim it didn't: each carries the text that says what happened.
    results = second['messages'][-1]['content']
    assert [(r['is_error'], [block['text'] for block in r['content']]) for r in results] == [
        (True, ['Ran 2 actions. Taking the screenshot failed: scrot: not found']),
        (True, ['Ran 2 actions. Taking the screenshot failed: scrot: not found']),
    ]


@pytest.mark.parametrize(
    'bad, message',
    [
        (('wait', {'duration': 30}), 'less than or equal to 10'),
        (('left_click', {'coordinate': [5]}), 'valid integer'),
    ],
    ids=['out-of-range', 'malformed-coordinate'],
)
async def test_an_invalid_member_is_refused_and_nothing_runs(
    bad: tuple[str, dict[str, Any]], message: str, allow_model_requests: None
) -> None:
    provider = InMemoryProvider()
    client = MockAnthropic.create_mock(
        [
            _turn(
                _tool_use('toolu_1', 'left_click', {'coordinate': [5, 6]}),
                _tool_use('toolu_2', *bad),
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
    assert all(message in r['content'][0]['text'] for r in results)


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


async def test_an_older_model_gets_a_function(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='done', type='text')))
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    await _agent(model, InMemoryProvider()).run('Go.')

    (request,) = get_mock_chat_completion_kwargs(client)
    assert [(tool['name'], 'input_schema' in tool) for tool in request['tools']] == [('computer', True)]


def test_the_toolset_goes_to_the_claude_api_and_google_cloud_only() -> None:
    pytest.importorskip('botocore')
    params = ModelRequestParameters(function_tools=[ComputerUse().get_toolset().tools['computer'].tool_def])
    native: dict[str, bool] = {}
    for name, client in (
        ('direct', None),
        ('vertex', AsyncAnthropicVertex(project_id='p', region='us-central1', access_token='x')),
        ('bedrock', AsyncAnthropicBedrock(aws_access_key='x', aws_secret_key='y', aws_region='us-east-1')),
        ('mantle', AsyncAnthropicBedrockMantle(aws_region='us-east-1', aws_access_key='x', aws_secret_key='y')),
        ('foundry', AsyncAnthropicFoundry(api_key='x', base_url='https://example.com')),
    ):
        provider = AnthropicProvider(api_key='x') if client is None else AnthropicProvider(anthropic_client=client)
        model = AnthropicModel('claude-sonnet-5-5', provider=provider)
        native[name] = model._native_computer_tool(params) is not None  # pyright: ignore[reportPrivateUsage]
    assert native == {'direct': True, 'vertex': True, 'bedrock': False, 'mantle': False, 'foundry': False}


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
        ('mouse_move', {'coordinate': [3, 4]}, {'type': 'move', 'x': 3, 'y': 4, 'modifiers': []}),
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
        (
            'left_click_drag',
            {'start_coordinate': 7, 'coordinate': [3, 4]},
            {'type': 'drag', 'path': [{'x': 7}, {'x': 3, 'y': 4}], 'modifiers': []},
        ),
        ('left_click', {'coordinate': [5]}, {'type': 'click', 'x': [5], 'button': 'left', 'count': 1, 'modifiers': []}),
        ('mouse_move', {'coordinate': 'here'}, {'type': 'move', 'x': 'here', 'modifiers': []}),
        ('left_mouse_down', {}, {'type': 'mouse_down'}),
        ('left_mouse_up', {}, {'type': 'mouse_up'}),
        (
            'scroll',
            {'scroll_direction': 'down', 'scroll_amount': 80, 'coordinate': [1, 2]},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'down', 'amount': 80, 'modifiers': []},
        ),
        ('type', {'text': 'hi'}, {'type': 'type', 'text': 'hi'}),
        ('key', {'text': 'ctrl+s', 'repeat': 2}, {'type': 'keypress', 'keys': ['ctrl+s'], 'repeat': 2}),
        ('hold_key', {'text': 'shift', 'duration': 30}, {'type': 'hold_key', 'keys': ['shift'], 'seconds': 30}),
        ('scroll', {'scroll_direction': 'up'}, {'type': 'scroll', 'direction': 'up', 'amount': 3, 'modifiers': []}),
        ('key', {'text': 'a'}, {'type': 'keypress', 'keys': ['a'], 'repeat': 1}),
        ('wait', {'duration': 2}, {'type': 'wait', 'seconds': 2}),
        ('screenshot', {}, {'type': 'screenshot'}),
        ('zoom', {'region': [0, 0, 10, 10]}, {'type': 'zoom', 'region': [0, 0, 10, 10]}),
    ],
)
def test_toolset_members_map_onto_the_computer_tool(name: str, input: dict[str, Any], action: dict[str, Any]) -> None:
    assert _map_computer_member(name, input) == action


@pytest.mark.parametrize(
    'model_name, native',
    [
        ('claude-fable-5-1', True),
        ('claude-opus-5-5', True),
        ('claude-sonnet-5', True),
        ('claude-haiku-5-5', True),
        ('claude-opus-4-8', True),
        ('claude-opus-4-7', False),
        ('claude-sonnet-4-6', False),
        ('claude-haiku-4-5', False),
    ],
)
def test_the_toolset_goes_to_the_models_that_accept_it(model_name: str, native: bool) -> None:
    profile = anthropic_model_profile(model_name)
    assert profile is not None
    assert profile.get('anthropic_supports_computer_toolset', False) is native


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


async def test_a_responses_member_calls_are_one_call_streamed_or_not(allow_model_requests: None) -> None:
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
                        {'type': 'keypress', 'keys': ['Return'], 'repeat': 1},
                    ]
                },
            ),
        ]
    )
    assert [type(part) for part in streamed.parts] == [type(part) for part in whole.parts] == [TextPart, ToolCallPart]
    starts = [
        event.part for event in events if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart)
    ]
    assert [part.tool_call_id for part in starts] == ['toolu_1']


async def test_the_toolset_follows_tool_choice(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock([_turn(BetaTextBlock(text='ok', type='text'))] * 2)
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=client))
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    other = ToolDefinition(name='other', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(function_tools=[COMPUTER, note, other], allow_text_output=True)
    prompt: list[ModelMessage] = [ModelRequest.user_text_prompt('Go.')]

    await model.request(prompt, {'tool_choice': ['computer']}, params)
    await model.request(prompt, {'tool_choice': ['note', 'other']}, params)

    forced, without = get_mock_chat_completion_kwargs(client)
    # The toolset can't be named, so forcing the computer offers it alone and requires a tool.
    assert [tool['type'] for tool in forced['tools']] == ['computer_toolset_20260801']
    assert forced['tool_choice'] == {'type': 'any'}
    assert [tool.get('name') for tool in without['tools']] == ['note', 'other']


async def test_a_result_without_its_metadata_reads_as_every_call_having_run() -> None:
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='unused'))
    blocks: list[dict[str, Any]] = [
        {'id': 'toolu_1', 'type': 'tool_use', 'name': 'left_click', 'input': {}, 'toolset_name': 'computer'},
        {'id': 'toolu_2', 'type': 'tool_use', 'name': 'type', 'input': {'text': 'hi'}, 'toolset_name': 'computer'},
    ]
    part = ToolReturnPart('computer', 'Ran 2 actions.', tool_call_id='toolu_1')

    results = await model._map_computer_results(part, blocks)  # pyright: ignore[reportPrivateUsage]

    assert [(r['tool_use_id'], r.get('is_error'), r.get('content')) for r in results] == [
        ('toolu_1', False, [{'text': 'OK', 'type': 'text'}]),
        ('toolu_2', False, [{'text': 'OK', 'type': 'text'}]),
    ]


async def test_without_the_toolset_past_member_calls_replay_as_plain_calls(allow_model_requests: None) -> None:
    client = MockAnthropic.create_mock(_turn(BetaTextBlock(text='ok', type='text')))
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=client))
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    other = ToolDefinition(name='other', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(function_tools=[COMPUTER, note, other], allow_text_output=True)
    block: dict[str, Any] = {
        'id': 'toolu_1',
        'type': 'tool_use',
        'name': 'screenshot',
        'input': {},
        'toolset_name': 'computer',
    }
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
    assert _members(body, 'tool_use') == [] and _members(body, 'tool_result') == []
    (call,) = [
        b for m in body['messages'] if isinstance(m['content'], list) for b in m['content'] if b['type'] == 'tool_use'
    ]
    assert (call['name'], call['input']) == ('computer', {'actions': [{'type': 'screenshot'}]})
