"""Anthropic's `computer_toolset_20260801` behind the `ComputerUse` capability."""

from __future__ import annotations as _annotations

from typing import Any

import pytest
from cassetter import Cassette

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, ComputerUse
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
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
    from anthropic.types.beta import BetaTextBlock, BetaToolUseBlock, BetaUsage

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
    errors = [result for result in results if result['is_error']]
    assert errors and all(block['type'] == 'text' for error in errors for block in error['content'])
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
        ('toolu_3', False, ['text']),
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


async def test_a_failed_call_is_an_error_on_its_first_member(allow_model_requests: None) -> None:
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
    first, rest = second['messages'][-1]['content']
    assert (first['is_error'], first['content'][0]['text']) == (
        True,
        'Ran 2 actions. Taking the screenshot failed: scrot: not found',
    )
    assert (rest['is_error'], rest['content'][0]['text']) == (False, NOT_EXECUTED)


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
        ('mouse_move', {'coordinate': [3, 4]}, {'type': 'move', 'x': 3, 'y': 4}),
        (
            'left_click_drag',
            {'start_coordinate': [1, 2], 'coordinate': [3, 4]},
            {'type': 'drag', 'path': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}], 'modifiers': []},
        ),
        ('left_click_drag', {'coordinate': [3, 4]}, {'type': 'drag', 'path': [{'x': 3, 'y': 4}], 'modifiers': []}),
        ('left_mouse_down', {}, {'type': 'mouse_down'}),
        ('left_mouse_up', {}, {'type': 'mouse_up'}),
        (
            'scroll',
            {'scroll_direction': 'down', 'scroll_amount': 80, 'coordinate': [1, 2]},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'down', 'amount': 50, 'modifiers': []},
        ),
        ('type', {'text': 'hi'}, {'type': 'type', 'text': 'hi'}),
        ('key', {'text': 'ctrl+s', 'repeat': 2}, {'type': 'keypress', 'keys': ['ctrl+s'], 'repeat': 2}),
        ('hold_key', {'text': 'shift', 'duration': 30}, {'type': 'hold_key', 'keys': ['shift'], 'seconds': 10}),
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
