"""OpenAI Responses' native `computer` tool behind the `ComputerUse` capability."""

from __future__ import annotations

from typing import Any

import pytest
from cassetter import Cassette

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, ToolCallPart, ToolDenied
from pydantic_ai.capabilities import AbstractCapability, ComputerUse
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart, UserPromptPart
from pydantic_ai.workspaces import Click, TypeText, WorkspaceBackend, WorkspaceError, WorkspaceRef

from ..cassette_utils import request_json
from ..conftest import try_import
from ..workspace_fakes import InMemoryProvider, ProviderBackend
from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

with try_import() as imports_successful:
    from openai.types import responses as resp

    from pydantic_ai.models.openai import OpenAIResponsesModel, _map_computer_action  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.profiles.openai import openai_model_profile
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='openai not installed'), pytest.mark.anyio]

TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'


def _bodies(cassette: Cassette) -> list[dict[str, Any]]:
    return [request_json(request) for request in cassette.requests]


def _items(body: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [item for item in body['input'] if item.get('type') == kind]


def _agent(openai_api_key: str, provider: InMemoryProvider, **settings: Any) -> Agent[None, str]:
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key=openai_api_key))
    return Agent(model, capabilities=[provider.capability(**settings), ComputerUse(settle_seconds=0)])


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
async def test_the_computer_tool_is_sent_natively_and_replayed(
    allow_model_requests: None, openai_api_key: str, vcr: Cassette, stream: bool
) -> None:
    provider = InMemoryProvider()
    agent = _agent(openai_api_key, provider)

    async def drain(ctx: Any, events: Any) -> None:
        async for _ in events:
            pass

    result = await agent.run(TASK, event_stream_handler=drain if stream else None)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, *later = _bodies(vcr)
    assert {'type': 'computer'} in first['tools']
    assert not [tool for tool in first['tools'] if tool.get('name') == 'computer']
    # Every later request sends each computer call back as it came, answered with a screenshot.
    last = later[-1]
    calls, outputs = _items(last, 'computer_call'), _items(last, 'computer_call_output')
    assert [call['call_id'] for call in calls] == [output['call_id'] for output in outputs]
    assert all(output['output']['image_url'].startswith('data:image/png;base64,') for output in outputs)
    assert all(
        isinstance(part, ToolCallPart) and part.provider_details and 'computer_call' in part.provider_details
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    )


@pytest.mark.vcr
async def test_a_failed_action_is_explained_after_its_screenshot(
    allow_model_requests: None, openai_api_key: str, vcr: Cassette
) -> None:
    provider = InMemoryProvider()
    agent = _agent(openai_api_key, provider, read_only=True)
    await agent.run('Click at x=100 y=200 once, then reply "done" whatever happens.')

    body = _bodies(vcr)[-1]
    looked, refused = _items(body, 'computer_call_output')
    after = [body['input'][body['input'].index(output) + 1] for output in (looked, refused)]
    # The screenshot-only call ran, so nothing follows it; the refused click is explained right after its output.
    assert after[0].get('role') != 'user'
    assert after[1]['role'] == 'user'
    assert after[1]['content'].startswith('Action 1 of 1 (click) failed: This workspace is read-only')
    assert provider.actions == []


def _computer_call(*actions: dict[str, Any], safety_checks: list[dict[str, Any]] | None = None) -> resp.Response:
    item = resp.ResponseComputerToolCall.model_validate(
        {
            'id': 'cu_1',
            'call_id': 'call_1',
            'type': 'computer_call',
            'status': 'completed',
            'actions': list(actions),
            'pending_safety_checks': safety_checks or [],
        }
    )
    return response_message([item])


def _done() -> resp.Response:
    return response_message(
        [
            resp.ResponseOutputMessage.model_validate(
                {
                    'id': 'msg_1',
                    'type': 'message',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_text', 'text': 'done', 'annotations': []}],
                }
            )
        ]
    )


CHECK = {'id': 'cu_sc_1', 'code': 'malicious_instructions', 'message': 'The page asks you to send a password.'}


@pytest.mark.parametrize('approve', [True, False], ids=['approved', 'denied'])
async def test_a_safety_check_waits_for_approval_and_is_acknowledged_only_if_approved(
    allow_model_requests: None, approve: bool
) -> None:
    provider = InMemoryProvider()
    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}, safety_checks=[CHECK]), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(
        model,
        capabilities=[provider.capability(), ComputerUse(settle_seconds=0)],
        output_type=[str, DeferredToolRequests],
    )

    paused = await agent.run('Go.')
    assert isinstance(paused.output, DeferredToolRequests)
    (approval,) = paused.output.approvals
    assert paused.output.metadata == {approval.tool_call_id: {'safety_checks': [CHECK]}}
    assert provider.actions == []

    decision = True if approve else ToolDenied('Not on this page.')
    resumed = await agent.run(
        message_history=paused.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={approval.tool_call_id: decision}),
    )
    assert resumed.output == 'done'
    _, second = get_mock_responses_kwargs(client)
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    if approve:
        assert output['acknowledged_safety_checks'] == [CHECK]
        assert provider.actions == [Click(x=5, y=6)]
    else:
        assert 'acknowledged_safety_checks' not in output
        note = second['input'][second['input'].index(output) + 1]
        assert note == {'role': 'user', 'content': 'Not on this page.'}
        assert provider.actions == []


async def test_a_failed_call_still_sends_a_screenshot_and_says_why(allow_model_requests: None) -> None:
    provider = InMemoryProvider()

    class NoScreenshots(ProviderBackend):
        async def screenshot(self) -> bytes:
            raise WorkspaceError('scrot: not found')

    class Supplier(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return NoScreenshots(provider, ref)

    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    await Agent(model, capabilities=[Supplier(), ComputerUse(settle_seconds=0)]).run('Go.')

    _, second = get_mock_responses_kwargs(client)
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    assert output['output']['image_url'].startswith('data:image/png;base64,')
    note = second['input'][second['input'].index(output) + 1]
    assert note == {'role': 'user', 'content': 'Ran 1 action. Taking the screenshot failed: scrot: not found'}


async def test_a_model_without_the_native_tool_gets_a_function(allow_model_requests: None) -> None:
    client = MockOpenAIResponses.create_mock(_done())
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(openai_client=client))
    await Agent(model, capabilities=[InMemoryProvider().capability(), ComputerUse()]).run('Go.')

    (request,) = get_mock_responses_kwargs(client)
    assert [tool['name'] for tool in request['tools']] == ['computer']
    assert request['tools'][0]['type'] == 'function'


async def test_computer_calls_from_another_provider_replay_as_function_calls(allow_model_requests: None) -> None:
    client = MockOpenAIResponses.create_mock(_done())
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    history = [
        ModelRequest(parts=[UserPromptPart('Look.')]),
        ModelResponse(
            parts=[ToolCallPart('computer', {'actions': [{'type': 'screenshot'}]}, tool_call_id='toolu_1')],
            provider_name='anthropic',
        ),
        ModelRequest(parts=[ToolReturnPart('computer', ['Ran 1 action.'], tool_call_id='toolu_1')]),
    ]
    agent = Agent(model, capabilities=[InMemoryProvider().capability(), ComputerUse()])
    await agent.run('Go on.', message_history=history)

    (request,) = get_mock_responses_kwargs(client)
    kinds = [item.get('type') for item in request['input']]
    assert 'function_call' in kinds and 'function_call_output' in kinds
    assert 'computer_call' not in kinds


@pytest.mark.parametrize(
    'action, mapped',
    [
        (
            {'type': 'click', 'x': 1, 'y': 2, 'button': 'wheel', 'keys': ['CTRL']},
            {'type': 'click', 'x': 1, 'y': 2, 'button': 'middle', 'modifiers': ['CTRL']},
        ),
        ({'type': 'double_click', 'x': 1, 'y': 2}, {'type': 'click', 'x': 1, 'y': 2, 'count': 2, 'modifiers': []}),
        (
            {'type': 'drag', 'path': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]},
            {'type': 'drag', 'path': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}], 'modifiers': []},
        ),
        ({'type': 'move', 'x': 1, 'y': 2}, {'type': 'move', 'x': 1, 'y': 2, 'modifiers': []}),
        (
            {'type': 'scroll', 'x': 1, 'y': 2, 'scroll_x': 0, 'scroll_y': 340},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'down', 'amount': 3, 'modifiers': []},
        ),
        (
            {'type': 'scroll', 'x': 1, 'y': 2, 'scroll_x': -9000, 'scroll_y': 10},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'left', 'amount': 50, 'modifiers': []},
        ),
        (
            {'type': 'scroll', 'x': 1, 'y': 2, 'scroll_x': 0, 'scroll_y': -20},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'up', 'amount': 1, 'modifiers': []},
        ),
        ({'type': 'keypress', 'keys': ['CTRL', 'S']}, {'type': 'keypress', 'keys': ['CTRL', 'S']}),
        ({'type': 'type', 'text': 'hi'}, {'type': 'type', 'text': 'hi'}),
        ({'type': 'wait'}, {'type': 'wait'}),
        ({'type': 'screenshot'}, {'type': 'screenshot'}),
        ({'type': 'teleport', 'x': 1}, {'type': 'teleport', 'x': 1}),
    ],
)
def test_openai_actions_map_onto_the_computer_tool(action: dict[str, Any], mapped: dict[str, Any]) -> None:
    assert _map_computer_action(action) == mapped


@pytest.mark.parametrize(
    'model_name, native',
    [
        ('gpt-5.4', True),
        ('gpt-5.4-mini', True),
        ('gpt-5.5', True),
        ('gpt-5.6', True),
        ('gpt-6-luna', True),
        ('gpt-6.1-sol', True),
        ('gpt-5.4-nano', False),
        ('gpt-5.2', False),
        ('gpt-4.1', False),
    ],
)
def test_the_native_tool_goes_to_the_models_that_accept_it(model_name: str, native: bool) -> None:
    assert openai_model_profile(model_name).get('openai_supports_computer_tool', False) is native
