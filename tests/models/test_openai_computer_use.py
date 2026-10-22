"""OpenAI Responses' native `computer` tool behind the `ComputerUse` capability."""

from __future__ import annotations

from typing import Any

import pytest
from cassetter import Cassette

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, ToolCallPart, ToolDenied
from pydantic_ai.capabilities import AbstractCapability, ComputerUse
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import Click, TypeText, WorkspaceBackend, WorkspaceError, WorkspaceRef

from ..conftest import try_import
from ..workspace_fakes import InMemoryProvider, ProviderBackend
from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

with try_import() as imports_successful:
    from openai.types import responses as resp
    from openai.types.responses.response import Conversation

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings, _map_computer_action  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.profiles.openai import openai_model_profile
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='openai not installed'), pytest.mark.anyio]

TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'


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

    # GPT-5.5's real `computer_call`s became `computer` calls that ran, and keep the item to send back.
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    assert all(
        isinstance(part, ToolCallPart) and part.provider_details and 'computer_call' in part.provider_details
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    )


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


async def test_a_safety_check_is_acknowledged_when_the_call_runs(allow_model_requests: None) -> None:
    provider = InMemoryProvider()
    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}, safety_checks=[CHECK]), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(model, capabilities=[provider.capability(), ComputerUse(settle_seconds=0)])

    assert (await agent.run('Go.')).output == 'done'
    _, second = get_mock_responses_kwargs(client)
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    assert output['acknowledged_safety_checks'] == [CHECK]
    assert provider.actions == [Click(x=5, y=6)]


@pytest.mark.parametrize('approve', [True, False], ids=['approved', 'denied'])
async def test_with_approval_a_reviewer_sees_the_safety_checks_and_a_denied_call_acknowledges_none(
    allow_model_requests: None, approve: bool
) -> None:
    provider = InMemoryProvider()
    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}, safety_checks=[CHECK]), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(
        model,
        capabilities=[provider.capability(), ComputerUse(require_approval=True, settle_seconds=0)],
        output_type=[str, DeferredToolRequests],
    )

    paused = await agent.run('Go.')
    assert isinstance(paused.output, DeferredToolRequests)
    (approval,) = paused.output.approvals
    assert (approval.provider_details or {})['safety_checks'] == [CHECK]
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


async def test_a_call_with_no_actions_takes_a_screenshot(allow_model_requests: None) -> None:
    provider = InMemoryProvider()
    client = MockOpenAIResponses.create_mock([_computer_call(), _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(model, capabilities=[provider.capability(), ComputerUse(settle_seconds=0)])

    result = await agent.run('Look.')
    (call,) = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)]
    assert call.args_as_dict() == {'actions': [{'type': 'screenshot'}]}
    _, second = get_mock_responses_kwargs(client)
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    assert output['output']['image_url'].startswith('data:image/png;base64,')


@pytest.mark.parametrize(
    'settings',
    [{'openai_previous_response_id': 'auto'}, {'openai_conversation_id': 'conv_1'}],
    ids=['previous-response', 'conversation'],
)
async def test_a_result_whose_call_is_stored_server_side_is_a_computer_call_output(
    allow_model_requests: None, settings: OpenAIResponsesModelSettings
) -> None:
    provider = InMemoryProvider()
    first = _computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'})
    first.id, first.conversation = 'resp_1', Conversation(id='conv_1')
    client = MockOpenAIResponses.create_mock([first, _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(model, capabilities=[provider.capability(), ComputerUse(settle_seconds=0)], model_settings=settings)

    assert (await agent.run('Go.')).output == 'done'
    _, second = get_mock_responses_kwargs(client)
    # The call itself is not sent again: the API has it in the stored response or conversation.
    assert [item['type'] for item in second['input'] if 'type' in item] == ['computer_call_output']
    assert second['input'][0]['call_id'] == 'call_1'


async def test_a_call_with_safety_checks_is_sent_again_to_acknowledge_them(allow_model_requests: None) -> None:
    provider = InMemoryProvider()
    first = _computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}, safety_checks=[CHECK])
    first.id = 'resp_1'
    client = MockOpenAIResponses.create_mock([first, _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(
        model,
        capabilities=[provider.capability(), ComputerUse(settle_seconds=0)],
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto'),
    )

    assert (await agent.run('Go.')).output == 'done'
    _, second = get_mock_responses_kwargs(client)
    assert 'previous_response_id' not in second or not second['previous_response_id']
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    assert output['acknowledged_safety_checks'] == [CHECK]


async def test_tool_choice_naming_the_computer_names_the_native_tool(allow_model_requests: None) -> None:
    client = MockOpenAIResponses.create_mock([_done(), _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    computer = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'}, tool_kind='computer')
    note = ToolDefinition(name='note', parameters_json_schema={'type': 'object'})
    other = ToolDefinition(name='other', parameters_json_schema={'type': 'object'})
    params = ModelRequestParameters(function_tools=[computer, note, other], allow_text_output=True)
    prompt: list[ModelMessage] = [ModelRequest.user_text_prompt('Look.')]

    await model.request(prompt, {'tool_choice': ['computer']}, params)
    await model.request(prompt, {'tool_choice': ['computer', 'note']}, params)

    forced, allowed = get_mock_responses_kwargs(client)
    assert forced['tool_choice'] == {'type': 'computer'}
    assert allowed['tool_choice']['tools'] == [{'type': 'computer'}, {'type': 'function', 'name': 'note'}]


async def test_a_computer_tool_waiting_to_be_loaded_is_not_sent(allow_model_requests: None) -> None:
    client = MockOpenAIResponses.create_mock(_done())
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    computer = ToolDefinition(
        name='computer', parameters_json_schema={'type': 'object'}, tool_kind='computer', defer_loading=True
    )
    params = ModelRequestParameters(function_tools=[computer], allow_text_output=True)
    await model.request([ModelRequest.user_text_prompt('Look.')], None, params)

    (request,) = get_mock_responses_kwargs(client)
    assert 'tools' not in request or not request['tools']


async def test_the_native_tool_is_sent_and_its_call_answered_with_a_screenshot(allow_model_requests: None) -> None:
    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    await Agent(model, capabilities=[InMemoryProvider().capability(), ComputerUse(settle_seconds=0)]).run('Go.')

    first, second = get_mock_responses_kwargs(client)
    assert first['tools'] == [{'type': 'computer'}]
    (call,) = [item for item in second['input'] if item.get('type') == 'computer_call']
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    assert call['call_id'] == output['call_id'] == 'call_1'
    assert output['output']['image_url'].startswith('data:image/png;base64,')
    # A call that ran needs no explanation after its output.
    assert second['input'][-1] is output


async def test_a_screenshot_left_out_of_the_request_is_explained(allow_model_requests: None) -> None:
    look = {'type': 'screenshot'}
    client = MockOpenAIResponses.create_mock([_computer_call(look), _computer_call(look), _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    capabilities = [InMemoryProvider().capability(), ComputerUse(keep_screenshots=1)]
    await Agent(model, capabilities=capabilities).run('Look twice.')

    *_, last = get_mock_responses_kwargs(client)
    older, newest = [index for index, item in enumerate(last['input']) if item.get('type') == 'computer_call_output']
    # The older output's image is blank, so a note says why; the newest carries the real screenshot.
    assert last['input'][older + 1]['content'].endswith(
        '[Older screenshot removed to save context; the latest screenshot shows the current screen.]'
    )
    assert last['input'][newest]['output']['image_url'].startswith('data:image/png;base64,iVBOR')
    assert newest == len(last['input']) - 1


async def test_a_refused_action_is_explained_after_its_screenshot(allow_model_requests: None) -> None:
    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    provider = InMemoryProvider()
    await Agent(model, capabilities=[provider.capability(read_only=True), ComputerUse(settle_seconds=0)]).run('Go.')

    _, second = get_mock_responses_kwargs(client)
    output, note = second['input'][-2:]
    assert output['type'] == 'computer_call_output'
    assert note['role'] == 'user'
    assert note['content'].startswith('Action 1 of 1 (click) failed: This workspace is read-only')
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
            {'type': 'scroll', 'x': 1, 'y': 2, 'scroll_x': 0, 'scroll_y': -20},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'up', 'amount': 1, 'modifiers': []},
        ),
        # A scroll by nothing only moves the pointer there.
        (
            {'type': 'scroll', 'x': 1, 'y': 2, 'scroll_x': 0, 'scroll_y': 0},
            {'type': 'move', 'x': 1, 'y': 2, 'modifiers': []},
        ),
        ({'type': 'keypress', 'keys': ['CTRL', 'S']}, {'type': 'keypress', 'keys': ['CTRL', 'S']}),
        ({'type': 'type', 'text': 'hi'}, {'type': 'type', 'text': 'hi'}),
        ({'type': 'wait'}, {'type': 'wait'}),
        ({'type': 'screenshot'}, {'type': 'screenshot'}),
        ({'type': 'teleport', 'x': 1}, {'type': 'teleport', 'x': 1}),
    ],
)
def test_openai_actions_map_onto_the_computer_tool(action: dict[str, Any], mapped: dict[str, Any]) -> None:
    assert _map_computer_action(action) == [mapped]


def test_a_diagonal_scroll_scrolls_both_ways() -> None:
    action = {'type': 'scroll', 'x': 1, 'y': 2, 'scroll_x': -9000, 'scroll_y': 10}
    assert _map_computer_action(action) == [
        {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'left', 'amount': 50, 'modifiers': []},
        {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'down', 'amount': 1, 'modifiers': []},
    ]


@pytest.mark.parametrize(
    'model_name, native',
    [
        # The profile resolution matrix covers the rest.
        ('gpt-5.4-mini', True),
        ('gpt-6-luna', True),
        ('gpt-6.1-sol', True),
        ('gpt-5.2', False),
    ],
)
def test_the_native_tool_goes_to_the_models_that_accept_it(model_name: str, native: bool) -> None:
    assert openai_model_profile(model_name).get('openai_supports_computer_tool', False) is native
