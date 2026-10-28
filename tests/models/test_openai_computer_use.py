"""OpenAI Responses' native `computer` tool, sent for a tool of the `computer` kind."""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, Tool, ToolCallPart, ToolDenied
from pydantic_ai.computer_use import COMPUTER_KIND, ComputerToolAction
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import (
    BinaryImage,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import Click, TypeText

from ..conftest import RequestCapture, try_import
from ..workspace_fakes import FAKE_SCREENSHOT, InMemoryProvider, computer_toolset
from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

with try_import() as imports_successful:
    from openai.types import responses as resp
    from openai.types.responses.response import Conversation

    from pydantic_ai.models.openai import (
        _BLANK_SCREENSHOT,  # pyright: ignore[reportPrivateUsage]
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
        _map_computer_action,  # pyright: ignore[reportPrivateUsage]
        _map_computer_call_output,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='openai not installed'), pytest.mark.anyio]

TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'


def _agent(openai_api_key: str, provider: InMemoryProvider, **settings: Any) -> Agent[None, str]:
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key=openai_api_key))
    return Agent(model, toolsets=[computer_toolset()], capabilities=[provider.capability(**settings)])


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
async def test_gpt_computer_calls_run_on_the_workspace(
    allow_model_requests: None, openai_api_key: str, stream: bool
) -> None:
    provider = InMemoryProvider()
    agent = _agent(openai_api_key, provider)

    async def drain(ctx: Any, events: Any) -> None:
        async for _ in events:
            pass

    await agent.run(TASK, event_stream_handler=drain if stream else None)

    # GPT-5.5's real `computer_call`s became `computer` calls that ran.
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the second request is from an agent without the computer tool')
async def test_computer_calls_go_back_to_a_request_without_the_computer_tool(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = InMemoryProvider()
    result = await _agent(openai_api_key, provider).run(TASK)
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]

    # An agent without the computer tool, such as an advisor that is forwarded the history, sends the calls back.
    model = OpenAIResponsesModel(
        'gpt-5.5', provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )
    answer = await Agent(model).run(
        'Where did you click? Reply with just the coordinates.', message_history=result.all_messages()
    )
    assert '100' in answer.output and '200' in answer.output

    # As function calls without their `cu_` ids, which the API refuses on a function call.
    calls = [
        item
        for item in cast(list[dict[str, Any]], request_capture.body('/responses')['input'])
        if item.get('type') == 'function_call'
    ]
    assert calls and all(call['name'] == 'computer' and 'id' not in call for call in calls)


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
    response = response_message([item])
    response.id = 'resp_1'
    return response


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
async def test_a_reviewer_sees_the_safety_checks_and_approving_acknowledges_them(
    allow_model_requests: None, approve: bool
) -> None:
    provider = InMemoryProvider()
    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}, safety_checks=[CHECK]), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(
        model,
        toolsets=[computer_toolset(requires_approval=True)],
        capabilities=[provider.capability()],
        output_type=[str, DeferredToolRequests],
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto'),
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
    # The flagged call is sent again rather than continued from, so its output can acknowledge or refuse it.
    assert 'previous_response_id' not in second
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    if approve:
        assert output['acknowledged_safety_checks'] == [CHECK]
        assert provider.actions == [Click(x=5, y=6)]
    else:
        assert 'acknowledged_safety_checks' not in output
        note = second['input'][second['input'].index(output) + 1]
        assert note == {'role': 'user', 'content': 'Not on this page.'}
        assert provider.actions == []


async def test_an_approved_call_in_a_conversation_acknowledges_its_safety_checks(allow_model_requests: None) -> None:
    provider = InMemoryProvider()
    first = _computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}, safety_checks=[CHECK])
    first.conversation = Conversation(id='conv_1')
    client = MockOpenAIResponses.create_mock([first, _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(
        model,
        toolsets=[computer_toolset(requires_approval=True)],
        capabilities=[provider.capability()],
        output_type=[str, DeferredToolRequests],
        model_settings=OpenAIResponsesModelSettings(openai_conversation_id='conv_1'),
    )
    paused = await agent.run('Go.')
    assert isinstance(paused.output, DeferredToolRequests)
    (approval,) = paused.output.approvals

    await agent.run(
        message_history=paused.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={approval.tool_call_id: True}),
    )
    _, second = get_mock_responses_kwargs(client)
    # The call stays in the conversation; its output still acknowledges the checks.
    (output,) = second['input']
    assert output['acknowledged_safety_checks'] == [CHECK]


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
    first.conversation = Conversation(id='conv_1')
    client = MockOpenAIResponses.create_mock([first, _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    agent = Agent(model, toolsets=[computer_toolset()], capabilities=[provider.capability()], model_settings=settings)

    assert (await agent.run('Go.')).output == 'done'
    _, second = get_mock_responses_kwargs(client)
    # The call itself is not sent again: the API has it in the stored response or conversation.
    assert [item['type'] for item in second['input'] if 'type' in item] == ['computer_call_output']
    assert second['input'][0]['call_id'] == 'call_1'


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
    assert 'tools' not in request


async def test_the_native_tool_is_sent_and_a_call_with_no_actions_is_answered_with_a_screenshot(
    allow_model_requests: None,
) -> None:
    client = MockOpenAIResponses.create_mock([_computer_call(), _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    result = await Agent(model, toolsets=[computer_toolset()], capabilities=[InMemoryProvider().capability()]).run(
        'Look.'
    )

    (call_part,) = [p for message in result.all_messages() for p in message.parts if isinstance(p, ToolCallPart)]
    assert call_part.args_as_dict() == {'actions': [{'type': 'screenshot'}]}
    first, second = get_mock_responses_kwargs(client)
    assert first['tools'] == [{'type': 'computer'}]
    (call,) = [item for item in second['input'] if item.get('type') == 'computer_call']
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    assert call['call_id'] == output['call_id'] == 'call_1'
    assert output['output']['image_url'] == f'data:image/png;base64,{base64.b64encode(FAKE_SCREENSHOT).decode()}'
    # A call that ran needs no explanation after its output.
    assert second['input'][-1] is output


async def test_a_call_with_a_single_action_runs_it(allow_model_requests: None) -> None:
    response = _computer_call()
    item = cast(resp.ResponseComputerToolCall, response.output[0])
    item.actions = None
    item.action = resp.response_computer_tool_call.ActionClick(type='click', x=5, y=6, button='left')
    provider = InMemoryProvider()
    client = MockOpenAIResponses.create_mock([response, _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    await Agent(model, toolsets=[computer_toolset()], capabilities=[provider.capability()]).run('Go.')
    assert provider.actions == [Click(x=5, y=6)]


def test_a_result_without_its_screenshot_gets_a_blank_image_and_a_note() -> None:
    # As after `keep_screenshots` dropped the screenshot from this request: the output still needs an image.
    note = '[Older screenshot removed to save context; the latest screenshot shows the current screen.]'
    part = ToolReturnPart('computer', ['Ran 1 action.', note], tool_call_id='call_1', metadata={'performed': 1})
    output, message = cast(list[dict[str, Any]], _map_computer_call_output(part, 'call_1', None))
    assert output['output']['image_url'] == _BLANK_SCREENSHOT
    assert message == {'role': 'user', 'content': f'Ran 1 action. {note}'}


def test_the_last_image_goes_out_as_the_screen_at_full_detail() -> None:
    zoomed = BinaryImage(data=b'zoomed', media_type='image/png')
    screen = BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png')
    part = ToolReturnPart(
        'computer', ['Ran 1 action.', zoomed, screen], tool_call_id='call_1', metadata={'performed': 1}
    )
    (output,) = cast(list[dict[str, Any]], _map_computer_call_output(part, 'call_1', None))
    assert output['output'] == {'type': 'computer_screenshot', 'image_url': screen.data_uri, 'detail': 'original'}


@pytest.mark.parametrize(
    'text,metadata',
    [
        # As from a UI that rebuilt the result: its text may say an action failed.
        pytest.param('Action 1 of 1 (click) failed: busy.', None, id='no-metadata'),
        # Otherwise the model would take its own actions to have run.
        pytest.param(
            'Before it ran, this call was changed to: [...]. Ran 1 action.',
            {'performed': 1, 'changed': True},
            id='changed-before-it-ran',
        ),
    ],
)
def test_a_result_the_screenshot_does_not_tell_also_sends_its_text(text: str, metadata: dict[str, Any] | None) -> None:
    screen = BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png')
    part = ToolReturnPart('computer', [text, screen], tool_call_id='call_1', metadata=metadata)
    _, message = cast(list[dict[str, Any]], _map_computer_call_output(part, 'call_1', None))
    assert message == {'role': 'user', 'content': text}


async def test_an_incomplete_call_runs_nothing(allow_model_requests: None) -> None:
    response = _computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'})
    cast(resp.ResponseComputerToolCall, response.output[0]).status = 'incomplete'
    provider = InMemoryProvider()
    client = MockOpenAIResponses.create_mock([response, _done()])
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    await Agent(model, toolsets=[computer_toolset()], capabilities=[provider.capability()]).run('Go.')

    assert provider.actions == []
    _, second = get_mock_responses_kwargs(client)
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    assert output['output']['image_url'] == _BLANK_SCREENSHOT


async def test_an_action_with_a_field_that_does_not_map_runs_nothing(allow_model_requests: None) -> None:
    # A long press would otherwise run as a plain click, and the value would be stored in the history.
    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left', 'duration_ms': 800}), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    provider = InMemoryProvider()
    result = await Agent(model, toolsets=[computer_toolset()], capabilities=[provider.capability()]).run('Go.')

    assert provider.actions == []
    call = next(part for part in result.all_messages()[1].parts if isinstance(part, ToolCallPart))
    assert call.args_as_dict() == {'actions': [{'type': 'click', 'unsupported_fields': ['duration_ms']}]}


def test_the_blank_screenshot_is_a_valid_png() -> None:
    # OpenAI rejects an image whose chunk checksums don't match, failing the whole request.
    data = base64.b64decode(_BLANK_SCREENSHOT.split(',', 1)[1])
    position, chunk = 8, b''
    while position < len(data):
        (length,) = struct.unpack('>I', data[position : position + 4])
        chunk = data[position + 4 : position + 8 + length]
        assert struct.unpack('>I', data[position + 8 + length : position + 12 + length])[0] == zlib.crc32(chunk)
        position += 12 + length
    assert chunk == b'IEND'


async def test_a_refused_action_is_explained_after_its_screenshot(allow_model_requests: None) -> None:
    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    provider = InMemoryProvider()
    await Agent(model, toolsets=[computer_toolset()], capabilities=[provider.capability(read_only=True)]).run('Go.')

    _, second = get_mock_responses_kwargs(client)
    output, note = second['input'][-2:]
    assert output['type'] == 'computer_call_output'
    assert note['role'] == 'user'
    assert note['content'].startswith('Ran 0 of 1 actions. The next one failed: This workspace is read-only')
    assert provider.actions == []


async def test_a_failed_call_sends_a_blank_screenshot_and_says_why(allow_model_requests: None) -> None:
    async def computer(actions: list[ComputerToolAction]) -> ToolReturn:
        raise ToolFailed('Taking the screenshot failed: scrot: not found')

    client = MockOpenAIResponses.create_mock(
        [_computer_call({'type': 'click', 'x': 5, 'y': 6, 'button': 'left'}), _done()]
    )
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(openai_client=client))
    await Agent(model, tools=[Tool(computer, tool_kind=COMPUTER_KIND)]).run('Go.')

    _, second = get_mock_responses_kwargs(client)
    (output,) = [item for item in second['input'] if item.get('type') == 'computer_call_output']
    assert output['output']['image_url'] == _BLANK_SCREENSHOT
    note = second['input'][second['input'].index(output) + 1]
    assert note == {'role': 'user', 'content': 'Taking the screenshot failed: scrot: not found'}


async def test_a_model_without_the_native_tool_gets_a_function(allow_model_requests: None) -> None:
    client = MockOpenAIResponses.create_mock(_done())
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(openai_client=client))
    await Agent(model, toolsets=[computer_toolset()], capabilities=[InMemoryProvider().capability()]).run('Go.')

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
    agent = Agent(model, toolsets=[computer_toolset()], capabilities=[InMemoryProvider().capability()])
    await agent.run('Go on.', message_history=history)

    (request,) = get_mock_responses_kwargs(client)
    assert [item.get('type') for item in request['input']] == snapshot(
        [None, 'function_call', 'function_call_output', None]
    )


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
        (
            {'type': 'scroll', 'x': 1, 'y': 2, 'scroll_x': 200, 'scroll_y': 0},
            {'type': 'scroll', 'x': 1, 'y': 2, 'direction': 'right', 'amount': 2, 'modifiers': []},
        ),
        ({'type': 'keypress', 'keys': ['CTRL', 'S']}, {'type': 'keypress', 'keys': ['CTRL', 'S']}),
        ({'type': 'type', 'text': 'hi'}, {'type': 'type', 'text': 'hi'}),
        ({'type': 'wait'}, {'type': 'wait'}),
        ({'type': 'teleport', 'x': 1}, {'type': 'teleport', 'unsupported_fields': ['x']}),
        # Under the tool's own type, so the retry names the field rather than refusing the type.
        (
            {'type': 'double_click', 'x': 1, 'y': 2, 'duration_ms': 5},
            {'type': 'click', 'unsupported_fields': ['duration_ms']},
        ),
        (
            {'type': 'drag', 'path': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4, 'pressure': 1}]},
            {'type': 'drag', 'unsupported_fields': ['path']},
        ),
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
