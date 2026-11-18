"""`ComputerUse` against real provider APIs, sent as each provider's native computer tool.

The model adapter tests in `tests/models/` use a minimal `computer` tool so the core suite doesn't need the Harness;
these recordings run the real capability, with its instructions, scaling and results, end to end.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterable
from typing import TypeGuard

import pytest
from pydantic import JsonValue

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    ApprovalRequired,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolCallPart,
    ToolDenied,
    ToolReturn,
)
from pydantic_ai.capabilities import Hooks, ValidatedToolArgs
from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    Click,
    ComputerCallPart,
    Drag,
    MouseMove,
    Point,
    PressKeys,
    Scroll,
    TypeText,
)
from pydantic_ai.messages import BinaryImage, CompactionPart, ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters, infer_model
from pydantic_ai.native_tools import ComputerTool
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction
from tests.models.conftest import json_objects

from ...conftest import RequestCapture, try_import
from ..display_fakes import FAKE_DISPLAY_SIZE, ScreenProvider, png

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@pytest.fixture(scope='module')
def vcr_config(vcr_config: dict[str, object], pytestconfig: pytest.Config) -> dict[str, object]:
    """Replay matches each request's body too, so a request the code no longer sends fails instead of replaying.

    Recording keeps the default match: recorded with the body matcher, cassetter writes the interactions out of order.
    """
    if pytestconfig.getoption('--record-mode', default=None) not in (None, 'none'):
        return vcr_config
    return {**vcr_config, 'match_on': ['method', 'uri', 'body']}


TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'
DONE = [Click(x=100, y=200), TypeText(text='hello')]


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _items(body: dict[str, JsonValue], type: str) -> list[dict[str, JsonValue]]:
    return [item for item in json_objects(body['input']) if item.get('type') == type]


def _gpt(openai_api_key: str, capture: RequestCapture, name: str = 'gpt-5.5') -> OpenAIResponsesModel:
    return OpenAIResponsesModel(name, provider=OpenAIProvider(api_key=openai_api_key, http_client=capture.client))


def _kinds(body: dict[str, JsonValue]) -> list[JsonValue]:
    return [item.get('type', item.get('role')) for item in json_objects(body['input'])]


def _computer_request(*extra_tools: str) -> ModelRequestParameters:
    """A request with a `computer` tool sent as the native one, beside function tools with these names."""
    computer = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'}, tool_kind=COMPUTER_KIND.id)
    others = [ToolDefinition(name=name, parameters_json_schema={'type': 'object'}) for name in extra_tools]
    return ModelRequestParameters(
        function_tools=[computer, *others],
        native_tools=[ComputerTool(display_size=FAKE_DISPLAY_SIZE)],
        allow_text_output=True,
    )


@pytest.mark.vcr
async def test_gpt_drives_the_screen_through_the_native_computer_tool(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    """The recording's `gpt-5.5` resolved to the `gpt-5.5-2026-04-23` snapshot."""
    provider = ScreenProvider()
    agent = Agent(_gpt(openai_api_key, request_capture), capabilities=[provider.capability(), ComputerUse()])

    async def drain(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    result = await agent.run(TASK, event_stream_handler=drain)

    assert result.response.model_name == 'gpt-5.5-2026-04-23'
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, *_, last = request_capture.bodies('/responses')
    # The capability's `computer` tool went out as OpenAI's own, not as a function.
    assert {'type': 'computer'} in json_objects(first['tools'])
    assert not any(tool.get('name') == 'computer' for tool in json_objects(first['tools']))
    # And its results came back as screenshots in `computer_call_output`s, at full detail.
    outputs = _items(last, 'computer_call_output')
    assert outputs and all(
        isinstance(screenshot := output['output'], dict) and screenshot['detail'] == 'original' for output in outputs
    )


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='loading the capability adds the native computer tool to `tools`')
async def test_gpt_drives_the_screen_after_loading_a_deferred_computer(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    capabilities = [provider.capability(), ComputerUse(defer_loading=True)]
    await Agent(_gpt(openai_api_key, request_capture), capabilities=capabilities).run(TASK)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, second, *_ = bodies = request_capture.bodies('/responses')
    # Before loading, the model has only `load_capability`; after it, the native computer tool. The API refuses a
    # function named `computer`, in `tools` or in an `additional_tools` item.
    assert {'type': 'computer'} not in json_objects(first['tools'])
    assert {'type': 'computer'} in json_objects(second['tools'])
    for body in bodies:
        declared = json_objects(body['tools'])
        declared += [tool for item in _items(body, 'additional_tools') for tool in json_objects(item['tools'])]
        assert not any(tool.get('name') == 'computer' for tool in declared)


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the second request is from an agent without the computer tool')
async def test_gpt_computer_calls_go_back_to_a_request_without_the_computer_tool(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture)
    result = await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]

    # An agent without the computer tool, such as an advisor that is forwarded the history, sends the calls back.
    answer = await Agent(model).run(
        'Where did you click? Reply with just the coordinates.', message_history=result.all_messages()
    )
    assert '100' in answer.output and '200' in answer.output

    # As function calls without their `cu_` ids, which the API refuses on a function call, answered by function
    # call outputs.
    last = request_capture.bodies('/responses')[-1]
    kinds = [item.get('type') for item in json_objects(last['input'])]
    assert 'computer_call' not in kinds and 'computer_call_output' not in kinds
    calls = _items(last, 'function_call')
    assert calls and all(call['name'] == 'computer' and 'id' not in call for call in calls)


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the second agent sends the computer tool as a function, not the native tool')
async def test_gpt_continues_a_native_history_after_the_tool_opts_out(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    """A run recorded with the native tool goes on with `ComputerUse(native=False)`, and the API accepts it."""
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture)
    capabilities = [provider.capability(), ComputerUse()]
    result = await Agent(model, capabilities=capabilities).run(TASK)

    opted_out = [provider.capability(), ComputerUse(native=False)]
    await Agent(model, capabilities=opted_out).run(
        'Now press Enter, then reply "done". Do nothing else.', message_history=result.all_messages()
    )

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello'), PressKeys(keys=('enter',))]
    first_native, *_, last = request_capture.bodies('/responses')
    assert {'type': 'computer'} in json_objects(first_native['tools'])
    # The opted-out agent declares a function, and the earlier native calls go back as function calls without
    # their `cu_` ids, after the reasoning that made them.
    assert [tool['name'] for tool in json_objects(last['tools'])] == ['computer']
    kinds = [item.get('type') for item in json_objects(last['input'])]
    assert 'computer_call' not in kinds and 'computer_call_output' not in kinds
    calls = _items(last, 'function_call')
    assert calls and all(call['name'] == 'computer' for call in calls)
    assert not any(str(call.get('id', '')).startswith('cu_') for call in calls)
    assert 'reasoning' in kinds


@pytest.mark.vcr
async def test_gpt_on_bedrock_mantle_drives_the_screen_through_the_native_computer_tool(
    allow_model_requests: None, request_capture: RequestCapture
) -> None:
    mantle = BedrockMantleProvider(
        region_name='us-east-1',
        api_key=os.getenv('AWS_BEARER_TOKEN_BEDROCK', 'mock-api-key'),
        http_client=request_capture.client,
    )
    model = infer_model('bedrock-mantle:openai.gpt-5.5', lambda _: mantle)
    provider = ScreenProvider()
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)

    assert provider.actions == DONE
    first, *_, last = request_capture.bodies('/responses')
    assert {'type': 'computer'} in json_objects(first['tools'])
    assert _items(last, 'computer_call_output')


@pytest.mark.vcr
async def test_gpt_gets_the_whole_history_back_with_store_off(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture, 'gpt-5.4-mini')
    settings = OpenAIResponsesModelSettings(openai_store=False)
    await Agent(model, capabilities=[provider.capability(), ComputerUse()], model_settings=settings).run(TASK)

    assert provider.actions == DONE
    last = request_capture.bodies('/responses')[-1]
    # Nothing is stored, so every call goes back with its output.
    assert last['store'] is False and 'previous_response_id' not in last
    assert _kinds(last) == ['user', 'computer_call', 'computer_call_output', 'computer_call', 'computer_call_output']


@pytest.mark.vcr
async def test_gpt_gets_only_the_outputs_after_a_stored_response(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture, 'gpt-5.4-mini')
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    await Agent(model, capabilities=[provider.capability(), ComputerUse()], model_settings=settings).run(TASK)

    assert provider.actions == DONE
    _, *later = request_capture.bodies('/responses')
    # The server holds each call, so a request carries only the call's output.
    assert later and all('previous_response_id' in body and _kinds(body) == ['computer_call_output'] for body in later)


@pytest.mark.vcr
async def test_gpt_compacts_a_history_of_computer_calls_through_the_compact_endpoint(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture, 'gpt-5.4-mini')
    result = await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)
    # `OpenAICompaction` compacts all but the newest request, which leaves a call without its output mid-run, and the
    # endpoint refuses that for any tool, so this compacts the finished run.
    context = ModelRequestContext(
        model=model, messages=result.all_messages(), model_settings=None, model_request_parameters=_computer_request()
    )
    compacted = await model.compact_messages(context)

    assert isinstance(compacted.parts[0], CompactionPart)
    compact = request_capture.body('/responses/compact')
    # The endpoint refuses a `computer_call_output` unless the computer tool is enabled.
    assert compact['tools'] == [{'type': 'computer'}]
    assert 'computer_call_output' in _kinds(compact)


@pytest.mark.vcr
async def test_gpt_compacts_computer_and_function_calls_through_the_compact_endpoint(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture, 'gpt-5.4-mini')

    def note(text: str) -> str:
        """Write down a note."""
        return 'Noted.'

    agent = Agent(model, tools=[note], capabilities=[provider.capability(), ComputerUse()])
    result = await agent.run(
        'Take a screenshot, then call `note` with "seen", click at x=100 y=200, and reply "done". Do nothing else.'
    )
    assert provider.actions == [Click(x=100, y=200)]
    context = ModelRequestContext(
        model=model,
        messages=result.all_messages(),
        model_settings=None,
        model_request_parameters=_computer_request('note'),
    )
    compacted = await model.compact_messages(context)

    assert isinstance(compacted.parts[0], CompactionPart)
    compact = request_capture.body('/responses/compact')
    # The endpoint takes the function calls with only the computer tool declared.
    assert compact['tools'] == [{'type': 'computer'}]
    kinds = _kinds(compact)
    assert 'computer_call_output' in kinds and 'function_call' in kinds and 'function_call_output' in kinds


@pytest.mark.vcr
async def test_gpt_gets_only_the_outputs_in_a_conversation(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    from openai import AsyncOpenAI

    conversation = await AsyncOpenAI(api_key=openai_api_key, http_client=request_capture.client).conversations.create()
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture, 'gpt-5.4-mini')
    settings = OpenAIResponsesModelSettings(openai_conversation_id=conversation.id)
    await Agent(model, capabilities=[provider.capability(), ComputerUse()], model_settings=settings).run(TASK)

    assert provider.actions == DONE
    _, *later = request_capture.bodies('/responses')
    # The conversation holds each call, so a request carries only the call's output.
    assert later and all(
        body['conversation'] == conversation.id and _kinds(body) == ['computer_call_output'] for body in later
    )


@pytest.mark.vcr
async def test_gpt_compacts_computer_calls_the_server_holds_through_the_compact_endpoint(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture, 'gpt-5.4-mini')
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    capabilities = [provider.capability(), ComputerUse()]
    result = await Agent(model, capabilities=capabilities, model_settings=settings).run(TASK)
    assert provider.actions == DONE
    # Up to the last screenshot, so the window ends with a result of a call the server holds.
    *messages, final = result.all_messages()
    assert isinstance(final, ModelResponse) and not final.tool_calls
    context = ModelRequestContext(
        model=model, messages=messages, model_settings=settings, model_request_parameters=_computer_request()
    )
    compacted = await model.compact_messages(context)

    assert isinstance(compacted.parts[0], CompactionPart)
    compact = request_capture.body('/responses/compact')
    # Only the result goes, as a `computer_call_output` beside the computer tool, after the stored response.
    assert 'previous_response_id' in compact
    assert compact['tools'] == [{'type': 'computer'}]
    assert _kinds(compact) == ['computer_call_output']


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='compaction replaces the history')
async def test_gpt_compacts_a_history_of_computer_calls_on_the_server(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture, 'gpt-5.4-mini')
    capabilities = [provider.capability(), ComputerUse(), OpenAICompaction(token_threshold=1000)]
    result = await Agent(model, capabilities=capabilities).run(TASK)

    assert provider.actions == DONE
    assert any(isinstance(part, CompactionPart) for message in result.all_messages() for part in message.parts)
    # A later request starts from the compaction and goes on with the computer calls after it.
    last = request_capture.bodies('/responses')[-1]
    assert _kinds(last)[0] == 'compaction' and 'computer_call_output' in _kinds(last)


@pytest.mark.vcr
@pytest.mark.parametrize(
    ('tool_choice', 'tools', 'sent'),
    [
        pytest.param(['computer'], ('note',), {'type': 'computer'}, id='only-the-computer'),
        pytest.param(
            ['computer', 'note'],
            ('note', 'other'),
            {
                'type': 'allowed_tools',
                'mode': 'required',
                'tools': [{'type': 'computer'}, {'type': 'function', 'name': 'note'}],
            },
            id='the-computer-and-a-function',
        ),
    ],
)
async def test_gpt_calls_the_computer_when_tool_choice_names_it(
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
    tool_choice: list[str],
    tools: tuple[str, ...],
    sent: dict[str, JsonValue],
) -> None:
    model = _gpt(openai_api_key, request_capture, 'gpt-5.4-mini')
    response = await model.request(
        [ModelRequest.user_text_prompt('Look at the screen.')], {'tool_choice': tool_choice}, _computer_request(*tools)
    )

    assert request_capture.body('/responses')['tool_choice'] == sent
    assert [type(part) for part in response.parts if not part.part_kind == 'thinking'] == [ComputerCallPart]


@pytest.mark.vcr
async def test_gpt_uses_the_computer_beside_a_function_found_by_tool_search(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()

    def get_weather(city: str) -> str:
        """The weather in a city."""
        return 'sunny'

    agent = Agent(
        _gpt(openai_api_key, request_capture, 'gpt-5.4'),
        tools=[Tool(get_weather, defer_loading=True)],
        capabilities=[provider.capability(), ComputerUse()],
    )
    await agent.run('First find and call the weather tool for Paris. Then click at x=100 y=200, and reply "done".')

    assert provider.actions == [Click(x=100, y=200)]
    first, *_, last = request_capture.bodies('/responses')
    assert {'type': 'computer'} in json_objects(first['tools'])
    assert {'tool_search_call', 'function_call', 'computer_call_output'} <= set(_kinds(last))


@pytest.mark.vcr
async def test_gpt_is_told_why_a_denied_call_has_a_blank_screenshot(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    hooks = Hooks[object]()

    @hooks.on.before_tool_execute
    async def ask_before_acting(
        ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        if tool_def.tool_kind == COMPUTER_KIND.id and not ctx.tool_call_approved:
            if any(action.type != 'screenshot' for action in args['actions']):
                raise ApprovalRequired
        return args

    agent = Agent(
        _gpt(openai_api_key, request_capture, 'gpt-5.4-mini'),
        capabilities=[provider.capability(), ComputerUse(), hooks],
        output_type=[str, DeferredToolRequests],
    )
    paused = await agent.run('Click at x=100 y=200. If you cannot, reply with the reason you were given.')
    assert isinstance(paused.output, DeferredToolRequests)
    (approval,) = paused.output.approvals
    denied = DeferredToolResults(approvals={approval.tool_call_id: ToolDenied('The user said no.')})
    result = await agent.run(message_history=paused.all_messages(), deferred_tool_results=denied)

    assert provider.actions == []
    assert isinstance(result.output, str) and 'said no' in result.output.lower()
    last = request_capture.bodies('/responses')[-1]
    (output,) = [item for item in _items(last, 'computer_call_output') if item['call_id'] == approval.tool_call_id]
    note = json_objects(last['input'])[json_objects(last['input']).index(output) + 1]
    # The API needs an image, so a 1x1 one stands in, and the note says why.
    assert isinstance(screenshot := output['output'], dict)
    assert str(screenshot['image_url']).startswith('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB')
    assert note['role'] == 'user' and 'The user said no.' in str(note['content'])


@pytest.mark.vcr
async def test_gpt_sees_an_image_a_result_holds_beside_its_screenshot(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    hooks = Hooks[object]()
    picture = BinaryImage(data=png(64, 64, red_box=(0, 0, 64, 64)), media_type='image/png')

    @hooks.on.after_tool_execute
    async def add_picture(
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: ToolReturn,
    ) -> ToolReturn:
        # As an app might add an image of its own to the result, before the screenshot.
        assert _is_object_list(result.return_value)
        return ToolReturn([picture, *result.return_value], metadata=result.metadata)

    agent = Agent(
        _gpt(openai_api_key, request_capture, 'gpt-5.4-mini'),
        capabilities=[provider.capability(), ComputerUse(), hooks],
    )
    result = await agent.run(
        'Take one screenshot. Besides the screenshot you get a small square image: reply with its colour only.'
    )

    assert 'red' in result.output.lower()
    last = request_capture.bodies('/responses')[-1]
    message = json_objects(last['input'])[-1]
    # The output can carry only the screenshot, so the image goes in a message after it.
    assert isinstance(content := message['content'], list)
    assert [part['type'] for part in content if isinstance(part, dict)] == ['input_text', 'input_image']


@pytest.mark.vcr
async def test_gpt_is_told_how_a_pixel_scroll_ran(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    agent = Agent(
        _gpt(openai_api_key, request_capture, 'gpt-5.4-mini'), capabilities=[provider.capability(), ComputerUse()]
    )
    await agent.run('Scroll down by exactly 250 pixels at x=640 y=400 in one scroll action, then reply "done".')

    assert provider.actions == [Scroll(x=640, y=400, direction='down', amount=2)]
    last = request_capture.bodies('/responses')[-1]
    assert 'scroll_y=250 ran as 2 wheel clicks of about 114 pixels each.' in str(json_objects(last['input'])[-1])


@pytest.mark.vcr
async def test_gpt_computer_call_cut_off_by_the_token_limit_has_no_actions(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    model = _gpt(openai_api_key, request_capture, 'gpt-5.5')
    task = 'Click the search box and type this exact text: ' + ' '.join(f'word{i}' for i in range(200))
    settings = OpenAIResponsesModelSettings(max_tokens=16, openai_reasoning_effort='none', tool_choice=['computer'])
    response = await model.request([ModelRequest.user_text_prompt(task)], settings, _computer_request())

    assert response.finish_reason == 'length'
    (call,) = [part for part in response.parts if isinstance(part, ComputerCallPart)]
    # OpenAI sends the cut-off call with no actions.
    assert call.actions is None
    stored = (call.provider_details or {})['computer_call']
    assert stored['status'] == 'incomplete'
    assert 'actions' not in stored


# Each step and the action it should reach the display as, run in this order in one test to keep the recording small.
ACTION_STEPS: list[tuple[str, DisplayAction]] = [
    ('Double-click at x=100 y=200.', Click(x=100, y=200, count=2)),
    ('Right-click at x=100 y=200.', Click(x=100, y=200, button='right')),
    ('Middle-click at x=100 y=200.', Click(x=100, y=200, button='middle')),
    ('Press the mouse back button at x=100 y=200.', Click(x=100, y=200, button='back')),
    ('Hold shift and click at x=100 y=200.', Click(x=100, y=200, modifiers=('shift',))),
    ('Move the pointer to x=100 y=200 without clicking.', MouseMove(x=100, y=200)),
    ('Scroll down by 300 pixels at x=100 y=200.', Scroll(x=100, y=200, direction='down', amount=3)),
    (
        'Drag in a straight line from x=100 y=200 to x=300 y=200.',
        Drag(path=(Point(x=100, y=200), Point(x=300, y=200))),
    ),
    ('Press ctrl+a.', PressKeys(keys=('ctrl', 'a'))),
    ('Type "hello".', TypeText(text='hello')),
]


@pytest.mark.vcr
async def test_gpt_actions_reach_the_screen(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    """Each kind of action GPT sends reaches the display as the matching `DisplayAction`."""
    provider = ScreenProvider()
    agent = Agent(_gpt(openai_api_key, request_capture), capabilities=[provider.capability(), ComputerUse()])
    steps = '\n'.join(f'{number}. {task}' for number, (task, _) in enumerate(ACTION_STEPS, 1))
    await agent.run(
        'Take a screenshot, then do these steps in order, all in one computer call:\n'
        f'{steps}\nDo only that, then reply "done".'
    )

    assert provider.actions == [expected for _, expected in ACTION_STEPS]
