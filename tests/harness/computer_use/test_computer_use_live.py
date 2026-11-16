"""`ComputerUse` against real provider APIs, sent as each provider's native computer tool.

The model adapter tests in `tests/models/` use a minimal `computer` tool so the core suite doesn't need the Harness;
these recordings run the real capability, with its instructions, scaling and results, end to end.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterable
from typing import TypeGuard

import pytest
from inline_snapshot import snapshot
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
from pydantic_ai.capabilities import Hooks, NativeTool, ValidatedToolArgs
from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    Click,
    ComputerCallPart,
    Drag,
    MouseDown,
    MouseUp,
    MoveMouse,
    Point,
    PressKeys,
    Scroll,
    TypeText,
)
from pydantic_ai.messages import BinaryImage, CompactionPart, ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters, infer_model
from pydantic_ai.native_tools import AbstractNativeTool, ComputerTool, WebSearchTool
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction
from tests.models.conftest import content_blocks, json_objects

from ...conftest import RequestCapture, try_import
from ..display_fakes import FAKE_DISPLAY_SIZE, ScreenProvider, png

with try_import() as openai_installed:
    from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_installed:
    from anthropic import AsyncAnthropicBedrockMantle

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_installed:
    from anthropic import AsyncAnthropicBedrock

    # `AsyncAnthropicBedrock` imports this as a stream starts, and the test's event loop refuses an import there.
    importlib.import_module('botocore.parsers')

needs_openai = pytest.mark.skipif(not openai_installed(), reason='openai not installed')
needs_anthropic = pytest.mark.skipif(not anthropic_installed(), reason='anthropic not installed')
needs_bedrock = pytest.mark.skipif(not bedrock_installed(), reason='anthropic or botocore not installed')


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


def _objects(body: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    """The objects in a request body's list under `key`, such as `tools` or `input`."""
    value = body[key]
    assert isinstance(value, list)
    return [item for item in value if isinstance(item, dict)]


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _items(body: dict[str, JsonValue], type: str) -> list[dict[str, JsonValue]]:
    return [item for item in _objects(body, 'input') if item.get('type') == type]


def _gpt(openai_api_key: str, capture: RequestCapture, name: str = 'gpt-5.5') -> OpenAIResponsesModel:
    return OpenAIResponsesModel(name, provider=OpenAIProvider(api_key=openai_api_key, http_client=capture.client))


def _kinds(body: dict[str, JsonValue]) -> list[JsonValue]:
    return [item.get('type', item.get('role')) for item in _objects(body, 'input')]


def _computer_request(*extra_tools: str) -> ModelRequestParameters:
    """A request with a `computer` tool sent as the native one, beside function tools with these names."""
    computer = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'}, tool_kind=COMPUTER_KIND.id)
    others = [ToolDefinition(name=name, parameters_json_schema={'type': 'object'}) for name in extra_tools]
    return ModelRequestParameters(
        function_tools=[computer, *others],
        native_tools=[ComputerTool(display_size=FAKE_DISPLAY_SIZE)],
        allow_text_output=True,
    )


@needs_openai
@pytest.mark.vcr
async def test_gpt_drives_the_screen_through_the_native_computer_tool(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    agent = Agent(_gpt(openai_api_key, request_capture), capabilities=[provider.capability(), ComputerUse()])

    async def drain(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    await agent.run(TASK, event_stream_handler=drain)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, *_, last = request_capture.bodies('/responses')
    # The capability's `computer` tool went out as OpenAI's own, not as a function.
    assert {'type': 'computer'} in _objects(first, 'tools')
    assert not any(tool.get('name') == 'computer' for tool in _objects(first, 'tools'))
    # And its results came back as screenshots in `computer_call_output`s, at full detail.
    outputs = _items(last, 'computer_call_output')
    assert outputs and all(
        isinstance(screenshot := output['output'], dict) and screenshot['detail'] == 'original' for output in outputs
    )


@needs_openai
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
    assert {'type': 'computer'} not in _objects(first, 'tools')
    assert {'type': 'computer'} in _objects(second, 'tools')
    for body in bodies:
        declared = _objects(body, 'tools')
        declared += [tool for item in _items(body, 'additional_tools') for tool in _objects(item, 'tools')]
        assert not any(tool.get('name') == 'computer' for tool in declared)


@needs_openai
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
    kinds = [item.get('type') for item in _objects(last, 'input')]
    assert 'computer_call' not in kinds and 'computer_call_output' not in kinds
    calls = _items(last, 'function_call')
    assert calls and all(call['name'] == 'computer' and 'id' not in call for call in calls)


@needs_openai
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
    assert {'type': 'computer'} in _objects(first_native, 'tools')
    # The opted-out agent declares a function, and the earlier native calls go back as function calls without
    # their `cu_` ids, after the reasoning that made them.
    assert [tool['name'] for tool in _objects(last, 'tools')] == ['computer']
    kinds = [item.get('type') for item in _objects(last, 'input')]
    assert 'computer_call' not in kinds and 'computer_call_output' not in kinds
    calls = _items(last, 'function_call')
    assert calls and all(call['name'] == 'computer' for call in calls)
    assert not any(str(call.get('id', '')).startswith('cu_') for call in calls)
    assert 'reasoning' in kinds


def _claude(model_name: str, anthropic_api_key: str, capture: RequestCapture) -> AnthropicModel:
    return AnthropicModel(model_name, provider=AnthropicProvider(api_key=anthropic_api_key, http_client=capture.client))


def _tools(body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """The request's tools, without the prompt-cache breakpoint that lands on the last of them."""
    return [{k: v for k, v in tool.items() if k != 'cache_control'} for tool in json_objects(body['tools'])]


def _results(body: dict[str, JsonValue]) -> list[tuple[JsonValue, list[JsonValue]]]:
    """Each tool result's error flag and its blocks: a text block's text, any other block's type."""
    return [
        (result['is_error'], [block['text'] if block['type'] == 'text' else block['type'] for block in blocks])
        for result in content_blocks(body, 'tool_result')
        for blocks in [json_objects(result['content'])]
    ]


@needs_anthropic
@pytest.mark.vcr
async def test_claude_drives_the_screen_through_the_computer_toolset(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-sonnet-5-5', anthropic_api_key, request_capture)
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, *_, last = request_capture.bodies('/v1/messages')
    # The capability's `computer` tool went out as Claude's toolset, which has no name or schema of ours.
    assert _tools(first) == [{'type': 'computer_toolset_20260801'}]
    # Claude put a screenshot before its click and typing, and the batch ran in one go: one request after the first.
    assert len(request_capture.bodies('/v1/messages')) == 2
    # Every member call got its own result, in order and in the toolset: `OK`, with the screen as the screenshot call
    # found it, and the last call the screen after the batch.
    calls = content_blocks(last, 'tool_use')
    results = content_blocks(last, 'tool_result')
    assert [result['tool_use_id'] for result in results] == [call['id'] for call in calls]
    assert all(result['toolset_name'] == 'computer' for result in results)
    assert [call['name'] for call in calls] == snapshot(['screenshot', 'left_click', 'type'])
    assert _results(last) == snapshot([(False, ['OK', 'image']), (False, ['OK']), (False, ['OK', 'image'])])


@needs_anthropic
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='loading the capability swaps the deferred `computer` tool for the toolset')
async def test_claude_drives_the_screen_after_loading_a_deferred_computer(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-sonnet-5-5', anthropic_api_key, request_capture)
    await Agent(model, capabilities=[provider.capability(), ComputerUse(defer_loading=True)]).run(TASK)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, second, *_ = bodies = request_capture.bodies('/v1/messages')
    # Before loading, `computer` is a deferred function; after it, Claude's toolset, which history can't reference.
    assert {'name': 'computer', 'defer_loading': True}.items() <= _tools(first)[-1].items()
    assert {'type': 'computer_toolset_20260801'} in _tools(second)
    for body in bodies:
        assert content_blocks(body, 'tool_addition') == []


@needs_anthropic
@pytest.mark.vcr
async def test_claude_sonnet_4_6_gets_computer_20251124_sized_to_the_display(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-sonnet-4-6', anthropic_api_key, request_capture)
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    # The capability gave the display size the earlier tool must declare, and the beta it needs went out.
    assert _tools(request_capture.bodies('/v1/messages')[0]) == [
        {
            'type': 'computer_20251124',
            'name': 'computer',
            'display_width_px': 1280,
            'display_height_px': 800,
            'enable_zoom': True,
        }
    ]
    assert 'computer-use-2025-11-24' in request_capture.headers[0]['anthropic-beta']


@needs_anthropic
@pytest.mark.vcr
async def test_claude_haiku_4_5_gets_computer_20250124_sized_to_the_display(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-haiku-4-5', anthropic_api_key, request_capture)
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    assert _tools(request_capture.bodies('/v1/messages')[0]) == [
        {'type': 'computer_20250124', 'name': 'computer', 'display_width_px': 1280, 'display_height_px': 800}
    ]
    assert 'computer-use-2025-01-24' in request_capture.headers[0]['anthropic-beta']


@needs_anthropic
@pytest.mark.vcr
async def test_claude_drives_the_screen_through_the_streamed_computer_toolset(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-haiku-5-5', anthropic_api_key, request_capture)

    async def drain(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK, event_stream_handler=drain)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    last = request_capture.bodies('/v1/messages')[-1]
    # The streamed member calls went back as they came, each answered in order.
    calls = content_blocks(last, 'tool_use')
    results = content_blocks(last, 'tool_result')
    assert calls and all(call['toolset_name'] == 'computer' for call in calls)
    assert [result['tool_use_id'] for result in results] == [call['id'] for call in calls]


@needs_anthropic
@pytest.mark.vcr
async def test_claude_shift_clicks_through_computer_20251124(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-sonnet-4-6', anthropic_api_key, request_capture)
    task = 'Take a screenshot, then shift+click at x=100 y=200, then reply "done". Do nothing else.'
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(task)

    # Claude holds the modifier in the click's `text`, as the toolset does.
    assert provider.actions == [Click(x=100, y=200, modifiers=('shift',))]
    calls = content_blocks(request_capture.bodies('/v1/messages')[-1], 'tool_use')
    assert {'action': 'left_click', 'coordinate': [100, 200], 'text': 'shift'} in [call['input'] for call in calls]


@needs_anthropic
@pytest.mark.vcr
async def test_claude_sonnet_4_6_drags_with_a_button_held_across_calls(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-sonnet-4-6', anthropic_api_key, request_capture)
    task = (
        'Take a screenshot. Then, one action per step: press and hold the left mouse button at x=100 y=200 with '
        '`left_mouse_down`, move the pointer to x=300 y=400 with `mouse_move`, release it there with `left_mouse_up`, '
        'and reply "done". Do nothing else.'
    )
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(task)

    # Claude presses and releases at a point, in calls of their own: the button stays pressed between them.
    assert provider.actions == [
        MoveMouse(x=100, y=200),
        MouseDown(),
        MoveMouse(x=300, y=400),
        MoveMouse(x=300, y=400),
        MouseUp(),
    ]
    bodies = request_capture.bodies('/v1/messages')
    assert [call['input'] for call in content_blocks(bodies[-1], 'tool_use')][1:] == [
        {'action': 'left_mouse_down', 'coordinate': [100, 200]},
        {'action': 'mouse_move', 'coordinate': [300, 400]},
        {'action': 'left_mouse_up', 'coordinate': [300, 400]},
    ]
    assert [is_error for body in bodies[1:] for is_error, _ in _results(body)[-1:]] == [False] * 4


@needs_anthropic
@pytest.mark.vcr
@pytest.mark.parametrize(
    'calls, actions, results',
    [
        pytest.param(
            'click at x=100 y=200, hold the shift key for 1 second, then type "b"',
            [Click(x=100, y=200)],
            # The click ran and carries the screen, the hold failed, and the typing never ran.
            [
                (False, ['OK', 'image']),
                (True, ["the fake display cannot perform 'hold_key'"]),
                (True, ['Not executed: an earlier computer action in this turn failed.']),
            ],
            id='mid-batch',
        ),
        pytest.param(
            'hold the shift key for 1 second, then type "b"',
            [],
            # Nothing ran before the hold, so it carries the screen, which an error result can't hold.
            [
                (False, ["Failed: the fake display cannot perform 'hold_key'", 'image']),
                (True, ['Not executed: an earlier computer action in this turn failed.']),
            ],
            id='first-call',
        ),
    ],
)
async def test_claude_toolset_batch_stops_at_the_failed_call(
    allow_model_requests: None,
    anthropic_api_key: str,
    request_capture: RequestCapture,
    calls: str,
    actions: list[DisplayAction],
    results: list[tuple[bool, list[str]]],
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-haiku-5-5', anthropic_api_key, request_capture)
    task = f'Make these computer calls in one turn, in this order: {calls}. Then reply "done". Do nothing else, and do not retry anything.'
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(task)

    assert provider.actions == actions
    assert _results(request_capture.bodies('/v1/messages')[-1]) == results


@needs_anthropic
@pytest.mark.vcr
async def test_claude_zoom_and_cursor_position_answer_their_own_calls(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-haiku-5-5', anthropic_api_key, request_capture)
    task = (
        'In one turn, zoom into the region from (0, 0) to (200, 100), then read the pointer position. '
        'Then reply with the pointer position. Do nothing else.'
    )
    result = await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(task)

    assert '640' in result.output and '400' in result.output
    last = request_capture.bodies('/v1/messages')[-1]
    assert [call['name'] for call in content_blocks(last, 'tool_use')] == snapshot(['zoom', 'cursor_position'])
    assert _results(last) == snapshot(
        [(False, ['OK', 'image']), (False, ['OK', 'The pointer is at (640, 400).', 'image'])]
    )


def _force_the_computer_until_it_is_called(ctx: RunContext[object]) -> ModelSettings:
    called = any(isinstance(part, ComputerCallPart) for message in ctx.messages for part in message.parts)
    return {} if called else {'tool_choice': ['computer']}


def note(text: str) -> str:
    """Write down a note."""
    return 'Noted.'


@needs_anthropic
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='forcing the toolset offers it alone, and later requests offer every tool')
@pytest.mark.parametrize(
    'model_name, native_tools, task, tools, tool_choice',
    [
        # The toolset has no name to force it by, so it's offered alone and a call required.
        pytest.param(
            'claude-haiku-5-5', [], 'Reply "done".', ['computer_toolset_20260801'], {'type': 'any'}, id='toolset'
        ),
        pytest.param(
            'claude-sonnet-4-6',
            [],
            'Reply "done".',
            ['note', 'computer_20251124'],
            {'type': 'tool', 'name': 'computer'},
            id='computer_20251124',
        ),
        # Web search is left out too: requiring a tool would let Claude search instead.
        pytest.param(
            'claude-haiku-5-5',
            [WebSearchTool(max_uses=1)],
            'Search the web for the capital of France, then reply with it.',
            ['computer_toolset_20260801'],
            {'type': 'any'},
            id='toolset-beside-web-search',
        ),
    ],
)
async def test_claude_is_forced_to_call_the_computer(
    allow_model_requests: None,
    anthropic_api_key: str,
    request_capture: RequestCapture,
    model_name: str,
    native_tools: list[AbstractNativeTool],
    task: str,
    tools: list[str],
    tool_choice: dict[str, JsonValue],
) -> None:
    provider = ScreenProvider()
    model = _claude(model_name, anthropic_api_key, request_capture)
    capabilities = [provider.capability(), ComputerUse(), *(NativeTool(tool) for tool in native_tools)]
    result = await Agent(model, tools=[note], capabilities=capabilities).run(
        task, model_settings=_force_the_computer_until_it_is_called
    )

    first = request_capture.bodies('/v1/messages')[0]
    # A function tool by its name, any other by its type.
    sent = [tool.get('type', tool.get('name')) for tool in _tools(first)]
    assert (sent, first['tool_choice']) == (tools, tool_choice)
    # Claude's first response called the computer.
    assert any(isinstance(part, ComputerCallPart) for part in result.all_messages()[1].parts)


@needs_anthropic
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the second request is from an agent without the computer tool')
async def test_claude_toolset_calls_go_back_to_a_request_without_the_computer_tool(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-haiku-5-5', anthropic_api_key, request_capture)
    result = await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]

    # An agent without the computer tool, such as an advisor that is forwarded the history, sends the calls back.
    answer = await Agent(model).run(
        'Where did you click? Reply with just the coordinates.', message_history=result.all_messages()
    )
    assert '100' in answer.output and '200' in answer.output

    # As plain `computer` calls and results, outside the toolset, which isn't in the request.
    last = request_capture.bodies('/v1/messages')[-1]
    assert 'tools' not in last
    calls = content_blocks(last, 'tool_use')
    assert calls and all(call['name'] == 'computer' and 'toolset_name' not in call for call in calls)
    assert all('toolset_name' not in result for result in content_blocks(last, 'tool_result'))


@needs_anthropic
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the second agent is on a model with another computer use tool')
async def test_claude_toolset_calls_go_back_to_a_computer_20251124_model(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    capabilities = [provider.capability(), ComputerUse()]
    toolset_model = _claude('claude-haiku-5-5', anthropic_api_key, request_capture)
    result = await Agent(toolset_model, capabilities=capabilities).run(TASK)

    model = _claude('claude-sonnet-4-6', anthropic_api_key, request_capture)
    await Agent(model, capabilities=capabilities).run(
        'Now click at x=10 y=10, then reply "done". Do nothing else.', message_history=result.all_messages()
    )

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello'), Click(x=10, y=10)]
    last = request_capture.bodies('/v1/messages')[-1]
    assert [tool['type'] for tool in _tools(last)] == ['computer_20251124']
    # The toolset's calls went back as plain `computer` calls, and the new one as `computer_20251124`'s own.
    inputs = json_objects([call['input'] for call in content_blocks(last, 'tool_use')])
    assert 'actions' in inputs[0] and inputs[-1] == {'action': 'left_click', 'coordinate': [10, 10]}
    assert 'toolset_name' not in str(last['messages'])


@needs_anthropic
@needs_bedrock
@pytest.mark.vcr
async def test_claude_on_bedrock_drives_the_screen_through_the_computer_toolset(
    allow_model_requests: None, request_capture: RequestCapture
) -> None:
    client = AsyncAnthropicBedrock(
        api_key=os.environ.get('AWS_BEARER_TOKEN_BEDROCK', 'test-bedrock-token'),
        aws_region='us-east-1',
        http_client=request_capture.client,
    )
    provider = ScreenProvider()
    model = AnthropicModel('us.anthropic.claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=client))
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    # Bedrock takes the toolset, as the Claude API does.
    assert _tools(request_capture.bodies()[0]) == [{'type': 'computer_toolset_20260801'}]


@needs_anthropic
@pytest.mark.vcr
async def test_claude_on_the_bedrock_messages_api_drives_the_screen_through_the_computer_toolset(
    allow_model_requests: None, request_capture: RequestCapture
) -> None:
    client = AsyncAnthropicBedrockMantle(
        api_key=os.environ.get('AWS_BEARER_TOKEN_BEDROCK', 'test-bedrock-token'),
        aws_region='us-east-1',
        http_client=request_capture.client,
    )
    provider = ScreenProvider()
    model = AnthropicModel('anthropic.claude-opus-5-5', provider=AnthropicProvider(anthropic_client=client))
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(TASK)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    assert _tools(request_capture.bodies()[0]) == [{'type': 'computer_toolset_20260801'}]


@needs_openai
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
    assert {'type': 'computer'} in _objects(first, 'tools')
    assert _items(last, 'computer_call_output')


@needs_openai
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


@needs_openai
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


@needs_openai
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


@needs_openai
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


@needs_openai
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


@needs_openai
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
    assert {'type': 'computer'} in _objects(first, 'tools')
    assert {'tool_search_call', 'function_call', 'computer_call_output'} <= set(_kinds(last))


@needs_openai
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
    note = _objects(last, 'input')[_objects(last, 'input').index(output) + 1]
    # The API needs an image, so a 1x1 one stands in, and the note says why.
    assert isinstance(screenshot := output['output'], dict)
    assert str(screenshot['image_url']).startswith('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB')
    assert note['role'] == 'user' and 'The user said no.' in str(note['content'])


@needs_openai
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
    message = _objects(last, 'input')[-1]
    # The output can carry only the screenshot, so the image goes in a message after it.
    assert isinstance(content := message['content'], list)
    assert [part['type'] for part in content if isinstance(part, dict)] == ['input_text', 'input_image']


@needs_openai
@pytest.mark.vcr
async def test_gpt_is_told_how_a_pixel_scroll_ran(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    agent = Agent(
        _gpt(openai_api_key, request_capture, 'gpt-5.4-mini'), capabilities=[provider.capability(), ComputerUse()]
    )
    await agent.run('Scroll down by exactly 250 pixels at x=640 y=400 in one scroll action, then reply "done".')

    assert provider.actions == [Scroll(x=640, y=400, direction='down', amount=3)]
    last = request_capture.bodies('/responses')[-1]
    assert 'scroll_y=250 ran as 3 wheel clicks of about 100 pixels each.' in str(_objects(last, 'input')[-1])


@needs_openai
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


@needs_openai
@pytest.mark.vcr
@pytest.mark.parametrize(
    ('task', 'expected'),
    [
        pytest.param('Double-click at x=100 y=200.', Click(x=100, y=200, count=2), id='double-click'),
        pytest.param('Right-click at x=100 y=200.', Click(x=100, y=200, button='right'), id='right-click'),
        pytest.param('Middle-click at x=100 y=200.', Click(x=100, y=200, button='middle'), id='middle-click'),
        pytest.param('Press the mouse back button at x=100 y=200.', Click(x=100, y=200, button='back'), id='back'),
        pytest.param(
            'Hold shift and click at x=100 y=200.', Click(x=100, y=200, modifiers=('shift',)), id='shift-click'
        ),
        pytest.param('Move the pointer to x=100 y=200 without clicking.', MoveMouse(x=100, y=200), id='move'),
        pytest.param(
            'Scroll down by 300 pixels at x=100 y=200.',
            Scroll(x=100, y=200, direction='down', amount=3),
            id='scroll',
        ),
        pytest.param(
            'Drag in a straight line from x=100 y=200 to x=300 y=200.',
            Drag(path=(Point(x=100, y=200), Point(x=300, y=200))),
            id='drag',
        ),
        pytest.param('Press ctrl+a.', PressKeys(keys=('ctrl', 'a')), id='keypress'),
        pytest.param('Type "hello".', TypeText(text='hello'), id='type'),
    ],
)
async def test_gpt_actions_reach_the_screen(
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
    task: str,
    expected: DisplayAction,
) -> None:
    provider = ScreenProvider()
    agent = Agent(_gpt(openai_api_key, request_capture), capabilities=[provider.capability(), ComputerUse()])
    await agent.run(f'{task} Do only that, then reply "done".')

    assert provider.actions == [expected]
