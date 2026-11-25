"""`ComputerUse` against real provider APIs, sent as each provider's native computer tool.

The model adapter tests in `tests/models/` use a minimal `computer` tool, so the core test suite doesn't need the
Harness. These recordings run the real capability end to end, with its instructions, screenshot scaling and results.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterable

import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, AgentStreamEvent, RunContext, ToolCallPart, ToolReturn
from pydantic_ai.capabilities import Hooks, NativeTool, ValidatedToolArgs
from pydantic_ai.computer_use import Click, ComputerCallPart, MouseDown, MouseMove, MouseUp, TypeText
from pydantic_ai.messages import BinaryImage
from pydantic_ai.native_tools import AbstractNativeTool, WebSearchTool
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction
from tests.models.conftest import content_blocks, json_objects

from ...conftest import RequestCapture, try_import
from .display_fakes import ScreenProvider, png
from .shared_live import DONE, TASK, is_object_list, vcr_config as vcr_config

with try_import() as anthropic_installed:
    from anthropic import AsyncAnthropicBedrockMantle

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_installed:
    from anthropic import AsyncAnthropicBedrock

    # `AsyncAnthropicBedrock` imports this when a stream starts, and the test's event loop refuses imports there.
    importlib.import_module('botocore.parsers')

needs_anthropic = pytest.mark.skipif(not anthropic_installed(), reason='anthropic not installed')
needs_bedrock = pytest.mark.skipif(not bedrock_installed(), reason='anthropic or botocore not installed')


def _claude(model_name: str, anthropic_api_key: str, capture: RequestCapture) -> AnthropicModel:
    return AnthropicModel(model_name, provider=AnthropicProvider(api_key=anthropic_api_key, http_client=capture.client))


def _tools(body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """The request's tools, without the prompt-cache breakpoint that lands on the last of them."""
    return [{k: v for k, v in tool.items() if k != 'cache_control'} for tool in json_objects(body['tools'])]


def _results(body: dict[str, JsonValue]) -> list[tuple[JsonValue, list[JsonValue]]]:
    """Each tool result's error flag and its blocks: the text of a text block, and the type of any other block."""
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
    # Claude's toolset has no name or schema from us.
    assert _tools(first) == [{'type': 'computer_toolset_20260801'}]
    # Each toolset call is one `computer` call, answered by its own result.
    calls = content_blocks(last, 'tool_use')
    results = content_blocks(last, 'tool_result')
    assert [result['tool_use_id'] for result in results] == [call['id'] for call in calls]
    assert all(result['toolset_name'] == 'computer' for result in results)
    assert [call['name'] for call in calls] == snapshot(['screenshot', 'left_click', 'type'])
    assert _results(last) == snapshot(
        [
            (False, ['Ran 1 action. Screenshot attached (1280x800).', 'image']),
            (False, ['Ran 1 action. Screenshot attached (1280x800).', 'image']),
            (False, ['Ran 1 action. Screenshot attached (1280x800).', 'image']),
        ]
    )


@needs_anthropic
@pytest.mark.vcr
async def test_claude_sees_an_image_a_result_holds_beside_its_screenshot(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
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
        assert is_object_list(result.return_value)
        return ToolReturn([picture, *result.return_value], metadata=result.metadata)

    agent = Agent(
        _claude('claude-sonnet-5-5', anthropic_api_key, request_capture),
        capabilities=[provider.capability(), ComputerUse(), hooks],
    )
    result = await agent.run(
        'Take one screenshot. Besides the screenshot you get a small square image: reply with its colour only.'
    )

    assert 'red' in result.output.lower()
    last = request_capture.bodies('/v1/messages')[-1]
    # The call's result goes back unchanged: the image, the summary and the screen.
    *_, last_result = _results(last)
    assert last_result == (False, ['image', 'Ran 1 action. Screenshot attached (1280x800).', 'image'])


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
    # Before it is loaded, `computer` is a deferred function tool. After, it is Claude's toolset, which the history
    # can't name.
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
    # The capability sent the display size the earlier tool requires, and the beta it needs.
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

    # Claude puts the held key in the click's `text` field, as the toolset does.
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

    # Claude presses and releases the button at a point, in separate calls. The button stays pressed between them.
    assert provider.actions == [
        MouseMove(x=100, y=200),
        MouseDown(),
        MouseMove(x=300, y=400),
        MouseMove(x=300, y=400),
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
    'calls, actions',
    [
        pytest.param(
            'click at x=100 y=200, hold the shift key for 1 second, then type "b"', [Click(x=100, y=200)], id='mid-turn'
        ),
        pytest.param('hold the shift key for 1 second, then type "b"', [], id='first-call'),
    ],
)
async def test_claude_toolset_calls_stop_at_the_failed_call(
    allow_model_requests: None,
    anthropic_api_key: str,
    request_capture: RequestCapture,
    calls: str,
    actions: list[DisplayAction],
) -> None:
    """The key hold fails, and the calls after it in the same response don't run."""
    provider = ScreenProvider()
    model = _claude('claude-haiku-5-5', anthropic_api_key, request_capture)
    task = f'Make these computer calls in one turn, in this order: {calls}. Then reply "done". Do nothing else, and do not retry anything.'
    await Agent(model, capabilities=[provider.capability(), ComputerUse()]).run(task)

    assert provider.actions == actions
    # The failed call and the call after it are both answered as errors, with text only.
    *_, failed, not_run = _results(request_capture.bodies('/v1/messages')[1])
    assert failed == (True, ["the fake display cannot perform 'hold_keys'"])
    assert not_run == (True, ['Not executed: an earlier computer action in this turn failed.'])


@needs_anthropic
@pytest.mark.vcr
async def test_claude_reads_a_zoom_and_the_pointer_position_from_their_results(
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
        [
            (False, ['Ran 1 action. Screenshot attached (1280x800).', 'image', 'image']),
            (False, ['Ran 1 action. Screenshot attached (1280x800).', 'The pointer is at (640, 400).', 'image']),
        ]
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
        # The toolset has no name to force it with, so it is offered alone and a tool call is required.
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
        # Web search is left out too. Otherwise Claude could meet the requirement by searching.
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
    sent = [tool.get('type', tool.get('name')) for tool in _tools(first)]
    assert (sent, first['tool_choice']) == (tools, tool_choice)
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

    # They go back as plain `computer` calls and results, outside the toolset, which isn't in the request.
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
    # The toolset's calls went back as plain `computer` calls, and the new call as a `computer_20251124` call.
    inputs = json_objects([call['input'] for call in content_blocks(last, 'tool_use')])
    assert 'actions' in inputs[0] and inputs[-1] == {'action': 'left_click', 'coordinate': [10, 10]}
    assert 'toolset_name' not in str(last['messages'])


@needs_anthropic
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the second agent is on a model with another computer use tool')
async def test_claude_computer_20251124_calls_go_back_to_a_toolset_model(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    capabilities = [provider.capability(), ComputerUse()]
    earlier_model = _claude('claude-sonnet-4-6', anthropic_api_key, request_capture)
    result = await Agent(earlier_model, capabilities=capabilities).run(TASK)

    model = _claude('claude-haiku-5-5', anthropic_api_key, request_capture)
    await Agent(model, capabilities=capabilities).run(
        'Now click at x=10 y=10, then reply "done". Do nothing else.', message_history=result.all_messages()
    )

    assert provider.actions == [*DONE, Click(x=10, y=10)]
    last = request_capture.bodies('/v1/messages')[-1]
    assert _tools(last) == [{'type': 'computer_toolset_20260801'}]
    # The `computer_20251124` calls went back as plain `computer` calls, and the new call as a toolset call.
    calls = content_blocks(last, 'tool_use')
    assert 'actions' in json_objects([calls[0]['input']])[0] and 'toolset_name' not in calls[0]
    assert calls[-1]['toolset_name'] == 'computer'


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
