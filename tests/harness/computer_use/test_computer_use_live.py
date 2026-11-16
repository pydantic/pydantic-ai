"""`ComputerUse` against real provider APIs, sent as each provider's native computer tool.

The model adapter tests in `tests/models/` use a minimal `computer` tool so the core suite doesn't need the Harness;
these recordings run the real capability, with its instructions, scaling and results, end to end.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterable

import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.computer_use import Click, ComputerCallPart, PressKeys, TypeText
from pydantic_ai.native_tools import AbstractNativeTool, WebSearchTool
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction
from tests.models.conftest import content_blocks, json_objects

from ...conftest import RequestCapture, try_import
from ..display_fakes import ScreenProvider

with try_import() as openai_installed:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_installed:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_installed:
    from anthropic import AsyncAnthropicBedrock

    # `AsyncAnthropicBedrock` imports this as a stream starts, and the test's event loop refuses an import there.
    importlib.import_module('botocore.parsers')

needs_openai = pytest.mark.skipif(not openai_installed(), reason='openai not installed')
needs_anthropic = pytest.mark.skipif(not anthropic_installed(), reason='anthropic not installed')
needs_bedrock = pytest.mark.skipif(not bedrock_installed(), reason='anthropic or botocore not installed')

TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'


def _objects(body: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    """The objects in a request body's list under `key`, such as `tools` or `input`."""
    value = body[key]
    assert isinstance(value, list)
    return [item for item in value if isinstance(item, dict)]


def _items(body: dict[str, JsonValue], type: str) -> list[dict[str, JsonValue]]:
    return [item for item in _objects(body, 'input') if item.get('type') == type]


def _gpt(openai_api_key: str, capture: RequestCapture) -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key=openai_api_key, http_client=capture.client))


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
    # Every member call got its own result, in order and in the toolset. Claude's first batch put a screenshot before
    # its click, which was refused, as the screen is captured only after a batch; it then took the screenshot alone and
    # clicked and typed in a second batch. Each call that ran got `OK`, and the last of a batch the screenshot.
    calls = content_blocks(last, 'tool_use')
    results = content_blocks(last, 'tool_result')
    assert [result['tool_use_id'] for result in results] == [call['id'] for call in calls]
    assert all(result['toolset_name'] == 'computer' for result in results)
    assert [[block['type'] for block in json_objects(result['content'])] for result in results] == snapshot(
        [['text'], ['text'], ['text'], ['text', 'image'], ['text'], ['text', 'image']]
    )


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
    calls = content_blocks(last, 'tool_use')
    results = content_blocks(last, 'tool_result')
    assert [
        (call['name'], [block['type'] for block in json_objects(result['content'])])
        for call, result in zip(calls, results)
    ] == snapshot([('zoom', ['text', 'image']), ('cursor_position', ['text', 'text', 'image'])])


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
