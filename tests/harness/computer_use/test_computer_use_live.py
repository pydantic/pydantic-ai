"""`ComputerUse` against real provider APIs, sent as each provider's native computer tool.

The model adapter tests in `tests/models/` use a minimal `computer` tool so the core suite doesn't need the Harness;
these recordings run the real capability, with its instructions, scaling and results, end to end.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.computer_use import Click, PressKeys, TypeText
from pydantic_ai_harness.computer_use import ComputerUse
from tests.models.conftest import content_blocks

from ...conftest import RequestCapture, try_import
from ..display_fakes import ScreenProvider

with try_import() as openai_installed:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_installed:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

needs_openai = pytest.mark.skipif(not openai_installed(), reason='openai not installed')
needs_anthropic = pytest.mark.skipif(not anthropic_installed(), reason='anthropic not installed')

TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'


def _items(body: dict[str, Any], type: str) -> list[dict[str, Any]]:
    return [item for item in cast(list[dict[str, Any]], body['input']) if item.get('type') == type]


def _gpt(openai_api_key: str, capture: RequestCapture) -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key=openai_api_key, http_client=capture.client))


@needs_openai
@pytest.mark.vcr
async def test_gpt_drives_the_screen_through_the_native_computer_tool(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    agent = Agent(
        _gpt(openai_api_key, request_capture), capabilities=[provider.capability(), ComputerUse(settle_seconds=0)]
    )

    async def drain(ctx: Any, events: Any) -> None:
        async for _ in events:
            pass

    await agent.run(TASK, event_stream_handler=drain)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, *_, last = request_capture.bodies('/responses')
    # The capability's `computer` tool went out as OpenAI's own, not as a function.
    assert {'type': 'computer'} in cast(list[Any], first['tools'])
    assert not any(tool.get('name') == 'computer' for tool in cast(list[dict[str, Any]], first['tools']))
    # And its results came back as screenshots in `computer_call_output`s, at full detail.
    outputs = _items(last, 'computer_call_output')
    assert outputs and all(output['output']['detail'] == 'original' for output in outputs)


@needs_openai
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='the second request is from an agent without the computer tool')
async def test_gpt_computer_calls_go_back_to_a_request_without_the_computer_tool(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _gpt(openai_api_key, request_capture)
    result = await Agent(model, capabilities=[provider.capability(), ComputerUse(settle_seconds=0)]).run(TASK)
    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]

    # An agent without the computer tool, such as an advisor that is forwarded the history, sends the calls back.
    answer = await Agent(model).run(
        'Where did you click? Reply with just the coordinates.', message_history=result.all_messages()
    )
    assert '100' in answer.output and '200' in answer.output

    # As function calls without their `cu_` ids, which the API refuses on a function call.
    calls = _items(request_capture.bodies('/responses')[-1], 'function_call')
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
    capabilities = [provider.capability(), ComputerUse(settle_seconds=0)]
    result = await Agent(model, capabilities=capabilities).run(TASK)

    opted_out = [provider.capability(), ComputerUse(settle_seconds=0, native=False)]
    await Agent(model, capabilities=opted_out).run(
        'Now press Enter, then reply "done". Do nothing else.', message_history=result.all_messages()
    )

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello'), PressKeys(keys=('enter',))]
    first_native, *_, last = request_capture.bodies('/responses')
    assert {'type': 'computer'} in cast(list[Any], first_native['tools'])
    # The opted-out agent declares a function, and the earlier native calls go back as function calls without
    # their `cu_` ids, after the reasoning that made them.
    assert [tool['name'] for tool in cast(list[dict[str, Any]], last['tools'])] == ['computer']
    kinds = [item.get('type') for item in cast(list[dict[str, Any]], last['input'])]
    assert 'computer_call' not in kinds and 'computer_call_output' not in kinds
    calls = _items(last, 'function_call')
    assert calls and all(call['name'] == 'computer' for call in calls)
    assert not any(str(call.get('id', '')).startswith('cu_') for call in calls)
    assert 'reasoning' in kinds


def _claude(model_name: str, anthropic_api_key: str, capture: RequestCapture) -> AnthropicModel:
    return AnthropicModel(model_name, provider=AnthropicProvider(api_key=anthropic_api_key, http_client=capture.client))


def _tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The request's tools, without the prompt-cache breakpoint that lands on the last of them."""
    return [
        {k: v for k, v in tool.items() if k != 'cache_control'} for tool in cast(list[dict[str, Any]], body['tools'])
    ]


@needs_anthropic
@pytest.mark.vcr
async def test_claude_drives_the_screen_through_the_computer_toolset(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-sonnet-5-5', anthropic_api_key, request_capture)
    await Agent(model, capabilities=[provider.capability(), ComputerUse(settle_seconds=0)]).run(TASK)

    assert provider.actions == [Click(x=100, y=200), TypeText(text='hello')]
    first, *_, last = request_capture.bodies('/v1/messages')
    # The capability's `computer` tool went out as Claude's toolset, which has no name or schema of ours.
    assert _tools(first) == [{'type': 'computer_toolset_20260801'}]
    # Every member call got its own result, in order and in the toolset: `OK` on each that ran, and the screenshot on
    # the last of a turn's calls.
    calls = content_blocks(last, 'tool_use')
    results = content_blocks(last, 'tool_result')
    assert [result['tool_use_id'] for result in results] == [call['id'] for call in calls]
    assert all(result['toolset_name'] == 'computer' for result in results)
    assert [
        [cast(dict[str, Any], block)['type'] for block in cast(list[Any], result['content'])] for result in results
    ] == snapshot([['text'], ['text'], ['text', 'image']])


@needs_anthropic
@pytest.mark.vcr
async def test_claude_sonnet_4_6_gets_computer_20251124_sized_to_the_display(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
) -> None:
    provider = ScreenProvider()
    model = _claude('claude-sonnet-4-6', anthropic_api_key, request_capture)
    await Agent(model, capabilities=[provider.capability(), ComputerUse(settle_seconds=0)]).run(TASK)

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
