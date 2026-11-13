"""`ComputerUse` against real provider APIs, sent as each provider's native computer tool.

The model adapter tests in `tests/models/` use a minimal `computer` tool so the core suite doesn't need the Harness;
these recordings run the real capability, with its instructions, scaling and results, end to end.
"""

from __future__ import annotations

from collections.abc import AsyncIterable

import pytest
from pydantic import JsonValue

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.computer_use import Click, PressKeys, TypeText
from pydantic_ai_harness.computer_use import ComputerUse

from ...conftest import RequestCapture, try_import
from ..display_fakes import ScreenProvider

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

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

    # As function calls without their `cu_` ids, which the API refuses on a function call.
    calls = _items(request_capture.bodies('/responses')[-1], 'function_call')
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
