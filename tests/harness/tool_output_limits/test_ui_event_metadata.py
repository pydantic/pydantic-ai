"""A spill must not drop the UI events a tool returns in `ToolReturn.metadata`.

AG-UI and Vercel AI both read a list of events/chunks from `metadata` and skip any other item.
Before the fix, `ToolOutputLimits` wrapped such a list in a dict under `original_metadata`, so
every event disappeared as soon as the return crossed a `Spill` band.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.tool_output_limits import Band, LocalFileStore, Spill, ToolOutputLimits
from pydantic_ai_harness.tool_output_limits._capability import _with_handles  # pyright: ignore[reportPrivateUsage]
from tests.conftest import try_import

with try_import() as ag_ui_imports_successful:
    from ag_ui.core import CustomEvent, EventType, RunAgentInput, UserMessage

    from pydantic_ai.ui.ag_ui import AGUIAdapter

with try_import() as vercel_imports_successful:
    from pydantic_ai.ui.vercel_ai._utils import iter_metadata_chunks
    from pydantic_ai.ui.vercel_ai.response_types import DataChunk


def test_with_handles_appends_to_a_list() -> None:
    marker = object()
    assert _with_handles([marker], 'h/1.0', 42) == [marker, {'overflow_handle': 'h/1.0', 'overflow_bytes': 42}]


def test_with_handles_turns_a_tuple_into_a_list() -> None:
    assert _with_handles(('a', 'b'), None, 0, 'h/1.0.content') == [
        'a',
        'b',
        {'overflow_content_handle': 'h/1.0.content'},
    ]


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart(tool_name='search', args={}, tool_call_id='c1')])
    return ModelResponse(parts=[TextPart('done')])


async def _stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='search', json_args='{}', tool_call_id='c1')}
    else:
        yield 'done'


def _agent(tmp_path: Path, metadata: object) -> Agent[None, str]:
    cap: ToolOutputLimits[object] = ToolOutputLimits(
        bands=[Band(over=100, action=Spill())], store=LocalFileStore(base_dir=tmp_path)
    )
    agent = Agent(FunctionModel(_respond, stream_function=_stream), capabilities=[cap])

    @agent.tool_plain
    def search() -> ToolReturn:
        return ToolReturn(return_value={'rows': list(range(100))}, metadata=metadata)

    return agent


@pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed')
async def test_spilled_return_keeps_ag_ui_events(tmp_path: Path) -> None:
    event = CustomEvent(type=EventType.CUSTOM, name='cards', value={'n': 100})
    run_input = RunAgentInput(
        thread_id='t',
        run_id='r',
        messages=[UserMessage(id='u', role='user', content='hi')],
        state={},
        tools=[],
        context=[],
        forwarded_props={},
    )
    adapter = AGUIAdapter(agent=_agent(tmp_path, [event]), run_input=run_input)
    events = [e async for e in adapter.run_stream()]

    customs = [e for e in events if isinstance(e, CustomEvent)]
    assert [(e.name, e.value) for e in customs] == [('cards', {'n': 100})]
    assert list(tmp_path.rglob('*')), 'the return was not spilled, so this test proves nothing'


@pytest.mark.skipif(not vercel_imports_successful(), reason='vercel ai dependencies not installed')
async def test_spilled_return_keeps_vercel_ai_chunks(tmp_path: Path) -> None:
    chunk = DataChunk(type='data-search-results', data={'count': 100})
    result = await _agent(tmp_path, [chunk]).run('hi')

    part = next(
        p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart) and p.tool_name == 'search'
    )
    assert list(iter_metadata_chunks(part)) == [chunk]
    metadata: list[object] = part.metadata
    assert metadata[0] == chunk
    assert isinstance(metadata[-1], dict) and 'overflow_handle' in metadata[-1]
