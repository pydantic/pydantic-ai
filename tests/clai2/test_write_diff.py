"""Successful filesystem writes display their native proposed diff."""

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai_harness.coder import Coder
from rich.console import Console

from pydantic_clai2 import Session, StreamRenderer


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('existing', [False, True])
async def test_write_diff_through_agent(tmp_path: Path, existing: bool) -> None:
    path = tmp_path / 'example.txt'
    if existing:
        path.write_text('old content\n')
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=True)
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            yield {0: DeltaToolCall(name='write_file', json_args='{"path":"example.txt","content":"new content\\n"}')}
        else:
            yield 'done'

    agent = Agent(FunctionModel(stream_function=respond), capabilities=[Coder(tmp_path)])
    session = Session(agent, deps=None, on_stream_event=renderer.on_stream_event)
    await session.prompt('write the file')
    assert path.read_text() == 'new content\n'
    assert '+new content' in output.getvalue()
    assert output.getvalue().count('● write_file example.txt') == 1
    assert 'Wrote ' not in output.getvalue()
    assert ('-old content' in output.getvalue()) == existing
