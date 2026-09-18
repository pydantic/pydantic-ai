"""Grep previews retain invocation context and distinguish truncation sources."""

import io

import pytest
from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from rich.console import Console

from pydantic_clai2 import StreamRenderer


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('tool_truncated', [False, True])
async def test_grep_preview(tool_truncated: bool) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=True, grep_lines=1)
    await renderer.on_stream_event(
        FunctionToolCallEvent(
            part=ToolCallPart(
                tool_name='grep',
                args={'pattern': 'needle', 'path': '/tmp/example'},
                tool_call_id='search',
            )
        )
    )
    assert "grep 'needle' in '/tmp/example'" in output.getvalue()
    result = 'example:1:first\nexample:2:second\nexample:3:third'
    if tool_truncated:
        result += '\n[truncated; narrow the search]'
    await renderer.on_stream_event(
        FunctionToolResultEvent(
            part=ToolReturnPart(
                tool_name='grep',
                content=result,
                tool_call_id='search',
            )
        )
    )
    text = output.getvalue()
    assert 'example:1:first' in text and 'example:2:second' not in text
    assert 'Truncated 2 result lines' in text
    assert ('additional result count unknown' in text) == tool_truncated
    assert text.endswith('\n\n')


async def test_grep_no_matches() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=True)
    await renderer.on_stream_event(
        FunctionToolCallEvent(
            part=ToolCallPart(
                tool_name='grep',
                args={'pattern': 'missing'},
                tool_call_id='search',
            )
        )
    )
    await renderer.on_stream_event(
        FunctionToolResultEvent(
            part=ToolReturnPart(
                tool_name='grep',
                content='',
                tool_call_id='search',
            )
        )
    )
    assert 'No matches.' in output.getvalue()
