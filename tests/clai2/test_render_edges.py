"""Malformed and unmatched tool events remain safe to display."""

import io
from dataclasses import dataclass

import pytest
from pydantic_ai import CapabilityEvent, FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent, PartStartEvent
from pydantic_ai.messages import (
    RetryPromptPart,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileEditedEvent, FileWrittenEvent
from rich.console import Console

from pydantic_clai2 import StreamRenderer


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass(kw_only=True)
class Notice(CapabilityEvent, namespace='test'):
    pass


@pytest.mark.parametrize('show_tool_output', [False, True])
async def test_render_edge_events(show_tool_output: bool) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=show_tool_output)
    await renderer.on_stream_event(Notice())
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 123})))
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('edit_file', {'path': 'file'}, tool_call_id='edit'))
    )
    await renderer.on_stream_event(
        FileEditedEvent(
            path='file', root_dir='/tmp', content_hash='hash', diff='diff', truncated=True, tool_call_id='edit'
        )
    )
    for tool in ('grep', 'shell'):
        await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(tool, '{', tool_call_id=tool)))
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart('read_file', {})))
    for content in (
        RetryPromptPart('retry', tool_name='grep', tool_call_id='g'),
        ToolReturnPart('grep', {}, tool_call_id='g'),
    ):
        await renderer.on_stream_event(
            FunctionToolCallEvent(part=ToolCallPart('grep', {'pattern': 'x'}, tool_call_id='g'))
        )
        await renderer.on_stream_event(FunctionToolResultEvent(part=content))
    for operation in ('write', 'create_directory'):
        await renderer.on_stream_event(
            FileChangeRequestEvent(
                path='file',
                root_dir='/tmp',
                operation='write' if operation == 'write' else 'create_directory',
                diff='',
                truncated=False,
                tool_call_id='write',
            )
        )
    await renderer.on_stream_event(
        FileWrittenEvent(path='file', root_dir='/tmp', content_hash='hash', tool_call_id='write')
    )
    await renderer.on_stream_event(FileWrittenEvent(path='other', root_dir='/tmp', content_hash='hash'))
    assert 'Wrote' not in output.getvalue()
    assert 'write_file' in output.getvalue()


@pytest.mark.parametrize('abort', [False, True])
async def test_refused_write_releases_diff(abort: bool) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=True)
    for call_id in ('refused', 'pending'):
        await renderer.on_stream_event(
            FileChangeRequestEvent(
                path='file',
                root_dir='/tmp',
                operation='write',
                diff='sentinel proposed diff',
                truncated=False,
                tool_call_id=call_id,
            )
        )
    if abort:
        await renderer.abort()
    else:
        await renderer.on_stream_event(
            FunctionToolResultEvent(part=ToolReturnPart('write_file', 'refused', tool_call_id='refused'))
        )
    await renderer.on_stream_event(
        FileWrittenEvent(path='file', root_dir='/tmp', content_hash='hash', tool_call_id='refused')
    )
    assert 'sentinel proposed diff' not in output.getvalue()
    await renderer.on_stream_event(
        FileWrittenEvent(path='file', root_dir='/tmp', content_hash='hash', tool_call_id='pending')
    )
    assert ('sentinel proposed diff' in output.getvalue()) == (not abort)


async def test_thinking_deltas_and_abort() -> None:
    renderer = StreamRenderer(Console(file=io.StringIO(), force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart('start')))
    await renderer.on_stream_event(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' more')))
    await renderer.on_stream_event(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{}')))
    await renderer.abort()
