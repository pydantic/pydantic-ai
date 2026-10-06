"""Malformed and unmatched tool events remain safe to display."""

import asyncio
import io
from dataclasses import dataclass

import pytest
from rich.console import Console
from rich.text import Text

from pydantic_ai import (
    CapabilityEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)
from pydantic_ai.messages import (
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileEditedEvent, FileWrittenEvent
from pydantic_clai2 import StreamRenderer


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
@pytest.mark.parametrize('show_tool_output', [False, True])
async def test_refused_write_releases_diff(abort: bool, show_tool_output: bool) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=show_tool_output)
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


@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
async def test_echo_prompt_resumes_active_part(thinking: bool, terminal: bool) -> None:
    """Steering drains pending Markdown, then keeps same-index deltas and later parts."""
    tasks = asyncio.all_tasks()
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=terminal), stop_loading=lambda: None)
    part = ThinkingPart('before\nunfinished') if thinking else TextPart('before\nunfinished')
    await renderer.on_stream_event(PartStartEvent(index=7, part=part))
    await renderer.echo_prompt('[bold]steer[/bold]\x1b[2J\r\x00')
    echoed = Text.from_ansi(output.getvalue()).plain
    assert 'before' in echoed
    assert 'unfinished' in echoed
    assert '> [bold]steer[/bold]\\x1b[2J\\x0d\\x00' in echoed
    assert output.getvalue().endswith('\n\n')
    assert echoed.index('before') < echoed.index('unfinished') < echoed.index('> ')

    # A delta for another index must not enter the restarted pipeline.
    await renderer.on_stream_event(PartDeltaEvent(index=6, delta=TextPartDelta('wrong index')))
    delta = ThinkingPartDelta(content_delta='continued') if thinking else TextPartDelta('continued')
    await renderer.on_stream_event(PartDeltaEvent(index=7, delta=delta))
    await renderer.on_stream_event(PartEndEvent(index=7, part=part))
    await renderer.on_stream_event(PartDeltaEvent(index=7, delta=TextPartDelta('ended part')))
    await renderer.on_stream_event(PartStartEvent(index=8, part=TextPart('future')))
    await renderer.on_stream_event(PartDeltaEvent(index=8, delta=TextPartDelta(' delta')))
    await renderer.finish()
    rendered = Text.from_ansi(output.getvalue()).plain
    assert 'continued' in rendered
    assert 'future delta' in rendered
    assert rendered.index('> ') < rendered.index('continued') < rendered.index('future delta')
    assert 'wrong index' not in rendered
    assert 'ended part' not in rendered
    assert rendered.count('Thinking ') == (2 if thinking else 0)
    assert asyncio.all_tasks() == tasks


@pytest.mark.parametrize('show_thinking', [False, True])
async def test_echo_prompt_without_active_part(show_thinking: bool) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output), stop_loading=lambda: None, show_thinking=show_thinking, tool_calls='grouped'
    )
    if not show_thinking:
        await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart('hidden reasoning')))
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart('read_file', {'path': 'file'})))
    await renderer.echo_prompt('steer')
    echoed = output.getvalue()
    assert 'read_file' in echoed
    assert echoed.endswith('> steer\n\n')
    await renderer.on_stream_event(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='hidden delta')))
    await renderer.on_stream_event(PartStartEvent(index=1, part=TextPart('answer')))
    await renderer.finish()
    assert 'answer' in output.getvalue()
    assert 'hidden' not in output.getvalue()
    assert 'Thinking' not in output.getvalue()


async def test_echo_prompt_idle_is_literal() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.echo_prompt('[bold]first[/bold]\nsecond\tline\x07')
    await renderer.finish()
    assert output.getvalue() == '> [bold]first[/bold]\nsecond  line\\x07\n\n'
