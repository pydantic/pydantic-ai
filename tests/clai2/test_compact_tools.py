"""Default tool rendering stays on one line without changing model results."""

import io

import pytest
from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from rich.console import Console

from pydantic_clai2 import StreamRenderer
from pydantic_clai2.config import Settings, resolve_settings


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('exit_code', [0, 1, None])
async def test_shell_only_prints_invocation(exit_code: int | None) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=40), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'echo secret'}, tool_call_id='shell'))
    )
    await renderer.on_stream_event(CommandStartedEvent(command='echo secret', pid=1, tool_call_id='shell'))
    await renderer.on_stream_event(CommandOutputEvent(text='hidden output\n' * 100, tool_call_id='shell'))
    await renderer.on_stream_event(
        CommandFinishedEvent(
            pid=1,
            output_path='/tmp/out',
            status_path='/tmp/status',
            exit_code=exit_code,
            total_lines=100,
            truncated=True,
            tool_call_id='shell',
        )
    )
    part = ToolReturnPart('shell', 'full model result', tool_call_id='shell')
    await renderer.on_stream_event(FunctionToolResultEvent(part=part))
    assert part.content == 'full model result'
    assert output.getvalue() == '● shell echo secret\n'


@pytest.mark.parametrize('operation', ['write', 'edit'])
async def test_file_changes_hide_diffs(operation: str) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart(f'{operation}_file', {'path': 'file.py'}, tool_call_id='file'))
    )
    if operation == 'write':
        await renderer.on_stream_event(
            FileChangeRequestEvent(
                path='file.py',
                root_dir='/tmp',
                operation='write',
                diff='secret diff',
                truncated=False,
                tool_call_id='file',
            )
        )
        await renderer.on_stream_event(
            FileWrittenEvent(path='file.py', root_dir='/tmp', content_hash='hash', tool_call_id='file')
        )
    else:
        await renderer.on_stream_event(
            FileEditedEvent(
                path='file.py',
                root_dir='/tmp',
                content_hash='hash',
                diff='secret diff',
                truncated=True,
                tool_call_id='file',
            )
        )
    assert output.getvalue() == f'● {operation}_file file.py\n'


async def test_grep_summary_does_not_wrap_or_print_results() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=30), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('grep', {'pattern': 'needle' * 40}, tool_call_id='grep'))
    )
    part = ToolReturnPart('grep', 'secret match\n' * 30, tool_call_id='grep')
    await renderer.on_stream_event(FunctionToolResultEvent(part=part))
    assert len(output.getvalue().splitlines()) == 1
    assert 'secret match' not in output.getvalue()
    assert part.content == 'secret match\n' * 30


def test_output_setting_is_opt_in() -> None:
    assert not Settings().tool_output
    assert resolve_settings({'display.tool_output': True}).tool_output
