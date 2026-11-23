"""Compact tool rendering keeps file diffs visible without changing model results."""

import io

import pytest
from rich.color import Color
from rich.console import Console
from rich.text import Text

from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai_harness.file_system import FileChangeRequestedEvent, FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from pydantic_ai_pax import StreamRenderer
from pydantic_ai_pax.config import Settings, resolve_settings
from pydantic_ai_pax.ui.rendering import theme


@pytest.mark.parametrize(
    ('exit_code', 'truncated', 'rows'),
    [
        (0, False, ['  ⎿ exit 0 · 100 lines']),
        # The events carry output before the harness masks secrets, so none of it is shown.
        (1, False, ['  ⎿ exit 1 · 100 lines']),
        # A background command keeps writing, so its row points at the log.
        (None, False, ['  ⎿ running in background · 100 lines · log /tmp/out']),
        # Output the harness truncated points at the log.
        (0, True, ['  ⎿ exit 0 · 100 lines · log /tmp/out']),
        (1, True, ['  ⎿ exit 1 · 100 lines · log /tmp/out']),
    ],
)
async def test_shell_shows_one_result_row_without_its_output(
    exit_code: int | None, truncated: bool, rows: list[str]
) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'echo secret'}, tool_call_id='shell'))
    )
    await renderer.on_stream_event(CommandStartedEvent(command='echo secret', pid=1, tool_call_id='shell'))
    await renderer.on_stream_event(CommandOutputEvent(text='hidden output\n' * 99 + 'last line', tool_call_id='shell'))
    await renderer.on_stream_event(
        CommandFinishedEvent(
            pid=1,
            output_path='/tmp/out',
            status_path='/tmp/status',
            exit_code=exit_code,
            line_count=100,
            truncated=truncated,
            tool_call_id='shell',
        )
    )
    part = ToolReturnPart('shell', 'full model result', tool_call_id='shell')
    await renderer.on_stream_event(FunctionToolResultEvent(part=part))
    await renderer.finish()
    assert part.content == 'full model result'
    assert 'hidden output' not in output.getvalue()
    assert output.getvalue() == '\n'.join(['● shell echo secret', *rows, '', ''])


@pytest.mark.parametrize('operation', ['write', 'edit'])
async def test_file_changes_show_diffs(operation: str) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart(f'{operation}_file', {'path': 'file.py'}, tool_call_id='file'))
    )
    if operation == 'write':
        await renderer.on_stream_event(
            FileChangeRequestedEvent(
                path='file.py',
                root_dir='/tmp',
                operation='write',
                diff='visible diff',
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
                diff='visible diff',
                truncated=True,
                tool_call_id='file',
            )
        )
    notice = 'Diff truncated.\n' if operation == 'edit' else ''
    assert output.getvalue() == f'● {operation}_file file.py\n\nvisible diff\n{notice}\n'


async def test_a_write_without_a_diff_keeps_one_blank_row() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('write_file', {'path': 'file.py'}, tool_call_id='file'))
    )
    await renderer.on_stream_event(
        FileWrittenEvent(path='file.py', root_dir='/tmp', content_hash='hash', tool_call_id='file')
    )
    assert output.getvalue() == '● write_file file.py\n\n'


async def test_grep_summary_does_not_wrap_or_print_results() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=30), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('grep', {'pattern': 'needle' * 40}, tool_call_id='grep'))
    )
    part = ToolReturnPart('grep', 'secret match\n' * 30, tool_call_id='grep')
    await renderer.on_stream_event(FunctionToolResultEvent(part=part))
    await renderer.finish()
    assert len(output.getvalue().splitlines()) == 2
    assert output.getvalue().endswith('\n\n')
    assert 'secret match' not in output.getvalue()
    assert part.content == 'secret match\n' * 30


@pytest.mark.parametrize('show_tool_output', [False, True])
@pytest.mark.parametrize(
    ('name', 'arguments'),
    [
        ('shell', {'command': 'echo hello'}),
        ('write_file', {'path': 'file.py'}),
        ('edit_file', {'path': 'file.py'}),
        ('read_file', {'path': 'file.py'}),
        ('list_files', {'path': 'src'}),
        ('grep', {'pattern': 'hello'}),
        ('custom_tool', {}),
    ],
)
async def test_tool_header_colors(name: str, arguments: dict[str, str], show_tool_output: bool) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor')
    renderer = StreamRenderer(console, stop_loading=lambda: None, show_tool_output=show_tool_output)
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(name, arguments)))
    await renderer.finish()
    # Text.from_ansi drops the final newline; check spacing on the terminal output itself.
    assert output.getvalue().endswith('\n\n')
    text = Text.from_ansi(output.getvalue())
    assert text.get_style_at_offset(console, 0).color == console.get_style(theme.MUTED).color
    assert text.get_style_at_offset(console, 2).color == console.get_style(theme.ACCENT).color
    if name == 'shell':
        # The command is highlighted as Bash: `echo` is a builtin.
        assert text.get_style_at_offset(console, 3 + len(name)).color == Color.parse('color(6)')
    elif arguments:
        assert text.get_style_at_offset(console, 3 + len(name)).color == console.get_style(theme.MUTED).color


def test_output_setting_is_opt_in() -> None:
    assert not Settings().tool_output
    assert resolve_settings({'display.tool_output': True}).tool_output
