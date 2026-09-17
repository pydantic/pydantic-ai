"""Native capability events drive terminal-safe specialized output."""

import io

import pytest
from pydantic_ai import FunctionToolCallEvent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.filesystem import FileEditedEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from rich.console import Console
from rich.text import Text

from pydantic_clai2 import StreamRenderer


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_shell_header_includes_argument_once() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(
            part=ToolCallPart(
                'shell',
                {'command': 'ls /tmp'},
                tool_call_id='shell-call',
            )
        )
    )
    await renderer.on_stream_event(
        CommandStartedEvent(
            tool_call_id='shell-call',
            command='ls /tmp',
            pid=1,
        )
    )
    assert output.getvalue() == '● shell ls /tmp\n\n'


async def test_shell_sgr_colors_across_chunks_and_lines() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output, force_terminal=True, color_system='truecolor'), stop_loading=lambda: None
    )
    for chunk in ('\x1b[1;', '35mMAGENTA\n', 'STILL MAGENTA\x1b[0m\n', '\x1b[2J\x1b]52;c;payload\x07safe\n'):
        await renderer.on_stream_event(CommandOutputEvent(tool_call_id='colors', text=chunk))
    rendered = output.getvalue()
    plain = Text.from_ansi(rendered).plain
    assert 'MAGENTA\nSTILL MAGENTA' in plain
    assert '\\x1b[1;35m' not in plain
    assert '\x1b[2J' not in rendered and '\x1b]52;' not in rendered
    assert '35m' in rendered
    assert '\\x1b[2J' in plain


@pytest.mark.parametrize(
    ('name', 'args', 'expected'),
    [
        ('list_files', {}, "● list_files '.' recursive=true limit=200"),
        (
            'list_files',
            {'path': '/tmp', 'glob': '*.cpp', 'limit': 10},
            "● list_files '/tmp' recursive=true limit=10 glob='*.cpp'",
        ),
        ('read_file', {'path': '/tmp/example.cpp'}, "● read_file '/tmp/example.cpp' offset=0 limit=2000 lines"),
        ('read_file', {'path': 'main.py', 'offset': 20, 'limit': 15}, "● read_file 'main.py' offset=20 limit=15 lines"),
        (
            'read_file',
            {'path': 'main.py', 'limit': 5000},
            "● read_file 'main.py' offset=0 limit=2000 lines (requested 5000)",
        ),
    ],
)
async def test_inspection_headers(name: str, args: dict[str, object], expected: str) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=160), stop_loading=lambda: None)
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(name, args, tool_call_id='inspect')))
    assert output.getvalue() == expected + '\n\n'


async def test_shell_event_display() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(CommandStartedEvent(tool_call_id='1', command='printf hello', pid=12))
    await renderer.on_stream_event(CommandOutputEvent(tool_call_id='1', text='hello\x1b]52;c;payload\x07'))
    await renderer.on_stream_event(
        CommandFinishedEvent(
            tool_call_id='1',
            pid=12,
            output_path='/tmp/output.log',
            status_path='/tmp/status.json',
            exit_code=0,
            truncated=True,
        )
    )
    assert '● shell printf hello' in output.getvalue()
    assert 'hello' in output.getvalue()
    assert '\x1b]52;' not in output.getvalue()
    assert 'truncated' in output.getvalue()


@pytest.mark.parametrize('limit', [0, 1, 2])
async def test_shell_line_limit_across_chunks(limit: int) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, shell_lines=limit)
    for chunk in ('fir', 'st\nsecond', '\nthird\nfourth'):
        await renderer.on_stream_event(CommandOutputEvent(tool_call_id='test', text=chunk))
    await renderer.on_stream_event(
        CommandFinishedEvent(
            tool_call_id='test',
            pid=1,
            output_path='/tmp/out',
            status_path='/tmp/status',
            exit_code=0,
            truncated=False,
            total_lines=4,
        )
    )
    text = output.getvalue()
    assert f'Truncated {4 - limit} lines' in text
    assert ('first' in text) == (limit >= 1)
    assert ('second' in text) == (limit >= 2)
    assert 'third' not in text and 'fourth' not in text


async def test_shell_progress_replaces_carriage_return_frames() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None)
    await renderer.on_stream_event(
        CommandStartedEvent(
            tool_call_id='progress',
            command='python3 - <<PY\nprint(123)\nPY',
            pid=1,
        )
    )
    for chunk in ('header\r', '\n10%', '\r50%', '\r', '100%\n', 'done'):
        await renderer.on_stream_event(CommandOutputEvent(tool_call_id='progress', text=chunk))
    await renderer.on_stream_event(
        CommandFinishedEvent(
            tool_call_id='progress',
            pid=1,
            output_path='/tmp/out',
            status_path='/tmp/status',
            exit_code=0,
            truncated=False,
            total_lines=3,
        )
    )
    text = output.getvalue()
    assert 'header\n100%\ndone\n' in text
    assert '10%' not in text and '50%' not in text and '\\x0d' not in text
    assert '(+2 command lines)' in text and 'print(123)' not in text
    assert 'Truncated' not in text


async def test_edit_uses_termflow_diff_renderer() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FileEditedEvent(
            path='demo.py',
            root_dir='/tmp',
            content_hash='hash',
            diff='--- a/demo.py\n+++ b/demo.py\n@@ -1 +1 @@\n-old\n+new\n',
            truncated=False,
        )
    )
    text = output.getvalue()
    assert '● edit_file demo.py' in Text.from_ansi(text).plain
    assert 'old' in text and 'new' in text
    assert '\x1b[48;2;70;82;88m' in text  # addition rows: Aqua over Dark Purple
    assert '\x1b[48;2;104;43;54m' in text  # deletion rows: Calcium over Dark Purple
