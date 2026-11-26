"""Native capability events drive terminal-safe specialized output."""

import io
import os
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Literal

import pytest
from rich.color import Color
from rich.console import Console
from rich.style import Style
from rich.text import Text
from termflow.themes import PALETTES

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    PartStartEvent,
    RetryInfo,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo
from pydantic_ai_harness.file_system import FileEditedEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent, Shell
from pydantic_ai_pax import StreamRenderer
from pydantic_ai_pax.runtime.sandbox_calls import DelegationToolCallEvent
from pydantic_ai_pax.ui.prompt.prompt_surface import PromptSurface
from pydantic_ai_pax.ui.rendering import theme
from tests.pax.test_speculative_mode import streamed


async def test_shell_header_includes_argument_once() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=True)
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
    await renderer.finish()
    assert output.getvalue() == '● shell ls /tmp\n\n'


async def test_shell_header_without_a_command_shows_only_the_tool_name() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=True)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': ''}, tool_call_id='empty-shell'))
    )
    await renderer.finish()
    assert output.getvalue() == '● shell\n\n'


async def test_shell_sgr_colors_across_chunks_and_lines() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output, force_terminal=True, color_system='truecolor'),
        stop_loading=lambda: None,
        show_tool_output=True,
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
        ('list_files', {}, '● list_files .'),
        ('list_files', {'path': '/tmp', 'glob': '*.cpp', 'limit': 10}, '● list_files /tmp *.cpp first 10'),
        ('list_files', {'recursive': False}, '● list_files . not recursive'),
        ('read_file', {'path': '/tmp/example.cpp'}, '● read_file /tmp/example.cpp'),
        ('read_file', {'path': 'main.py', 'offset': 20, 'limit': 15}, '● read_file main.py lines 20-34'),
        ('read_file', {'path': 'main.py', 'offset': 20}, '● read_file main.py from line 20'),
        ('read_file', {'path': 'main.py', 'limit': 5000}, '● read_file main.py lines 1-2000'),
    ],
)
async def test_inspection_headers(name: str, args: dict[str, object], expected: str) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=160), stop_loading=lambda: None, show_tool_output=True)
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(name, args, tool_call_id='inspect')))
    await renderer.finish()
    assert output.getvalue() == expected + '\n\n'


async def test_shell_event_display() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=True)
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
    assert output.getvalue().endswith('  ⎿ exit 0 · log /tmp/output.log\n\n')


@pytest.mark.parametrize('limit', [0, 1, 2])
async def test_shell_line_limit_across_chunks(limit: int) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=True, shell_lines=limit)
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
            line_count=4,
        )
    )
    text = output.getvalue()
    assert f'  ⎿ exit 0 · 4 lines, {limit} shown · log /tmp/out\n' in text
    assert ('first' in text) == (limit >= 1)
    assert ('second' in text) == (limit >= 2)
    assert 'third' not in text and 'fourth' not in text


async def test_shell_progress_replaces_carriage_return_frames() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None, show_tool_output=True)
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
            line_count=3,
        )
    )
    text = output.getvalue()
    assert 'header\n100%\ndone\n' in text
    assert '10%' not in text and '50%' not in text and '\\x0d' not in text
    assert text.startswith('● shell python3 - <<PY\n        print(123)\n        PY\nheader\n')
    assert text.endswith('done\n  ⎿ exit 0 · 3 lines\n\n')


async def test_long_shell_command_wraps_under_a_hanging_indent() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=40), stop_loading=lambda: None)
    command = 'cd /repo/src; cat pkg/module.py; grep -n "needle" -r pkg | head -40'
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': command}, tool_call_id='long'))
    )
    await renderer.finish()
    assert output.getvalue() == (
        '● shell cd /repo/src; cat pkg/module.py;\n        grep -n "needle" -r pkg | head\n        -40\n\n'
    )


async def test_shell_command_wraps_rather_than_clips_on_a_narrow_console() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=16), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'echo abcdefghijkl'}, tool_call_id='narrow'))
    )
    await renderer.finish()
    assert output.getvalue() == '● shell echo\n        abcdefgh\n        ijkl\n\n'


async def test_shell_command_rows_beyond_the_limit_are_counted() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None)
    command = '\n'.join(f'echo {index}' for index in range(15))
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': command}, tool_call_id='script'))
    )
    await renderer.finish()
    lines = output.getvalue().splitlines()
    assert lines[0] == '● shell echo 0'
    assert lines[9] == '        echo 9'
    assert lines[10:] == ['        … +5 lines', '']


async def test_edit_uses_termflow_diff_renderer() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output, force_terminal=True), stop_loading=lambda: None, show_tool_output=True
    )
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
    assert '\x1b[48;2;34;92;43m' in text  # addition rows: green
    assert '\x1b[48;2;122;41;54m' in text  # deletion rows: red


async def test_a_truncated_diff_of_headers_only_keeps_one_blank_row() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=True, color_system=None), stop_loading=lambda: None)
    await renderer.on_stream_event(
        FileEditedEvent(
            path='big.py', root_dir='/tmp', content_hash='hash', diff='--- a/big.py\n+++ b/big.py\n', truncated=True
        )
    )
    assert output.getvalue() == '● edit_file big.py\n\nDiff truncated.\n\n'


async def _print_shell_call(console: Console, command: str) -> None:
    renderer = StreamRenderer(console, stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': command}, tool_call_id='call'))
    )
    await renderer.finish()


# Rich 15.0.0's `from_ansi` keeps the trailing newline that earlier versions drop
# (https://github.com/Textualize/rich/pull/4076), so tests strip it before counting rows.
async def _highlighted_header(command: str, *, width: int = 100) -> tuple[Console, Text]:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=width)
    await _print_shell_call(console, command)
    return console, Text.from_ansi(output.getvalue())


def _style(console: Console, text: Text, fragment: str) -> Style:
    return text.get_style_at_offset(console, text.plain.index(fragment))


def _color(console: Console, text: Text, fragment: str) -> Color | None:
    return _style(console, text, fragment).color


KEYWORD = Color.parse('color(4)')
STRING = Color.parse('color(3)')
BUILTIN = Color.parse('color(6)')


@pytest.mark.parametrize(
    ('command', 'keyword'),
    [
        pytest.param("cat > /tmp/probe.py <<'PYEOF'\nimport os\nPYEOF", 'import', id='heredoc-to-a-python-file'),
        pytest.param("python - <<'EOF'\nfor line in lines: pass\nEOF", 'for', id='python-heredoc'),
        pytest.param('uv run python <<EOF\nimport os\nEOF', 'import', id='uv-run-python-heredoc'),
        pytest.param("psql <<'SQL'\nSELECT 1;\nSQL", 'SELECT', id='sql-heredoc'),
        pytest.param("python3 -c 'import sys; print(sys.argv)'", 'import', id='python-c'),
        pytest.param('node -e "const answer = 42"', 'const', id='node-e'),
    ],
)
async def test_shell_header_highlights_embedded_code_in_its_language(command: str, keyword: str) -> None:
    console, text = await _highlighted_header(command)
    assert text.plain.startswith(f'● shell {command.splitlines()[0]}')
    assert _color(console, text, keyword) == KEYWORD


async def test_shell_header_highlights_the_shell_as_bash() -> None:
    console, text = await _highlighted_header('cd /tmp && echo "$HOME" | python3 -c \'import sys\'\nEOF')
    assert _color(console, text, 'cd') == BUILTIN
    assert _color(console, text, 'echo') == BUILTIN
    assert _color(console, text, '"$HOME"') == STRING
    # The quotes around inline code are Bash; what they hold is Python.
    assert _color(console, text, "'import") == STRING
    assert _color(console, text, 'import') == KEYWORD
    assert _color(console, text, 'EOF') is None, 'a name stays in the terminal foreground'


async def test_shell_header_mutes_a_heredoc_in_an_unknown_language() -> None:
    console, text = await _highlighted_header("git commit -F - <<'EOF'\nimport this\nEOF")
    assert _color(console, text, 'import') == _color(console, text, '●'), 'muted, like the marker'
    assert _color(console, text, "'EOF'") == STRING
    assert _color(console, text, 'git') is None


async def test_shell_header_dims_highlighted_code_without_underlines() -> None:
    console, text = await _highlighted_header(
        "cd /tmp && uv run python - <<'EOF'\nfrom pydantic_ai.messages import ModelRequest\nclass Probe: pass\nEOF\n"
        "git commit -F - <<'EOF'\nnotes\nEOF"
    )
    for fragment in ('cd', "'EOF'", 'from', 'pydantic_ai.messages', 'ModelRequest', 'Probe'):
        style = _style(console, text, fragment)
        assert style.dim, fragment
        assert not style.underline, fragment
    assert _color(console, text, 'from') == KEYWORD, 'still highlighted'
    for fragment in ('●', 'shell', 'notes'):
        assert not _style(console, text, fragment).dim, 'the tool name and muted text are as before'


async def test_streamed_code_fence_keeps_syntax_underlines() -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=100)
    renderer = StreamRenderer(console, stop_loading=lambda: None)
    source = '```python\nfrom pydantic_ai.messages import ModelRequest\n```\n'
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(source)))
    await renderer.finish()
    text = Text.from_ansi(output.getvalue())
    assert _style(console, text, 'pydantic_ai').underline, 'only shell call headers drop underlines'
    assert not _style(console, text, 'from').dim


async def test_shell_header_escapes_controls_before_highlighting() -> None:
    command = (
        "python3 - <<'EOF'\nprint('\x1b[2J\x1b]52;c;cGF5bG9hZA==\x07')\nimport os\nEOF\n"
        'python3 -c \'x = "\x1b]8;;https://example.com\x1b\\\\"\'\r\x9b31m'
    )
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=100)
    await _print_shell_call(console, command)
    text = Text.from_ansi(output.getvalue())
    for control in ('\x1b[2J', '\x1b]', '\x07', '\r', '\x9b'):
        assert control not in output.getvalue()
    for escaped in (r'\x1b[2J', r'\x1b]52;c;', r'\x07', r'\x1b]8;;', r'\x0d', r'\x9b'):
        assert escaped in text.plain
    assert _color(console, text, 'import') == KEYWORD


async def test_highlighted_shell_header_keeps_the_hanging_indent_and_row_cap() -> None:
    command = (
        'python3 - <<EOF\nx = "' + 'a' * 60 + '"\n' + '\n'.join(f'import m{index}' for index in range(12)) + '\nEOF'
    )
    plain = io.StringIO()
    await _print_shell_call(Console(file=plain, width=40), command)
    console, text = await _highlighted_header(command, width=40)
    assert text.plain == Text.from_ansi(plain.getvalue()).plain, 'highlighting moves no character'
    assert text.plain.rstrip('\n').splitlines() == [
        '● shell python3 - <<EOF',
        '        x =',
        '        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        '        aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
        '        import m0',
        '        import m1',
        '        import m2',
        '        import m3',
        '        import m4',
        '        import m5',
        '        … +7 lines',
    ]
    assert _color(console, text, 'import') == KEYWORD
    assert _color(console, text, '"aaa') == STRING
    assert _color(console, text, '…') == _color(console, text, '●')


async def test_shell_header_stops_highlighting_inline_code_at_the_row_cap() -> None:
    command = "python3 -c '\n" + '\n'.join(f'import m{index}' for index in range(12)) + "\n'"
    console, text = await _highlighted_header(command)
    assert text.plain.rstrip('\n').splitlines()[-1] == '        … +4 lines'
    assert _color(console, text, 'import m8') == KEYWORD
    _, empty = await _highlighted_header("python3 -c ''")
    assert empty.plain.rstrip('\n') == "● shell python3 -c ''"


async def test_shell_header_lexes_a_long_one_liner_only_as_far_as_its_shown_rows() -> None:
    command = "python3 -c '" + 'x = 1; ' * 2000 + "'"
    plain = io.StringIO()
    await _print_shell_call(Console(file=plain, width=60), command)
    console, text = await _highlighted_header(command, width=60)
    lines = text.plain.rstrip('\n').splitlines()
    assert len(lines) == 11
    assert lines[-1] == f'        … +{len(Text(command).wrap(console, 52, overflow="fold")) - 10} lines'
    assert text.plain == Text.from_ansi(plain.getvalue()).plain
    assert _color(console, text, 'x = 1') is None, 'Python, not a Bash string'


@pytest.mark.parametrize('name', ['default', 'github_light', 'tokyo_night'])
async def test_shell_header_uses_the_palette_syntax_colours(name: str) -> None:
    with theme.use(lambda: name):
        console, text = await _highlighted_header("python3 - <<'EOF'\nimport os\nvalue = os.sep\nEOF")
    keyword = PALETTES[name].ansi[4] if name != 'default' else 'color(4)'
    assert _color(console, text, 'import') == Color.parse(keyword)
    assert _color(console, text, 'value') is None, 'names use the terminal foreground'
    assert _style(console, text, 'import').dim
    assert _style(console, text, 'value').dim
    assert not _style(console, text, 'os').underline


async def test_live_prompt_repaints_the_shell_header_for_a_new_theme_and_width() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (80, 24))
    console = Console(file=surface, force_terminal=True, color_system='truecolor', width=80)
    selected = ['tokyo_night']
    command = "python3 - <<'EOF'\nimport os\nprint(os.getcwd())\nEOF"

    def rows(*, width: int = 80) -> list[str]:
        return list(surface.transcript.frame(width=width, height=24).rows)

    def colour(fragment: str) -> Color | None:
        row = next(Text.from_ansi(row) for row in rows() if fragment in Text.from_ansi(row).plain)
        return _color(console, row, fragment)

    with theme.use(lambda: selected[0]):
        await _print_shell_call(console, command)
        console.print('after')
        assert colour('import') == Color.parse(PALETTES['tokyo_night'].ansi[4])
        selected[0] = 'default'
        # Rendered again, not recoloured: the palette's blue is also the slot of a brand colour.
        assert colour('import') == KEYWORD
        assert _style(console, Text.from_ansi(rows()[1]), 'import').dim, 'still dimmed'
        retained = rows()[:5]
        await _print_shell_call(console, command)
        assert rows()[-6:-1] == retained, 'the retained header matches one printed in the new theme'
        assert [Text.from_ansi(row).plain for row in rows(width=20)][:5] == [
            '● shell python3 -',
            "        <<'EOF'",
            '        import os',
            '        print(os.get',
            '        cwd())',
        ]
        assert [Text.from_ansi(row).plain for row in rows()][4:6] == ['', 'after']


def _finished(tool_call_id: str, *, exit_code: int | None, line_count: int) -> CommandFinishedEvent:
    return CommandFinishedEvent(
        tool_call_id=tool_call_id,
        pid=1,
        output_path='/tmp/out',
        status_path='/tmp/status',
        exit_code=exit_code,
        truncated=False,
        line_count=line_count,
    )


@pytest.mark.parametrize(
    ('outcome', 'cause', 'reason'),
    [
        ('failed', None, 'Not allowed here.'),
        ('denied', None, 'Not allowed here.'),
        ('interrupted', None, 'Not allowed here.'),
        ('retried', 'model_retry', 'Not allowed here.'),
        ('retried', 'validation_error', 'invalid arguments'),
    ],
)
async def test_a_failed_or_refused_call_is_marked_under_its_header(
    outcome: Literal['failed', 'denied', 'retried', 'interrupted'],
    cause: Literal['model_retry', 'validation_error'] | None,
    reason: str,
) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=80)
    renderer = StreamRenderer(console, stop_loading=lambda: None)
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart('deploy', {}, tool_call_id='d')))
    part = ToolReturnPart(
        'deploy',
        'Not allowed here.\nMore detail.',
        tool_call_id='d',
        outcome=outcome,
        retry=RetryInfo(cause=cause) if cause else None,
    )
    await renderer.on_stream_event(FunctionToolResultEvent(part=part))
    await renderer.finish()
    text = Text.from_ansi(output.getvalue())
    assert text.plain.rstrip('\n') == f'● deploy\n  ⎿ {reason}'
    assert _color(console, text, reason) == console.get_style(theme.ERROR).color


async def test_a_failed_delegation_is_left_to_its_task_row() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None)
    call = ToolCallPart('delegate_task', {}, tool_call_id='t')
    await renderer.on_stream_event(DelegationToolCallEvent(call))
    part = ToolReturnPart('delegate_task', "Sub-agent 'x' failed", tool_call_id='t', outcome='retried')
    await renderer.on_stream_event(FunctionToolResultEvent(part=part))
    assert output.getvalue() == ''


@pytest.mark.parametrize(('exit_code', 'color'), [(0, theme.MUTED), (1, theme.ERROR)])
async def test_a_shell_result_row_is_red_only_when_the_command_fails(exit_code: int, color: str) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=80)
    renderer = StreamRenderer(console, stop_loading=lambda: None)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'make'}, tool_call_id='m'))
    )
    await renderer.on_stream_event(_finished('m', exit_code=exit_code, line_count=2))
    assert _color(console, Text.from_ansi(output.getvalue()), f'exit {exit_code}') == console.get_style(color).color


async def test_a_failure_of_a_call_whose_header_a_plugin_drew_is_still_marked() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None)
    part = ToolReturnPart('deploy', '', tool_call_id='d', outcome='failed')
    await renderer.on_stream_event(FunctionToolResultEvent(part=part))
    assert output.getvalue() == '  ⎿ failed · deploy\n\n'


async def test_a_narrow_detached_row_clips_its_command_not_its_exit_code() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=28), stop_loading=lambda: None)
    for call_id in ('a', 'b'):
        await renderer.on_stream_event(
            FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'echo example'}, tool_call_id=call_id))
        )
    await renderer.on_stream_event(_finished('a', exit_code=123, line_count=1))
    assert output.getvalue().endswith('  ⎿ exit 123 · 1 line · ech…\n\n')


async def test_a_row_under_another_calls_header_names_its_command() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None)
    for call_id, command in (('a', 'git status'), ('b', 'git log -1')):
        await renderer.on_stream_event(
            FunctionToolCallEvent(part=ToolCallPart('shell', {'command': command}, tool_call_id=call_id))
        )
    await renderer.on_stream_event(_finished('a', exit_code=0, line_count=2))
    await renderer.on_stream_event(_finished('b', exit_code=0, line_count=1))
    await renderer.finish()
    assert output.getvalue() == (
        '● shell git status\n\n● shell git log -1\n\n  ⎿ exit 0 · 2 lines · git status\n\n'
        '  ⎿ exit 0 · 1 line · git log -1\n\n'
    )


async def test_a_preview_line_cut_at_the_terminal_edge_keeps_the_log_path() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=40), stop_loading=lambda: None, show_tool_output=True)
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'cat wide'}, tool_call_id='w'))
    )
    await renderer.on_stream_event(CommandOutputEvent(tool_call_id='w', text='x' * 60 + '\n'))
    log = '/private/tmp/pax-logs/session-1234/commands/w.log'
    await renderer.on_stream_event(
        CommandFinishedEvent(
            tool_call_id='w', pid=1, output_path=log, status_path='/s', exit_code=0, truncated=False, line_count=1
        )
    )
    # The path wraps rather than clips, so it can still be copied whole.
    assert output.getvalue().replace('\n', '').endswith(f'  ⎿ exit 0 · 1 line · log {log}')


async def test_a_live_failed_row_takes_the_new_themes_error_colour() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (80, 24))
    console = Console(file=surface, force_terminal=True, color_system='truecolor', width=80)
    renderer = StreamRenderer(console, stop_loading=lambda: None)
    selected = ['default']
    with theme.use(lambda: selected[0]):
        await renderer.on_stream_event(
            FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'make'}, tool_call_id='m'))
        )
        await renderer.on_stream_event(_finished('m', exit_code=2, line_count=4))
        selected[0] = 'tokyo_night'
        row = next(
            Text.from_ansi(row)
            for row in surface.transcript.frame(width=80, height=24).rows
            if 'exit 2' in Text.from_ansi(row).plain
        )
        assert _color(console, row, 'exit 2') == console.get_style(theme.color(theme.ERROR)).color


async def test_a_live_background_command_row_offers_nothing_to_expand() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (80, 24))
    renderer = StreamRenderer(
        Console(file=surface, force_terminal=True, color_system='truecolor', width=80), stop_loading=lambda: None
    )
    call = ToolCallPart('shell', {'command': 'npm run dev', 'mode': 'background'}, tool_call_id='b')
    await renderer.on_stream_event(FunctionToolCallEvent(part=call))
    await renderer.on_stream_event(_finished('b', exit_code=None, line_count=3))
    rows = [Text.from_ansi(row).plain.rstrip() for row in surface.transcript.frame(width=80, height=24).rows]
    assert '  ⎿ running in background · 3 lines · log /tmp/out' in rows


async def test_a_live_command_whose_result_never_arrives_expands_to_its_row_alone() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (80, 24))
    renderer = StreamRenderer(
        Console(file=surface, force_terminal=True, color_system='truecolor', width=80), stop_loading=lambda: None
    )
    await renderer.on_stream_event(
        FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'make'}, tool_call_id='m'))
    )
    await renderer.on_stream_event(_finished('m', exit_code=0, line_count=4))
    surface.toggle_details()
    rows = [Text.from_ansi(row).plain.rstrip() for row in surface.transcript.frame(width=80, height=24).rows]
    assert rows == ['● shell make', '  ⎿ exit 0 · 4 lines', '', '']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX shell command')
async def test_a_live_command_expands_to_what_the_shell_tool_returned_without_its_handles(tmp_path: Path) -> None:
    """Runs the real `shell` tool, so a change to the handles that follow its output fails here."""
    calls = iter([ToolCallPart('shell', {'command': 'printf "one\\ntwo\\nthree\\n"'}, tool_call_id='s')])

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[next(calls, TextPart('done'))])

    surface = PromptSurface(output=io.StringIO(), size=lambda: (80, 24))
    renderer = StreamRenderer(
        Console(file=surface, force_terminal=True, color_system='truecolor', width=80), stop_loading=lambda: None
    )

    async def render(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            await renderer.on_stream_event(event)

    shell = Shell[object](tools=['shell'], denied_commands=[])
    await Agent(streamed(respond), capabilities=[LocalWorkspace(tmp_path), shell]).run(
        'go', event_stream_handler=render
    )
    await renderer.finish()
    surface.toggle_details()
    rows = [Text.from_ansi(row).plain.rstrip() for row in surface.transcript.frame(width=80, height=24).rows]
    start = next(index for index, row in enumerate(rows) if row.startswith('  ⎿ exit 0'))
    assert rows[start + 1 : start + 5] == ['    one', '    two', '    three', '']
