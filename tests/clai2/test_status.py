"""Footer accounting and terminal restoration without provider calls."""

import asyncio
import io
import re
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent, PartStartEvent
from pydantic_ai.messages import NativeToolCallPart, TextPart, ToolCallPart, ToolCallPartDelta, ToolReturnPart
from rich.console import Console

from pydantic_clai2._app import _reset_status  # pyright: ignore[reportPrivateUsage]
from pydantic_clai2.status import Status, StatusLine
from pydantic_clai2.theme import MUTED, WARNING, sgr


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def test_estimate_includes_tool_argument_deltas() -> None:
    status = Status(model='test')
    status.observe(PartStartEvent(index=0, part=TextPart(content='abcd')))
    status.observe(PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='12345678')))
    assert '~3 streamed tokens' in status.text()
    assert 'context: ?' in status.text()
    status.context_tokens = 1000
    status.output_tokens = 20
    assert 'context: 1,000 tokens' in status.text()
    assert '20 output tokens' in status.text()


def test_workspace_follows_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, 'home', lambda: Path('/home/me'))
    status = Status(model='m', workspace='/home/me/code/app')
    assert status.text().startswith('m | ~/code/app | context: ')
    status.workspace = '/home/me/' + 'deep/' * 10 + 'app'
    shown = status.text().split(' | ')[1]
    assert shown.startswith('\u2026') and shown.endswith('/deep/app') and len(shown) == 40
    status.workspace = '/home/meow/app'
    assert ' | /home/meow/app | ' in status.text()


def test_workspace_control_characters_are_inert() -> None:
    status = Status(model='m', workspace='/tmp/a\nb\x1b[31m\u00e9')
    assert ' | /tmp/a\\x0ab\\x1b[31m\u00e9 | ' in status.text()
    assert ' | /tmp/a\\x0ab\\x1b[31m\u00e9 | ' in ''.join(text for _, text in status.toolbar())


def test_workspace_without_a_home_directory_is_shown_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_home() -> Path:
        raise RuntimeError('Could not determine home directory.')

    monkeypatch.setattr(Path, 'home', no_home)
    assert ' | /srv/app | ' in Status(model='m', workspace='/srv/app').text()


def test_toolbar_paints_the_context_figure_on_alert() -> None:
    status = Status(model='m', context_tokens=90, context_alert=True)
    assert status.toolbar() == [('', 'm | context: '), (WARNING, '90'), ('', ' tokens | ~0 streamed tokens | ready')]
    status.context_alert = False
    assert status.toolbar()[1] == ('', '90')
    assert ''.join(text for _, text in status.toolbar()) == status.text()
    status.cost = Decimal('0.0123')
    status.context_alert = True
    assert status.toolbar()[1] == (WARNING, '90')
    assert '$0.0123' in status.toolbar()[2][1]
    assert ''.join(text for _, text in status.toolbar()) == status.text()


async def test_footer_paints_the_context_figure_on_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COLORTERM', 'truecolor')
    output = io.StringIO()
    status = Status(model='m', context_tokens=90, context_alert=True)
    async with StatusLine(Console(file=output, force_terminal=True, width=80, height=24), status):
        pass
    painted = output.getvalue()
    assert f'{sgr(WARNING)}9{sgr(WARNING)}0' in painted and f'{sgr(WARNING)}m' not in painted


def test_new_resets_the_figures_whatever_follows_it() -> None:
    status = Status(context_tokens=90, context_alert=True, output_tokens=5, streamed_chars=8)
    _reset_status('/new please', status)
    assert status == Status()
    status.context_alert = True
    _reset_status('/newer', status)
    assert status.context_alert


def test_tool_status_transitions() -> None:
    status = Status()
    status.observe(PartStartEvent(index=0, part=NativeToolCallPart('web_search', {})))
    call = ToolCallPart('shell', {})
    status.observe(PartStartEvent(index=0, part=call))
    assert status.activity == 'tool: shell'
    status.observe(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={})))
    status.observe(FunctionToolCallEvent(part=call))
    assert status.activity == 'running: shell'
    status.observe(FunctionToolResultEvent(part=ToolReturnPart('shell', 'done')))
    assert status.activity == 'working'


@pytest.mark.parametrize('truecolor', [False, True])
async def test_shimmer_without_spinner(monkeypatch: pytest.MonkeyPatch, truecolor: bool) -> None:
    monkeypatch.setenv('COLORTERM', 'truecolor' if truecolor else '')
    output = io.StringIO()
    frames: list[str] = []
    original_sleep = asyncio.sleep

    async def tick(delay: float) -> None:
        frames.append(output.getvalue().split('\x1b[2K')[-1])
        if len(frames) == 11:
            raise asyncio.CancelledError
        await original_sleep(0)

    monkeypatch.setattr('pydantic_clai2.status.asyncio.sleep', tick)
    async with StatusLine(Console(file=output, force_terminal=True, width=40, height=24), Status(model='test\x1b\n')):
        while len(frames) < 11:
            await original_sleep(0)
    plain = [re.sub(r'\x1b\[[0-9;]*m|\x1b8', '', frame) for frame in frames]
    assert plain[0].startswith('test?? | context:')
    assert all(frame == plain[0] for frame in plain)
    assert all(len(frame) == 39 for frame in plain)
    assert frames[0] != frames[10]
    assert ('38;2;' in frames[0]) == truecolor
    assert ('\x1b[38;2;155;119;255m' if truecolor else '\x1b[35m') in frames[0]
    assert ('\x1b[38;2;0;255;235m' if truecolor else '\x1b[96m') not in output.getvalue()
    assert '\n' not in output.getvalue()


async def test_row_reserved_before_margins_and_again_on_resize() -> None:
    painted = asyncio.Event()

    class Output(io.StringIO):
        def flush(self) -> None:
            painted.set()

    output = Output()
    console = Console(file=output, force_terminal=True, width=40, height=24)
    async with StatusLine(console, Status()):
        painted.clear()
        await asyncio.wait_for(painted.wait(), timeout=5)
        first = output.getvalue()
        assert first.count('\x1bD' * 4 + '\x1b[4A\x1b7\x1b[1;20r') == 1
        assert '│> Working... Ctrl-C to interrupt' in first
        assert first.count('\x1b[24;1H') >= 2
        console.height = 30
        painted.clear()
        await asyncio.wait_for(painted.wait(), timeout=5)
        resized = output.getvalue()[len(first) :]
        assert resized.count('\x1bD' * 4 + '\x1b[4A\x1b7\x1b[1;26r') == 1
        assert resized.count('\x1b[30;1H') >= 1
        console.height = 2
        painted.clear()
        before = len(output.getvalue())
        await asyncio.wait_for(painted.wait(), timeout=5)
        assert '\x1b[r' in output.getvalue()[before:]
        console.height = 24
        painted.clear()
        await asyncio.wait_for(painted.wait(), timeout=5)
        assert 'Working... Ctrl-C to interrupt' in output.getvalue()[before:]


async def test_tiny_terminal() -> None:
    output = io.StringIO()
    async with StatusLine(Console(file=output, force_terminal=True, height=2), Status()):
        pass
    assert '\x1b[1;' not in output.getvalue()
    assert 'Working' not in output.getvalue()


@pytest.mark.parametrize(('width', 'height'), [(80, 5), (3, 24), (1, 3)])
async def test_small_terminal_keeps_only_the_status_row(width: int, height: int) -> None:
    output = io.StringIO()
    async with StatusLine(Console(file=output, force_terminal=True, width=width, height=height), Status()):
        assert f'\x1b[1;{height - 1}r' in output.getvalue()
        assert '┌' not in output.getvalue()
    assert output.getvalue().endswith('\x1b8\x1b[?25h')


async def test_redirected_output_has_no_footer() -> None:
    output = io.StringIO()
    async with StatusLine(Console(file=output, force_terminal=False), Status()):
        pass
    assert output.getvalue() == ''


@pytest.mark.parametrize('fail', [False, True])
async def test_cursor_restored_after_run(fail: bool) -> None:
    output = io.StringIO()
    try:
        async with StatusLine(Console(file=output, force_terminal=True, width=80, height=24), Status()):
            assert '\x1b[?25l' in output.getvalue()
            assert '\x1b[?25h' not in output.getvalue()
            if fail:
                raise ValueError('run failed')
    except ValueError:
        assert fail
    assert output.getvalue().endswith('\x1b[?25h')


async def test_cancellation_restores_scroll_region() -> None:
    output = io.StringIO()
    entered = asyncio.Event()

    async def run() -> None:
        async with StatusLine(Console(file=output, force_terminal=True, width=80, height=24), Status()):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert output.getvalue().startswith('\x1b[?25l' + '\x1bD' * 4 + '\x1b[4A\x1b7\x1b[1;20r')
    cleared = ''.join(f'\x1b[{row};1H\x1b[2K' for row in range(21, 25))
    assert f'\x1b[r{cleared}\x1b8' in output.getvalue()
    assert '\n' not in output.getvalue()
    assert '\x1b[r' in output.getvalue()
    assert output.getvalue().endswith('\x1b8\x1b[?25h')


def test_plugin_segments_are_appended_and_empties_are_skipped() -> None:
    status = Status(model='m', status_segments=(lambda: '', lambda: '/tmp/work'))
    assert status.text().endswith('ready | /tmp/work')
    assert status.toolbar()[-1] == (MUTED, ' | /tmp/work')
    assert ''.join(text for _, text in status.toolbar()) == status.text()
    status.context_alert = True
    assert ''.join(text for _, text in status.toolbar()) == status.text()


def test_a_failing_segment_reports_itself_instead_of_breaking_the_row() -> None:
    def broken() -> str:
        raise RuntimeError('no directory')

    status = Status(model='m', status_segments=(broken, lambda: 'last'))
    assert status.text().endswith('ready | !RuntimeError | last')


def test_a_segment_returning_a_non_string_is_reported_not_joined() -> None:
    status = Status(model='m', status_segments=(lambda: cast(str, 42), lambda: cast(str, None), lambda: 'ok'))
    assert status.text().endswith('ready | !int | ok')


def test_a_fragment_cannot_break_the_prompt_row() -> None:
    status = Status(model='m', status_segments=(lambda: 'a\nb\x1b[31mc',))
    assert '\n' not in status.text()
    assert '\x1b' not in status.text()
    assert ''.join(text for _, text in status.toolbar()) == status.text()


async def test_plugin_segments_are_painted_muted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COLORTERM', 'truecolor')
    output = io.StringIO()
    status = Status(model='m', status_segments=(lambda: 'cwd: /tmp',))
    async with StatusLine(Console(file=output, force_terminal=True, width=80, height=24), status):
        pass
    muted = sgr(MUTED)
    painted = output.getvalue()
    assert f'{muted}c{muted}w{muted}d' in painted
    assert f'{muted}m{muted}p' in painted
    assert sgr(WARNING) not in painted


async def test_a_fragment_wider_than_the_terminal_still_mutes_only_itself() -> None:
    output = io.StringIO()
    status = Status(model='m', status_segments=(lambda: 'x' * 200,))
    async with StatusLine(Console(file=output, force_terminal=True, width=60, height=24), status):
        pass
    painted = output.getvalue()
    assert f'{sgr(MUTED)}m' not in painted
    assert f'{sgr(MUTED)}x' in painted
