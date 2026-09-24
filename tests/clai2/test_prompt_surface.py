"""Blank while resizing, then replay output without stale-coordinate erases."""

import io
from typing import IO

import pytest
from surface_terminal import SurfaceTerminal

from pydantic_clai2 import prompt_surface
from pydantic_clai2.prompt_surface import PromptSurface

ROWS = ('TOP', 'DRAFT', 'BOTTOM', 'FOOTER')


class Screen:
    def __init__(self) -> None:
        self.now = 0.0
        self.terminal = SurfaceTerminal(width=80, height=24)
        self.surface = PromptSurface(
            output=self.terminal,
            size=lambda: (self.terminal.width, self.terminal.height),
            clock=lambda: self.now,
        )
        self.surface.paint(ROWS)

    def resize(self, *, width: int, height: int) -> None:
        self.terminal.resize(width=width, height=height)
        self.surface.paint(ROWS)

    def settle(self, *, rows: tuple[str, ...] = ROWS) -> None:
        self.now += 0.3
        self.surface.paint(rows)


@pytest.mark.parametrize('tty', [False, True])
async def test_streaming_does_not_touch_editor(tty: bool) -> None:
    class Output(io.StringIO):
        def isatty(self) -> bool:
            return tty

    output = Output()
    surface = PromptSurface(output=output, size=lambda: (80, 24))
    surface.paint(ROWS)
    start = len(output.getvalue())
    for text in ('one', ' two', '\n', 'next'):
        surface.write(text)
        surface.flush()
        surface.paint(ROWS)
    await surface.drain()
    assert output.getvalue()[start:] == ('one two\r\nnext\r\n' if tty else 'one two\nnext\n')
    assert surface.isatty() is tty
    surface.release()
    start = len(output.getvalue())
    surface.release()
    assert output.getvalue()[start:] == ''


def test_typing_changes_only_the_draft_row_without_showing_cursor() -> None:
    screen = Screen()
    start = len(screen.terminal.getvalue())
    screen.surface.paint(('TOP', 'DRAFT!', 'BOTTOM', 'FOOTER'))
    update = screen.terminal.getvalue()[start:]
    assert '\x1b[22;1H' in update and 'DRAFT!' in update
    assert 'TOP' not in update and 'BOTTOM' not in update and 'FOOTER' not in update
    assert '\x1b[?25' not in update and '\x1b[2J' not in update


def test_editor_growth_is_not_a_physical_resize() -> None:
    screen = Screen()
    start = len(screen.terminal.getvalue())
    screen.surface.paint(('TOP', 'DRAFT', 'SECOND LINE', 'BOTTOM', 'FOOTER'))
    screen.surface.paint(ROWS)
    assert '\x1b[2J' not in screen.terminal.getvalue()[start:]
    assert screen.terminal.lines()[-4:] == list(ROWS)


def test_resize_blanks_viewport_and_resets_quiet_timer_until_settled() -> None:
    screen = Screen()
    screen.surface.write('before resize\npartial')
    screen.resize(width=100, height=40)
    assert not any(screen.terminal.lines())
    screen.surface.write(' continuation\nnew output\n')
    assert not any(screen.terminal.lines())
    for width, height in ((50, 18), (120, 45), (80, 24)):
        screen.now += 0.15
        screen.resize(width=width, height=height)
        assert not any(screen.terminal.lines())
    screen.now += 0.249
    screen.surface.paint(('TOP', 'LATEST DRAFT', 'BOTTOM', 'FOOTER'))
    assert not any(screen.terminal.lines())
    screen.now += 0.002
    screen.surface.paint(('TOP', 'LATEST DRAFT', 'BOTTOM', 'FOOTER'))
    lines = screen.terminal.lines()
    assert lines[-4:] == ['TOP', 'LATEST DRAFT', 'BOTTOM', 'FOOTER']
    assert 'before resize' in lines and 'partial continuation' in lines and 'new output' in lines
    assert 'DRAFT' not in lines
    assert '\x1b[3J' not in screen.terminal.getvalue()
    start = len(screen.terminal.getvalue())
    screen.surface.paint(('TOP', 'LATEST DRAFT', 'BOTTOM', 'FOOTER'))
    assert screen.terminal.getvalue()[start:] == ''


def test_signal_notices_keep_screen_blank_even_when_reported_size_lags() -> None:
    screen = Screen()
    screen.surface.resize_notice()
    screen.surface.paint(ROWS)
    assert not any(screen.terminal.lines())
    screen.now += 0.2
    screen.surface.resize_notice()
    screen.now += 0.2
    screen.surface.paint(ROWS)
    assert not any(screen.terminal.lines())
    screen.settle()
    assert screen.terminal.lines()[-4:] == list(ROWS)


def test_resize_during_write_does_not_require_a_paint_to_pause() -> None:
    screen = Screen()
    screen.terminal.resize(width=100, height=40)
    screen.surface.write('queued without a poll\n')
    assert not any(screen.terminal.lines())
    screen.settle()
    assert 'queued without a poll' in screen.terminal.lines()


def test_replay_does_not_itself_append_to_or_clear_native_scrollback() -> None:
    screen = Screen()
    for index in range(50):
        screen.surface.write(f'line {index:02d}\n')
    screen.resize(width=80, height=40)
    history = screen.terminal.history.copy()
    screen.settle()
    assert screen.terminal.history == history
    assert 'line 49' in screen.terminal.lines()
    assert not any('DRAFT' in line for line in screen.terminal.history)


@pytest.mark.parametrize('ending', ['partial', 'a' * 80, 'line\n'])
def test_streaming_continues_at_replayed_cursor(ending: str) -> None:
    screen = Screen()
    screen.surface.write(ending)
    screen.resize(width=80, height=40)
    screen.settle()
    screen.surface.write('tail')
    lines = screen.terminal.lines()
    if ending == 'partial':
        assert 'partialtail' in lines
    else:
        assert 'tail' in lines
        assert ending.rstrip('\n') in lines


def test_tiny_terminal_then_grow_keeps_transcript_and_draft() -> None:
    screen = Screen()
    screen.surface.write('retained\n')
    screen.resize(width=10, height=2)
    screen.settle(rows=('DRAFT',))
    assert 'retained' in screen.terminal.lines()
    assert 'DRAFT' not in screen.terminal.lines()
    screen.resize(width=80, height=24)
    screen.settle()
    assert screen.terminal.lines()[-4:] == list(ROWS)
    assert 'retained' in screen.terminal.lines()


@pytest.mark.parametrize('pending', [False, True])
def test_release_during_resize_flushes_output_and_restores_terminal(pending: bool) -> None:
    screen = Screen()
    screen.terminal.resize(width=60, height=20)
    if pending:
        screen.surface.write('queued before exit\n')
    screen.surface.release()
    assert screen.terminal.getvalue().endswith('\x1b[?25h\x1b[?2026l')
    if pending:
        assert 'queued before exit' in screen.terminal.lines()
    screen.surface.resize_notice()
    screen.surface.write('menu\n')
    screen.surface.paint(ROWS)
    assert screen.terminal.lines()[-4:] == list(ROWS)
    assert screen.terminal.getvalue().count('\x1b[?25l') >= 2


def test_long_resize_spools_all_new_output_and_closes_the_spool(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[IO[str]] = []
    original = prompt_surface.SpooledTemporaryFile

    def spool(**kwargs: object) -> IO[str]:
        stream = original(max_size=16, mode='w+t', encoding='utf-8', newline='')
        opened.append(stream)
        return stream

    monkeypatch.setattr(prompt_surface, 'SpooledTemporaryFile', spool)
    output = io.StringIO()
    now = 0.0
    size = (80, 24)
    surface = PromptSurface(output=output, size=lambda: size, clock=lambda: now)
    surface.paint(ROWS)
    size = (100, 40)
    content = 'tool output line\n' * 10000
    surface.write(content)
    assert content not in output.getvalue()
    now = 0.3
    surface.paint(ROWS)
    assert output.getvalue().endswith(content)
    assert len(opened) == 1 and opened[0].closed


async def test_empty_output_and_empty_drain() -> None:
    output = io.StringIO()
    surface = PromptSurface(output=output, size=lambda: (1, 2))
    assert surface.write('') == 0
    await surface.drain()
    assert output.getvalue() == ''


def test_failed_resize_release_still_closes_output_spool(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[io.StringIO] = []

    def spool(**kwargs: object) -> IO[str]:
        stream = io.StringIO()
        opened.append(stream)
        return stream

    class Output(io.StringIO):
        broken = False

        def write(self, text: str) -> int:
            if self.broken:
                raise OSError('terminal gone')
            return super().write(text)

    monkeypatch.setattr(prompt_surface, 'SpooledTemporaryFile', spool)
    output = Output()
    size = (80, 24)
    surface = PromptSurface(output=output, size=lambda: size)
    surface.paint(ROWS)
    size = (100, 40)
    surface.write('queued')
    output.broken = True
    with pytest.raises(OSError, match='terminal gone'):
        surface.release()
    assert len(opened) == 1 and opened[0].closed
    surface.release()


def test_replacement_editor_can_reuse_transcript_after_reload() -> None:
    screen = Screen()
    screen.surface.write('retained across reload\n')
    screen.surface.release()
    replacement = PromptSurface(
        output=screen.terminal,
        size=lambda: (screen.terminal.width, screen.terminal.height),
        clock=lambda: screen.now,
        transcript=screen.surface.transcript,
    )
    replacement.paint(ROWS)
    screen.terminal.resize(width=100, height=40)
    replacement.paint(ROWS)
    screen.now += 0.3
    replacement.paint(ROWS)
    assert 'retained across reload' in screen.terminal.lines()


def test_popup_reopening_reuses_gap_without_scrolling_blank_lines() -> None:
    screen = Screen()
    screen.surface.write('transcript tail\npartial')
    popup = ('TOP', 'DRAFT', 'BOTTOM', 'one', 'two', 'three', 'FOOTER')
    screen.surface.paint(popup)
    history = screen.terminal.history.copy()
    transcript_rows = screen.terminal.lines()[: -len(popup)]
    for count in (1, 2, 3, 0, 3, 1, 0, 3):
        screen.surface.paint((*ROWS[:-1], *('suggestion' for _ in range(count)), ROWS[-1]))
        assert screen.terminal.history == history
    assert screen.terminal.lines()[: -len(popup)] == transcript_rows
    screen.surface.write(' continuation')
    assert 'partial continuation' in screen.terminal.lines()


@pytest.mark.parametrize('resize', [False, True])
def test_release_finishes_partial_output_before_shell_can_overwrite_it(resize: bool) -> None:
    screen = Screen()
    screen.surface.write('partial streamed response')
    if resize:
        screen.resize(width=100, height=40)
    screen.surface.release()
    screen.terminal.write('shell prompt')
    assert 'partial streamed response' in screen.terminal.lines()
    assert 'shell prompt' in screen.terminal.lines()


@pytest.mark.parametrize('released', [False, True])
def test_held_output_survives_resize_and_release_until_the_outer_hold_exits(released: bool) -> None:
    output = io.StringIO()
    now = 0.0
    size = (80, 24)
    surface = PromptSurface(output=output, size=lambda: size, clock=lambda: now)
    surface.paint(ROWS)
    surface.write('before\n')
    with surface.held():
        surface.write('one\n')
        with surface.held():
            surface.write('two\n')
        assert 'one' not in output.getvalue()
        size = (100, 40)
        surface.write('three\n')
        now = 0.3
        surface.paint(ROWS)
        if released:
            surface.release()
        assert 'one' not in output.getvalue()
    text = output.getvalue()
    assert text.index('before') < text.index('one\n') < text.index('two\n') < text.index('three\n')


def test_hold_ending_mid_resize_replays_once_the_viewport_settles() -> None:
    output = io.StringIO()
    now = 0.0
    size = (80, 24)
    surface = PromptSurface(output=output, size=lambda: size, clock=lambda: now)
    surface.paint(ROWS)
    with surface.held():
        surface.write('held\n')
        size = (100, 40)
        surface.paint(ROWS)
    assert 'held' not in output.getvalue()
    now = 0.3
    surface.paint(ROWS)
    assert 'held\n' in output.getvalue()
