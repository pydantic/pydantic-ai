"""The live panel paints frames from the transcript, on the alternate screen, and prints the session on close."""

import io

import pytest
from rich.console import Console
from rich.text import Text
from termflow.themes import reset_palette

from pydantic_clai2.ui.prompt.prompt_surface import ENTER, FRAME_INTERVAL, LEAVE, MODES_OFF, MODES_ON, PromptSurface
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.prompt.transcript_view import SCROLLED_HINT
from pydantic_clai2.ui.rendering import theme
from tests.clai2.surface_terminal import SurfaceTerminal

ROWS = ('TOP', 'DRAFT', 'BOTTOM', 'FOOTER')


class Screen:
    def __init__(self, *, width: int = 80, height: int = 24) -> None:
        self.now = 0.0
        self.terminal = SurfaceTerminal(width=width, height=height)
        self.surface = PromptSurface(
            output=self.terminal,
            size=lambda: (self.terminal.width, self.terminal.height),
            clock=lambda: self.now,
        )

    def write(self, text: str) -> None:
        """Write a frame interval apart, so every write paints."""
        self.now += FRAME_INTERVAL
        self.surface.write(text)

    def lines(self) -> list[str]:
        return self.terminal.lines()


def test_modified_key_and_mouse_reporting_are_scoped_to_editor_ownership() -> None:
    output = io.StringIO()
    surface = PromptSurface(output=output, size=lambda: (80, 24))
    for activation in range(1, 3):
        surface.paint(ROWS)
        surface.paint((*ROWS, 'EXTRA ROW'))
        assert output.getvalue().count(MODES_ON) == activation
        assert output.getvalue().count(MODES_OFF) == activation - 1
        surface.release()
        surface.release()
        assert output.getvalue().count(MODES_OFF) == activation
    # Kitty keeps a flag stack per screen: push after entering, pop before leaving.
    assert output.getvalue().index(ENTER) < output.getvalue().index(MODES_ON)
    surface.restore()
    assert output.getvalue().rindex(MODES_OFF) < output.getvalue().rindex(LEAVE)
    assert output.getvalue().count(ENTER) == output.getvalue().count(LEAVE) == 1


def test_output_fills_the_panel_above_the_pinned_rows() -> None:
    screen = Screen()
    screen.write('before the editor\n')
    assert screen.terminal.getvalue() == '', 'nothing paints before the editor opens'
    screen.surface.paint(ROWS)
    assert screen.terminal.alternate
    for chunk in ('one', ' two', '\n', 'next'):
        screen.write(chunk)
    lines = screen.lines()
    assert lines[:3] == ['before the editor', 'one two', 'next']
    assert lines[-4:] == list(ROWS)
    assert not any(lines[3:-4])


def test_typing_sends_only_the_changed_cells_without_showing_the_cursor() -> None:
    screen = Screen()
    screen.write('transcript\n')
    screen.surface.paint(ROWS)
    start = len(screen.terminal.getvalue())
    screen.surface.paint(('TOP', 'DRAFT!', 'BOTTOM', 'FOOTER'))
    update = screen.terminal.getvalue()[start:]
    assert '!' in update and 'DRAFT' not in update and 'transcript' not in update and 'FOOTER' not in update
    assert '\x1b[?25h' not in update and '\x1b[2J' not in update
    start = len(screen.terminal.getvalue())
    screen.surface.paint(('TOP', 'DRAFT!', 'BOTTOM', 'FOOTER'))
    assert screen.terminal.getvalue()[start:] == ''


def test_editor_growth_takes_rows_from_the_transcript_and_gives_them_back() -> None:
    screen = Screen(height=8)
    screen.surface.paint(ROWS)
    for index in range(10):
        screen.write(f'line {index}\n')
    screen.surface.paint(('TOP', 'DRAFT', 'SECOND LINE', 'BOTTOM', 'FOOTER'))
    assert screen.lines() == ['line 8', 'line 9', '', 'TOP', 'DRAFT', 'SECOND LINE', 'BOTTOM', 'FOOTER']
    screen.surface.paint(ROWS)
    assert screen.lines() == ['line 7', 'line 8', 'line 9', '', *ROWS]


def test_writes_inside_a_frame_interval_wait_for_refresh() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.write('first\n')
    screen.surface.write('second\n')
    assert 'second' not in screen.lines()
    screen.surface.refresh()
    assert 'second' in screen.lines()
    start = len(screen.terminal.getvalue())
    screen.surface.refresh()
    assert screen.terminal.getvalue()[start:] == ''


def test_resize_rewraps_the_transcript_from_the_retained_output() -> None:
    screen = Screen(width=40)
    screen.surface.paint(ROWS)
    screen.write('x' * 60 + '\n')
    assert screen.lines()[:2] == ['x' * 40, 'x' * 20]
    screen.terminal.resize(width=80, height=30)
    screen.surface.resize_notice()
    screen.surface.paint(ROWS)
    assert screen.lines()[0] == 'x' * 60
    assert screen.lines()[-4:] == list(ROWS)
    assert '\x1b[3J' not in screen.terminal.getvalue()


def test_resize_notice_repaints_every_cell() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    start = len(screen.terminal.getvalue())
    screen.surface.resize_notice()
    screen.surface.paint(ROWS)
    assert '\x1b[2J' in screen.terminal.getvalue()[start:]


def test_menus_get_the_main_screen_and_held_output_shows_afterwards() -> None:
    screen = Screen()
    screen.write('conversation\n')
    screen.surface.paint(ROWS)
    screen.surface.release()
    with screen.surface.held(), screen.surface.held():
        assert not screen.terminal.alternate
        start = len(screen.terminal.getvalue())
        screen.write('streamed while the menu was open\n')
        screen.surface.refresh()
        assert screen.terminal.getvalue()[start:] == ''
    screen.surface.paint(ROWS)
    assert screen.terminal.alternate
    assert screen.lines()[:2] == ['conversation', 'streamed while the menu was open']


def test_inline_widgets_keep_the_panel_on_screen() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.surface.release()
    with screen.surface.held(leave_screen=False):
        screen.write('question\n')
        assert 'question' not in screen.lines()
        screen.surface.paint(('1. Patch',))
        assert screen.terminal.alternate
        assert screen.lines()[0] == 'question' and screen.lines()[-1] == '1. Patch'


def test_released_panel_shows_command_output_without_editor_rows() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.write('partial')
    screen.surface.release()
    assert screen.lines()[0] == 'partial'
    assert not any(row in screen.lines() for row in ROWS)
    screen.write('command output\n')
    assert screen.lines()[:2] == ['partial', 'command output']
    assert screen.terminal.alternate


async def test_close_prints_the_session_into_the_main_screen_once() -> None:
    screen = Screen()
    screen.terminal.write('$ clai2\r\n')
    transcript = screen.surface.transcript
    console = Console(file=screen.terminal, force_terminal=True)
    with transcript.capture(console):
        console.print('banner')
    screen.terminal.write('\r')  # Startup runs in cooked mode, where the newline also returned the carriage.
    screen.surface.paint(ROWS)
    screen.write('\x1b]8;;https://example.com\x1b\\linked\x1b]8;;\x1b\\ answer')
    await screen.surface.drain()
    screen.surface.restore()
    assert not screen.terminal.alternate
    assert screen.lines()[:3] == ['$ clai2', 'banner', 'linked answer']
    printed = Text.from_ansi(screen.terminal.getvalue().rsplit(LEAVE, 1)[1])
    assert printed.get_style_at_offset(Console(), 0).link == 'https://example.com', 'scrollback keeps links'
    screen.surface.restore()
    assert screen.lines()[:4] == ['$ clai2', 'banner', 'linked answer', '']


def test_close_keeps_lines_hidden_by_clear_in_scrollback() -> None:
    output = io.StringIO()
    surface = PromptSurface(output=output, size=lambda: (80, 24))
    surface.write('before clear\n')
    surface.transcript.clear()
    surface.write('after clear\n')
    assert [Text.from_ansi(row).plain for row in surface.transcript.frame(width=80, height=5).rows] == [
        'after clear',
        '',
    ]
    surface.restore()
    assert output.getvalue() == 'before clear\nafter clear\n'


@pytest.mark.parametrize('terminator', ['\x07', '\x1b\\'])
def test_palette_controls_reach_the_terminal_and_repaint(terminator: str) -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    start = len(screen.terminal.getvalue())
    screen.write(f'\x1b]11;#0a1929{terminator}')
    assert screen.terminal.getvalue()[start:] == f'\x1b]11;#0a1929{terminator}', 'the frame waits for the next refresh'
    screen.surface.refresh()
    assert '\x1b[2J' in screen.terminal.getvalue()[start:], 'every cell repaints in the new colours'
    assert screen.surface.transcript.frame(width=80, height=2).rows == ('',)


async def test_palette_reset_reaches_the_terminal_unsplit_on_a_slow_runner() -> None:
    """Termflow writes a reset as three controls; no frame may paint between them, however slow."""
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.write('partial')
    await screen.surface.drain()
    start = len(screen.terminal.getvalue())

    class Slow(io.StringIO):
        def write(self, text: str) -> int:
            screen.now += 1  # Every write lands after the frame interval.
            return screen.surface.write(text)

    reset_palette(output=Slow())
    update = screen.terminal.getvalue()[start:]
    assert update == '\x1b]104\x07\x1b]111\x07\x1b]110\x07'
    screen.surface.refresh()
    assert screen.terminal.getvalue()[start:].count('\x1b[2J') == 1, 'one repaint, not one per control'
    await screen.surface.drain()
    rows = [Text.from_ansi(row).plain for row in screen.surface.transcript.frame(width=80, height=5).rows]
    assert rows == ['partial', ''], 'controls are not content, so no blank line follows'


def test_scrolling_holds_the_view_while_output_arrives_and_returns_to_follow() -> None:
    screen = Screen(height=10)
    screen.surface.paint(ROWS)
    for index in range(20):
        screen.write(f'line {index}\n')
    assert screen.lines()[:6] == [f'line {index}' for index in range(15, 20)] + ['']
    assert screen.surface.page == 5
    screen.surface.scroll(screen.surface.page)
    view = screen.lines()[:6]
    assert view[:5] == [f'line {index}' for index in range(10, 15)]
    assert view[5].endswith(SCROLLED_HINT.rstrip())
    screen.write('line 20\n')
    assert screen.lines()[:5] == view[:5]
    screen.surface.scroll(100)
    assert screen.lines()[0] == 'line 0', 'scrolling stops at the oldest row'
    screen.surface.scroll(-100)
    assert screen.surface.view.anchor is None
    assert screen.lines()[:6] == [f'line {index}' for index in range(16, 21)] + ['']


def test_short_transcripts_do_not_scroll() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.write('only line\n')
    screen.surface.scroll(5)
    assert screen.surface.view.anchor is None
    assert screen.lines()[0] == 'only line'


def test_scroll_anchor_survives_eviction_by_following_again() -> None:
    terminal = SurfaceTerminal(width=40, height=8)
    surface = PromptSurface(
        output=terminal, size=lambda: (40, 8), transcript=TranscriptBuffer(max_lines=10), clock=lambda: 1.0
    )
    surface.paint(('EDITOR',))
    for index in range(10):
        surface.write(f'line {index}\n')
    surface.scroll(3)
    assert surface.view.anchor is not None
    for index in range(10, 30):
        surface.write(f'line {index}\n')
    surface.refresh()
    assert surface.view.anchor is None
    assert 'line 29' in terminal.lines()


def test_markdown_renders_again_for_a_new_width_or_theme() -> None:
    calls: list[tuple[int, str]] = []

    def render(*, source: str, width: int) -> str:
        calls.append((width, theme.name()))
        return f'{theme.name()} {width}: {source}\n'

    screen = Screen(width=40)
    screen.surface.paint(ROWS)
    block = screen.surface.markdown(render=render, width=40)
    block.extend('**bold**')
    block.write('streamed bold\n')
    screen.surface.refresh()
    assert screen.lines()[0] == 'streamed bold'
    assert calls == []
    screen.terminal.resize(width=60, height=24)
    screen.surface.paint(ROWS)
    assert screen.lines()[0] == 'default 60: **bold**'
    with theme.use(lambda: 'github_light'):
        screen.surface.paint(ROWS)
        assert screen.lines()[0] == 'github_light 60: **bold**'
        screen.surface.paint(ROWS)
    assert calls == [(60, 'default'), (60, 'github_light')]
    block.freeze()
    screen.surface.paint(ROWS)
    assert screen.lines()[0] == 'streamed bold', 'an aborted part keeps what it showed'


async def test_drain_settles_a_partial_line_once() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (80, 24))
    surface.write('partial')
    await surface.drain()
    await surface.drain()
    assert [Text.from_ansi(row).plain for row in surface.transcript.frame(width=80, height=5).rows] == ['partial', '']
    assert surface.isatty() is False


def test_scroll_before_open_and_during_a_hold_does_not_paint() -> None:
    screen = Screen()
    screen.surface.scroll(1)
    assert screen.terminal.getvalue() == ''
    screen.surface.paint(ROWS)
    with screen.surface.held():
        start = len(screen.terminal.getvalue())
        screen.surface.scroll(1)
        assert screen.terminal.getvalue()[start:] == ''


def test_transcript_scrolls_rows_within_one_wrapped_item_and_skips_empty_parts() -> None:
    from pydantic_clai2.ui.prompt.transcript_view import TranscriptView

    buffer = TranscriptBuffer()
    view = TranscriptView(buffer)
    assert view.window(width=4, height=0) == []

    # Empty Markdown parts have no rows, but still have an id. They were streamed at another
    # width, so each window renders them again, and they stay empty.
    def render(*, source: str, width: int) -> str:
        assert (source, width) == ('', 4)
        return ''

    buffer.markdown(render=render, width=1, changed=lambda: None)
    buffer.write('abcdefghijklmnopqrstuvwxyz\n')
    buffer.markdown(render=render, width=1, changed=lambda: None)
    assert view.window(width=4, height=3) == ['uvwx', 'yz', '']
    view.scroll(3)
    assert view.window(width=4, height=3) == ['ijkl', 'mnop', 'qrst']
    view.scroll(-1)
    assert view.window(width=4, height=3) == ['mnop', 'qrst', 'uvwx']
    view.anchor = (2, 0)  # A part that is still empty.
    assert view.window(width=4, height=3) == ['qrst', 'uvwx', 'yz']
    view.scroll(-100)
    assert view.anchor is None
    # An empty anchor at the very beginning falls back to the last row.
    view.anchor = (0, 0)
    assert view.window(width=4, height=3) == ['uvwx', 'yz', '']
