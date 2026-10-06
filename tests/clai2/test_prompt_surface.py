"""The live panel paints frames from the transcript, on the alternate screen, and prints the session on close."""

import io

import pytest
from rich.console import Console
from rich.text import Text
from termflow.ansi import make_clipboard_copy
from termflow.live import ScreenBuffer
from termflow.live.buffer import REVERSE
from termflow.themes import PALETTES, reset_palette

from pydantic_clai2.ui.prompt.prompt_selection import MouseReport, Selection, mouse_report
from pydantic_clai2.ui.prompt.prompt_surface import (
    ENTER,
    FRAME_INTERVAL,
    LEAVE,
    MAX_HELD_OSC,
    MODES_OFF,
    MODES_ON,
    WHEEL_ROWS,
    PromptSurface,
)
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


def test_close_leaves_lines_forgotten_by_clear_out_of_scrollback() -> None:
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
    assert output.getvalue() == 'after clear\n'


def test_clear_blanks_the_panel_and_follows_new_output() -> None:
    screen = Screen(height=10)
    screen.surface.paint(ROWS)
    for index in range(20):
        screen.write(f'line {index}\n')
    screen.write('partial')
    screen.surface.scroll(3)
    start = len(screen.terminal.getvalue())
    screen.surface.clear()
    screen.surface.paint(ROWS)
    assert '\x1b[2J' in screen.terminal.getvalue()[start:], 'every cell repaints'
    assert '\x1b[3J' not in screen.terminal.getvalue(), 'the terminal scrollback is left alone'
    assert screen.surface.view.anchor is None
    assert screen.lines() == [''] * 6 + list(ROWS)
    screen.write('after\n')
    screen.surface.refresh()
    assert screen.lines()[:2] == ['after', '']
    screen.surface.restore()
    assert not screen.terminal.alternate
    assert screen.lines()[0] == 'after', 'cleared output does not reach the scrollback on exit'


def test_clear_keeping_current_output_lets_a_running_turn_finish_its_line_or_part() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.write('earlier\n')
    screen.write('\x1b[1mbold ')
    screen.surface.clear(keep_current=True)
    screen.write('line\n')
    assert screen.lines()[:2] == ['bold line', '']

    def render(*, source: str, width: int) -> str:
        # The width and theme never change, so the streamed rows show and nothing renders again.
        raise NotImplementedError

    block = screen.surface.markdown(render=render, width=80)
    block.write('streamed ')
    screen.surface.clear(keep_current=True)
    block.write('answer\n')
    screen.surface.refresh()
    assert screen.lines()[:2] == ['streamed answer', '']

    screen.write('tool output\n')
    screen.surface.clear(keep_current=True)
    screen.surface.paint(ROWS)
    assert not any(screen.lines()[:-4]), 'a part followed by other output is finished'


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


def test_a_theme_change_repaints_earlier_tool_output_in_the_new_palette() -> None:
    def truecolor(colour: str) -> str:
        red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
        return f'38;2;{red};{green};{blue}m'

    selected = ['default']
    screen = Screen(width=40)
    with theme.use(lambda: selected[0]):
        screen.write(f'\x1b[{truecolor(theme.color(theme.MUTED))}earlier tool output\x1b[0m\n')
        screen.surface.paint(ROWS)
        painted = len(screen.terminal.getvalue())
        selected[0] = 'tokyo_night'
        screen.surface.paint(ROWS)
    assert truecolor(PALETTES['tokyo_night'].ansi[8]) in screen.terminal.getvalue()[painted:]
    assert screen.lines()[0] == 'earlier tool output'


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


@pytest.mark.parametrize('terminator', ['\x07', '\x1b\\'])
def test_palette_controls_split_across_writes_reach_the_terminal_whole(terminator: str) -> None:
    control = f'\x1b]11;#0a1929{terminator}'
    for split in range(1, len(control)):
        screen = Screen()
        screen.surface.paint(ROWS)
        start = len(screen.terminal.getvalue())
        screen.write('text ' + control[:split])
        screen.write(control[split:] + 'more\n')
        assert screen.terminal.getvalue()[start:].count(control) == 1, split
        assert [Text.from_ansi(row).plain for row in screen.surface.transcript.frame(width=80, height=5).rows] == [
            'text more',
            '',
        ]


def test_controls_only_split_does_not_mark_a_partial_line() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.write('\x1b]104')
    screen.write('\x07')
    assert '\x1b]104\x07' in screen.terminal.getvalue()
    assert not screen.surface._partial  # pyright: ignore[reportPrivateUsage]


def test_overlong_unterminated_control_is_dropped_not_held_forever() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.write('\x1b]11;' + 'x' * (MAX_HELD_OSC + 1))
    screen.write('\x07after\n')
    assert '\x1b]11;' not in screen.terminal.getvalue()
    assert screen.surface._held == ''  # pyright: ignore[reportPrivateUsage]


def test_split_hyperlinks_stay_in_the_transcript_and_are_not_forwarded() -> None:
    screen = Screen()
    screen.surface.paint(ROWS)
    screen.write('\x1b]8;;https://example.com')
    screen.write('\x1b\\link\x1b]8;;\x1b\\\n')
    assert '\x1b]8;;https://example.com' not in screen.terminal.getvalue()
    assert Text.from_ansi(screen.surface.transcript.frame(width=80, height=5).rows[0]).plain == 'link'


def press(column: int, row: int) -> str:
    return f'\x1b[<0;{column};{row}M'


def drag(column: int, row: int) -> str:
    return f'\x1b[<32;{column};{row}M'


def release(column: int, row: int) -> str:
    return f'\x1b[<0;{column};{row}m'


def highlighted(surface: PromptSurface) -> list[str]:
    """The text of each painted row's reverse-video cells."""
    frame = surface._frame  # pyright: ignore[reportPrivateUsage]
    assert frame is not None
    rows: list[str] = []
    for row in range(frame.height):
        cells = range(row * frame.width, (row + 1) * frame.width)
        text = ''.join(frame.chars[index] for index in cells if frame.attrs[index] & REVERSE)
        if text and SCROLLED_HINT.strip() not in text:  # The hint is reverse video of its own.
            rows.append(text)
    return rows


def test_dragging_highlights_cells_and_releasing_copies_them() -> None:
    """The panel reports the mouse for the wheel, so most terminals no longer select text themselves."""
    screen = Screen(width=40, height=10)
    screen.surface.paint(ROWS)
    screen.write('first line here\nsecond line\nthird\n')
    assert screen.surface.transcript_key('mouse', press(7, 1)) is None
    assert highlighted(screen.surface) == [], 'a press alone selects nothing'
    assert screen.surface.transcript_key('mouse', drag(10, 1)) is None
    assert highlighted(screen.surface) == ['line']
    assert screen.surface.transcript_key('mouse', drag(3, 3)) is None
    assert highlighted(screen.surface) == ['line here' + ' ' * 25, 'second line' + ' ' * 29, 'thi']
    start = len(screen.terminal.getvalue())
    assert screen.surface.transcript_key('mouse', release(3, 3)) == 'line here\nsecond line\nthi'
    assert screen.terminal.getvalue()[start:] == make_clipboard_copy('line here\nsecond line\nthi')
    assert highlighted(screen.surface), 'the copied cells stay highlighted'
    assert screen.lines()[:3] == ['first line here', 'second line', 'third'], 'selection never changes text'


def test_dragging_backwards_selects_the_same_cells_in_reading_order() -> None:
    screen = Screen(width=40, height=10)
    screen.surface.paint(ROWS)
    screen.write('alpha beta\ngamma\n')
    for report in (press(3, 2), drag(1, 1), drag(7, 1), release(7, 1)):
        copied = screen.surface.transcript_key('mouse', report)
    assert copied == 'beta\ngam'


def test_clicks_other_buttons_and_blank_drags_copy_nothing() -> None:
    screen = Screen(width=40, height=10)
    screen.surface.paint(ROWS)
    screen.write('text\n')
    start = len(screen.terminal.getvalue())
    # A click, a drag that ends on its own cell, a release without a press, a right drag, junk.
    for report in (
        press(2, 1),
        release(2, 1),
        release(2, 1),
        '\x1b[<2;2;1M',
        '\x1b[<34;4;1M',
        '\x1b[<2;4;1m',
        'not a mouse report',
    ):
        assert screen.surface.transcript_key('mouse', report) is None
    assert highlighted(screen.surface) == []
    for report in (press(20, 3), drag(30, 4)):
        screen.surface.transcript_key('mouse', report)
    assert screen.surface.transcript_key('mouse', release(30, 4)) is None, 'blank cells have nothing to copy'
    assert '\x1b]52;' not in screen.terminal.getvalue()[start:]


def test_scrolling_resizing_and_releasing_the_panel_clear_the_highlight() -> None:
    screen = Screen(width=40, height=10)
    screen.surface.paint(ROWS)
    for index in range(20):
        screen.write(f'line {index}\n')

    def select() -> None:
        for report in (press(1, 1), drag(4, 1), release(4, 1)):
            screen.surface.transcript_key('mouse', report)
        assert highlighted(screen.surface) == ['line']

    select()
    screen.surface.transcript_key('mouse', '\x1b[<64;1;1M')
    assert screen.surface.view.anchor is not None, 'the wheel still scrolls'
    assert highlighted(screen.surface) == [], 'the selected text moved away'
    select()
    screen.surface.transcript_key('pagedown')
    assert screen.surface.view.anchor is None and highlighted(screen.surface) == []
    select()
    screen.surface.resize_notice()
    screen.surface.paint(ROWS)
    assert highlighted(screen.surface) == []
    select()
    screen.surface.clear()  # Ctrl+L.
    assert screen.surface.selection.anchor is None
    for index in range(20):
        screen.write(f'line {index}\n')
    select()
    screen.surface.release()
    assert highlighted(screen.surface) == [], 'a menu or command never shows a stale highlight'
    screen.surface.release()


def test_output_that_moves_the_selected_text_drops_the_selection() -> None:
    """Screen cells, not transcript rows: a release must never copy text the user did not drag over."""
    screen = Screen(width=40, height=10)
    screen.surface.paint(ROWS)
    for index in range(4):
        screen.write(f'line {index}\n')
    for report in (press(1, 1), drag(6, 1)):
        screen.surface.transcript_key('mouse', report)
    assert highlighted(screen.surface) == ['line 0']
    screen.write('line 4\n')  # Below the selection: the selected cells still show the same text.
    assert highlighted(screen.surface) == ['line 0']
    screen.write('line 5\nline 6\n')  # Following the newest output scrolls `line 0` away.
    assert screen.lines()[0] != 'line 0'
    assert highlighted(screen.surface) == []
    assert screen.surface.transcript_key('mouse', drag(6, 1)) is None, 'the drag ended with the selection'
    assert screen.surface.transcript_key('mouse', release(6, 1)) is None
    assert '\x1b]52;' not in screen.terminal.getvalue()


def test_a_drag_during_a_hold_highlights_on_the_next_frame() -> None:
    """An inline question repaints the panel itself after each report."""
    screen = Screen(width=40, height=10)
    screen.surface.paint(ROWS)
    screen.write('context\n')
    screen.surface.release()
    with screen.surface.held(leave_screen=False):
        screen.surface.paint(('1. Patch',))
        for report in (press(1, 1), drag(3, 1)):
            screen.surface.transcript_key('mouse', report)
        assert highlighted(screen.surface) == []
        screen.surface.paint(('1. Patch',))
        assert highlighted(screen.surface) == ['con']
        assert screen.surface.transcript_key('mouse', release(3, 1)) == 'con'


def test_a_release_before_any_frame_copies_nothing() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (40, 10))
    surface.selection = Selection(anchor=(0, 0), head=(0, 3), held=True)
    assert surface.transcript_key('mouse', release(4, 1)) is None


def test_cells_already_in_reverse_video_still_look_selected() -> None:
    frame = ScreenBuffer(4, 1)
    frame.attrs[1] = REVERSE  # The editor's painted cursor, for example.
    Selection(anchor=(0, 0), head=(0, 2)).highlight(frame, rows=1, previous=None)
    assert [attrs & REVERSE for attrs in frame.attrs] == [REVERSE, REVERSE, REVERSE, 0]


def test_releasing_another_button_mid_drag_does_not_end_it() -> None:
    screen = Screen(width=40, height=10)
    screen.surface.paint(ROWS)
    screen.write('alpha beta\n')
    for report in (press(1, 1), drag(3, 1), '\x1b[<2;3;1M'):
        screen.surface.transcript_key('mouse', report)
    assert screen.surface.transcript_key('mouse', '\x1b[<2;3;1m') is None, 'the right button let go'
    assert screen.surface.transcript_key('mouse', drag(5, 1)) is None
    assert screen.surface.transcript_key('mouse', release(5, 1)) == 'alpha'


def test_malformed_mouse_reports_are_ignored() -> None:
    assert mouse_report('\x1b[<' + '9' * 5000 + ';1;1M') is None, 'longer than `int` accepts'
    assert mouse_report('\x1b[<0;1M') is None
    assert mouse_report('\x1b[<0;10;5M') == MouseReport(button=0, cell=(4, 9), released=False)


def test_selection_spans_clamp_to_the_frame() -> None:
    selection = Selection(anchor=(-1, -5), head=(99, 99))
    assert selection.span(width=4, rows=3) == range(0, 12)
    assert selection.span(width=4, rows=2) == range(0, 8), 'rows below the transcript are not selectable'
    assert Selection(anchor=(0, 0)).span(width=4, rows=3) == range(0)
    assert Selection(anchor=(0, 0)).text(ScreenBuffer(4, 3), rows=3) == '', 'a click selects nothing'


def test_a_drag_into_the_editor_copies_only_the_transcript() -> None:
    screen = Screen(width=40, height=10)
    screen.surface.paint(ROWS)
    screen.write('answer\n')
    for report in (press(1, 1), drag(5, 9)):  # Row 9 is the editor's `BOTTOM` row.
        screen.surface.transcript_key('mouse', report)
    assert not any(row in highlighted(screen.surface) for row in ROWS)
    assert screen.surface.transcript_key('mouse', release(5, 9)) == 'answer'


def test_the_wheel_scrolls_both_ways_with_or_without_modifiers() -> None:
    screen = Screen(height=10)
    screen.surface.paint(ROWS)
    for index in range(20):
        screen.write(f'line {index}\n')
    newest = screen.lines()[0]
    screen.surface.transcript_key('mouse', '\x1b[<68;1;1M')  # Shift+wheel up.
    assert screen.lines()[0] == f'line {int(newest.split()[1]) - WHEEL_ROWS}'
    screen.surface.transcript_key('mouse', '\x1b[<65;1;1M')
    assert screen.surface.view.anchor is None
