"""Bounded transcript replay preserves text, wrapping, and styling, not controls."""

import io
from collections import deque
from collections.abc import Iterable
from typing import cast

import pytest
from rich.color import ColorSystem
from rich.console import Console
from rich.style import Style
from rich.text import Span, Text
from termflow.ansi.utils import visible_length
from termflow.themes import PALETTES

from pydantic_clai2.ui.prompt.prompt_transcript import (
    MarkdownBlock,
    TranscriptBuffer,
    TranscriptDecoder,
    render_ansi,
    style_prefix,
)
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._branding import print_banner
from pydantic_clai2.ui.rendering.recolor import recolor


def plain(buffer: TranscriptBuffer, *, width: int = 80, height: int = 24) -> list[str]:
    return [Text.from_ansi(row).plain for row in buffer.frame(width=width, height=height).rows]


def test_completed_lines_partial_tail_and_exact_cell_wrapping() -> None:
    buffer = TranscriptBuffer()
    buffer.write('first\n    indented\n界界x')
    assert plain(buffer, width=6) == ['first', '    in', 'dented', '界界x']
    buffer.write('\n')
    assert plain(buffer)[-1] == ''
    assert plain(buffer, height=2) == ['界界x', '']
    assert all(visible_length(row) <= 6 for row in buffer.frame(width=6, height=24).rows)


def test_style_split_across_writes_and_lines_survives_replay() -> None:
    buffer = TranscriptBuffer()
    buffer.write('\x1b[38;2;229;')
    buffer.write('32;233mPink\npartial')
    snapshot = buffer.frame(width=20, height=10)
    assert '229;32;233' in snapshot.rows[0]
    assert '229;32;233' in snapshot.rows[1]
    assert '229;32;233' in snapshot.continuation_style
    buffer.write('\x1b[0m normal')
    assert plain(buffer) == ['Pink', 'partial normal']
    assert buffer.frame(width=20, height=10).continuation_style == ''


def test_tabs_carriage_returns_crlf_and_non_sgr_controls() -> None:
    buffer = TranscriptBuffer()
    buffer.write('before\rafter\r\n\tindented\n\x1b[2Jliteral')
    assert plain(buffer) == ['after', '        indented', 'literal']
    assert '\x1b[2J' not in ''.join(buffer.frame(width=80, height=24).rows)


def test_clear_forgets_lines_partial_tail_and_style() -> None:
    buffer = TranscriptBuffer()
    buffer.write('\x1b[1mold\npartial')
    buffer.clear()
    assert plain(buffer) == ['']
    buffer.write('new\n')
    snapshot = buffer.frame(width=80, height=24)
    assert [Text.from_ansi(row).plain for row in snapshot.rows] == ['new', '']
    assert snapshot.continuation_style == ''


@pytest.mark.parametrize('terminator', ['\x07', '\x1b\\'])
@pytest.mark.parametrize('split', [False, True])
def test_hyperlinks_survive_wrapping_and_replay(*, terminator: str, split: bool) -> None:
    buffer = TranscriptBuffer()
    value = f'\x1b]8;id=pr;https://example.com{terminator}PR #1006'
    for chunk in list(value) if split else [value]:
        buffer.write(chunk)
    snapshot = buffer.frame(width=4, height=10)
    assert plain(buffer, width=4) == ['PR #', '1006']
    console = Console()
    for row in snapshot.rows:
        text = Text.from_ansi(row)
        assert text.get_style_at_offset(console, 0).link == 'https://example.com'
        assert row.endswith('\x1b]8;;\x1b\\')
    assert '\x1b]' not in snapshot.continuation_style
    buffer.write(f'\x1b]8;;{terminator} plain\nnext')
    snapshot = buffer.frame(width=80, height=10)
    assert plain(buffer) == ['PR #1006 plain', 'next']
    text = Text.from_ansi(snapshot.rows[0])
    assert text.get_style_at_offset(console, 0).link == 'https://example.com'
    assert text.get_style_at_offset(console, 9).link is None
    assert '\x1b]' not in snapshot.rows[1]


@pytest.mark.parametrize('payload', ['52;c;Y2xpcGJvYXJk', '0;title', '8;malformed', '8;;https://bad\x01url'])
def test_replay_drops_other_or_malformed_osc(*, payload: str) -> None:
    buffer = TranscriptBuffer()
    buffer.write(f'before\x1b]{payload}\x1b\\after')
    snapshot = buffer.frame(width=80, height=10)
    assert plain(buffer) == ['beforeafter']
    assert '\x1b]' not in ''.join(snapshot.rows)


def test_link_style_prefix_restores_only_sgr() -> None:
    prefix = style_prefix(Style(color='red', link='https://example.com'))
    assert prefix == '\x1b[31m'


def test_limits_cover_completed_and_unterminated_lines() -> None:
    buffer = TranscriptBuffer(max_lines=2, max_chars=10)
    buffer.write('one\ntwo\nthree\n')
    assert plain(buffer) == ['two', 'three', '']
    buffer.write('four\nfive\n')
    assert plain(buffer) == ['four', 'five', '']
    buffer.write('0123456789ABCD')
    assert plain(buffer)[-1] == '456789ABCD'
    buffer.write('\n')
    assert plain(buffer) == ['456789ABCD', '']
    buffer.write('x' * 20 + '\n')
    assert plain(buffer) == ['x' * 10, '']


@pytest.mark.parametrize(('lines', 'chars'), [(0, 1), (1, 0)])
def test_invalid_limits(lines: int, chars: int) -> None:
    with pytest.raises(ValueError, match='positive'):
        TranscriptBuffer(max_lines=lines, max_chars=chars)


def test_empty_buffer_has_a_writer_position() -> None:
    assert plain(TranscriptBuffer()) == ['']


def test_replay_never_executes_embedded_control_characters() -> None:
    buffer = TranscriptBuffer()
    buffer.write('a\bb\x07c')
    replay = ''.join(buffer.frame(width=20, height=10).rows)
    assert '\b' not in replay and '\x07' not in replay


def test_wide_character_in_one_column_cannot_scroll_the_replayed_screen() -> None:
    buffer = TranscriptBuffer()
    buffer.write('界x')
    assert all(visible_length(row) <= 1 for row in buffer.frame(width=1, height=10).rows)
    assert plain(buffer) == ['界x']


def test_partial_truncation_never_splits_ansi_tokens() -> None:
    buffer = TranscriptBuffer(max_chars=8)
    buffer.write('prefix\x1b[31mTAIL')
    assert plain(buffer) == ['TAIL']
    assert '\x1b[31m' in buffer.frame(width=40, height=10).rows[0]
    buffer.write('\nnext')
    assert plain(buffer) == ['TAIL', 'next']
    assert '\x1b[31m' in buffer.frame(width=40, height=10).rows[-1]


def test_partial_escape_is_retained_until_completed_without_leaking_bytes() -> None:
    buffer = TranscriptBuffer(max_chars=4)
    buffer.write('prefix\x1b[38;2;229;')
    assert '[38;' not in ''.join(plain(buffer))
    buffer.write('32;233mTAIL')
    assert plain(buffer) == ['TAIL']
    assert '229;32;233' in buffer.frame(width=40, height=10).rows[0]


def test_malformed_unbounded_control_cannot_grow_the_partial_cache() -> None:
    buffer = TranscriptBuffer(max_chars=4)
    buffer.write('prefix\x1b[' + '1;' * 3000)
    buffer.write('still an unclosed control')
    assert plain(buffer) == ['efix']
    buffer.write('\nnext')
    assert plain(buffer) == ['efix', 'next']


def test_replay_ignores_richs_previously_cached_16_color_encoding() -> None:

    style = Style(color='#e520e9', bold=True)
    assert '\x1b[1;95m' in style.render('cached', color_system=ColorSystem.STANDARD)
    assert '38;2;229;32;233' in render_ansi(text='replay', style=style)


def test_capture_forwards_and_restores_console_on_error() -> None:

    output = io.StringIO()
    console = Console(file=output)
    buffer = TranscriptBuffer()
    with pytest.raises(ValueError):
        with buffer.capture(console):
            assert not console.file.isatty()
            console.print('startup notice')
            console.file.flush()
            raise ValueError('startup failed')
    assert console.file is output
    assert output.getvalue() == 'startup notice\n'
    assert plain(buffer) == ['startup notice', '']
    assert not output.closed


@pytest.mark.parametrize('terminator', ['\x07', '\x1b\\'])
@pytest.mark.parametrize('split', [False, True])
def test_palette_controls_never_become_transcript_text(terminator: str, split: bool) -> None:
    buffer = TranscriptBuffer()
    buffer.write('\x1b[31mbefore')
    for payload in ('11;#0a1929', '10;#d6eaf8', '4;0;#0a1929', '104', '111', '110'):
        control = f'\x1b]{payload}{terminator}'
        chunks = list(control) if split else [control]
        for chunk in chunks:
            buffer.write(chunk)
            assert plain(buffer) == ['before']
    buffer.write(' after\nnext')
    assert plain(buffer) == ['before after', 'next']
    assert '\x1b[31m' in buffer.frame(width=80, height=24).rows[1]


def test_real_palette_output_is_forwarded_but_not_replayed() -> None:
    buffer = TranscriptBuffer()
    output = io.StringIO()
    console = Console(file=output)
    with buffer.capture(console):
        theme.apply('github_light', output=console.file)
        console.print('conversation')
        theme.apply('default', output=console.file)
    assert '\x1b]' in output.getvalue()
    assert plain(buffer) == ['conversation', '']


def test_adjacent_palette_controls_do_not_swallow_visible_text() -> None:
    buffer = TranscriptBuffer()
    buffer.write('before\x1b]11;#ffffff\x07middle\x1b]104\x07after\n')
    assert plain(buffer) == ['beforemiddleafter', '']


def test_colour_reset_does_not_close_a_hyperlink_across_lines() -> None:
    buffer = TranscriptBuffer()
    buffer.write('\x1b]8;;https://example.com\x1b\\\x1b[31mred\x1b[0m plain\nnext')
    console = Console()
    for row in buffer.frame(width=80, height=10).rows:
        text = Text.from_ansi(row)
        for index in range(len(text)):
            assert text.get_style_at_offset(console, index).link == 'https://example.com'
    buffer.write('\x1b]8;;\x1b\\ unlinked')
    text = Text.from_ansi(buffer.frame(width=80, height=10).rows[-1])
    assert text.get_style_at_offset(console, -1).link is None


def _block(buffer: TranscriptBuffer, *, width: int = 40) -> MarkdownBlock:
    def render(*, source: str, width: int) -> str:
        return f'rendered at {width}: {source}\n'

    return buffer.markdown(render=render, width=width, changed=lambda: None)


def test_markdown_part_starts_on_its_own_row_and_keeps_its_id() -> None:
    buffer = TranscriptBuffer()
    buffer.write('tool header')
    block = _block(buffer)
    assert buffer.ids() == range(0, 3)
    block.extend('**hi**')
    block.write('streamed\npartial')
    block.flush()
    buffer.write('after\n')
    assert plain(buffer, width=40) == ['tool header', 'streamed', 'partial', 'after', '']
    assert plain(buffer, width=30) == ['tool header', 'rendered at 30: **hi**', 'after', '']
    assert [Text.from_ansi(row).plain for row in buffer.rows(1, width=40)] == ['streamed', 'partial']


def test_printed_skips_output_already_in_scrollback_and_cleared_output() -> None:
    buffer = TranscriptBuffer()
    buffer.write('startup\n')
    buffer.mark_printed()
    buffer.write('\tconversation\n')
    block = _block(buffer)
    block.extend('answer')
    block.write('answer\n')
    buffer.clear()
    buffer.write('after clear\n')
    assert Text.from_ansi(buffer.printed(width=40)).plain == 'after clear'
    assert buffer.printed(width=40) == ''
    block.freeze()
    assert plain(buffer) == ['after clear', '']


def test_markdown_parts_count_towards_the_line_limit() -> None:
    buffer = TranscriptBuffer(max_lines=2)
    _block(buffer)
    buffer.write('one\ntwo\n')
    assert buffer.ids() == range(1, 4)
    assert plain(buffer) == ['one', 'two', '']


def test_oversized_markdown_keeps_a_bounded_tail_and_does_not_evict_itself() -> None:
    buffer = TranscriptBuffer(max_chars=10, max_lines=2)
    block = _block(buffer)
    block.extend('source much larger than the limit')
    block.extend('ignored after freezing')
    assert block.source is None
    block.write('one\ntwo\nthree\n')
    assert plain(buffer, width=40) == ['two', 'three', '']
    block.write('four\n')
    assert plain(buffer, width=40) == ['three', 'four', ''], 'a same-length tail still invalidates its cache'
    block.write('pending-too-long')
    assert block.chars <= 10
    assert plain(buffer, width=40) == ['g-too-long', '']
    buffer.write('new\nnext\n')
    block.write('evicted block still receiving output')
    assert plain(buffer, width=40) == ['new', 'next', '']


def test_markdown_drops_source_when_rendered_output_exceeds_the_shared_budget() -> None:
    buffer = TranscriptBuffer(max_chars=10)
    block = _block(buffer)
    block.extend('source')
    block.write('answer\n')
    assert block.source is None
    assert plain(buffer, width=40) == ['answer', '']


class _PreLiveTranscript:
    """The pre-1.0 runtime state: styled Text lines and an unfinished ANSI stream.

    It deliberately has no live-panel methods, including `mark_printed`.
    """

    def __init__(self, *, pending: str = 'pending\x1b[32', discard: bool = False) -> None:
        self.max_lines = 2000
        self.max_chars = 1_000_000
        self._lines = deque([Text.from_ansi('\x1b[31mold output\x1b[0m')])
        self._pending = pending
        self._chars = len(self._lines[0])
        self._discard_until_newline = discard
        self._decoder = TranscriptDecoder()


def test_pre_live_transcript_migrates_in_place_preserving_styles_and_partial_ansi() -> None:
    legacy = _PreLiveTranscript()
    retained = cast(TranscriptBuffer, legacy)
    transcript = TranscriptBuffer.rebind(retained)
    assert transcript is retained
    assert type(transcript) is TranscriptBuffer
    assert transcript.max_lines == 2000 and transcript.max_chars == 1_000_000
    assert plain(transcript) == ['old output', 'pending']
    assert '\x1b[31m' in transcript.frame(width=80, height=24).rows[0]
    transcript.write('m green\x1b[0m\n')
    assert plain(transcript) == ['old output', 'pending green', '']
    assert '\x1b[32m' in transcript.frame(width=80, height=24).rows[1]
    assert Text.from_ansi(transcript.printed(width=80)).plain == 'pending green', 'old output was already emitted'
    output = io.StringIO()
    console = Console(file=output)
    with transcript.capture(console):
        console.print('after reload')
    assert console.file is output
    assert output.getvalue() == 'after reload\n'
    assert plain(transcript) == ['old output', 'pending green', 'after reload', '']
    assert transcript.printed(width=80) == '', 'capture does not duplicate lifecycle output'


def test_pre_live_transcript_preserves_discarding_a_malformed_escape() -> None:
    legacy = _PreLiveTranscript(pending='', discard=True)
    transcript = TranscriptBuffer.rebind(cast(TranscriptBuffer, legacy))
    transcript.write('discarded')
    assert plain(transcript) == ['old output', '']
    transcript.write('discarded\nnew\n')
    assert plain(transcript) == ['old output', '', 'new', '']


def test_live_transcript_rebinds_nested_markdown_state_without_resetting_output() -> None:
    transcript = TranscriptBuffer(max_lines=3)
    transcript.write('notice\n')
    block = _block(transcript)
    block.extend('answer')
    block.write('answer\n')
    for _ in range(2):
        assert TranscriptBuffer.rebind(transcript) is transcript
        assert plain(transcript, width=40) == ['notice', 'answer', '']
    block.extend(' more')
    block.write('more\n')
    assert plain(transcript, width=40) == ['notice', 'answer', 'more', '']
    transcript.max_lines = 1
    transcript.write('latest\n')
    assert plain(transcript, width=40) == ['latest', '']
    assert transcript.printed(width=40) == 'latest\n'


@pytest.mark.parametrize('repaint', ['width', 'theme'])
def test_markdown_repaint_respects_configured_character_and_line_limits(repaint: str) -> None:
    transcript = TranscriptBuffer(max_chars=8, max_lines=1)

    def render(*, source: str, width: int) -> str:
        return 'one\ntwo\nthree\n'

    block = transcript.markdown(render=render, width=40, changed=lambda: None)
    block.extend('x')
    block.write('ok\n')
    assert plain(transcript, width=40) == ['ok', '']
    with theme.use(lambda: 'github_light' if repaint == 'theme' else 'default'):
        assert plain(transcript, width=20 if repaint == 'width' else 40) == ['three', '']


def _painted(text: str) -> TranscriptBuffer:
    buffer = TranscriptBuffer()
    buffer.write(text)
    return buffer


def _printed(styles: list[Style]) -> str:
    """Truecolor output, not `console.print`: Rich reuses an SGR another test cached on a shared style."""
    return ''.join(render_ansi(text=f'line {index}', style=style) + '\n' for index, style in enumerate(styles))


def _colours(rows: Iterable[str]) -> list[tuple[str | None, str | None]]:
    console = Console()
    styles = [Text.from_ansi(row).get_style_at_offset(console, 0) for row in rows if row]
    return [
        (None if style.color is None else style.color.name, None if style.bgcolor is None else style.bgcolor.name)
        for style in styles
    ]


def test_styled_lines_repaint_in_a_newly_selected_theme_and_back() -> None:
    styles = [
        Style(color=theme.color(theme.MUTED)),
        Style(bgcolor=theme.diff_theme().addition),
        Style(color='green'),
        Style(color='#123456', bgcolor='#abcdef'),
        Style(bold=True),
    ]
    buffer = _painted(_printed(styles))
    original = _colours(buffer.frame(width=80, height=24).rows)
    tokyo = PALETTES['tokyo_night']
    with theme.use(lambda: 'tokyo_night'):
        expected = [
            (tokyo.ansi[8], None),
            (None, theme.diff_theme().addition.lower()),
            ('color(2)', None),
            ('#123456', '#abcdef'),
            (None, None),
        ]
        assert _colours(buffer.frame(width=80, height=24).rows) == expected
        assert _colours(buffer.printed(width=80).splitlines()) == expected
    assert _colours(buffer.frame(width=80, height=24).rows) == original
    assert plain(buffer) == [f'line {index}' for index in range(5)] + ['']


def test_palette_output_repaints_in_another_palette_or_the_default_theme() -> None:
    light = PALETTES['github_light']
    tokyo = PALETTES['tokyo_night']
    with theme.use(lambda: 'github_light'):
        styles = [
            Style(color=theme.color(theme.MUTED)),
            Style(color=light.ansi[2], bgcolor=light.bg),
            Style(color=theme.color(theme.ERROR)),
        ]
        buffer = _painted(_printed(styles))
    with theme.use(lambda: 'tokyo_night'):
        assert _colours(buffer.frame(width=80, height=24).rows) == [
            (tokyo.ansi[8], None),
            (tokyo.ansi[2], tokyo.bg),
            (tokyo.ansi[1], None),
        ]
    assert _colours(buffer.frame(width=80, height=24).rows) == [
        (theme.GREY.lower(), None),
        ('color(2)', 'default'),
        (theme.CALCIUM.lower(), None),
    ]


def test_recolor_accepts_spans_styled_by_name() -> None:
    text = Text('muted', spans=[Span(0, 5, f'bold {theme.GREY}')])
    themed = recolor(text, source='default', target='tokyo_night')
    style = themed.spans[0].style
    assert isinstance(style, Style)
    assert style.bold and style.color is not None and style.color.name == PALETTES['tokyo_night'].ansi[8]
    assert text.spans[0].style == f'bold {theme.GREY}', 'the retained line is unchanged'


def test_rebind_assumes_the_current_theme_for_lines_retained_without_one() -> None:
    transcript = _painted(_printed([Style(color=theme.color(theme.MUTED))]))
    block = _block(transcript)
    block.write(_printed([Style(color=theme.color(theme.MUTED))]))
    block.freeze()
    unfinished = _printed([Style(color=theme.color(theme.MUTED))]).removesuffix('\n')
    block.write(unfinished)
    transcript.write(unfinished)
    stream = cast(object, vars(block)['_stream'])
    lines = [*cast(deque[object], vars(transcript)['_items']), *cast(list[object], vars(stream)['lines'])]
    for line in lines:
        vars(line).pop('theme_name', None)
    vars(transcript).pop('_pending_theme')
    vars(stream).pop('pending_theme')
    with theme.use(lambda: 'tokyo_night'):
        TranscriptBuffer.rebind(transcript)
        assert _colours(transcript.frame(width=80, height=24).rows) == [(theme.GREY.lower(), None)] * 4
    assert _colours(transcript.frame(width=80, height=24).rows) == [(theme.GREY.lower(), None)] * 4


def test_an_unfinished_line_keeps_the_theme_it_started_in() -> None:
    tokyo = PALETTES['tokyo_night']
    selected = ['default']
    with theme.use(lambda: selected[0]):
        buffer = _painted(_printed([Style(color=theme.color(theme.MUTED))]).removesuffix('\n'))
        selected[0] = 'tokyo_night'
        assert _colours(buffer.frame(width=80, height=24).rows) == [(tokyo.ansi[8], None)]
        buffer.write(' done\n' + _printed([Style(color=theme.color(theme.MUTED))]))
        assert _colours(buffer.frame(width=80, height=24).rows) == [(tokyo.ansi[8], None)] * 2
        selected[0] = 'default'
        assert _colours(buffer.frame(width=80, height=24).rows) == [(theme.GREY.lower(), None)] * 2
    assert plain(buffer) == ['line 0 done', 'line 0', '']


def test_an_unfinished_markdown_line_keeps_the_theme_it_started_in() -> None:
    tokyo = PALETTES['tokyo_night']
    buffer = TranscriptBuffer()
    block = _block(buffer)
    block.write(_printed([Style(color=theme.color(theme.MUTED))]).removesuffix('\n'))
    block.freeze()
    with theme.use(lambda: 'tokyo_night'):
        assert _colours(buffer.rows(0, width=80)) == [(tokyo.ansi[8], None)]
        block.write(' more\nnext')
        assert _colours(buffer.rows(0, width=80)) == [(tokyo.ansi[8], None), (None, None)]


def test_branding_keeps_its_colours_when_the_theme_changes() -> None:
    buffer = TranscriptBuffer()
    console = Console(file=io.StringIO(), force_terminal=True, color_system='truecolor', width=20)
    with buffer.capture(console):
        print_banner(console)
    with theme.branded():
        buffer.write(render_ansi(text='logo', style=Style(color=theme.LITHIUM)) + '\n')
    buffer.write(render_ansi(text='accent', style=Style.parse(theme.color(theme.ACCENT))) + '\n')
    banner = _colours(buffer.frame(width=20, height=24).rows)[0]
    with theme.use(lambda: 'tokyo_night'):
        assert _colours(buffer.frame(width=20, height=24).rows) == [
            banner,
            (theme.LITHIUM.lower(), None),
            (PALETTES['tokyo_night'].ansi[12], None),
        ]
