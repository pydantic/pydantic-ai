"""Bounded transcript replay preserves text, wrapping, and styling, not controls."""

import io

import pytest
from rich.color import ColorSystem
from rich.console import Console
from rich.style import Style
from rich.text import Text
from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import theme
from pydantic_clai2.prompt_transcript import TranscriptBuffer, render_ansi


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


def test_hyperlinks_are_not_replayed_as_terminal_commands() -> None:
    buffer = TranscriptBuffer()
    buffer.write('\x1b]8;;https://example.com\x1b\\link')
    snapshot = buffer.frame(width=20, height=10)
    assert plain(buffer) == ['link']
    assert '\x1b]' not in ''.join(snapshot.rows) + snapshot.continuation_style


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
