"""Pure editing state, independent of terminal timing or ownership."""

import pytest
from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.prompt_buffer import PromptBuffer


@pytest.mark.parametrize(
    ('keys', 'expected', 'cursor'),
    [
        (['left', 'left', 'backspace'], 'one wo', 4),
        (['ctrl-a', 'delete'], 'ne two', 0),
        (['home', 'right', 'end'], 'one two', 7),
        (['ctrl-w'], 'one ', 4),
        (['ctrl-u'], '', 0),
        (['alt-b', 'ctrl-k'], 'one ', 4),
        (['ctrl-a', 'alt-f'], 'one two', 3),
        (['ctrl-a', 'ctrl-right', 'ctrl-right'], 'one two', 7),
        (['ctrl-left', 'ctrl-left'], 'one two', 0),
        (['ctrl-e', '!'], 'one two!', 8),
    ],
)
def test_editing(keys: list[str], expected: str, cursor: int) -> None:
    buffer = PromptBuffer()
    buffer.replace('one two')
    for key in keys:
        assert buffer.edit(key)
    assert buffer.text == expected
    assert buffer.cursor == cursor
    assert not buffer.edit('unknown-key')


def test_empty_edges_and_single_word_deletion() -> None:
    buffer = PromptBuffer()
    for key in ('backspace', 'delete', 'left', 'right', 'home', 'end', 'ctrl-k', 'alt-f'):
        assert buffer.edit(key)
    assert buffer.text == '' and buffer.cursor == 0
    buffer.replace('word')
    buffer.edit('ctrl-w')
    assert buffer.text == ''


@pytest.mark.parametrize('key', ['ctrl-w', 'alt-backspace'])
@pytest.mark.parametrize(
    ('text', 'cursor', 'expected', 'expected_cursor'),
    [
        ('one two', 7, 'one ', 4),
        ('one two   ', 10, 'one ', 4),
        ('one two three', 7, 'one  three', 4),
        ('one two', 6, 'one o', 4),
        ('word', 4, '', 0),
        ('', 0, '', 0),
        ('one two', 0, 'one two', 0),
        ('   ', 3, '', 0),
        ('hello 世界', 8, 'hello ', 6),
        ('one\ntwo', 7, 'one\n', 4),
        ('one\ttwo', 7, 'one\t', 4),
        ('one\u2003two', 7, 'one\u2003', 4),
        ('one\n\t two \t\n', 12, 'one\n\t ', 6),
        ('one\ntwo three', 7, 'one\n three', 4),
        ('one\ntwo', 6, 'one\no', 4),
        ('\n\t', 2, '', 0),
    ],
)
def test_delete_previous_word(key: str, text: str, cursor: int, expected: str, expected_cursor: int) -> None:
    buffer = PromptBuffer(text=text, cursor=cursor)
    assert buffer.edit(key)
    assert buffer.text == expected
    assert buffer.cursor == expected_cursor


def test_multiline_navigation_and_history_restore_draft() -> None:
    buffer = PromptBuffer(history=['first', 'second'])
    buffer.replace('one\ntwo')
    buffer.edit('up')
    assert buffer.cursor == 3
    buffer.edit('down')
    assert buffer.cursor == 7
    buffer.edit('home')
    assert buffer.cursor == 4
    buffer.edit('ctrl-a')
    buffer.edit('up')
    buffer.edit('up')
    assert buffer.text == 'second'
    buffer.edit('up')
    assert buffer.text == 'first'
    buffer.edit('down')
    buffer.edit('down')
    assert buffer.text == 'one\ntwo'


def test_reverse_search_accept_cancel_backspace_and_repeat() -> None:
    buffer = PromptBuffer(history=['alpha', 'beta', 'alphabet'])
    buffer.replace('draft')
    buffer.edit('ctrl-r')
    for key in 'alph':
        buffer.edit(key)
    assert buffer.text == 'alphabet'
    buffer.edit('ctrl-r')
    assert buffer.text == 'alpha'
    buffer.edit('backspace')
    buffer.edit('enter')
    assert buffer.search is None and buffer.text == 'alphabet'
    buffer.edit('ctrl-r')
    buffer.edit('Z')
    buffer.edit('ctrl-g')
    assert buffer.text == 'alphabet'
    buffer.edit('ctrl-r')
    buffer.edit('escape')
    assert buffer.search is None


def test_unicode_wrapping_and_nonblinking_cursor() -> None:
    buffer = PromptBuffer()
    buffer.insert('界界\r\nhello\x1b\tworld')
    assert '\r' not in buffer.text and '\x1b' not in buffer.text
    rows = buffer.rows(width=8, limit=2)
    assert len(rows) == 2
    assert all(visible_length(row) <= 8 for row in rows)
    assert any('\x1b[7m' in row for row in rows)
    buffer.replace('abcd\nef')
    assert len(buffer.rows(width=4, limit=5)) == 2
    buffer.cursor = 4
    assert any('\x1b[7m' in row for row in buffer.rows(width=4, limit=5))
    buffer.cursor = 0
    assert len(buffer.rows(width=4, limit=5)) == 2


def test_empty_history_and_line_end_movement() -> None:
    buffer = PromptBuffer()
    buffer.recall(backwards=True)
    assert buffer.text == ''
    buffer.replace('a\nb')
    buffer.cursor = 0
    buffer.edit('end')
    assert buffer.cursor == 1
    buffer.replace('    word')
    buffer.cursor = 0
    buffer.edit('alt-f')
    assert buffer.cursor == len(buffer.text)


def test_history_control_bytes_are_not_terminal_instructions() -> None:
    buffer = PromptBuffer()
    buffer.replace('bad\x1b[2J')
    rows = buffer.rows(width=40, limit=3)
    assert not any('\x1b[2J' in row for row in rows)
    assert 'bad?[2J' in rows[0]
