"""Pure editing state, independent of terminal timing or ownership."""

import pytest
from termflow.ansi.utils import visible_length

from pydantic_clai2.ui.prompt.prompt_buffer import UNDO_LIMIT, PromptBuffer


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


@pytest.mark.parametrize('key', ['backspace', 'ctrl-w', 'x'])
def test_any_edit_to_recalled_history_becomes_the_restored_draft(key: str) -> None:
    buffer = PromptBuffer(history=['older', 'recalled word'])
    buffer.edit('up')
    buffer.edit(key)
    edited = buffer.text
    assert edited != 'recalled word'
    buffer.edit('up')
    assert buffer.text == 'recalled word'
    buffer.edit('down')
    assert buffer.text == edited


def test_history_control_bytes_are_not_terminal_instructions() -> None:
    buffer = PromptBuffer()
    buffer.replace('bad\x1b[2J')
    rows = buffer.rows(width=40, limit=3)
    assert not any('\x1b[2J' in row for row in rows)
    assert 'bad?[2J' in rows[0]


def typed(buffer: PromptBuffer, text: str) -> None:
    for key in text:
        assert buffer.edit(key)


@pytest.mark.parametrize(
    ('keys', 'edited'),
    [
        (['backspace'], 'one tw'),
        (['ctrl-a', 'delete'], 'ne two'),
        (['ctrl-w'], 'one '),
        (['alt-backspace'], 'one '),
        (['ctrl-u'], ''),
        (['alt-b', 'ctrl-k'], 'one '),
    ],
)
@pytest.mark.parametrize(
    ('undo', 'redo'), [('ctrl-z', 'ctrl-y'), ('super-z', 'super-shift-z'), ('ctrl-z', 'ctrl-shift-z')]
)
def test_undo_restores_deleted_text_and_redo_deletes_it_again(
    keys: list[str], edited: str, undo: str, redo: str
) -> None:
    buffer = PromptBuffer()
    typed(buffer, 'one two')
    for key in keys:
        buffer.edit(key)
    assert buffer.text == edited
    cursor = buffer.cursor
    before = {'ctrl-k': 4, 'delete': 0}.get(keys[-1], 7)
    assert buffer.edit(undo)
    assert (buffer.text, buffer.cursor) == ('one two', before)
    assert buffer.edit(redo)
    assert (buffer.text, buffer.cursor) == (edited, cursor)


def test_undo_steps_group_words_and_runs_of_deletion() -> None:
    buffer = PromptBuffer()
    typed(buffer, 'hello world')
    for key in ('backspace', 'backspace', 'backspace', 'ctrl-w'):
        buffer.edit(key)
    assert buffer.text == 'hello '
    undone: list[str] = []
    for _ in range(5):
        buffer.undo()
        undone.append(buffer.text)
    assert undone == ['hello wo', 'hello world', 'hello ', '', '']
    redone: list[str] = []
    for _ in range(5):
        buffer.redo()
        redone.append(buffer.text)
    assert redone == ['hello ', 'hello world', 'hello wo', 'hello ', 'hello ']


def test_moving_the_cursor_or_switching_direction_starts_a_new_step() -> None:
    buffer = PromptBuffer()
    typed(buffer, 'abc')
    buffer.edit('left')
    typed(buffer, 'X')
    buffer.edit('backspace')
    buffer.edit('delete')
    assert buffer.text == 'ab'
    steps: list[str] = []
    while buffer.text:
        buffer.undo()
        steps.append(buffer.text)
    assert steps == ['abc', 'abXc', 'abc', '']
    typed(buffer, 'new')
    buffer.redo()
    assert buffer.text == 'new', 'a new change discards the redo steps'


def test_undo_with_nothing_to_undo_or_redo_is_a_no_op() -> None:
    buffer = PromptBuffer(text='kept', cursor=2)
    for key in ('ctrl-z', 'super-z', 'ctrl-y', 'super-shift-z', 'ctrl-shift-z'):
        assert buffer.edit(key)
        assert (buffer.text, buffer.cursor) == ('kept', 2)
    buffer.edit('backspace')
    buffer.edit('left')
    buffer.edit('backspace')
    assert (buffer.text, buffer.cursor) == ('kpt', 0)
    buffer.undo()
    assert (buffer.text, buffer.cursor) == ('kept', 2)
    buffer.undo()
    assert (buffer.text, buffer.cursor) == ('kept', 2), 'the text it started with is the oldest step'


def test_undo_restores_a_folded_paste_and_inserted_strings_are_steps() -> None:
    buffer = PromptBuffer()
    pasted = '\n'.join(f'line {index}' for index in range(5))
    buffer.insert('before ')
    buffer.insert(pasted, paste=True)
    buffer.edit('ctrl-u')
    assert buffer.text == ''
    buffer.undo()
    assert buffer.display() == ('before [paste 5 lines]', 22)
    buffer.undo()
    assert buffer.text == 'before '
    buffer.redo()
    assert buffer.display()[0] == 'before [paste 5 lines]'


def test_undo_keeps_only_the_newest_steps() -> None:
    buffer = PromptBuffer()
    for index in range(UNDO_LIMIT + 20):
        buffer.insert(f'<{index}>')
    while True:
        text = buffer.text
        buffer.undo()
        if buffer.text == text:
            break
    assert buffer.text == ''.join(f'<{index}>' for index in range(20))


def test_history_walks_and_searches_are_single_steps_that_end_on_undo() -> None:
    buffer = PromptBuffer(history=['first', 'second', 'third'])
    typed(buffer, 'draft')
    for key in ('up', 'up', 'up'):
        buffer.edit(key)
    assert buffer.text == 'first'
    buffer.undo()
    assert buffer.text == 'draft' and buffer.history_index is None
    buffer.edit('up')
    buffer.edit('down')
    assert buffer.text == 'draft'
    buffer.undo()
    assert buffer.text == '', 'returning to the draft leaves no step to undo'
    buffer.redo()
    buffer.edit('ctrl-r')
    typed(buffer, 'sec')
    buffer.edit('enter')
    assert buffer.text == 'second'
    buffer.undo()
    assert buffer.text == 'draft'


def test_reset_forgets_the_submitted_draft() -> None:
    buffer = PromptBuffer()
    typed(buffer, 'sent')
    buffer.edit('backspace')
    buffer.reset()
    for key in ('ctrl-z', 'ctrl-y'):
        buffer.edit(key)
        assert buffer.text == ''
