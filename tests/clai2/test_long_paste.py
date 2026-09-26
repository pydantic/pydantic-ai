"""Long paste display retains literal input through edits and submission."""

import pytest
from rich.text import Text

from pydantic_clai2.prompt_buffer import PromptBuffer

PASTE = '\n'.join(f'line {index}' for index in range(5))


@pytest.mark.parametrize(
    ('text', 'label'),
    [(PASTE, '[paste 5 lines]'), (PASTE + '\n', '[paste 5 lines]'), ('x' * 1000, '[paste 1 lines]')],
)
def test_long_paste(text: str, label: str) -> None:
    buffer = PromptBuffer()
    buffer.insert(text, paste=True)
    assert buffer.text == text
    assert buffer.display() == (label, len(label))
    assert Text.from_ansi(''.join(buffer.rows(width=80, limit=3))).plain == label + ' '
    assert all(len(Text.from_ansi(row).plain) <= 6 for row in buffer.rows(width=6, limit=10))


@pytest.mark.parametrize('text', ['short', 'a\nb\nc\nd', 'x' * 999, ''])
def test_short_paste(text: str) -> None:
    buffer = PromptBuffer()
    buffer.insert(text, paste=True)
    assert buffer.display() == (text, len(text))


def test_typed_multiline_is_not_folded() -> None:
    buffer = PromptBuffer()
    buffer.insert(PASTE)
    assert buffer.display() == (PASTE, len(PASTE))


def test_surrounding_edits_and_multiple_pastes() -> None:
    buffer = PromptBuffer()
    buffer.insert(PASTE, paste=True)
    buffer.insert(' tail')
    buffer.cursor = 0
    buffer.insert('prefix ')
    assert buffer.display() == ('prefix [paste 5 lines] tail', 7)
    buffer.edit('backspace')
    buffer.edit('delete')
    assert buffer.display()[0] == 'prefix' + PASTE[1:] + ' tail'
    buffer.replace('')
    buffer.insert(PASTE, paste=True)
    buffer.insert(' / ')
    buffer.insert(PASTE, paste=True)
    assert buffer.display()[0] == '[paste 5 lines] / [paste 5 lines]'
    buffer.edit('left')
    assert buffer.display()[0] == '[paste 5 lines] / ' + PASTE
    buffer.edit('right')
    assert buffer.display()[0] == '[paste 5 lines] / ' + PASTE


@pytest.mark.parametrize('key', ['left', 'up', 'home', 'alt-b'])
def test_navigation_reveals_paste_for_editing(key: str) -> None:
    buffer = PromptBuffer()
    buffer.insert(PASTE, paste=True)
    buffer.edit(key)
    assert buffer.display() == (PASTE, buffer.cursor)
    buffer.insert('!')
    assert buffer.text[buffer.cursor - 1] == '!'


@pytest.mark.parametrize('key', ['backspace', 'ctrl-w', 'ctrl-u'])
def test_deleting_paste_reveals_remaining_text(key: str) -> None:
    buffer = PromptBuffer()
    buffer.insert(PASTE, paste=True)
    buffer.edit(key)
    assert buffer.display() == (buffer.text, buffer.cursor)


def test_replace_completion_and_history_clear_or_shift_folds() -> None:
    buffer = PromptBuffer(history=['previous'])
    buffer.insert('prefix ')
    buffer.insert(PASTE, paste=True)
    buffer.replace_range(0, 6, 'new')
    assert buffer.display()[0] == 'new [paste 5 lines]'
    buffer.cursor = 4
    buffer.edit('ctrl-k')
    assert buffer.display() == ('new ', 4)
    buffer.insert(PASTE, paste=True)
    buffer.recall(backwards=True)
    assert buffer.display() == ('previous', 8)
    buffer.recall(backwards=False)
    assert buffer.display()[0] == 'new ' + PASTE
    buffer.replace('[paste 5 lines]')
    assert buffer.text == '[paste 5 lines]'


@pytest.mark.parametrize('key', ['delete', 'backspace'])
def test_noop_deletion_preserves_history_navigation(key: str) -> None:
    buffer = PromptBuffer(history=['older', 'newer'])
    buffer.recall(backwards=True)
    if key == 'backspace':
        buffer.cursor = 0
    buffer.edit(key)
    buffer.recall(backwards=True)
    assert buffer.text == 'older'


def test_many_pastes_render_in_order() -> None:
    buffer = PromptBuffer()
    for index in range(100):
        buffer.insert(f'{index}:')
        buffer.insert(PASTE, paste=True)
    expected = ''.join(f'{index}:[paste 5 lines]' for index in range(100))
    assert buffer.display() == (expected, len(expected))


def test_paste_normalization_precedes_folding() -> None:
    buffer = PromptBuffer()
    buffer.insert(PASTE.replace('\n', '\r\n') + '\x1b', paste=True)
    assert buffer.text == PASTE
    assert buffer.display()[0] == '[paste 5 lines]'
