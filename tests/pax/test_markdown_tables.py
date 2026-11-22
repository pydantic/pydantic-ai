"""Markdown tables wrap between words at any width, and fall back to one block per row."""

import io

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text
from termflow import Parser, Renderer

from pydantic_ai_pax.ui.rendering.rendering import markdown_style, render_markdown

ERRORS = r"""| Variant | Meaning | Example | Fix |
|---|---|---|---|
| UnexpectedEof | Input ends before all 4 hex digits of a \u escape | `\u00` | `UnexpectedEof` |
| InvalidHexDigit(char) | A non-hex character inside a \u escape | `\u00G1` | Keep `InvalidHexDigit(char)` |
| InvalidCodePoint(u16) | A lone surrogate that does not pair with another escape to form a code point | `\uD800` | Report InvalidCodePoint(u16) |"""


def _render(source: str, width: int) -> str:
    return render_markdown(source=source, width=width, thinking=False, colors='truecolor')


@pytest.mark.parametrize(('width', 'stacked'), [(60, True), (80, False), (120, False)])
def test_table_cells_keep_their_words_whole(*, width: int, stacked: bool) -> None:
    text = Text.from_ansi(_render(ERRORS, width))
    lines = text.plain.splitlines()
    assert all(len(line) <= width for line in lines)
    assert lines[0].startswith('┌') is not stacked
    for word in ERRORS.replace('`', ' ').replace('|', ' ').split():
        if set(word) != {'-'}:
            assert any(word in line.split() or f'{word}:' in line.split() for line in lines), word
    # A code span keeps its style wherever the layout moves it.
    code = text.plain.index('InvalidHexDigit(char)', text.plain.index('Keep'))
    assert text.get_style_at_offset(Console(), code).dim
    if stacked:
        assert lines[:4] == [
            'Variant: UnexpectedEof',
            'Meaning: Input ends before all 4 hex digits of a \\u escape',
            'Example: \\u00',
            'Fix: UnexpectedEof',
        ]
        assert lines[4] == '─' * width


def test_a_table_that_fits_is_drawn_as_before() -> None:
    source = '| Variant | Meaning |\n|:---|---:|\n| `UnexpectedEof` | Input **ends** early |'
    output = io.StringIO()
    renderer = Renderer(output=output, width=80, style=markdown_style())
    parser = Parser()
    for line in source.split('\n'):
        for event in parser.parse_line(line):
            renderer.render(event)
    for event in parser.finalize():
        renderer.render(event)
    assert _render(source, 80) == output.getvalue()


def test_a_header_only_table_too_narrow_for_its_words_keeps_them() -> None:
    lines = Text.from_ansi(_render('| Variant | InvalidCodePoint(u16) |\n|---|---|', 24)).plain.splitlines()
    assert lines == ['Variant', 'InvalidCodePoint(u16)']


def test_wide_characters_may_wrap_between_any_two() -> None:
    source = '| 名前 | 説明 |\n|---|---|\n| 東京タワー | 東京都港区にある電波塔で、高さは333メートルです 🚀 |'
    lines = Text.from_ansi(_render(source, 40)).plain.splitlines()
    assert lines[0].startswith('┌')
    assert all(cell_len(line) <= 40 for line in lines)
