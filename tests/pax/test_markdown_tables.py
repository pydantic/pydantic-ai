"""Markdown tables wrap between words at any width, and fall back to one block per row."""

import io

import pytest
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


def test_wide_characters_join_the_word_around_them() -> None:
    lines = Text.from_ansi(_render('| Name | Code |\n|---|---|\n| 東京hello | 12345678 |', 20)).plain.splitlines()
    assert any('東京hello' in line for line in lines)


@pytest.mark.parametrize('cap', ['40', ' +40 '])
def test_termflow_table_width_cap_applies_to_every_layout(monkeypatch: pytest.MonkeyPatch, cap: str) -> None:
    monkeypatch.setenv('TERMFLOW_MAX_TABLE_WIDTH', cap)
    lines = Text.from_ansi(_render(ERRORS, 80)).plain.splitlines()
    assert lines[0] == 'Variant: UnexpectedEof'
    assert max(len(line) for line in lines) == 40


def test_a_stacked_row_names_the_cells_it_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TERMFLOW_MAX_TABLE_WIDTH', 'wide')  # Not a number, so no cap.
    lines = Text.from_ansi(_render('| Variant | InvalidCodePoint(u16) |\n|---|---|\n| UnexpectedEof |', 24)).plain
    assert lines.splitlines() == ['Variant: UnexpectedEof', 'InvalidCodePoint(u16):']
