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


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        pytest.param(
            '| Variant | InvalidCodePoint(u16) |\n|---|---|\n| UnexpectedEof |\n| a | b | extra |',
            [
                'Variant: UnexpectedEof',
                'InvalidCodePoint(u16):',
                '─' * 24,
                'Variant: a',
                'InvalidCodePoint(u16): b',
                'extra',
            ],
            id='ragged-rows',
        ),
        pytest.param(
            '| Variant | InvalidCodePoint(u16) |\n|---|---|', ['Variant:', 'InvalidCodePoint(u16):'], id='header-only'
        ),
    ],
)
def test_a_table_too_narrow_for_its_words_stacks_its_rows(*, source: str, expected: list[str]) -> None:
    assert Text.from_ansi(_render(source, 24)).plain.splitlines() == expected


@pytest.mark.parametrize(('cap', 'width'), [(' +40 ', 40), ('0', 80), ('wide', 80)])
def test_termflow_table_width_cap_applies_when_it_leaves_room(
    monkeypatch: pytest.MonkeyPatch, *, cap: str, width: int
) -> None:
    monkeypatch.setenv('TERMFLOW_MAX_TABLE_WIDTH', cap)
    assert max(len(line) for line in Text.from_ansi(_render(ERRORS, 80)).plain.splitlines()) == width
