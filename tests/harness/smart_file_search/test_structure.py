"""Edge cases of `SmartFileSearch`'s language-agnostic structural chunking, driven by a minimal `Syntax`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai_harness.smart_file_search._chunks import (
    Chunk,
    _chunks_from_ranges as chunks_from_ranges,  # pyright: ignore[reportPrivateUsage]
    source_chunks,
    windows,
)
from pydantic_ai_harness.smart_file_search._structure import SPLIT_LINES, Range, structure_ranges


@dataclass
class Node:
    first: int
    last: int
    name: str | None = None
    body: Sequence[Node] = ()
    members: Sequence[Node] | None = None


class NodeSyntax:
    def span(self, node: Node) -> tuple[int, int]:
        return node.first, node.last

    def symbol(self, node: Node) -> str | None:
        return node.name

    def members(self, node: Node) -> Sequence[Node] | None:
        return node.members

    def statements(self, node: Node) -> Sequence[Node] | None:
        return node.body

    def broken(self, node: Node) -> bool:
        return False


def _spans(nodes: list[Node]) -> list[tuple[int, int, str | None]]:
    return [(r.start, r.end, r.symbol) for r in structure_ranges(nodes, NodeSyntax())]


def test_long_body_groups_short_statements_and_isolates_blocks() -> None:
    body = [Node(2, 2), Node(3, 3), Node(4, 9), Node(10, 10), Node(11, 40)]
    assert _spans([Node(1, 40, 'f', body)]) == [(1, 3, 'f'), (4, 9, 'f'), (10, 10, 'f'), (11, 40, 'f')]


def test_declarations_of_split_lines_or_more_are_split() -> None:
    def spans(lines: int) -> list[tuple[int, int, str | None]]:
        return _spans([Node(1, lines, 'f', [Node(2, 2), Node(3, lines)])])

    assert spans(SPLIT_LINES - 1) == [(1, SPLIT_LINES - 1, 'f')]
    assert spans(SPLIT_LINES) == [(1, 2, 'f'), (3, SPLIT_LINES, 'f')]


def test_body_whose_block_ends_on_the_last_line_stays_whole() -> None:
    body = [Node(2, 30), Node(30, 30)]  # e.g. `} y(); }`: a statement after the block, on its closing line
    assert _spans([Node(1, 30, 'f', body)]) == [(1, 30, 'f')]


def test_container_whose_first_member_shares_its_line_has_no_header() -> None:
    container = Node(1, 10, 'A', members=[Node(1, 5, 'm'), Node(6, 10, 'n')])
    assert _spans([container]) == [(1, 5, 'A.m'), (6, 10, 'A.n')]


def test_whitespace_only_windows_are_dropped() -> None:
    assert windows(['', '   '], 'blank.txt') == []


def test_ranges_past_the_last_line_are_ignored() -> None:
    chunks = chunks_from_ranges(
        ['a = 1'], 'x.py', [Range(start=1, end=1, symbol='a'), Range(start=3, end=4, symbol=None)]
    )
    assert chunks == [Chunk(path='x.py', line=1, end_line=1, text='a = 1', symbol='a')]


def test_python_without_statements_uses_windows() -> None:
    assert source_chunks('# only a comment\n', 'empty.py')[1] == 'overlapping-lines'
