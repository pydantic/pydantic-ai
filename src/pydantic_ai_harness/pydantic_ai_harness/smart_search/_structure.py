"""Structural chunking: declarations -> line ranges.

The algorithm reads a syntax tree through `Syntax`, which `PythonSyntax`
implements over the stdlib `ast`:

* containers (classes) are split into a header plus one
  range per member, named `Container.member`;
* declarations of `SPLIT_LINES` or more are split into the blocks inside
  them, so each range is one behavior. Splits cover every line of the
  original range, so coverage is unchanged;
* runs of short neighbors (imports, one-liners) merge into one range,
  since each range costs a relevance judgment.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

SPLIT_LINES = 24
_MAX_DEPTH = 3
_BLOCK_LINES = 6
"""A statement this long is a block of its own when a long body is split."""
_GROUP_LINES = 20
"""Shorter statements group into one range until it would pass this many lines."""
_MERGE_LINES = 12
"""Short neighbors merge only while the merged range spans fewer lines than this."""
_MERGE_SYMBOL_CHARS = 60
"""Short neighbors merge only while their joined symbol names stay shorter than this."""

NodeT = TypeVar('NodeT')


@dataclass(kw_only=True)
class Range:
    """A 1-based, inclusive line range to chunk, and the symbol it belongs to."""

    start: int
    end: int
    symbol: str | None


class Syntax(Protocol[NodeT]):
    """What the chunker needs to know about one language's tree."""

    def span(self, node: NodeT) -> tuple[int, int]:
        """1-based first and last line, including decorators/wrappers."""
        ...

    def symbol(self, node: NodeT) -> str | None:
        """The declared name, if any."""
        ...

    def members(self, node: NodeT) -> Sequence[NodeT] | None:
        """Member declarations when `node` is a container, else `None`."""
        ...

    def statements(self, node: NodeT) -> Sequence[NodeT] | None:
        """The statements of `node`'s body, if it has one."""
        ...


def _partition(
    node: NodeT, start: int, end: int, symbol: str | None, syntax: Syntax[NodeT], depth: int = 0
) -> list[Range] | None:
    """Split a long body into its blocks, covering every line of `[start, end]`."""
    body = syntax.statements(node)
    if not body or end - start + 1 < SPLIT_LINES:  # `[start, end]` is inclusive
        return None
    if len(body) == 1:
        if depth >= _MAX_DEPTH:
            return None
        return _partition(body[0], start, end, symbol, syntax, depth + 1)

    out: list[Range] = []
    cursor = start
    group: list[NodeT] = []

    def flush(*, last: bool) -> None:
        nonlocal cursor
        stop = end if last else min(end, syntax.span(group[-1])[1])
        if stop >= cursor:
            inner = None
            if len(group) == 1 and depth < _MAX_DEPTH:
                inner = _partition(group[0], cursor, stop, symbol, syntax, depth + 1)
            out.extend(inner or [Range(start=cursor, end=stop, symbol=symbol)])
            cursor = stop + 1
        group.clear()

    for index, stmt in enumerate(body):
        first, last = syntax.span(stmt)
        span = last - first + 1
        if group and (span >= _BLOCK_LINES or last - cursor + 1 > _GROUP_LINES):
            flush(last=False)
        group.append(stmt)
        if span >= _BLOCK_LINES and index != len(body) - 1:
            flush(last=False)
    flush(last=True)
    return out if len(out) > 1 else None


def structure_ranges(nodes: Sequence[NodeT], syntax: Syntax[NodeT]) -> list[Range]:
    """Ranges for a file's top-level nodes."""
    ranges: list[Range] = []

    def walk(node: NodeT, prefix: str | None) -> None:
        start, end = syntax.span(node)
        name = syntax.symbol(node)
        symbol = f'{prefix}.{name or "body"}' if prefix else name
        members = syntax.members(node)
        if members:
            header_end = syntax.span(members[0])[0] - 1
            if header_end >= start:
                ranges.append(Range(start=start, end=header_end, symbol=symbol))
            for member in members:
                walk(member, symbol or 'body')
            return
        ranges.extend(_partition(node, start, end, symbol, syntax) or [Range(start=start, end=end, symbol=symbol)])

    for node in nodes:
        walk(node, None)
    return _merge_short_neighbors(ranges)


def _merge_short_neighbors(ranges: list[Range]) -> list[Range]:
    """Fold runs of imports/one-liners into one target (each costs a judgment).

    Members (dotted symbols) never merge, so a method is never folded into its
    container's header or a neighbor.
    """

    def mergeable(r: Range) -> bool:
        return r.end - r.start <= 1 and '.' not in (r.symbol or '')

    merged: list[Range] = []
    for r in ranges:
        last = merged[-1] if merged else None
        if (
            last
            and mergeable(last)
            and mergeable(r)
            and r.start <= last.end + 1
            and r.end - last.start < _MERGE_LINES
            and len(f'{last.symbol} {r.symbol}') < _MERGE_SYMBOL_CHARS
        ):
            names = dict.fromkeys(s for s in (last.symbol, r.symbol) if s)
            last.end, last.symbol = r.end, ', '.join(names) or None
        else:
            merged.append(Range(start=r.start, end=r.end, symbol=r.symbol))
    return merged


_Declaration = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
_Block = _Declaration | ast.If | ast.For | ast.AsyncFor | ast.While | ast.With | ast.AsyncWith | ast.Try | ast.TryStar
"""The statements with a `body` of statements to split into blocks; `match` has none, so it stays whole."""


class PythonSyntax:
    """`Syntax` over the standard library `ast`."""

    def span(self, node: ast.stmt) -> tuple[int, int]:
        decorators = node.decorator_list if isinstance(node, _Declaration) else []
        return min([node.lineno, *(d.lineno for d in decorators)]), node.end_lineno or node.lineno

    def symbol(self, node: ast.stmt) -> str | None:
        return node.name if isinstance(node, _Declaration) else None

    def members(self, node: ast.stmt) -> Sequence[ast.stmt] | None:
        return node.body if isinstance(node, ast.ClassDef) else None

    def statements(self, node: ast.stmt) -> Sequence[ast.stmt] | None:
        return node.body if isinstance(node, _Block) else None


def python_ranges(text: str) -> list[Range]:
    """Ranges for Python source. Raises `SyntaxError`/`ValueError`/`RecursionError` on unparsable source."""
    return structure_ranges(ast.parse(text).body, PythonSyntax())
