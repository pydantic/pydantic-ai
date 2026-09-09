"""Parse partial `run_code` arguments for eager execution."""

from __future__ import annotations

import ast

from pydantic import TypeAdapter, ValidationError
from pydantic_core import from_json
from typing_extensions import NotRequired, TypedDict


class PartialArgs(TypedDict):
    """Lenient view of partially streamed `run_code` arguments: only the keys eager scanning reads."""

    code: NotRequired[object]
    restart: NotRequired[object]


_PARTIAL_ARGS_ADAPTER: TypeAdapter[PartialArgs] = TypeAdapter(PartialArgs)

MAX_SCAN_CHARS = 1 << 18
"""Largest streamed prefix (in characters) the host will decode or `ast.parse`.

Larger snippets run whole at dispatch. This bounds one parse; the eager coordinator also
bounds cumulative parsing across all deltas because repeatedly parsing a growing prefix is
quadratic. Characters are the honest unit because parser work scales with them; the
equivalent UTF-8 byte size can be up to four times larger.
"""


def decode_partial_args(args_text: str) -> PartialArgs | None:
    """Recover what has streamed so far of the `run_code` arguments, or `None` if undecodable.

    The caller keeps `args_text` within `MAX_SCAN_CHARS`, so one decode is bounded.
    """
    try:
        return _PARTIAL_ARGS_ADAPTER.validate_python(from_json(args_text, allow_partial='trailing-strings'))
    except (ValueError, ValidationError):
        return None


def closed_statements(code: str) -> list[ast.stmt]:
    """Return the top-level statements in `code` that can no longer change as the stream grows.

    Only fully streamed lines participate, and the final parsed statement always stays
    provisional: a trailing compound (`for`, `if`, `try`) can still grow an indented body, so a
    statement counts as closed only once a later top-level statement starts on a later line. A prefix that
    does not parse yields nothing -- either an open bracket/triple-quote closes later, or the
    model wrote broken code and the real run will surface the error.

    The caller limits cumulative parsing work across deltas and falls back to normal dispatch
    when that limit is reached.
    """
    # Only `\n` terminates lines here. A snippet using lone `\r` separators parses (the AST
    # counts them as lines) but the executed-prefix slicing is `\n`-based, so treating `\r`
    # as a boundary would feed misaligned slices; such snippets stay whole until dispatch.
    end = code.rfind('\n')
    if end < 0 or end >= MAX_SCAN_CHARS:
        # Oversized prefixes skip host-side parsing entirely: the dispatch hands the code to
        # the sandbox parser, which applies its own resource limits. The cost is losing
        # eagerness for snippets this large, not correctness.
        return []
    try:
        tree = ast.parse(code[: end + 1])
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        # `ValueError` covers source `ast.parse` rejects before parsing, such as a NUL
        # character that JSON happily encodes; `RecursionError`/`MemoryError` cover parser
        # depth limits on adversarial nesting. The dispatch feed surfaces the real error.
        return []
    closed: list[ast.stmt] = []
    for stmt, following in zip(tree.body, tree.body[1:]):
        if following.lineno <= (stmt.end_lineno or stmt.lineno):
            # Semicolon-separated statements share a line, and the executed prefix is a
            # line slice: feeding this statement would drag the rest of its line (possibly
            # the provisional final expression) along with it. Hold the whole line back.
            break
        closed.append(stmt)
    return closed
