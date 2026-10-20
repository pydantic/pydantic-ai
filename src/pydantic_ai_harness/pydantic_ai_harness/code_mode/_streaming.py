"""Parse partial `run_code` arguments for eager execution."""

from __future__ import annotations

import ast
import warnings
from typing import Literal, NotRequired, overload

from pydantic import TypeAdapter, ValidationError
from pydantic_core import from_json
from typing_extensions import TypedDict


class PartialArgs(TypedDict):
    """Lenient view of partially streamed `run_code` arguments: only the keys eager scanning reads."""

    code: NotRequired[object]
    restart: NotRequired[object]


_PARTIAL_ARGS_ADAPTER: TypeAdapter[PartialArgs] = TypeAdapter(PartialArgs)
_ANALYSIS_FILENAME = '<pydantic-ai-code-mode-analysis>'

MAX_SCAN_CHARS = 1 << 18
"""Largest streamed prefix (in characters) the host will decode or `ast.parse`.

Larger snippets run whole at dispatch. This bounds one parse; the eager coordinator also
bounds cumulative parsing across all deltas because repeatedly parsing a growing prefix is
quadratic. Characters are the honest unit because parser work scales with them; the
equivalent UTF-8 byte size can be up to four times larger.
"""

MAX_SCAN_WORK_CHARS = 1 << 20
"""Cumulative characters a streamed call may hand to host parsers."""

CANCEL_TIMEOUT_SECONDS = 5.0
"""How long to wait for cancelled streamed work to release a non-cooperative nested tool before
abandoning it. Abandoning is safe: the cancelled feed or launch starts no further tool calls."""


def decode_partial_args(args_text: str) -> PartialArgs | None:
    """Recover what has streamed so far of the `run_code` arguments, or `None` if undecodable.

    The caller keeps `args_text` within `MAX_SCAN_CHARS`, so one decode is bounded.
    """
    try:
        return _PARTIAL_ARGS_ADAPTER.validate_python(from_json(args_text, allow_partial='trailing-strings'))
    except (TypeError, ValueError, ValidationError):
        # `TypeError` covers raw surrogate code points in the streamed text: pydantic-core's
        # JSON decoder rejects them before any value exists, and `from_json` raises rather
        # than returning a parse error. Undecodable is the same as partial: no launch.
        return None


@overload
def parse_code(code: str, *, mode: Literal['exec'] = 'exec') -> ast.Module: ...


@overload
def parse_code(code: str, *, mode: Literal['eval']) -> ast.Expression: ...


def parse_code(code: str, *, mode: Literal['exec', 'eval'] = 'exec') -> ast.Module | ast.Expression:
    """Parse model code for host-side analysis, leaving execution diagnostics to the sandbox."""
    # Each streamed delta can reparse the same literal. CPython's invalid-escape advisory
    # would flood stderr (or become an error under strict warning filters), even though
    # these escapes preserve their backslashes. This is analysis, not Python execution.
    # Warning filters are process-global on older Python. The unique filename limits
    # suppression to our analysis, not another thread compiling unrelated source.
    with warnings.catch_warnings():
        for category in (SyntaxWarning, DeprecationWarning):
            warnings.filterwarnings(
                'ignore', message=r'.*invalid escape sequence', category=category, module=rf'^{_ANALYSIS_FILENAME}$'
            )
        tree = ast.parse(code, filename=_ANALYSIS_FILENAME, mode=mode)
    assert isinstance(tree, (ast.Module, ast.Expression))
    return tree


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
        tree = parse_code(code[: end + 1])
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
