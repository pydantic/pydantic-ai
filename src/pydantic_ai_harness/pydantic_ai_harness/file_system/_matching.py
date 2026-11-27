"""Regex and glob matching for the filesystem tools' searches."""

from __future__ import annotations

import fnmatch
import functools
from collections.abc import Callable, Sequence

from pydantic_core import SchemaError, SchemaValidator, core_schema

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai_harness.file_system._changes import split_lines


def matching_lines(text: str, search: Callable[[str], bool], rel_str: str, limit: int) -> tuple[list[str], bool]:
    """Match one file's lines with `search`, keeping at most `limit` of them.

    Returns the formatted matches and whether a further match had to be
    dropped, so the caller reports truncation only when output was cut. A
    `limit` of zero or less keeps nothing.
    """
    matches: list[str] = []
    for line_num, line in enumerate(split_lines(text), start=1):
        line = line.removesuffix('\n').removesuffix('\r')
        if search(line):
            if len(matches) >= limit:
                return matches, True
            matches.append(f'{rel_str}:{line_num}:{line}')
    return matches, False


def linear_time_search(pattern: str) -> Callable[[str], bool]:
    """A function that reports whether `pattern` matches anywhere in a line, in time linear in the line's length.

    It uses pydantic-core's Rust regex engine rather than Python's backtracking `re`, which can take exponential
    time on patterns like `(a|a)*b` and has no timeout, so one model-chosen pattern could stall the event loop.
    Raises `ValueError` for a pattern that engine can't compile, such as one with a lookaround, a backreference or
    an atomic group, or one too large for it.
    """
    try:
        validator = SchemaValidator(core_schema.str_schema(pattern=pattern, regex_engine='rust-regex'))
    except SchemaError as exc:
        raise ValueError(
            "This regex uses syntax the linear-time engine this workspace searches with doesn't support (for "
            'example lookarounds such as `(?=`, backreferences such as `\\1`, `\\Z`, atomic groups or possessive '
            f'quantifiers), or is too large for it. Rewrite it without them. Details: {exc}'
        ) from exc
    return validator.isinstance_python


def glob_parts(pattern: str) -> tuple[list[str], bool]:
    """Split a relative glob into its components and whether it only matches directories (a trailing `/`)."""
    parts = [part for part in pattern.split('/') if part not in ('', '.')]
    if not parts or '..' in parts:
        raise ModelRetry(f'Pattern {pattern!r} is not a valid glob pattern.')
    return parts, pattern.endswith('/')


def glob_match(pattern: Sequence[str], path: Sequence[str]) -> bool:
    """Match path components against glob components the way `Path.glob` does.

    `*`, `?`, and `[...]` stay within one component; a `**` component matches
    zero or more whole components.
    """

    # Memoized on (pattern index, path index): each `**` tries every split, so plain
    # recursion is exponential in the number of `**` components.
    @functools.cache
    def match(p: int, q: int) -> bool:
        if p == len(pattern):
            return q == len(path)
        if pattern[p] == '**':
            return any(match(p + 1, index) for index in range(q, len(path) + 1))
        return q < len(path) and fnmatch.fnmatchcase(path[q], pattern[p]) and match(p + 1, q + 1)

    return match(0, 0)
