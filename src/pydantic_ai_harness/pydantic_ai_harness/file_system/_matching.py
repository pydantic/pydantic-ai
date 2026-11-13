"""Regex and glob matching for the filesystem tools' searches."""

from __future__ import annotations

import fnmatch
import functools
import math
import re
from collections.abc import Sequence

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai_harness.file_system._changes import split_lines


def matching_lines(text: str, compiled: re.Pattern[str], rel_str: str, limit: int) -> tuple[list[str], bool]:
    """Match one file's lines, keeping at most `limit` of them.

    Returns the formatted matches and whether a further match had to be
    dropped, so the caller reports truncation only when output was cut. A
    `limit` of zero or less keeps nothing.
    """
    matches: list[str] = []
    for line_num, line in enumerate(split_lines(text), start=1):
        line = line.removesuffix('\n').removesuffix('\r')
        if compiled.search(line):
            if len(matches) >= limit:
                return matches, True
            matches.append(f'{rel_str}:{line_num}:{line}')
    return matches, False


_QUANTIFIER = re.compile(r'[*+?]|\{(?P<count>\d+)\}|\{(?P<low>\d*),(?P<high>\d*)\}')


def _quantifier_at(pattern: str, index: int) -> tuple[int, float, int] | None:
    """The fewest and most repetitions of a quantifier starting at `index`, and the index after it.

    The index skips a lazy `?` or possessive `+` suffix, which changes the order matches are tried
    in, not how many there are.
    """
    if (quantifier := _QUANTIFIER.match(pattern, index)) is None:
        return None
    end = quantifier.end() + (pattern[quantifier.end() : quantifier.end() + 1] in ('?', '+'))
    if quantifier[0] in '*+?':
        return int(quantifier[0] == '+'), 1 if quantifier[0] == '?' else math.inf, end
    if (count := quantifier['count']) is not None:
        return int(count), int(count), end
    high = quantifier['high']
    return int(quantifier['low'] or 0), int(high) if high else math.inf, end


_MAX_BOUNDED_PATHS = 64
"""Most ways the bounded repeated groups of a pattern may match a text, together, before they count as nested repetition."""


def _power(base: float, exponent: float) -> float:
    """`base ** exponent`, as infinity where that would overflow a float."""
    if base > 1 and exponent * math.log(base) > 700:
        return math.inf
    return base**exponent


_SCOPED_FLAGS = re.compile(r'\(\?(?P<on>[a-zA-Z]*)(?:-(?P<off>[a-zA-Z]*))?:')
"""The opening of a group with scoped inline flags, such as `(?x:` or `(?i-x:`."""


def _skip_insignificant(pattern: str, index: int, verbose: bool) -> int:
    """The index of the next character at or after `index` that `re` does not ignore.

    `re` ignores an inline `(?#...)` comment, and in verbose mode whitespace and `#` comments too.
    """
    while index < len(pattern):
        if pattern.startswith('(?#', index):
            index = _inline_comment_end(pattern, index)
        elif verbose and pattern[index].isspace():
            index += 1
        elif verbose and pattern[index] == '#':
            end = pattern.find('\n', index)
            index = len(pattern) if end < 0 else end + 1
        else:
            break
    return index


def _inline_comment_end(pattern: str, index: int) -> int:
    """The index after the `(?#...)` comment that opens at `index`."""
    # The comment runs to the first unescaped `)`; the pattern compiled, so there is one.
    index += 3
    while pattern[index] != ')':
        index += 2 if pattern[index] == '\\' else 1
    return index + 1


def has_nested_quantifier(pattern: str, *, verbose: bool) -> bool:
    """Whether a repeated group itself contains a repetition, as `(a+)+` and `(.*a){12}` do.

    That shape can make a backtracking engine such as Python's `re` take exponential or high
    polynomial time on a short line, and `re` has no timeout, so a search run in Python refuses it.
    A group repeated a bounded number of times whose inside is bounded too, such as
    `([0-9]{1,3}[.]){3}`, is allowed while the ways all such groups can match, together, stay within `_MAX_BOUNDED_PATHS`.
    An optional part counts too: `(a?)+` is refused. This is a heuristic for the common shape: it also
    refuses some harmless patterns, such as `([a-z]+,)+x`, and misses overlapping alternatives, such as
    `(a|a)+`. `verbose` says whether
    the compiled pattern is verbose as a whole; a scoped `(?x:...)` group is tracked here. Where
    verbose mode applies, whitespace and comments are skipped as `re` skips them, so they cannot
    hide a quantifier from the check.
    """
    # Each open group saves the enclosing group's state and verbose mode, restored when it closes.
    enclosing: list[tuple[bool, float, bool]] = []
    # Whether the innermost open group can match a text more than one way, through a quantifier
    # whose count varies (`?`, `{1,3}`) or that repeats (`+`, `{2}`, which `|` can make ambiguous).
    varies = False
    paths = 1.0  # a bound on the ways the innermost open group can match a given text so far
    allowed = 1.0  # the product of the ways every bounded repeated group allowed so far can match
    index = 0
    while (index := _skip_insignificant(pattern, index, verbose)) < len(pattern):
        char = pattern[index]
        if char == '\\':
            index += 2
            continue
        if char == '[':
            index = _class_end(pattern, index)
        elif char == '(':
            enclosing.append((varies, paths, verbose))
            varies, paths = False, 1.0
            verbose = _scoped_verbose(pattern, index, verbose)
            index += pattern.startswith('?', index + 1)  # `(?:`, `(?P<name>`, ...: not a quantifier
        elif char == '|':
            paths = math.inf  # Alternatives can overlap, which this does not track.
        elif char == ')':  # the pattern compiled, and comments are skipped, so a group is open
            outer_varies, outer_paths, verbose = enclosing.pop()
            index, varies, paths, counted = _close_group(pattern, index, verbose, varies, paths)
            # Groups one after another multiply, so they share one budget.
            allowed *= counted
            if allowed > _MAX_BOUNDED_PATHS:
                return True
            varies, paths = outer_varies or varies, outer_paths * paths
        elif (quantifier := _quantifier_at(pattern, index)) is not None:
            low, high, index = quantifier
            index -= 1
            varies = varies or high > 1 or high > low
            paths *= high - low + 1
        index += 1
    return False


def _class_end(pattern: str, index: int) -> int:
    """The index of the `]` that closes the character class opening at `index`, or the pattern's length."""
    # A `]` first in the class (after any `^`) is literal.
    index += 1
    index += pattern.startswith('^', index)
    index += pattern.startswith(']', index)
    while index < len(pattern) and pattern[index] != ']':
        index += 2 if pattern[index] == '\\' else 1
    return index


def _scoped_verbose(pattern: str, index: int, verbose: bool) -> bool:
    """The verbose mode inside the group opening at `index`, which scoped inline flags such as `(?x:` can set."""
    if (flags := _SCOPED_FLAGS.match(pattern, index)) is None:
        return verbose
    if 'x' in flags['on']:
        return True
    if 'x' in (flags['off'] or ''):
        return False
    return verbose


def _close_group(pattern: str, index: int, verbose: bool, varies: bool, paths: float) -> tuple[int, bool, float, float]:
    """Repeat the group that the `)` at `index` closes by the quantifier after it, if there is one.

    Returns the index of the quantifier's last character (`index` without one), the group's `varies` and
    `paths` once repeated, and the ways it counts against `_MAX_BOUNDED_PATHS`: 1 unless the group
    varies and repeats.
    """
    # The quantifier after `)` is read in the enclosing group's verbose mode.
    quantifier = _quantifier_at(pattern, _skip_insignificant(pattern, index + 1, verbose))
    if quantifier is None:
        return index, varies, paths, 1.0
    low, high, end = quantifier
    repeated = _power(paths, high) * (high - low + 1)
    return end - 1, varies or high > 1 or high > low, repeated, repeated if varies and high > 1 else 1.0


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
