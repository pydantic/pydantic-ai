"""Count the steps a backtracking regex match takes, so a test can show a pattern scans in linear time.

`re` reports no step count, and a time bound flakes on a shared runner (see "Test cost" in `tests/AGENTS.md`).
`StepCounter` runs a pattern on the tree `re` parses it into and backtracks through it in `re`'s order: alternatives
left to right, greedy repeats longest first, lazy ones shortest first, possessive repeats and atomic groups without
giving anything back. It counts one step per node it tries, so a count is exact, and the same on any runner.

Only that control flow is reimplemented. Whether a character matches a literal, a set or `.`, and whether an anchor
holds, is asked of `re` itself by compiling that one node, so flags and Unicode categories behave exactly as in `re`.
The tree and the one-node compile come from `re._parser` and `re._compiler`, which `re.compile` is built on and which
publish no types, so their values are read as `object` and narrowed. A construct this module does not model, such as
a backreference, raises `NotImplementedError` rather than being counted wrongly.
"""

from __future__ import annotations

import functools
import importlib
import re
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NamedTuple, TypeGuard

_parser = importlib.import_module('re._parser')
_compiler = importlib.import_module('re._compiler')
_UNBOUNDED: int = importlib.import_module('re._constants').MAXREPEAT
"""The upper bound `re` gives `*`, `+` and `{n,}`."""

_SINGLE_CHARACTER = frozenset({'LITERAL', 'NOT_LITERAL', 'ANY', 'IN'})
_REPEATS = frozenset({'MAX_REPEAT', 'MIN_REPEAT', 'POSSESSIVE_REPEAT'})

Continuation = Callable[[int], bool]
"""What to match after a node, given the position the node ended at; `True` once the whole pattern matched."""
Matcher = Callable[[int, Continuation], bool]
"""Match a node at a position, handing the end of each way it matches to the continuation, in `re`'s order."""


class Node(NamedTuple):
    name: str
    """The opcode's name, such as `'MAX_REPEAT'`."""
    argument: object
    """The opcode's argument, as `re._parser` left it."""
    entry: object
    """The whole node as `re._parser` left it, to compile on its own."""


class StepBudgetExceeded(Exception):
    """A scan took more steps than its budget allowed."""


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)


def _is_str_pattern(value: object) -> TypeGuard[re.Pattern[str]]:
    # A pattern compiled from a parse tree keeps no source text to check, but every tree here came from a `str`.
    return isinstance(value, re.Pattern)


def _int(value: object) -> int:
    assert isinstance(value, int)
    return value


def _pair(value: object) -> tuple[object, object]:
    assert _is_object_tuple(value) and len(value) == 2
    return value[0], value[1]


def _nodes(subpattern: object) -> list[Node]:
    """The nodes of a parsed `SubPattern`, which keeps them in `data`, or of a plain list of them."""
    data: object = getattr(subpattern, 'data', subpattern)
    assert _is_object_list(data)
    nodes: list[Node] = []
    for entry in data:
        assert _is_object_tuple(entry)
        opcode, argument = entry
        nodes.append(Node(str(opcode), argument, entry))
    return nodes


def _repeat(argument: object) -> tuple[int, int, list[Node]]:
    assert _is_object_tuple(argument)
    low, high, body = argument
    return _int(low), _int(high), _nodes(body)


def _group(argument: object, flags: int) -> tuple[list[Node], int]:
    """The body of a `(...)` group, and the flags inside it after its `(?i:...)`-style flag changes.

    `re` combines them, so a scoped `(?a:...)` replaces the pattern's Unicode matching rather than adding to it.
    """
    assert _is_object_tuple(argument)
    _, add_flags, remove_flags, body = argument
    return _nodes(body), _int(_compiler._combine_flags(flags, add_flags, remove_flags))


def parse(pattern: str, flags: int) -> tuple[list[Node], int]:
    """The nodes `re` parses `pattern` into, and the flags it ends up with, inline flags included."""
    parsed: object = _parser.parse(pattern, flags)
    return _nodes(parsed), _int(getattr(getattr(parsed, 'state'), 'flags'))


def _compile_node(node: Node, flags: int) -> re.Pattern[str]:
    # A fresh `State` carries no pattern-wide flags, so `flags` alone applies, including a `(?-i:...)` that
    # removes a flag the whole pattern was compiled with.
    compiled: object = _compiler.compile(_parser.SubPattern(_parser.State(), [node.entry]), flags)
    assert _is_str_pattern(compiled)
    return compiled


def character_test(node: Node, flags: int) -> Callable[[str], bool] | None:
    """Whether `re` matches a character with `node`, for a node that matches exactly one character; else `None`."""
    if node.name not in _SINGLE_CHARACTER:
        return None
    compiled = _compile_node(node, flags)

    @functools.cache
    def matches(character: str) -> bool:
        return compiled.fullmatch(character) is not None

    return matches


@dataclass
class Scan:
    spans: list[tuple[int, int]]
    """The span of each match, as `re.finditer` reports them."""
    steps: int
    """How many nodes the scan tried, counting every backtrack."""


@contextmanager
def _recursion_limit(limit: int) -> Generator[None]:
    # Each node a match attempt gets past nests the continuation after it, so an attempt over a few hundred
    # characters recurses deeper than the default limit. Python frames, not the C stack, carry it on 3.11+.
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, limit))
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


class StepCounter:
    """Scan text with a pattern the way `re.finditer` does, counting the steps the backtracking takes."""

    def __init__(self, pattern: str, flags: int = 0) -> None:
        nodes, pattern_flags = parse(pattern, flags)
        self._text = ''
        self._steps = 0
        self._budget = 0
        self._match = self._sequence(nodes, pattern_flags)

    def scan(self, text: str, budget: int) -> Scan:
        """Find every match in `text`; raise `StepBudgetExceeded` once the scan takes more than `budget` steps."""
        self._text, self._steps, self._budget = text, 0, budget
        spans: list[tuple[int, int]] = []
        position, after_empty_match = 0, False
        with _recursion_limit(50 * len(text) + 1_000):
            while (span := self._search(position, after_empty_match)) is not None:
                spans.append(span)
                position, after_empty_match = span[1], span[0] == span[1]
        return Scan(spans, self._steps)

    def _search(self, position: int, after_empty_match: bool) -> tuple[int, int] | None:
        """The first match at or after `position`; right after an empty match, not another empty one there."""
        for start in range(position, len(self._text) + 1):
            ends: list[int] = []
            must_advance = after_empty_match and start == position

            def accept(end: int, start: int = start, must_advance: bool = must_advance) -> bool:
                if must_advance and end == start:
                    return False
                ends.append(end)
                return True

            if self._match(start, accept):
                return start, ends[0]
        return None

    def _step(self) -> None:
        self._steps += 1
        if self._steps > self._budget:
            raise StepBudgetExceeded(f'more than {self._budget} steps')

    def _sequence(self, nodes: list[Node], flags: int) -> Matcher:
        matchers = [self._node(node, flags) for node in nodes]

        def chain(index: int) -> Matcher:
            if index == len(matchers):
                return lambda position, then: then(position)
            first, rest = matchers[index], chain(index + 1)
            return lambda position, then: first(position, lambda end: rest(end, then))

        return chain(0)

    def _node(self, node: Node, flags: int) -> Matcher:
        test = character_test(node, flags)
        if test is not None:
            return self._character(test)
        match node.name:
            case 'AT':
                return self._anchor(_compile_node(node, flags))
            case 'BRANCH':
                _, alternatives = _pair(node.argument)
                assert _is_object_list(alternatives)
                return self._branch([self._sequence(_nodes(alternative), flags) for alternative in alternatives])
            case 'SUBPATTERN':
                return self._sequence(*_group(node.argument, flags))
            case 'ATOMIC_GROUP':
                return self._atomic(self._sequence(_nodes(node.argument), flags))
            case 'ASSERT' | 'ASSERT_NOT':
                direction, body = _pair(node.argument)
                width: object = getattr(body, 'getwidth')()
                low, high = (_int(bound) for bound in _pair(width))
                behind = 0 if _int(direction) == 1 else low
                assert behind == high or not behind, 'a lookbehind has a fixed width'
                return self._lookaround(self._sequence(_nodes(body), flags), behind, want=node.name == 'ASSERT')
            case 'MAX_REPEAT' | 'MIN_REPEAT' | 'POSSESSIVE_REPEAT':
                low, high, body = _repeat(node.argument)
                single = character_test(body[0], flags) if len(body) == 1 else None
                if single is not None:
                    return self._repeat_character(single, low, high, node.name)
                return self._repeat_sequence(self._sequence(body, flags), low, high, node.name)
            case _:
                raise NotImplementedError(f'`StepCounter` does not model {node.name}')

    def _character(self, test: Callable[[str], bool]) -> Matcher:
        def character(position: int, then: Continuation) -> bool:
            self._step()
            return position < len(self._text) and test(self._text[position]) and then(position + 1)

        return character

    def _anchor(self, anchor: re.Pattern[str]) -> Matcher:
        def holds(position: int, then: Continuation) -> bool:
            self._step()
            return anchor.match(self._text, position) is not None and then(position)

        return holds

    def _branch(self, alternatives: list[Matcher]) -> Matcher:
        def branch(position: int, then: Continuation) -> bool:
            self._step()
            for alternative in alternatives:
                if alternative(position, then):
                    return True
            return False

        return branch

    def _first_end(self, matcher: Matcher, position: int) -> int | None:
        """Where `matcher`'s first match at `position` ends, without backtracking into it later."""
        ends: list[int] = []

        def first(end: int) -> bool:
            ends.append(end)
            return True

        matcher(position, first)
        return ends[0] if ends else None

    def _atomic(self, body: Matcher) -> Matcher:
        def atomic(position: int, then: Continuation) -> bool:
            self._step()
            end = self._first_end(body, position)
            return end is not None and then(end)

        return atomic

    def _lookaround(self, body: Matcher, behind: int, want: bool) -> Matcher:
        """A lookahead, or a lookbehind `behind` characters wide, which must end where it is asserted."""

        def lookaround(position: int, then: Continuation) -> bool:
            self._step()
            start = position - behind
            found = start >= 0 and body(start, lambda end: end == position or not behind)
            return found == want and then(position)

        return lookaround

    def _repeat_character(self, test: Callable[[str], bool], low: int, high: int, kind: str) -> Matcher:
        """A repeat of one character, which `re` runs as one loop over the run of characters it matches."""

        def repeat(position: int, then: Continuation) -> bool:
            self._step()
            text = self._text
            limit = len(text) if high == _UNBOUNDED else min(len(text), position + high)
            end = position
            if kind == 'MIN_REPEAT':
                while True:
                    self._step()
                    if end - position >= low and then(end):
                        return True
                    if end >= limit or not test(text[end]):
                        return False
                    end += 1
            while end < limit and test(text[end]):
                self._step()
                end += 1
            if kind == 'POSSESSIVE_REPEAT':
                return end - position >= low and then(end)
            while end - position >= low:
                self._step()
                if then(end):
                    return True
                end -= 1
            return False

        return repeat

    def _repeat_sequence(self, body: Matcher, low: int, high: int, kind: str) -> Matcher:
        """A repeat of a group; an iteration past `low` that matches nothing ends the repeat there, as in `re`."""

        def possessive(position: int, then: Continuation) -> bool:
            self._step()
            count = 0
            while count < high:
                end = self._first_end(body, position)
                if end is None or (end == position and count >= low):
                    break
                position, count = end, count + 1
            return count >= low and then(position)

        def iterate(position: int, then: Continuation, count: int) -> bool:
            self._step()

            def another() -> bool:
                return count < high and body(
                    position,
                    lambda end: then(end) if end == position and count >= low else iterate(end, then, count + 1),
                )

            if kind == 'MIN_REPEAT':
                return (count >= low and then(position)) or another()
            return another() or (count >= low and then(position))

        if kind == 'POSSESSIVE_REPEAT':
            return possessive
        return lambda position, then: iterate(position, then, 0)


_SAMPLE = 'aA0 _-./:=@"\'<>\\\n\t!#\x1b\x00\u200b'
"""Characters to try in a set or wildcard, besides the ones a pattern names."""
GROWTH_LIMIT = 3
"""Doubling the input doubles a linear scan's added steps and quadruples a quadratic one's; this sits between."""


@dataclass
class Shape:
    units: list[str]
    """Strings to repeat into an input: what precedes each repeat in the pattern, and what the repeat matches."""
    reach: int
    """How far one match attempt can read through the pattern's bounded parts, counting each unbounded repeat once."""
    unbounded: bool
    """Whether some repeat has no upper bound. Without one, an attempt reads at most `reach` characters."""
    nodes: int
    """How many nodes the pattern parses into."""


@dataclass
class _Walk:
    units: set[str] = field(default_factory=set[str])
    nodes: int = 0
    admitting: bool = False
    """Whether each one-character lookbehind carries a character it admits into the units after it."""


def _representatives(node: Node, test: Callable[[str], bool]) -> list[str]:
    """Characters `node` matches: the ones it names first, then a few from `_SAMPLE`."""
    named: list[str] = []
    if node.name == 'LITERAL':
        named.append(chr(_int(node.argument)))
    elif node.name == 'IN':
        for item in _nodes(node.argument):
            if item.name == 'LITERAL':
                named.append(chr(_int(item.argument)))
            elif item.name == 'RANGE':
                named.append(chr(_int(_pair(item.argument)[0])))
    sampled = [c for c in _SAMPLE if test(c)][:3]
    return [c for c in dict.fromkeys([*named, *sampled]) if test(c)]


def _walk(nodes: list[Node], flags: int, prefix: str, walk: _Walk) -> tuple[int, bool]:
    """Add the pump units of `nodes` to `walk`, carrying in the literal text before them; return reach and unboundedness."""
    reach, unbounded = 0, False
    for index, node in enumerate(nodes):
        walk.nodes += 1
        test = character_test(node, flags)
        if node.name in _REPEATS:
            low, high, body = _repeat(node.argument)
            single = character_test(body[0], flags) if len(body) == 1 else None
            if single is not None:
                walk.nodes += 1
                for character in _representatives(body[0], single):
                    walk.units.update({character, prefix + character})
                body_reach, body_unbounded = 1, False
            else:
                body_reach, body_unbounded = _walk(body, flags, prefix, walk)
            walk.units.add(prefix)
            unbounded |= body_unbounded or high == _UNBOUNDED
            reach += body_reach * (max(low, 1) if high == _UNBOUNDED else high)
            prefix = prefix if low == 0 else ''
        elif test is not None:
            prefix += ''.join(_representatives(node, test)[:1])
            reach += 1
        elif node.name == 'BRANCH':
            _, alternatives = _pair(node.argument)
            assert _is_object_list(alternatives)
            walks = [_walk(_nodes(alternative), flags, prefix, walk) for alternative in alternatives]
            reach += max(alternative_reach for alternative_reach, _ in walks)
            unbounded |= any(alternative_unbounded for _, alternative_unbounded in walks)
            prefix = ''
        elif node.name in ('SUBPATTERN', 'ATOMIC_GROUP'):
            body, body_flags = (
                _group(node.argument, flags) if node.name == 'SUBPATTERN' else (_nodes(node.argument), flags)
            )
            body_reach, body_unbounded = _walk(body, body_flags, prefix, walk)
            reach, unbounded, prefix = reach + body_reach, unbounded or body_unbounded, ''
        elif node.name in ('ASSERT', 'ASSERT_NOT'):
            direction, body = _pair(node.argument)
            body_nodes = _nodes(body)
            body_reach, body_unbounded = _walk(body_nodes, flags, '', walk)
            reach += body_reach if _int(direction) == 1 else 0
            unbounded |= body_unbounded
            # A one-character lookbehind lets an attempt start only after a character it admits, or at the start of
            # the text. `(?<!\S)#` restarts at each `#` of ` # # #`, not of `###`, while `(?<!\S)(?:ab)*(?:ab)*c`
            # backtracks through `abab` from the start. So the rest of the sequence is walked as is, and once more
            # admitting from here on: one extra walk per lookbehind, not one per combination of them.
            behind = character_test(body_nodes[0], flags) if _int(direction) == -1 and len(body_nodes) == 1 else None
            admits = [c for c in _SAMPLE if behind(c) == (node.name == 'ASSERT')] if behind is not None else []
            if admits and walk.admitting:
                prefix += admits[0]
            elif admits:
                rest = _Walk(admitting=True)
                _walk(nodes[index + 1 :], flags, prefix + admits[0], rest)
                walk.units |= rest.units
    walk.units.add(prefix)
    walk.units.discard('')
    return reach, unbounded


def shape(pattern: str, flags: int) -> Shape:
    nodes, pattern_flags = parse(pattern, flags)
    walk = _Walk()
    reach, unbounded = _walk(nodes, pattern_flags, '', walk)
    return Shape(sorted(walk.units), reach, unbounded, walk.nodes)


def linearity_failure(pattern: str, flags: int) -> str | None:
    """How scanning repeated adversarial input with `pattern` fails to stay linear, or `None` if it stays linear.

    Each pump unit is repeated into an input long enough to read past every bounded part of a match attempt, then
    followed by `!`, which few patterns accept, so the last attempts fail. Each scan must find what `re.finditer`
    finds, and stay within `4 * nodes * (reach + 1)` steps per character: a linear scan tries each node about once
    per character an attempt reads. A pattern without an unbounded repeat reads at most `reach` characters per
    attempt, so that one scan is enough. One with an unbounded repeat is also scanned at twice and four times that
    length, and must add no more than `GROWTH_LIMIT` times as many steps for the second doubling as for the first.
    """
    counter = StepCounter(pattern, flags)
    compiled = re.compile(pattern, flags)
    pattern_shape = shape(pattern, flags)
    sizes = (1, 2, 4) if pattern_shape.unbounded else (1,)
    for unit in pattern_shape.units:
        repeats = max(4, -(-(pattern_shape.reach + 16) // len(unit)))
        steps: list[int] = []
        for size in sizes:
            text = unit * repeats * size + '!'
            budget = len(text) * 4 * pattern_shape.nodes * (pattern_shape.reach + 1)
            try:
                scan = counter.scan(text, budget)
            except StepBudgetExceeded:
                return f'{unit!r} * {repeats * size}: over {budget} steps'
            assert scan.spans == [m.span() for m in compiled.finditer(text)], (
                f'`StepCounter` disagrees with `re` on {text!r}'
            )
            steps.append(scan.steps)
        if pattern_shape.unbounded and steps[2] - steps[1] > GROWTH_LIMIT * (steps[1] - steps[0]):
            return f'{unit!r} * {repeats}, {2 * repeats}, {4 * repeats}: {steps} steps'
    return None
