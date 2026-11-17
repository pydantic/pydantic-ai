"""Allow, ask about, or deny tool calls and shell commands by rule."""

from __future__ import annotations

import itertools
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal

from pydantic_ai.exceptions import UserError

PermissionVerdict = Literal['allow', 'ask', 'deny']
"""What a `PermissionRule` does with the calls it matches: run them, ask a person first, or refuse them."""

_SEVERITY: dict[PermissionVerdict, int] = {'allow': 0, 'ask': 1, 'deny': 2}
_SEPARATORS = frozenset({';', '&', '&&', '|', '||', '|&'})
_OPERATOR_CHARS = frozenset('();<>|&')
_UNSPLITTABLE = ('$(', '`', '<(', '>(', '\n')
_UNSPLITTABLE_OPERATOR = re.compile(r'[()]|<<|;;')
_HARMLESS_REDIRECTS = frozenset({('>', '/dev/null'), *(('>&', str(fd)) for fd in range(10))})


@dataclass(frozen=True)
class PermissionRule:
    """Allow, ask about, or deny the tool calls that match every field set.

    When several rules match a call, `'deny'` beats `'ask'`, which beats `'allow'`, whatever their order;
    among rules with the same verdict, the first one gives the reason.
    """

    verdict: PermissionVerdict
    """What to do with a matching call."""

    _: KW_ONLY

    tool: str = '*'
    """A glob for the tool name, such as `'run_command'` or `'mcp_*'`."""

    command: str | None = None
    """The words of a shell command, such as `'git push'`, matched against each simple command in a compound one.

    Only tools that declare a shell command argument (`code_arg_language='shell'` and a `code_arg_name` in their
    metadata) match a rule with a `command`. An `'allow'` rule matches a simple command that starts with these
    words and has no redirection other than to `/dev/null` or a file descriptor. An `'ask'` or `'deny'` rule
    matches one that contains them anywhere, so `sudo git push` matches `'git push'`, and its first word also
    matches a path ending in it, so `/bin/rm -rf x` matches `'rm -rf'`.
    """

    args: Mapping[str, str] = field(default_factory=dict[str, str])
    """Globs for string arguments by name, such as `{'path': 'docs/*'}`. A missing or non-string argument does not match."""

    reason: str | None = None
    """Why the rule asks or denies, shown to the person asked or to the model when denied."""

    def __post_init__(self) -> None:
        """Fail at construction on a `command` with no words or unbalanced quotes."""
        if self.command is not None and not _words(self.command):
            raise UserError(f'PermissionRule.command must be a shell command, got {self.command!r}.')


@dataclass
class _Segment:
    words: list[str] = field(default_factory=list[str])
    redirected: bool = False


def _words(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _tokens(command: str) -> list[str] | None:
    """`command`'s words and operators, or `None` when it has syntax the lexer cannot follow."""
    if any(marker in command for marker in _UNSPLITTABLE):
        return None
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ''
    try:
        return list(lexer)
    except ValueError:
        return None


def _segments(command: str) -> list[_Segment] | None:
    """The simple commands in `command`, or `None` when its syntax is too complex to split safely."""
    tokens = _tokens(command)
    if tokens is None:
        return None
    segments = [_Segment()]
    target = False
    for token, following in itertools.pairwise([*tokens, '']):
        if target:
            target = False
        elif token in _SEPARATORS:
            segments.append(_Segment())
        elif not set(token) <= _OPERATOR_CHARS:
            segments[-1].words.append(token)
        elif _UNSPLITTABLE_OPERATOR.search(token):
            return None
        else:
            target = True
            segments[-1].redirected |= (token, following) not in _HARMLESS_REDIRECTS
    return [segment for segment in segments if segment.words]


def _contains(words: Sequence[str], rule: Sequence[str]) -> bool:
    head, rest = rule[0], list(rule[1:])
    return any(
        (word == head or word.endswith(f'/{head}')) and list(words[i + 1 : i + len(rule)]) == rest
        for i, word in enumerate(words)
    )


def _matches_segment(rule: PermissionRule, segment: _Segment) -> bool:
    words = _words(rule.command or '')
    if rule.verdict == 'allow':
        return not segment.redirected and segment.words[: len(words)] == words
    return _contains(segment.words, words)


def _strictest(rules: Iterable[PermissionRule]) -> PermissionRule | None:
    found: PermissionRule | None = None
    for rule in rules:
        if found is None or _SEVERITY[rule.verdict] > _SEVERITY[found.verdict]:
            found = rule
    return found


_UNSPLITTABLE_RULE = PermissionRule(
    'ask', reason='Permission rules cannot split this command into simple commands, so it asks first.'
)


def _unsplittable_rule(rules: Sequence[PermissionRule], command: str) -> PermissionRule | None:
    """Match the words of a command too complex to split against the `'ask'` and `'deny'` rules.

    When none matches but an `'allow'` rule could have vouched for the command, it asks.
    """
    words = re.split(r'[\s;&|()<>`$]+', command)
    restrictive = _strictest(
        rule for rule in rules if rule.verdict != 'allow' and _contains(words, _words(rule.command or ''))
    )
    if restrictive is not None or all(rule.verdict != 'allow' for rule in rules):
        return restrictive
    return _UNSPLITTABLE_RULE


def _command_rule(rules: Sequence[PermissionRule], command: str) -> PermissionRule | None:
    """The rule deciding a shell command: the strictest any simple command matches, `'allow'` only if all do."""
    segments = _segments(command)
    if segments is None:
        return _unsplittable_rule(rules, command)
    deciders = [_strictest(rule for rule in rules if _matches_segment(rule, segment)) for segment in segments]
    restrictive = _strictest(rule for rule in deciders if rule is not None and rule.verdict != 'allow')
    if restrictive is not None or None in deciders:
        return restrictive
    return deciders[0] if deciders else None


def decide(
    rules: Sequence[PermissionRule], *, tool_name: str, command: str | None, args: Mapping[str, Any]
) -> PermissionRule | None:
    """The rule deciding a call, or `None` when no rule does."""
    applicable = [
        rule
        for rule in rules
        if fnmatchcase(tool_name, rule.tool)
        and all(isinstance(args.get(name), str) and fnmatchcase(args[name], glob) for name, glob in rule.args.items())
    ]
    command_rules = [rule for rule in applicable if rule.command is not None]
    found = [rule for rule in applicable if rule.command is None]
    if command is not None and command_rules:
        decided = _command_rule(command_rules, command)
        if decided is not None:
            found.append(decided)
    return _strictest(found)
