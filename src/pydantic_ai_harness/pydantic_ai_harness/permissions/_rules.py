"""Allow, ask about, or deny tool calls and shell commands by rule."""

from __future__ import annotations

import itertools
import posixpath
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
_UNSPLITTABLE = ('$(', '${', "$'", '$"', '`', '<(', '>(', '\n')
_LOOSE_SPLIT = re.compile(r'[\s;&|()<>`${}]+')
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
    """Globs for string arguments by name, such as `{'path': 'docs/*'}`. A missing or non-string argument does not match.

    Each value is matched as given and as a normalized path: an `'allow'` rule needs both to match, others either.
    """

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
    return [segment for segment in segments if segment.words or segment.redirected]


def _loose(words: Iterable[str]) -> list[str]:
    """`words` split again at whitespace and shell punctuation, so a quoted command such as `sh -c '...'` shows."""
    return [part for word in words for part in _LOOSE_SPLIT.split(word) if part]


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
    return _contains(_loose(segment.words), _loose(words))


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
    """Match a command too complex to split word by word, quotes removed, asking where a rule would allow it."""
    words = _loose([re.sub(r'[\'"\\]', '', command)])
    matched = _strictest(rule for rule in rules if _contains(words, _loose(_words(rule.command or ''))))
    return _UNSPLITTABLE_RULE if matched is not None and matched.verdict == 'allow' else matched


def _command_rule(
    rules: Sequence[PermissionRule], command: str, segments: list[_Segment] | None
) -> PermissionRule | None:
    """The rule deciding a shell command: the strictest any simple command matches, `'allow'` only if all do."""
    if segments is None:
        return _unsplittable_rule(rules, command)
    deciders = [_strictest(rule for rule in rules if _matches_segment(rule, segment)) for segment in segments]
    restrictive = _strictest(rule for rule in deciders if rule is not None and rule.verdict != 'allow')
    if restrictive is not None or None in deciders:
        return restrictive
    return deciders[0] if deciders else None


def _arg_matches(rule: PermissionRule, value: object, glob: str) -> bool:
    """Whether a string argument matches `glob`, as given and as a normalized path: both for `'allow'`, either otherwise."""
    if not isinstance(value, str):
        return False
    matches = [fnmatchcase(value, glob), fnmatchcase(posixpath.normpath(value), glob)]
    return all(matches) if rule.verdict == 'allow' else any(matches)


def decide(
    rules: Sequence[PermissionRule], *, tool_name: str, command: str | None, args: Mapping[str, Any]
) -> PermissionRule | None:
    """The rule deciding a call, or `None` when no rule does."""
    applicable = [
        rule
        for rule in rules
        if fnmatchcase(tool_name, rule.tool)
        and all(_arg_matches(rule, args.get(name), glob) for name, glob in rule.args.items())
    ]
    found = [rule for rule in applicable if rule.command is None]
    if command is None:
        return _strictest(found)
    segments = _segments(command)
    decided = _command_rule([rule for rule in applicable if rule.command is not None], command, segments)
    rule = _strictest([*found, decided] if decided is not None else found)
    if segments is None and rule is not None and rule.verdict == 'allow':
        return _UNSPLITTABLE_RULE
    return rule
