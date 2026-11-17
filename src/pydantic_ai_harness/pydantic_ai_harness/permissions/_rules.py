"""Allow, ask about, or deny tool calls and shell commands by rule."""

from __future__ import annotations

import itertools
import ntpath
import posixpath
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal, Required

from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolSelector
from pydantic_ai_harness._judge import SerializableToolSelector

PermissionVerdict = Literal['allow', 'ask', 'deny']
"""What a `PermissionRule` does with the calls it matches: run them, ask a person first, or refuse them."""

_SEVERITY: dict[PermissionVerdict, int] = {'allow': 0, 'ask': 1, 'deny': 2}
_SEPARATORS = frozenset({';', '&', '&&', '|', '||', '|&'})
_OPERATOR_CHARS = frozenset('();<>|&')
_UNSPLITTABLE = ('$', '`', '<(', '>(', '\n')
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

    tools: ToolSelector[Any] = 'all'
    """Which tools the rule applies to: `'all'`, a list of names, a metadata match, or a predicate."""

    command: str | None = None
    """The words of a shell command, such as `'git push'`, matched against each simple command in a compound one.

    Only tools that declare a shell command argument (`code_arg_language='shell'` and a `code_arg_name` in their
    metadata) match a rule with a `command`. An `'allow'` rule matches a simple command that starts with these
    words and has no redirection other than to `/dev/null` or a file descriptor. An `'ask'` or `'deny'` rule
    matches one that contains them anywhere, inside quotes too, so `sudo git push` matches `'git push'`, and its
    first word also matches a path ending in it, so `/bin/rm -rf x` matches `'rm -rf'`.
    """

    args: Mapping[str, str] = field(default_factory=dict[str, str])
    """Globs for string arguments by name, such as `{'path': 'docs/*'}`. A missing or non-string argument does not match.

    Each value is matched as given and as a normalized path: an `'allow'` rule needs both to match, others either.
    """

    reason: str | None = None
    """Why the rule asks or denies, shown to the person asked or to the model when denied."""

    def __post_init__(self) -> None:
        """Fail at construction on a `command` with no words or unbalanced quotes."""
        if self.command is not None and not _loose(_words(self.command)):
            raise UserError(f'PermissionRule.command must be a shell command, got {self.command!r}.')


class PermissionRuleSpec(TypedDict, total=False):
    """A `PermissionRule` in an agent spec, where `tools` cannot be a predicate."""

    verdict: Required[PermissionVerdict]
    tools: SerializableToolSelector
    command: str
    args: dict[str, str]
    reason: str


@dataclass
class _Segment:
    words: list[str] = field(default_factory=list[str])
    redirected: bool = False
    descriptors: set[int] = field(default_factory=set[int])
    """Indexes of `words` that may be the file descriptor of the redirection after them, as in `2>`."""


def _words(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _quotes_an_operator(command: str) -> bool:
    """Whether `command` quotes or escapes an operator character, which `shlex` would not tell apart from a real one."""
    quote = ''
    escaped = False
    for char in command:
        if escaped:
            escaped = False
            if char in _OPERATOR_CHARS:
                return True
        elif char == '\\' and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ''
            elif char in _OPERATOR_CHARS:
                return True
        elif char in '\'"':
            quote = char
    return False


def _tokens(command: str) -> list[str] | None:
    """`command`'s words and operators, or `None` when it has syntax the lexer cannot follow."""
    if any(marker in command for marker in _UNSPLITTABLE) or _quotes_an_operator(command):
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
        segment = segments[-1]
        if target:
            target = False
        elif token in _SEPARATORS:
            segments.append(_Segment())
        elif not set(token) <= _OPERATOR_CHARS:
            segment.words.append(token)
        elif _UNSPLITTABLE_OPERATOR.search(token):
            return None
        else:
            target = True
            segment.redirected |= (token, following) not in _HARMLESS_REDIRECTS
            if segment.words and segment.words[-1].isdigit():
                segment.descriptors.add(len(segment.words) - 1)
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
    arguments = (word for index, word in enumerate(segment.words) if index not in segment.descriptors)
    return _contains(_loose(arguments), _loose(words))


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
    """Match a command too complex to split word by word, quotes removed; with command rules, it asks at least."""
    if not rules:
        return None
    words = _loose([re.sub(r'[\'"\\]', '', command)])
    restrictive = _strictest(
        rule for rule in rules if rule.verdict != 'allow' and _contains(words, _loose(_words(rule.command or '')))
    )
    return restrictive or _UNSPLITTABLE_RULE


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


def _arg_matches(rule: PermissionRule, value: object, glob: str, *, posix: bool) -> bool:
    """Whether a string argument matches `glob`, as given and as a normalized path: both for `'allow'`, either otherwise.

    On Windows paths are compared case-insensitively, with `/` for separators.
    """
    if not isinstance(value, str):
        return False
    forms = [value, posixpath.normpath(value)]
    if not posix:
        forms = [form.replace('\\', '/').lower() for form in (value, ntpath.normpath(value))]
        glob = glob.lower()
    matches = [fnmatchcase(form, glob) for form in forms]
    return all(matches) if rule.verdict == 'allow' else any(matches)


def decide(
    rules: Sequence[PermissionRule], *, command: str | None, args: Mapping[str, Any], posix: bool
) -> PermissionRule | None:
    """The rule deciding a call to a tool `rules` all apply to, or `None` when no rule does.

    With `posix=False`, for a Windows workspace, a command is never split, since PowerShell quotes differently.
    """
    applicable = [
        rule
        for rule in rules
        if all(_arg_matches(rule, args.get(name), glob, posix=posix) for name, glob in rule.args.items())
    ]
    found = [rule for rule in applicable if rule.command is None]
    if command is None:
        return _strictest(found)
    segments = _segments(command) if posix else None
    decided = _command_rule([rule for rule in applicable if rule.command is not None], command, segments)
    rule = _strictest([*found, decided] if decided is not None else found)
    if segments is None and rule is not None and rule.verdict == 'allow':
        return _UNSPLITTABLE_RULE
    return rule
