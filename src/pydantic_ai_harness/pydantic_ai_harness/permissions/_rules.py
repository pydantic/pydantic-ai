"""Allow, ask about, or deny tool calls and shell commands by rule."""

from __future__ import annotations

import itertools
import ntpath
import posixpath
import re
import shlex
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from fnmatch import fnmatchcase
from typing import Any, Generic, Literal, NamedTuple, Required

from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, ToolSelector
from pydantic_ai_harness._judge import SerializableToolSelector

PermissionVerdict = Literal['allow', 'ask', 'deny']
"""What a `PermissionRule` does with the calls it matches: run them, ask a person first, or refuse them."""

_SEVERITY: dict[PermissionVerdict, int] = {'allow': 0, 'ask': 1, 'deny': 2}
_SEPARATORS = frozenset({';', '&', '&&', '|', '||', '|&'})
_OPERATOR_CHARS = frozenset('();<>|&')
_UNSPLITTABLE = ('$', '`', '<(', '>(', '\n', '\r')
_LOOSE_SPLIT = re.compile(r'[\s;&|()<>`${}]+')
_UNSPLITTABLE_OPERATOR = re.compile(r'[()]|<<|;;')
_GLOB = re.compile(r'[*?]|\[[^\[\]]*\]')
"""A pattern the shell expands: `*`, `?`, or a closed bracket expression, so the `[` builtin isn't one."""
_LITERAL = re.compile(r'[^*?\[\]]')
_OPTIONS = re.compile(r'-[A-Za-z]+')
_REDIRECTION = re.compile(r'(?<!\d)\d*[<>]+&?\s*[^\s;&|<>()]*')
_GLOB_LIMIT = 4096
"""Longer argument values aren't globbed, which could take quadratic time: ask and deny rules match them."""
COMMAND_LIMIT = 65536
"""Longer commands aren't parsed: no rule allows them, and they ask when an ask or deny command rule applies."""
_HARMLESS_REDIRECTS = frozenset({('>', '/dev/null'), *(('>&', str(fd)) for fd in range(10))})


@dataclass(frozen=True)
class PermissionRule(Generic[AgentDepsT]):
    """Allow, ask about, or deny the tool calls that match every field set.

    When several rules match a call, `'deny'` beats `'ask'`, which beats `'allow'`, whatever their order;
    among rules with the same verdict, the first one gives the reason.
    """

    verdict: PermissionVerdict
    """What to do with a matching call."""

    _: KW_ONLY

    tools: ToolSelector[AgentDepsT] = 'all'
    """Which tools the rule applies to: `'all'`, a list of names, a metadata match, or a predicate."""

    command: str | None = None
    """The words of a shell command, such as `'git push'`, matched against each simple command in a compound one.

    Only tools that declare a shell command argument (`code_arg_language='shell'` and a `code_arg_name` in their
    metadata) match a rule with a `command`. An `'allow'` rule matches a simple command that starts with these
    words and has no redirection other than to `/dev/null` or a file descriptor.

    An `'ask'` or `'deny'` rule matches a simple command that contains its first word followed by the others in order,
    other words allowed in between, so `sudo git -C x push` matches `'git push'`. A short-option cluster such as `-rf`
    matches those options in any grouping, the first word also matches a path ending in it, such as `/bin/rm`, and a
    pattern such as `r[m]` matches the words it could expand to. Quoted text that matches may only be data, so there a
    deny rule asks instead.
    """

    args: Mapping[str, str] = field(default_factory=dict[str, str])
    """Globs for string arguments by name, such as `{'path': 'docs/*'}`; a missing or non-string argument doesn't match.

    An `'allow'` rule that globs a shell tool's command argument never matches: use `command` to allow commands.

    Each value is matched as given and as a normalized path: an `'allow'` rule needs both to match, others either,
    and also match the end of a longer path, so `'secrets/*'` covers `/project/secrets/key`.
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


def _chars(command: str) -> Iterator[tuple[str, str]]:
    """Each character of `command` other than quoting syntax, with how it is quoted: `''`, a quote, or a backslash."""
    quote = ''
    escaped = comment = False
    previous = ' '
    for char in command:
        if comment:
            comment = char != '\n'
        elif escaped:
            escaped = False
            yield char, '\\'
        elif char == '\\' and quote != "'":
            escaped = True
        elif char == quote:
            quote = ''
        elif not quote and char in '\'"':
            quote = char
        elif not quote and char == '#' and (previous.isspace() or previous in _OPERATOR_CHARS):
            comment = True  # the shell ignores the rest of the line, quotes included
        else:
            yield char, quote
        previous = char


def _quotes_an_operator(command: str) -> bool:
    """Whether `command` quotes or escapes an operator character, which `shlex` would not tell apart from a real one."""
    return any(char in _OPERATOR_CHARS and how for char, how in _chars(command))


def _expands(command: str) -> bool:
    """Whether the shell expands part of `command`, which could hide a rule's words from a scan of its text.

    That is `$` or a backtick outside single quotes, an unquoted `{` (brace expansion), or a line continuation.
    """
    return any(
        (char in '$`' and how in ('', '"')) or (char == '{' and not how) or (char == '\n' and how == '\\')
        for char, how in _chars(command)
    )


def _tokens(command: str) -> list[str] | None:
    """`command`'s words and operators, or `None` when it has syntax the lexer cannot follow."""
    if any(marker in command for marker in _UNSPLITTABLE) or _quotes_an_operator(command) or _expands(command):
        return None
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ''
    try:
        return list(lexer)
    except ValueError:
        return None


def _is_word(token: str) -> bool:
    """Whether a token is a word rather than an operator; an empty quoted argument lexes as an empty word."""
    return not token or not set(token) <= _OPERATOR_CHARS


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
        elif _is_word(token):
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


def _matches_word(word: str, expected: str) -> bool:
    """Whether `word` is `expected`, or a pattern with a literal character that could expand into it, such as `r[m]`.

    A bare wildcard such as `*` stands for any file, not for the program or flag a rule names.
    """
    return word == expected or (
        _GLOB.search(word) is not None and _LITERAL.search(word) is not None and fnmatchcase(expected, word)
    )


def _contains(words: Sequence[str], rule: Sequence[str]) -> bool:
    """Whether `rule`'s first word is in `words`, followed by the rest in order, other words allowed in between.

    The first word also matches a path ending in it. A short-option cluster such as `-rf` matches when the options
    after the first word include its letters in any grouping or case, as in `-fr`, `-Rf`, or `-r -f`.
    """
    head, rest = rule[0], rule[1:]
    start = next(
        (
            i
            for i, word in enumerate(words)
            if _matches_word(word, head) or _matches_word(posixpath.basename(word), head)
        ),
        None,
    )
    if start is None:
        return False
    following = words[start + 1 :]
    letters = {letter for word in following if _OPTIONS.fullmatch(word) for letter in word[1:].lower()}
    remaining = iter(following)
    return all(
        set(expected[1:].lower()) <= letters or any(_matches_word(word, expected) for word in following)
        if _OPTIONS.fullmatch(expected)
        else any(_matches_word(word, expected) for word in remaining)
        for expected in rest
    )


class _CommandRule(NamedTuple):
    rule: PermissionRule[Any]
    words: list[str]
    loose: list[str]


def _as_data(rule: PermissionRule[Any] | None, *, strict: bool) -> PermissionRule[Any] | None:
    """`rule` for a match inside quoted text, which may only be data such as a commit message: a deny asks instead.

    A `strict` deny stays a deny, so that no approval can run what it refuses.
    """
    return replace(rule, verdict='ask') if rule is not None and rule.verdict == 'deny' and not strict else rule


def _segment_rule(rules: Sequence[_CommandRule], segment: _Segment, *, strict: bool) -> PermissionRule[Any] | None:
    """The strictest rule matching one simple command: allow rules by prefix, others in its words or a quoted one."""
    # A number before a redirection may be its file descriptor or an argument, so either reading can match.
    readings = (segment.words, [word for index, word in enumerate(segment.words) if index not in segment.descriptors])
    quoted = [_loose([word]) for word in segment.words if _LOOSE_SPLIT.search(word)]
    matched = (_match(parsed, segment, readings, quoted, strict=strict) for parsed in rules)
    return _strictest(rule for rule in matched if rule is not None)


def _match(
    parsed: _CommandRule, segment: _Segment, readings: Iterable[list[str]], quoted: Iterable[list[str]], *, strict: bool
) -> PermissionRule[Any] | None:
    """What `parsed` makes of one simple command: its rule, the rule as for data, or `None`."""
    if parsed.rule.verdict == 'allow':
        allowed = not segment.redirected and segment.words[: len(parsed.words)] == parsed.words
        return parsed.rule if allowed else None
    if any(_contains(reading, parsed.words) for reading in readings):
        return parsed.rule
    if any(_contains(parts, parsed.loose) for parts in quoted):
        return _as_data(parsed.rule, strict=strict)
    return None


def _strictest(rules: Iterable[PermissionRule[Any]]) -> PermissionRule[Any] | None:
    found: PermissionRule[Any] | None = None
    for rule in rules:
        if found is None or _SEVERITY[rule.verdict] > _SEVERITY[found.verdict]:
            found = rule
    return found


_UNSPLITTABLE_RULE = PermissionRule[Any](
    'ask', reason='Permission rules cannot split this command into simple commands, so it asks first.'
)


def _restricts(rules: Iterable[PermissionRule[Any]]) -> bool:
    return any(rule.verdict != 'allow' for rule in rules)


def _unquoted(command: str) -> str:
    """`command` with quoted spaces and punctuation as placeholders, so quoted text stays one word."""
    return ''.join('\0' if how and _LOOSE_SPLIT.match(char) else char for char, how in _chars(command))


def _globbed_name(segments: Iterable[_Segment]) -> bool:
    """Whether a command name is a pattern, such as `/bin/r[m]`, which the shell expands into the program run."""
    return any(_GLOB.search(segment.words[0]) for segment in segments if segment.words)


def _unsplittable_rule(
    rules: Sequence[PermissionRule[Any]], command: str, *, expands: bool, strict: bool, posix: bool = True
) -> PermissionRule[Any] | None:
    """Match a command too complex to split word by word, quotes removed; no rule allows it.

    When the shell `expands` part of it, a rule's words could be hidden, so an ask or deny rule makes it ask.
    PowerShell names are case-insensitive, so without `posix` words are compared ignoring case.
    """
    if not _restricts(rules):
        return None
    fold = str if posix else str.lower
    restrictive = [(rule, _loose(_words(fold(rule.command or '')))) for rule in rules if rule.verdict != 'allow']

    def strictest_in(text: str) -> PermissionRule[Any] | None:
        words = _loose([fold(_REDIRECTION.sub(' ', text))])
        return _strictest(rule for rule, rule_words in restrictive if _contains(words, rule_words))

    everywhere = _as_data(strictest_in(re.sub(r'[\'"\\]', '', command)), strict=strict)
    found = _strictest(rule for rule in (strictest_in(_unquoted(command)), everywhere) if rule is not None)
    return found or (_UNSPLITTABLE_RULE if expands else None)


def _command_rule(
    rules: Sequence[PermissionRule[Any]], command: str, *, posix: bool, strict: bool = False
) -> PermissionRule[Any] | None:
    """The rule deciding a shell command: the strictest any simple command matches, `'allow'` only if all do.

    With `strict`, a deny matched only inside quoted text stays a deny instead of asking.
    """
    if len(command) > COMMAND_LIMIT:
        return _UNSPLITTABLE_RULE if _restricts(rules) else None
    segments = _segments(command) if posix else None
    if segments is None:
        return _unsplittable_rule(rules, command, expands=_expands(command), strict=strict, posix=posix)
    if _globbed_name(segments):
        return _unsplittable_rule(rules, command, expands=True, strict=strict)
    parsed = [_CommandRule(rule, words := _words(rule.command or ''), _loose(words)) for rule in rules]
    deciders = [_segment_rule(parsed, segment, strict=strict) for segment in segments]
    restrictive = _strictest(rule for rule in deciders if rule is not None and rule.verdict != 'allow')
    if restrictive is not None or None in deciders:
        return restrictive
    return deciders[0] if deciders else None


def _arg_matches(rule: PermissionRule[Any], value: object, glob: str, *, posix: bool) -> bool:
    """Whether a string argument matches `glob` as given and as a normalized path: both for `'allow'`, else either.

    For `'ask'` and `'deny'` the glob also matches the end of a longer path, such as an absolute one. On Windows
    paths are compared case-insensitively, with `/` for separators.
    """
    if not isinstance(value, str):
        return False
    if len(value) > _GLOB_LIMIT:
        return rule.verdict != 'allow'
    forms = [value, posixpath.normpath(value)]
    if not posix:
        forms = [form.replace('\\', '/').lower() for form in (value, ntpath.normpath(value))]
        glob = glob.lower()
    if rule.verdict == 'allow':
        return all(fnmatchcase(form, glob) for form in forms)
    return any(fnmatchcase(form, glob) or fnmatchcase(form, f'*/{glob}') for form in forms)


def decide(
    rules: Sequence[PermissionRule[Any]],
    *,
    command: str | None,
    command_arg: str | None,
    args: Mapping[str, Any],
    posix: bool,
    strict: bool = False,
) -> PermissionRule[Any] | None:
    """The rule deciding a call to a tool `rules` all apply to, or `None` when no rule does.

    With `strict`, as for managed rules, a deny matched only inside quoted text in `command` stays a deny.

    An allow rule that globs the shell command argument (`command_arg`) is skipped: only `command` splits a command.
    With `posix=False`, for a Windows workspace, a command is never split, since PowerShell quotes differently.
    """
    applicable = [
        rule
        for rule in rules
        if not (rule.verdict == 'allow' and command_arg in rule.args)
        and all(_arg_matches(rule, args.get(name), glob, posix=posix) for name, glob in rule.args.items())
    ]
    found = [rule for rule in applicable if rule.command is None]
    command_rules = [rule for rule in applicable if rule.command is not None]
    if command is not None and command_rules:
        decided = _command_rule(command_rules, command, posix=posix, strict=strict)
        if decided is not None:
            found.append(decided)
    return _strictest(found)


def command_matches(rules: Sequence[PermissionRule[Any]], command: str, *, posix: bool) -> bool:
    """Whether one of the ask `rules` matches `command`, not counting the ask for a command too complex to split."""
    found = _command_rule(rules, command, posix=posix) if rules else None
    return found is not None and found is not _UNSPLITTABLE_RULE


def command_segments(command: str) -> list[tuple[list[str], bool]] | None:
    """The simple commands in `command`, each with whether it redirects; `None` when it is too complex to split."""
    if len(command) > COMMAND_LIMIT:
        return None
    segments = _segments(command)
    return None if segments is None else [(segment.words, segment.redirected) for segment in segments]
