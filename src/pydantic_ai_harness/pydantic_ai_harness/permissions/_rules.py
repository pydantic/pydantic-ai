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
from typing import Any, Generic, Literal, NamedTuple, Required, cast

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
_COMMAND_LIMIT = 65536
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


def _as_data(rule: PermissionRule[Any] | None) -> PermissionRule[Any] | None:
    """`rule` for a match inside quoted text, which may only be data such as a commit message: a deny asks instead."""
    return replace(rule, verdict='ask') if rule is not None and rule.verdict == 'deny' else rule


def _segment_rule(rules: Sequence[_CommandRule], segment: _Segment) -> PermissionRule[Any] | None:
    """The strictest rule matching one simple command: allow rules by prefix, others in its words or a quoted one."""
    # A number before a redirection may be its file descriptor or an argument, so either reading can match.
    readings = (segment.words, [word for index, word in enumerate(segment.words) if index not in segment.descriptors])
    quoted = [_loose([word]) for word in segment.words if _LOOSE_SPLIT.search(word)]
    matched = (_match(parsed, segment, readings, quoted) for parsed in rules)
    return _strictest(rule for rule in matched if rule is not None)


def _match(
    parsed: _CommandRule, segment: _Segment, readings: Iterable[list[str]], quoted: Iterable[list[str]]
) -> PermissionRule[Any] | None:
    """What `parsed` makes of one simple command: its rule, the rule as for data, or `None`."""
    if parsed.rule.verdict == 'allow':
        allowed = not segment.redirected and segment.words[: len(parsed.words)] == parsed.words
        return parsed.rule if allowed else None
    if any(_contains(reading, parsed.words) for reading in readings):
        return parsed.rule
    if any(_contains(parts, parsed.loose) for parts in quoted):
        return _as_data(parsed.rule)
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
    rules: Sequence[PermissionRule[Any]], command: str, *, expands: bool, posix: bool = True
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

    everywhere = _as_data(strictest_in(re.sub(r'[\'"\\]', '', command)))
    found = _strictest(rule for rule in (strictest_in(_unquoted(command)), everywhere) if rule is not None)
    return found or (_UNSPLITTABLE_RULE if expands else None)


def _command_rule(rules: Sequence[PermissionRule[Any]], command: str, *, posix: bool) -> PermissionRule[Any] | None:
    """The rule deciding a shell command: the strictest any simple command matches, `'allow'` only if all do."""
    if len(command) > _COMMAND_LIMIT:
        return _UNSPLITTABLE_RULE if _restricts(rules) else None
    segments = _segments(command) if posix else None
    if segments is None:
        return _unsplittable_rule(rules, command, expands=_expands(command), posix=posix)
    if _globbed_name(segments):
        return _unsplittable_rule(rules, command, expands=True)
    parsed = [_CommandRule(rule, words := _words(rule.command or ''), _loose(words)) for rule in rules]
    deciders = [_segment_rule(parsed, segment) for segment in segments]
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
) -> PermissionRule[Any] | None:
    """The rule deciding a call to a tool `rules` all apply to, or `None` when no rule does.

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
        decided = _command_rule(command_rules, command, posix=posix)
        if decided is not None:
            found.append(decided)
    return _strictest(found)


_SHELL_NAMES = frozenset({'Bash', 'Shell'})
_CLAUDE_CODE_TOOLS: dict[str, str] = {
    'Read': '`read_file`',
    'Edit': '`edit_file`',
    'MultiEdit': '`edit_file`',
    'Write': '`write_file`',
    'Glob': '`find_files`',
    'Grep': '`grep`',
    'WebFetch': '`web_fetch`',
    'WebSearch': '`web_search`',
}
"""Claude Code tool names a rule string can't use, with the tools to name instead."""
_SHELL_TOOL_NAMES = frozenset({'bash', 'shell', 'run_command', 'start_command'})
"""Lowercase names of shell tools, whose calls have no `path` for a pattern to match."""


def _split_rule(rule: str) -> tuple[str, str | None]:
    """A rule string's tool name and the pattern in its `()`, if any; raises `UserError` when it is malformed.

    Split by hand, not by regular expression: a rule string comes from a settings file, and backtracking on one
    would take time quadratic in its length.
    """
    name, paren, rest = rule.strip().partition('(')
    inner = rest[:-1] if paren and rest.endswith(')') else None
    if (
        not name
        or any(char.isspace() or char == ')' for char in name)
        or (paren and (inner is None or not inner.strip()))
    ):
        raise UserError(f'Permission rule {rule!r} must be a tool name, optionally followed by a pattern in `()`.')
    return name, inner


def parse_permission_rule(rule: str, verdict: PermissionVerdict, *, reason: str | None = None) -> PermissionRule[Any]:
    """A `PermissionRule` from a rule string in the style of Claude Code's permission rules.

    - `'Bash(git push:*)'` or `'Shell(git push:*)'`: shell commands that start with `git push`, as a
      `PermissionRule.command`, on any tool that declares a shell command argument. `'Bash'` alone matches every
      shell command.
    - `'Bash(git push *)'`: the same as `'Bash(git push:*)'`, and `'Bash(*)'` the same as `'Bash'`. A `*` anywhere
      else in a command raises `UserError`: a command rule matches words, not wildcards.
    - `'Bash(git push)'`, without `:*`: the same rule for `'ask'` and `'deny'`. For `'allow'` it raises
      `UserError`, since Claude Code reads it as one exact command and a command rule's allow matches every
      command that starts with its words.
    - `'write_file'`: every call to the tool of that name.
    - `'write_file(docs/**)'`: calls whose `path` argument matches the glob. A `*` matches across `/` here, so
      `docs/*` and `docs/**` mean the same thing. A leading `./` is dropped, and so is a leading `**/` for `'ask'`
      and `'deny'`, which match the end of a longer path anyway. A path starting with `/` or `~` raises
      `UserError`: Claude Code reads those against the project or home folder, while a glob here is matched as
      written.
    - `'web_fetch(url=https://example.com/*)'`: calls whose `url` argument matches the glob.

    A pattern ending in `:*` on any other name raises `UserError`, and so does a `path` pattern on a shell tool's
    lowercase name, such as `'bash(rm -rf)'` or `'run_command(rm -rf)'`, whose calls have no `path`. So does
    a wildcard in a tool name, such as `'mcp__github__*'`, or Claude Code's rule for a whole MCP server, such as
    `'mcp__github'`: a rule names tools exactly.

    The text between the first `(` and the last `)` is the pattern, so a command may hold parentheses. Claude Code's
    own tool names, such as `Read` or `Edit`, raise `UserError` naming the tool to use: no tool here has them, so
    a rule naming one would match nothing.
    """
    _check_verdict(verdict, f'Permission rule {rule!r}')
    name, inner = _split_rule(rule)
    if (instead := _CLAUDE_CODE_TOOLS.get(name)) is not None:
        raise UserError(f'Permission rule {rule!r} uses the Claude Code tool name {name!r}: name {instead} instead.')
    if any(char in name for char in '*?['):
        raise UserError(f'Permission rule {rule!r} has a wildcard in its tool name: name each tool in full.')
    if name.startswith('mcp__') and '__' not in name.removeprefix('mcp__'):
        # Claude Code's rule for every tool of one MCP server; tool names here carry no server by convention.
        raise UserError(
            f'Permission rule {rule!r} names an MCP server the way Claude Code does, which matches no tool here: '
            'name each tool as the agent sees it.'
        )
    inner = None if inner is None else inner.strip()
    if name in _SHELL_NAMES:
        return _shell_rule(rule, name, inner, verdict, reason=reason)
    if inner is None:
        return PermissionRule(verdict, tools=[name], reason=reason)
    arg, equals, glob = inner.partition('=')
    arg, glob = arg.strip(), glob.strip()
    arg, glob = (arg, glob) if equals and arg.isidentifier() and glob else ('path', inner)
    if arg == 'path' and inner.endswith(':*'):
        # Otherwise it would be a `path` glob that a shell tool's call never matches, so a deny would refuse nothing.
        raise UserError(f'Permission rule {rule!r} uses `:*`, which only `Bash(...)` and `Shell(...)` take.')
    if arg == 'path' and name.lower() in _SHELL_TOOL_NAMES:
        # A shell tool's call has no `path`, so the rule would match nothing.
        raise UserError(
            f'Permission rule {rule!r} matches a path, which a shell command has none of: '
            f'write `Bash({inner})` for a command rule.'
        )
    return PermissionRule(
        verdict, tools=[name], args={arg: _path_glob(rule, glob, verdict) if arg == 'path' else glob}, reason=reason
    )


def _shell_rule(
    rule: str, name: str, inner: str | None, verdict: PermissionVerdict, *, reason: str | None
) -> PermissionRule[Any]:
    """The rule for a `Bash(...)` or `Shell(...)` string, whose pattern is `inner`."""
    if inner is None or inner == '*':
        return PermissionRule(verdict, tools={'code_arg_language': 'shell'}, reason=reason)
    prefix = inner.endswith((':*', ' *'))
    command = inner[:-2].rstrip() if prefix else inner
    if '*' in command:
        # A command rule matches words in order, so the wildcard would be a literal word and the rule match nothing.
        raise UserError(f'Permission rule {rule!r} has a `*` that is not at its end: a command rule matches words.')
    if verdict == 'allow' and not prefix:
        raise UserError(
            f'Permission rule {rule!r} would allow every command starting with {inner!r}, not just that one: '
            f'write `{name}({inner}:*)` to allow them.'
        )
    return PermissionRule(verdict, command=command, reason=reason)


def _path_glob(rule: str, glob: str, verdict: PermissionVerdict) -> str:
    """A rule string's `path` glob as a `PermissionRule` matches it: without a leading `./`.

    An ask or deny glob also matches the end of a longer path, so a leading `**/` is dropped there: kept, it would
    miss a match at the top, such as `**/.env` missing `.env`. A glob ending in `/` names a directory and everything
    under it, as in Claude Code, so `secrets/` becomes `secrets/**`, and `./` becomes `**`.
    """
    if glob.startswith(('/', '~')):
        raise UserError(
            f'Permission rule {rule!r} starts its path with {glob[0]!r}: write it relative, such as `secrets/**`, '
            'which also matches the end of a longer path.'
        )
    while True:
        if glob.startswith('./'):
            glob = glob[2:]
        elif verdict != 'allow' and glob.startswith('**/'):
            glob = glob[3:]
        else:
            break
    # A path value is normalized without a trailing `/`, so `secrets/` as written would match nothing.
    return f'{glob}**' if not glob or glob.endswith('/') else glob


def _check_verdict(verdict: str, where: str) -> None:
    """Raise `UserError` unless `verdict` is `'allow'`, `'ask'` or `'deny'`.

    A typo such as `'denny'` would otherwise build a rule that neither allows nor denies, so it would ask.
    """
    if verdict not in _SEVERITY:
        raise UserError(f'{where} has the verdict {verdict!r}: use `allow`, `ask` or `deny`.')


def parse_permission_rules(rules: Mapping[PermissionVerdict, Sequence[str]]) -> list[PermissionRule[Any]]:
    """`PermissionRule`s from rule strings by verdict, such as `{'allow': ['Bash(git status:*)'], 'deny': [...]}`.

    Each string is parsed by `parse_permission_rule`. The order doesn't matter: `'deny'` beats `'ask'`, which beats
    `'allow'`, whatever the order. A key other than a verdict, or anything but a list of strings under one, such
    as a single string, raises `UserError`, since settings files that people write by hand aren't type-checked.
    """
    for verdict, strings in rules.items():
        _check_verdict(verdict, 'A permission rules mapping')
        _check_strings(verdict, strings)
    return [parse_permission_rule(rule, verdict) for verdict, strings in rules.items() for rule in strings]


def _check_strings(verdict: str, strings: object) -> None:
    """Raise `UserError` unless `strings` is a list or tuple of strings."""
    if isinstance(strings, (list, tuple)) and all(isinstance(rule, str) for rule in cast(Sequence[object], strings)):
        return
    raise UserError(f'Permission rules for {verdict!r} must be a list of strings: {strings!r}.')
