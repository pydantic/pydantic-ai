"""Find the protected paths a file write or shell command names, best effort."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence
from fnmatch import fnmatchcase
from typing import NamedTuple

from pydantic_ai.workspaces import Workspace, WorkspaceError, WorkspacePaths
from pydantic_ai_harness.permissions._rules import command_segments

_OPERATORS = ';&|()<>'
"""Characters the shell reads as operators outside quotes, which split words like whitespace does."""

_WORD_PARTS = re.compile(r'[`=]+')
"""Where a word splits further into words that may be paths: a backtick substitution and `--option=value`."""

_FALLBACK_SPLIT = re.compile(r'[\s;&|()<>`=]+')
"""How a command whose quotes don't balance is split, both as written and with its quotes removed."""

_QUOTES = re.compile(r'[\'"]')

_SHORT_OPTIONS = re.compile(r'-[A-Za-z]+')
"""A word's leading short options, as `-vt` in `-vt.git/hooks`, after any of whose letters a value may follow."""

_READING_PROGRAMS = frozenset(
    {
        'cat',
        'cmp',
        'diff',
        'file',
        'grep',
        'head',
        'ls',
        'md5sum',
        'readlink',
        'realpath',
        'sha256sum',
        'stat',
        'tail',
        'wc',
    }
)
"""Programs that only read the files they name, so a command made only of them, without redirection, changes none."""

_SHELLS = frozenset({'bash', 'dash', 'fish', 'ksh', 'sh', 'zsh'})
"""Shells whose `-c` runs the script that follows it."""

_SCRIPT_FLAG = re.compile(r'-[a-zA-Z]*c[a-zA-Z]*')
"""A shell option word that includes `-c`, as `-c`, `-lc` and `-ec` do."""

_SCRIPT_DEPTH_LIMIT = 3
"""How deep scripts run by a shell's `-c` or `eval` are looked into; a command nesting them deeper counts as protected."""

_LENGTH_LIMIT = 4096
"""Longer words aren't globbed, which could take quadratic time: they count as protected."""

_BRACE_LIMIT = 64
"""How many words a word's brace expansion is followed to; a word expanding to more counts as protected."""


class _Char(NamedTuple):
    """One part of a pattern: a character, or a wildcard for any one or, with `star`, any number of them."""

    char: str | None
    """The character this part matches, or `None` for a wildcard."""
    slash: bool = True
    """Whether a wildcard matches `/`, as an `fnmatch` one does and a shell's, which matches within a name, doesn't."""
    star: bool = False


def matches(location: str, globs: Sequence[str], *, posix: bool, pattern: bool = False) -> bool:
    """Whether `location` matches one of `globs`, or a relative glob matches its end.

    Paths are compared case-insensitively, as macOS and Windows file systems do by default, and on Windows with
    `/` for separators. This only adds asks: on a case-sensitive file system, `.PAX/x` asks like `.pax/x`.

    With `pattern`, as for a command's word, `location`'s `*`, `?` and bracket expressions are wildcards the shell
    expands within a name: it matches when a path it could expand to does, as `.git/hook?/x` does `.git/hooks/*`.
    """
    if len(location) > _LENGTH_LIMIT:
        return True
    location = (location if posix else location.replace('\\', '/')).lower()
    wildcards = _pattern(location, slash=False) if pattern else []
    if all(part.char is not None for part in wildcards):
        wildcards = []
    for glob in globs:
        glob = (glob if posix else glob.replace('\\', '/')).lower()
        for candidate in (glob, f'*/{glob}'):
            if wildcards and _overlaps(wildcards, _pattern(candidate, slash=True)):
                return True
            if not wildcards and fnmatchcase(location, candidate):
                return True
    return False


def _pattern(text: str, *, slash: bool) -> list[_Char]:
    """`text`'s parts as a pattern, its bracket expressions as wildcards for any one character, which only asks more.

    A `[` without a closing `]`, or with a `/` before it where the shell matches within a name, is a character.
    """
    parts: list[_Char] = []
    i = 0
    while i < len(text):
        char = text[i]
        end = _bracket_end(text, i) if char == '[' else None
        if char in '*?':
            parts.append(_Char(None, slash=slash, star=char == '*'))
        elif end is not None and (slash or '/' not in text[i:end]):
            parts.append(_Char(None, slash=slash))
            i = end
        else:
            parts.append(_Char(char))
        i += 1
    return parts


def _bracket_end(text: str, start: int) -> int | None:
    """Where the bracket expression opening at `start` closes, as in `[!]a]`, if it does."""
    i = start + 1
    if text[i : i + 1] in ('!', '^'):
        i += 1
    if text[i : i + 1] == ']':
        i += 1
    end = text.find(']', i)
    return None if end == -1 else end


def _same_char(a: _Char, b: _Char) -> bool:
    """Whether one character matches both `a` and `b`."""
    if a.char is not None and b.char is not None:
        return a.char == b.char
    if a.char is not None:
        return a.char != '/' or b.slash
    if b.char is not None:
        return b.char != '/' or a.slash
    return True


def _overlaps(a: list[_Char], b: list[_Char]) -> bool:
    """Whether some text matches both patterns, found by walking them side by side."""
    seen: set[tuple[int, int]] = set()
    todo = [(0, 0)]
    while todo:
        state = todo.pop()
        if state in seen:
            continue
        seen.add(state)
        if state == (len(a), len(b)):
            return True
        todo += _steps(a, b, *state)
    return False


def _steps(a: list[_Char], b: list[_Char], i: int, j: int) -> list[tuple[int, int]]:
    """Where walking `a` from `i` and `b` from `j` side by side goes next: past a star, or past a character both match."""
    steps: list[tuple[int, int]] = []
    if i < len(a) and a[i].star:
        steps.append((i + 1, j))
    if j < len(b) and b[j].star:
        steps.append((i, j + 1))
    if i < len(a) and j < len(b) and _same_char(a[i], b[j]):
        steps.append((i if a[i].star else i + 1, j if b[j].star else j + 1))
    return steps


async def write_locations(workspace: Workspace, path: str) -> list[str]:
    """Where a file write to `path` lands: as written, resolved against the working directory, and where it leads."""
    try:
        written = await workspace.resolve(path)
    except WorkspaceError:
        return [workspace.paths.normpath(path)]
    try:
        return [written, await workspace.realpath(written)]
    except (WorkspaceError, OSError):
        return [written]


async def command_locations(
    workspace: Workspace, command: str, *, posix: bool, expand: bool = True
) -> list[str] | None:
    """The paths `command`'s words could name, resolved against the working directory; none for a read-only command.

    Words are split as the shell would: quotes and, on POSIX, backslash escapes keep spaces in a word, and on
    Windows backslashes are separators. `~` and environment variables are expanded as in this process, and on POSIX
    brace expansions, as `a{b,c}`, too. Every word counts, redirection targets, `--option=value` values and what
    follows each letter of a short option such as `-o.pax/x` included. The paths may be patterns the shell expands,
    to match with `matches(..., pattern=True)`. A command made only of programs that read the files they name, such
    as `cat` or `grep`, run by name rather than path and without redirection, names none.

    `None` when the paths can't be known, which counts as protected: with a word whose braces expand to more than
    `_BRACE_LIMIT` words, or a pattern that could name `.` or `..`, as `.*` can, which moves where the rest of it
    leads. With `expand=False`, as inside a durable run, where this process's environment may differ on replay and
    is not the workspace's, nothing is expanded, and `None` too when a word would need it.
    """
    if posix and _only_reads(command):
        return []
    try:
        cwd: str | None = await workspace.working_dir()
    except WorkspaceError:
        cwd = None
    paths = workspace.paths
    words = _words(command, posix=posix)
    if words is None:
        return None
    expanded = _expanded(words, posix=posix, expand=expand)
    return None if expanded is None else _locations(expanded, paths, cwd, posix=posix)


def _locations(words: list[str], paths: WorkspacePaths, cwd: str | None, *, posix: bool) -> list[str] | None:
    """Where `words` lead from `cwd`, their braces expanded on POSIX; `None` when that can't be known."""
    locations: list[str] = []
    for word in words:
        braced = _braces(word) if posix else [word]
        if braced is None:
            return None
        for location in braced:
            location = location if cwd is None else paths.join(cwd, location)
            if _names_a_dot(location, posix=posix):
                return None
            locations.append(paths.normpath(location))
    return locations


def _braces(word: str) -> list[str] | None:
    """The words bash's brace expansion makes of `word`, as `a{b,c}` makes `ab` and `ac`; `None` past `_BRACE_LIMIT`.

    A sequence such as `{1..3}` becomes `*`, a pattern that matches its values.
    """
    if len(word) > _LENGTH_LIMIT:
        # `matches` counts it as protected as it is.
        return [word]
    pending = [word]
    done: list[str] = []
    while pending:
        current = pending.pop()
        group = _brace_group(current)
        if group is None:
            done.append(current)
        else:
            start, end, alternatives = group
            pending += [current[:start] + alternative + current[end + 1 :] for alternative in alternatives]
        if len(pending) + len(done) > _BRACE_LIMIT:
            return None
    return done


def _brace_group(word: str) -> tuple[int, int, list[str]] | None:
    """The first brace group `word` expands: where it opens and closes, and the words it stands for.

    A group expands when it has a comma outside the groups nested in it, or holds a sequence; others, like `{}`, are
    kept as they are, as is a `{` that doesn't close.
    """
    opened: list[int] = []
    closes: dict[int, int] = {}
    commas: dict[int, list[int]] = {}
    for i, char in enumerate(word):
        if char == '{':
            opened.append(i)
        elif char == '}' and opened:
            closes[opened.pop()] = i
        elif char == ',' and opened:
            commas.setdefault(opened[-1], []).append(i)
    for start in sorted(closes):
        end = closes[start]
        if start in commas:
            bounds = [start, *commas[start], end]
            return start, end, [word[a + 1 : b] for a, b in zip(bounds, bounds[1:])]
        if _is_sequence(word[start + 1 : end]):
            return start, end, ['*']
    return None


def _is_sequence(inner: str) -> bool:
    """Whether `inner` is the inside of a brace sequence, such as `{1..3}` or `{a..e..2}`, whose values hold no `/`.

    Read without a regular expression, which would backtrack over a long run of digits.
    """
    parts = inner.split('..')
    if len(parts) not in (2, 3) or (len(parts) == 3 and not _is_integer(parts[2])):
        return False
    first, last = parts[0], parts[1]
    letters = all(len(end) == 1 and end.isascii() and end.isalpha() for end in (first, last))
    return letters or (_is_integer(first) and _is_integer(last))


def _is_integer(text: str) -> bool:
    """Whether `text` is an optionally negative run of ASCII digits."""
    digits = text.removeprefix('-')
    return digits.isascii() and digits.isdecimal()


def _names_a_dot(location: str, *, posix: bool) -> bool:
    """Whether a name in `location` is a pattern that could expand to `.` or `..`, as `.*` can in some shells.

    Shells match a leading `.` only explicitly, and a bracket expression may count, so only such patterns can.
    """
    for name in location.split('/') if posix else re.split(r'[/\\]', location):
        parts = _pattern(name, slash=False)
        if (
            name.startswith(('.', '['))
            and any(part.char is None for part in parts)
            and (_overlaps(parts, _pattern('.', slash=False)) or _overlaps(parts, _pattern('..', slash=False)))
        ):
            return True
    return False


def _expanded(words: list[str], *, posix: bool, expand: bool) -> list[str] | None:
    """`words` with `~` and variables expanded as in this process; unexpanded, or `None` if one needs it, without `expand`."""
    if expand:
        return [os.path.expanduser(os.path.expandvars(word)) for word in words]
    if any(word.startswith('~') or '$' in word or (not posix and '%' in word) for word in words):
        return None
    return words


def _words(command: str, *, posix: bool, depth: int = 0) -> list[str] | None:
    """`command`'s words, their parts around a backtick or `=`, and what follows a short option's letters.

    The words of a script a shell runs with `-c`, or `eval` runs, count too, as the command runs them; `None` when
    such scripts nest deeper than `_SCRIPT_DEPTH_LIMIT`, as the paths they name aren't looked for.
    """
    tokens = [token for token in _tokens(command, posix=posix) if token and not set(token) <= set(_OPERATORS)]
    words = [word for token in tokens for word in _word_parts(token)]
    for script in _scripts(tokens):
        inner = None if depth >= _SCRIPT_DEPTH_LIMIT else _words(script, posix=posix, depth=depth + 1)
        if inner is None:
            return None
        words += inner
    return list(dict.fromkeys(word for word in words if word))


def _word_parts(token: str) -> list[str]:
    """`token`, its parts around a backtick or `=`, and, for a short option, what follows each of its letters."""
    parts = [token, *_WORD_PARTS.split(token)] if _WORD_PARTS.search(token) else [token]
    if options := _SHORT_OPTIONS.match(token):
        # A short option's value may follow it in the same word, as in `-o.pax/x`, after any of its letters.
        parts += [token[i:] for i in range(2, options.end() + 1)]
    return parts


def _tokens(command: str, *, posix: bool) -> list[str]:
    """`command` split as the shell would, operators included."""
    try:
        lexer = shlex.shlex(command, posix=posix, punctuation_chars=_OPERATORS)
        lexer.whitespace_split = True
        # A `#` starts a comment only at the start of a word, and an unquoted one inside a word, as in `x#y`, is
        # part of it: with no comments, the words after a real one count too, which only asks more.
        lexer.commenters = ''
        return [token if posix else _QUOTES.sub('', token) for token in lexer]
    except ValueError:
        # Quotes that don't balance: every way of reading the words counts.
        return [*_FALLBACK_SPLIT.split(command), *_FALLBACK_SPLIT.split(_QUOTES.sub('', command))]


def _scripts(words: list[str]) -> list[str]:
    """The scripts `words` may hand a shell's `-c`, such as `sh -c '...'` or `bash -lc -- '...'`, or `eval`.

    A shell or `eval` counts wherever it is among the words, so behind a prefix such as `env` or `sudo` too. Rather
    than parse a shell's options, every word after its `-c` counts as a script, which only checks more words. Only the
    first `eval`, or shell with a `-c`, is looked at: the words after it hold any later one, whose script is checked
    inside them. A shell without a `-c`, as in `sh build.sh`, runs no script of its words, but a later `eval` still
    counts.
    """
    shells = True
    for i, word in enumerate(words):
        program = word.replace('\\', '/').rsplit('/', 1)[-1]
        if program == 'eval':
            return [' '.join(words[i + 1 :])]
        if shells and program in _SHELLS:
            flag = next((j for j in range(i + 1, len(words)) if _SCRIPT_FLAG.fullmatch(words[j])), None)
            if flag is not None:
                return list(dict.fromkeys(words[flag + 1 :]))
            # No `-c` follows, so no later shell has one either.
            shells = False
    return []


def _only_reads(command: str) -> bool:
    """Whether `command` only runs reading programs by bare name, as a path such as `./cat` may be another program."""
    segments = command_segments(command)
    return bool(segments) and all(
        not redirected and words and words[0] in _READING_PROGRAMS for words, redirected in segments or ()
    )
