"""Find the protected paths a file write or shell command names, best effort."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence
from fnmatch import fnmatchcase

from pydantic_ai.workspaces import Workspace, WorkspaceError
from pydantic_ai_harness.permissions._rules import command_segments

_OPERATORS = ';&|()<>'
"""Characters the shell reads as operators outside quotes, which split words like whitespace does."""

_WORD_PARTS = re.compile(r'[`=]+')
"""Where a word splits further into words that may be paths: a backtick substitution and `--option=value`."""

_FALLBACK_SPLIT = re.compile(r'[\s;&|()<>`=]+')
"""How a command whose quotes don't balance is split, both as written and with its quotes removed."""

_QUOTES = re.compile(r'[\'"]')

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


def matches(location: str, globs: Sequence[str], *, posix: bool) -> bool:
    """Whether `location` matches one of `globs`, or a relative glob matches its end.

    Paths are compared case-insensitively, as macOS and Windows file systems do by default, and on Windows with
    `/` for separators. This only adds asks: on a case-sensitive file system, `.PAX/x` asks like `.pax/x`.
    """
    if len(location) > _LENGTH_LIMIT:
        return True
    location = (location if posix else location.replace('\\', '/')).lower()
    for glob in globs:
        pattern = (glob if posix else glob.replace('\\', '/')).lower()
        if fnmatchcase(location, pattern) or fnmatchcase(location, f'*/{pattern}'):
            return True
    return False


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
    Windows backslashes are separators. `~` and environment variables are expanded as in this process, and every
    word counts, redirection targets and `--option=value` values included. A command made only of programs that
    read the files they name, such as `cat` or `grep`, run by name rather than path and without redirection, names
    none.

    With `expand=False`, as inside a durable run, where this process's environment may differ on replay and is not
    the workspace's, nothing is expanded: `None` when a word would need it, as the paths it names can't be known.
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
    if expanded is None:
        return None
    return [paths.normpath(word if cwd is None else paths.join(cwd, word)) for word in expanded]


def _expanded(words: list[str], *, posix: bool, expand: bool) -> list[str] | None:
    """`words` with `~` and variables expanded as in this process; unexpanded, or `None` if one needs it, without `expand`."""
    if expand:
        return [os.path.expanduser(os.path.expandvars(word)) for word in words]
    if any(word.startswith('~') or '$' in word or (not posix and '%' in word) for word in words):
        return None
    return words


def _words(command: str, *, posix: bool, depth: int = 0) -> list[str] | None:
    """`command`'s words, and their parts around a backtick or `=`, without operators.

    The words of a script a shell runs with `-c`, or `eval` runs, count too, as the command runs them; `None` when
    such scripts nest deeper than `_SCRIPT_DEPTH_LIMIT`, as the paths they name aren't looked for.
    """
    tokens = [token for token in _tokens(command, posix=posix) if token and not set(token) <= set(_OPERATORS)]
    words: list[str] = []
    for token in tokens:
        words += [token, *_WORD_PARTS.split(token)] if _WORD_PARTS.search(token) else [token]
    for script in _scripts(tokens):
        inner = None if depth >= _SCRIPT_DEPTH_LIMIT else _words(script, posix=posix, depth=depth + 1)
        if inner is None:
            return None
        words += inner
    return list(dict.fromkeys(word for word in words if word))


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
    """The scripts `words` hand a shell's `-c`, such as `sh -c '...'` or `bash -lc '...'`, or `eval`.

    A shell or `eval` counts wherever it is among the words, so behind a prefix such as `env` or `sudo` too.
    """
    scripts: list[str] = []
    for i, word in enumerate(words):
        program = word.replace('\\', '/').rsplit('/', 1)[-1]
        if program == 'eval':
            scripts.append(' '.join(words[i + 1 :]))
        elif program in _SHELLS:
            flag = next((j for j in range(i + 1, len(words)) if _SCRIPT_FLAG.fullmatch(words[j])), len(words))
            scripts += words[flag + 1 : flag + 2]
    return scripts


def _only_reads(command: str) -> bool:
    """Whether `command` only runs reading programs by bare name, as a path such as `./cat` may be another program."""
    segments = command_segments(command)
    return bool(segments) and all(
        not redirected and words and words[0] in _READING_PROGRAMS for words, redirected in segments or ()
    )
