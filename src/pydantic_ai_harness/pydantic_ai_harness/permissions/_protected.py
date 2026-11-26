"""Find the protected paths a file write or shell command names, best effort."""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Sequence
from fnmatch import fnmatchcase

from pydantic_ai.workspaces import Workspace, WorkspaceError
from pydantic_ai_harness.permissions._rules import command_segments

_WORD_SPLIT = re.compile(r'[\s;&|()<>`=]+')
"""Where a command's text splits into words that may be paths: whitespace, operators, and `--option=value`."""

_QUOTING = re.compile(r'[\'"\\]')

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

_LENGTH_LIMIT = 4096
"""Longer words aren't globbed, which could take quadratic time: they count as protected."""


def matches(location: str, globs: Sequence[str], *, posix: bool) -> bool:
    """Whether `location` matches one of `globs`, or a relative glob matches its end.

    On Windows paths are compared case-insensitively, with `/` for separators.
    """
    if len(location) > _LENGTH_LIMIT:
        return True
    if not posix:
        location = location.replace('\\', '/').lower()
    for glob in globs:
        pattern = glob if posix else glob.replace('\\', '/').lower()
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


async def command_locations(workspace: Workspace, command: str, *, posix: bool) -> list[str]:
    """The paths `command`'s words could name, resolved against the working directory; none for a read-only command.

    Quotes are removed, `~` and environment variables are expanded as in this process, and every word counts,
    redirection targets and `--option=value` values included. A command made only of programs that read the
    files they name, such as `cat` or `grep`, without redirection, names none.
    """
    if posix and _only_reads(command):
        return []
    try:
        cwd: str | None = await workspace.working_dir()
    except WorkspaceError:
        cwd = None
    paths = workspace.paths
    expanded = [os.path.expanduser(os.path.expandvars(word)) for word in _WORD_SPLIT.split(_QUOTING.sub('', command))]
    return [paths.normpath(word if cwd is None else paths.join(cwd, word)) for word in expanded if word]


def _only_reads(command: str) -> bool:
    segments = command_segments(command)
    return bool(segments) and all(
        not redirected and words and posixpath.basename(words[0]) in _READING_PROGRAMS
        for words, redirected in segments or ()
    )
