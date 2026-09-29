"""Filesystem toolset providing file operations inside the run's workspace."""

from __future__ import annotations

import errno
import fnmatch
import functools
import hashlib
import logging
import os
import posixpath
import re
import shlex
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Concatenate, ParamSpec

import anyio
from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import (
    FileEntry,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceReadOnlyError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._events import event_ctx
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored, warn_argument_renamed
from pydantic_ai_harness._workspace import raise_tool_failure, supports_commands, workspace_path
from pydantic_ai_harness.filesystem._changes import Change
from pydantic_ai_harness.filesystem._command_search import run_posix_search
from pydantic_ai_harness.filesystem._events import (
    MAX_DIFF_SOURCE_CHARS,
    DirectoryCreatedEvent,
    DirectoryListedEvent,
    FileReadEvent,
    FilesSearchedEvent,
    FileWrittenEvent,
    SearchKind,
)
from pydantic_ai_harness.filesystem._ripgrep import Record, RipgrepMissing, Unreadable, run_ripgrep

_P = ParamSpec('_P')

logger = logging.getLogger(__name__)

DEFAULT_TOOL_NAMES: tuple[str, ...] = (
    'read_file',
    'write_file',
    'edit_file',
    'list_directory',
    'search_files',
    'find_files',
    'create_directory',
    'file_info',
)
"""The tools `FileSystem` registers by default; none needs a command-capable workspace."""

RIPGREP_TOOL_NAMES: tuple[str, ...] = ('list_files', 'grep')
"""Opt-in tools backed by the `rg` executable, which respects `.gitignore` and skips hidden files."""

FILE_SYSTEM_TOOL_NAMES: tuple[str, ...] = (*DEFAULT_TOOL_NAMES, *RIPGREP_TOOL_NAMES)
"""Every tool `FileSystem` can register, in registration order."""

_MAX_SEARCH_FILE_BYTES = 10 << 20
"""Avoid remote full-file downloads for large files during Python-side content searches."""

_MAX_UNREADABLE_NAMED = 5
"""Unreadable files a search result names before it only counts the rest."""

_FILE_TYPE_NEEDS_RIPGREP = '`file_type` needs ripgrep, which this workspace cannot run; use `glob` instead.'

_MAX_MATCH_COLUMNS = 4096
"""Bytes of a matching or context line `grep` shows before ripgrep cuts it with an omission marker."""

READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info', *RIPGREP_TOOL_NAMES}
)
"""Names of filesystem tools that do not modify the workspace."""

_READLINK_TIMEOUT = 10.0
"""Deadline for the `readlink` probe `file_info` runs to report a symlink target."""

_MAX_WALK_DIRECTORIES = 10_000
"""Directories one `search_files` or `find_files` walk lists before it stops.

The workspace filesystem API follows symlinked directories and cannot say an entry is a
symlink, so a link back to an ancestor is walked again under a longer path; two such links
grow the walk exponentially. The walk has no identity to detect a revisit with, so these
caps are the guard.
"""

_MAX_WALK_ENTRIES = 100_000
"""Entries one walk collects before it stops, for the same reason as `_MAX_WALK_DIRECTORIES`."""

_WALK_CUT_NOTICE = (
    f'[... walk cut short after {_MAX_WALK_DIRECTORIES} directories or {_MAX_WALK_ENTRIES} entries; narrow the path]'
)


@dataclass
class Replacement:
    """One exact replacement: `old_text` must occur exactly once and is replaced by `new_text`."""

    _: KW_ONLY
    old_text: str
    new_text: str


# Errors that mean "the model asked for something the tool couldn't do" -- a
# missing file, a denied path, a stale edit. pyai only feeds `ModelRetry` back
# to the model; any other exception aborts the whole run. `_recoverable`
# converts these so the agent can correct itself and continue.
_RECOVERABLE_ERRORS = (PermissionError, FileNotFoundError, NotADirectoryError, IsADirectoryError, ValueError)

# The same idea one level down, for failures the workspace raises as a bare
# `OSError` with no dedicated subclass for `_RECOVERABLE_ERRORS` to name.
# Entries are explicit so other errors keep aborting the run; for example,
# retrying cannot fix `ENOSPC` or `EROFS`.
#
# Keyed by `OSError.errno`, which the stdlib types as `int | None`.
_RECOVERABLE_ERRNOS: dict[int | None, str] = {
    errno.ENAMETOOLONG: 'The path name is too long.',
    errno.ELOOP: 'The path resolves through a symlink loop.',
    errno.EILSEQ: 'The path name contains a byte sequence the filesystem cannot represent.',
}
_WINDOWS_ERROR_INVALID_NAME = 123

_OUTSIDE_WORKSPACE = '<outside-workspace>'
"""Shown instead of an absolute path that is not inside the workspace root."""

_NOT_A_PATH = '<not-a-path>'
"""Shown when an error's `filename` is not a path value at all."""

_ABSENT_HASH = '<absent>'
"""The guard for a write announced against a path with nothing at it.

No file hashes to a bracketed marker, so one that appears while the create is
announced fails the check before the write even when it is empty -- which the
empty file's own hash would have matched.
"""


class _EventLocation(TypedDict):
    """The `path` and `root_dir` fields shared by every filesystem event."""

    path: str
    root_dir: str


@dataclass
class _Bounds:
    """A workspace's containment boundary and working directory, resolved on its first file operation."""

    root: str
    """The boundary: the real (symlink-free) workspace path of `root_dir`."""
    cwd: str
    """The workspace's working directory, which relative paths resolve from; inside `root`."""
    checks_realpath: bool
    """Whether targets are resolved through symlinks before use.

    A boundary at `/` contains everything, so only access patterns need the real path there.
    """
    lacks_ripgrep: bool = False
    """Set once `rg` is found missing, so later searches skip the rg probe."""


@dataclass
class _PathLock:
    """The lock one file's writes and edits take turns on, and how many calls hold or await it."""

    lock: anyio.Lock = field(default_factory=anyio.Lock)
    users: int = 0


@dataclass
class _Scope:
    """The workspace a call acts on, with its `_Bounds`."""

    workspace: Workspace
    bounds: _Bounds
    given: WorkspaceBackend
    """The workspace as the call was given it, which may be a backend `workspace` wraps."""

    @property
    def root(self) -> str:
        return self.bounds.root

    @property
    def cwd(self) -> str:
        return self.bounds.cwd

    @property
    def checks_realpath(self) -> bool:
        return self.bounds.checks_realpath

    @property
    def lacks_ripgrep(self) -> bool:
        return self.bounds.lacks_ripgrep

    @lacks_ripgrep.setter
    def lacks_ripgrep(self, value: bool) -> None:
        self.bounds.lacks_ripgrep = value


def _fall_back_from_ripgrep(scope: _Scope) -> None:
    """Search this workspace without `rg` from now on, saying so once, since the results differ slightly."""
    if not scope.lacks_ripgrep:
        logger.debug(
            '`rg` is not on the PATH of the workspace at %s; its searches use the POSIX fallback, which also '
            'returns tracked files that .gitignore lists, skips nested .ignore files, and without git '
            'applies no ignore files.',
            scope.cwd,
        )
    scope.lacks_ripgrep = True


def _contains(root: str, path: str) -> bool:
    """Whether the normalized absolute `path` is `root` or below it, compared as text."""
    return path == root or path.startswith(root.rstrip('/') + '/')


def _is_hidden(relative: str) -> bool:
    """Whether a root-relative path has a dot-prefixed component."""
    return relative != '.' and any(part.startswith('.') for part in relative.split('/'))


def _explicit_hidden(pattern: str) -> bool:
    """Whether the caller explicitly names a dot-prefixed glob component."""
    return any(part.startswith('.') for part in pattern.split('/'))


def _sort_key(path: str) -> list[str]:
    """Order paths component by component, as sorting `Path` objects does."""
    return path.split('/')


def _model_safe_filename(filename: str | bytes, root: str) -> str:
    """Return the path relative to the workspace root.

    Paths not inside the root become `_OUTSIDE_WORKSPACE`; values that are
    not paths at all become `_NOT_A_PATH`.
    """
    try:
        raw = os.fsdecode(filename)
    except TypeError:
        return _NOT_A_PATH
    if not posixpath.isabs(raw):
        return raw
    path = posixpath.normpath(raw)
    if _contains(root, path):
        return posixpath.relpath(path, root)
    return _OUTSIDE_WORKSPACE


def _sanitize_recoverable_error(error: BaseException, root: str) -> str:
    """Render a recoverable error without exposing absolute paths outside the root.

    Errors without an OS-supplied `filename` keep their original message.
    OS errors keep `errno` and `strerror`, with the path rewritten relative
    to `root` (see `_model_safe_filename` for the fallback placeholders).
    """
    if not isinstance(error, OSError) or error.filename is None:
        return str(error)

    filename = _model_safe_filename(error.filename, root)
    return f'[Errno {error.errno}] {error.strerror}: {filename!r}'


def _recoverable(
    fn: Callable[Concatenate[FileSystemToolset, _Scope, _P], Awaitable[str]],
) -> Callable[Concatenate[FileSystemToolset, _Scope, _P], Awaitable[str]]:
    """Surface model-correctable tool errors as `ModelRetry`, and workspace refusals as `ToolFailed`."""

    @functools.wraps(fn)
    async def wrapper(self: FileSystemToolset, scope: _Scope, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, scope, *args, **kwargs)
        # Before the recoverable tuple and `OSError`: `WorkspaceReadOnlyError` is a
        # `PermissionError` and `WorkspaceTimeoutError` a `TimeoutError`, and neither is
        # something the model fixes by changing its arguments.
        except WorkspaceError as e:
            raise_tool_failure(e)
        except _RECOVERABLE_ERRORS as e:
            raise ModelRetry(_sanitize_recoverable_error(e, scope.root)) from e
        except OSError as e:
            reason = _RECOVERABLE_ERRNOS.get(e.errno)
            if reason is None and getattr(e, 'winerror', None) == _WINDOWS_ERROR_INVALID_NAME:
                reason = 'The path name is invalid.'
            if reason is None:
                raise
            # The full error may embed an absolute path; the reason is path-free.
            raise ModelRetry(reason) from e

    return wrapper


_NOTICE_CHARS = 200
"""Room reserved for the longest continuation or oversized-line notice `_format_lines` appends."""


def _format_lines(lines: Sequence[str], offset: int, limit: int, max_chars: int | None = None) -> str:
    """Format pre-split lines with line numbers and continuation hint.

    With `max_chars`, the numbered lines end on the last complete one that
    fits, so the continuation offset always names the first line not shown.
    The notice appended after them is not counted; callers reserve `_NOTICE_CHARS`.
    """
    total = len(lines)

    if total == 0:
        return '(empty file)\n'

    if offset >= total:
        raise ValueError(f'Offset {offset} exceeds file length ({total} lines).')

    numbered: list[str] = []
    budget = max_chars
    for number, line in enumerate(lines[offset : offset + limit], start=offset + 1):
        rendered = f'{number:>6}\t{line}'
        if budget is not None and len(rendered) > budget:
            if not numbered:
                return (
                    f'... (Line {number} is {len(line):,} characters and does not fit the read window. '
                    f'Use offset={number} to skip it, or a shell byte range to inspect it.)\n'
                )
            break
        numbered.append(rendered)
        if budget is not None:
            budget -= len(rendered)
    result = ''.join(numbered)
    if not result.endswith('\n'):
        result += '\n'

    remaining = total - (offset + len(numbered))
    if remaining > 0:
        next_offset = offset + len(numbered)
        result += f'... ({remaining} more lines. Use offset={next_offset} to continue reading.)\n'

    return result


def _is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Detect binary content by checking for null bytes in the sample."""
    return b'\x00' in data[:sample_size]


def _matching_lines(text: str, compiled: re.Pattern[str], rel_str: str, limit: int) -> tuple[list[str], bool]:
    """Match one file's lines, keeping at most `limit` of them.

    Returns the formatted matches and whether a further match had to be
    dropped, so the caller reports truncation only when output was cut. A
    `limit` of zero or less keeps nothing.
    """
    matches: list[str] = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        if compiled.search(line):
            if len(matches) >= limit:
                return matches, True
            matches.append(f'{rel_str}:{line_num}:{line}')
    return matches, False


def _bytes_hash(data: bytes) -> str:
    """The content hash every tool reports: SHA-256 of the file's raw bytes, first 12 hex characters.

    `read_file`, `write_file`, `edit_file`, and `file_info` all hash the bytes
    in the workspace, so a hash from any of them identifies the same content and
    the `expected_hash` handshake holds whatever the line endings or encoding.
    """
    return hashlib.sha256(data).hexdigest()[:12]


def _content_hash(content: str) -> str:
    """The hash of `content` as the tools write it: its UTF-8 bytes."""
    return _bytes_hash(content.encode('utf-8'))


_DIFF_SOURCE_BYTES = 4 * MAX_DIFF_SOURCE_CHARS
"""Bytes that can hold `MAX_DIFF_SOURCE_CHARS` of UTF-8; a longer file is past the bound whatever it holds."""


def _check_expected_hash(path: str, current_hash: str, expected_hash: str) -> None:
    """Reject a write or edit whose `expected_hash` no longer matches the file."""
    if current_hash != expected_hash:
        raise ValueError(
            f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
            f'got hash:{current_hash}). Re-read the file and retry.'
        )


def _replacements(
    old_text: str | None, new_text: str | None, replacements: Sequence[Replacement] | None
) -> list[Replacement]:
    """Normalize the two argument forms of `edit_file` into one ordered list."""
    if replacements is None:
        if old_text is None or new_text is None:
            raise ModelRetry('Provide old_text and new_text, or a non-empty replacements list.')
        return [Replacement(old_text=old_text, new_text=new_text)]
    if old_text is not None or new_text is not None or not replacements:
        raise ModelRetry('Provide either old_text and new_text or a non-empty replacements list, not both.')
    return list(replacements)


def _apply_replacements(text: str, replacements: Sequence[Replacement], path: str) -> str:
    """Apply `replacements` in order, each matching exactly once; nothing is written on failure."""
    for index, replacement in enumerate(replacements, start=1):
        label = f'replacement {index}' if len(replacements) > 1 else 'old_text'
        if not replacement.old_text:
            raise ValueError(f'{label} is empty; old_text must be the exact text to replace.')
        count = text.count(replacement.old_text)
        if count == 0:
            raise ValueError(f'{label} not found in {path}. No changes were written.')
        if count > 1:
            raise ValueError(
                f'{label} found {count} times in {path}. Include more surrounding context to make the match '
                'unique. No changes were written.'
            )
        text = text.replace(replacement.old_text, replacement.new_text, 1)
    return text


def _glob_parts(pattern: str) -> tuple[list[str], bool]:
    """Split a relative glob into its components and whether it only matches directories (a trailing `/`)."""
    parts = [part for part in pattern.split('/') if part not in ('', '.')]
    if not parts or '..' in parts:
        raise ModelRetry(f'Pattern {pattern!r} is not a valid glob pattern.')
    return parts, pattern.endswith('/')


def _glob_match(pattern: Sequence[str], path: Sequence[str]) -> bool:
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


def _with_walk_notice(lines: list[str], walk_cut: bool, hidden_count: int = 0) -> str:
    """A walker's result, ending with notices for entries omitted and caps hit."""
    if hidden_count:
        lines.append(f'[{hidden_count} hidden entries omitted; name a hidden path explicitly to include it]')
    if walk_cut:
        return '\n'.join([*(lines or ['No matches found.']), _WALK_CUT_NOTICE])
    return '\n'.join(lines) if lines else 'No matches found.'


def root_spelling(root_dir: Path | None) -> str | None:
    """The workspace spelling of `root_dir`, rejecting a relative one that cannot contain the working directory.

    A relative root other than `.` or a chain of `..` names a directory below the working
    directory, which fails every run, so it is refused up front. An absolute root is checked
    against the workspace on the first file operation.
    """
    if root_dir is None:
        return None
    spelling = workspace_path(root_dir)
    normalized = posixpath.normpath(spelling)
    if not posixpath.isabs(normalized) and normalized != '.' and set(normalized.split('/')) != {'..'}:
        raise UserError(
            f'root_dir {spelling!r} is below the working directory, but root_dir must contain the working '
            'directory. To scope the tools to a subfolder, attach the workspace there, '
            "e.g. `LocalWorkspace('./src')`."
        )
    return spelling


class FileSystemToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing filesystem operations inside the run's workspace, scoped to a root directory.

    Every file operation goes through `ctx.workspace`. Guardrails for the model's file tools:
    - Relative paths resolve from the workspace's working directory. Each target must be inside
      `root_dir` both as written and once the workspace has resolved its symlinks, checked
      before each operation; a symlink swapped in between the check and the use is not caught.
    - Glob-based allow/deny filtering
    - Read-only path patterns (e.g. `.git/`, `.env`), matched against both spellings
    - Binary file detection blocks text operations

    These are guardrails, not isolation: the workspace is the isolation boundary.
    """

    def __init__(
        self,
        *,
        root_dir: Path | None = None,
        allowed_patterns: Sequence[str],
        denied_patterns: Sequence[str],
        read_only_patterns: Sequence[str] | None = None,
        max_read_lines: int,
        max_read_chars: int | None = None,
        max_list_results: int,
        max_search_results: int,
        max_find_results: int,
        id: str | None = None,
        content_hashes: bool = True,
        tools: Sequence[str] = DEFAULT_TOOL_NAMES,
        protected_patterns: Sequence[str] | None = None,
        cwd: Path | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__(id=id, max_retries=max_retries)
        if cwd is not None:
            warn_argument_ignored('FileSystemToolset', 'cwd', SET_WORKING_DIR_ON_THE_WORKSPACE, stacklevel=3)
        if protected_patterns is not None:
            if read_only_patterns is not None:
                raise TypeError('Pass `read_only_patterns` only: `protected_patterns` is its deprecated name.')
            warn_argument_renamed('FileSystemToolset', 'protected_patterns', 'read_only_patterns')
            read_only_patterns = protected_patterns
        # A workspace path, absolute or relative to the workspace's working directory, resolved
        # against the workspace a call acts on; `None` bounds calls by the working directory itself.
        self._root_spelling = root_spelling(root_dir)
        # The bounds each workspace's first call resolved, reused by later calls against it. Keyed
        # weakly by the workspace, so one toolset serves concurrent runs and holds no finished one.
        self._bounds: weakref.WeakKeyDictionary[Workspace, _Bounds] = weakref.WeakKeyDictionary()
        # One lock per file a write or edit is in progress on, keyed by the identity of the workspace
        # the call was given and the resolved path. An entry lives only while a call holds or awaits
        # it, and that call keeps the workspace alive, so the identity cannot be reused meanwhile.
        self._path_locks: dict[tuple[int, str], _PathLock] = {}
        self._allowed_patterns = list(allowed_patterns)
        self._denied_patterns = list(denied_patterns)
        self._read_only_patterns = list(read_only_patterns or ())
        self._max_read_lines = max_read_lines
        self._max_read_chars = max_read_chars
        self._max_list_results = max_list_results
        self._max_search_results = max_search_results
        self._max_find_results = max_find_results
        self._content_hashes = content_hashes
        self._tools = tuple(tools)
        if unknown := sorted(set(self._tools) - set(FILE_SYSTEM_TOOL_NAMES)):
            raise ValueError(
                f'Unknown filesystem tools: {", ".join(unknown)}. Available: {", ".join(FILE_SYSTEM_TOOL_NAMES)}.'
            )

        registrations: dict[str, Callable[..., Awaitable[str]]] = {
            'read_file': self._read_file_tool,
            'write_file': self._write_file_tool if content_hashes else self._write_file_tool_unhashed,
            'edit_file': self._edit_file_tool if content_hashes else self._edit_file_tool_unhashed,
            'list_directory': self._list_directory_tool,
            'search_files': self._search_files_tool,
            'find_files': self._find_files_tool,
            'create_directory': self._create_directory_tool,
            'file_info': self._file_info_tool,
            'list_files': self._list_files_tool,
            'grep': self._grep_tool,
        }
        for name in FILE_SYSTEM_TOOL_NAMES:
            if name in self._tools:
                # Approval must run in the workflow before the durable workspace write activity.
                metadata = {'temporal': False} if name in {'write_file', 'edit_file', 'create_directory'} else None
                self.add_function(registrations[name], name=name, metadata=metadata)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Offer only the tools the run's workspace can serve.

        A read-only workspace keeps only `READ_ONLY_TOOL_NAMES`. Where the workspace cannot run
        `rg`, the ripgrep tools walk its files instead.
        """
        tools = await super().get_tools(ctx)
        if ctx.workspace.read_only:
            tools = {name: tool for name, tool in tools.items() if name in READ_ONLY_TOOL_NAMES}
        return tools

    async def _scope(self, workspace: WorkspaceBackend) -> _Scope:
        """Resolve the boundary and working directory inside `workspace`, once per workspace.

        Resolution waits for the first file operation, so a run that never touches a file does no
        workspace I/O. A default boundary is the working directory with its symlinks resolved, and
        relative paths then resolve from that same spelling, so targets and boundary compare alike
        even on a backend whose working directory is a symlinked path. Raises `UserError` when the
        working directory is outside `root_dir`.
        """
        facade = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        if (bounds := self._bounds.get(facade)) is not None:
            return _Scope(facade, bounds, workspace)
        cwd = posixpath.normpath(await facade.working_dir())
        if self._root_spelling is None:
            root = cwd = cwd if cwd == '/' else await facade.realpath(cwd)
        else:
            root = await facade.resolve(self._root_spelling)
            if root != '/':
                root = await facade.realpath(root)
                # Compare like with like: a working directory reached through a symlink is still inside.
                cwd = await facade.realpath(cwd)
            if not _contains(root, cwd):
                raise UserError(
                    f'The working directory {cwd!r} is outside root_dir {root!r}. '
                    f'To work in {root!r}, attach the workspace there, such as `LocalWorkspace({root!r})` on this '
                    'machine; otherwise set `root_dir` to a directory that contains the working directory.'
                )
        has_patterns = bool(self._allowed_patterns or self._denied_patterns or self._read_only_patterns)
        bounds = _Bounds(root=root, cwd=cwd, checks_realpath=root != '/' or has_patterns)
        self._bounds[facade] = bounds
        return _Scope(facade, bounds, workspace)

    @asynccontextmanager
    async def _changing(self, scope: _Scope, real: str) -> AsyncGenerator[None]:
        """Let one write or edit at a time read and replace the file at `real` in this workspace.

        Workspace I/O suspends between a call's read and its write, so without this two concurrent
        edits both start from the same content and the later write drops the earlier change. Keyed
        by the real path, so a symlink and its target share a turn; where real paths go unchecked
        (`root_dir='/'` without patterns) it is the path as written.
        """
        key = (id(scope.given), real)
        entry = self._path_locks.setdefault(key, _PathLock())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if not entry.users:
                del self._path_locks[key]

    def _matches(self, path: str, pattern: str) -> bool:
        """Glob-match a relative path, treating a leading `**/` as 'any directory, including the root'.

        `fnmatch` has no recursive `**`, so a bare `**/secrets*` would miss a
        root-level `secrets.yaml` -- there's no leading directory to match.
        Retrying with the `**/` prefix stripped covers the zero-directory case.
        """
        if fnmatch.fnmatch(path, pattern):
            return True
        if pattern.startswith('**/'):
            return fnmatch.fnmatch(path, pattern[3:])
        return False

    def _first_matching_pattern(self, path: str, patterns: list[str], *, ancestors: bool = False) -> str | None:
        """Return the first pattern matching this path (or, for a directory policy, its ancestors)."""
        # A directory denial protects its descendants even when the pattern has no `/**` suffix.
        parts = path.split('/')
        paths = [path] if not ancestors else ['/'.join(parts[:end]) for end in range(1, len(parts) + 1)]
        return next((p for p in patterns if any(self._matches(candidate, p) for candidate in paths)), None)

    async def _resolve_path(self, scope: _Scope, path: str) -> tuple[str, str]:
        """Resolve path from the working directory, rejecting any that leads outside the root.

        Returns the path as written (textually resolved) and the real path the workspace
        reaches through symlinks; both must be inside the root. The operation then uses the
        path as written, so a symlink swapped in after this check is not caught.
        """
        resolved = await scope.workspace.resolve(path, base=scope.cwd)
        if not _contains(scope.root, resolved):
            raise PermissionError(
                f'`{resolved}` is outside {self._root_name(scope)}; the file tools only work inside it. '
                f'Create or clone it inside {self._root_place(scope)}, or use a shell tool if you have one.'
            )
        if not scope.checks_realpath:
            return resolved, resolved
        real = await scope.workspace.realpath(resolved)
        if not _contains(scope.root, real):
            # The link's target is not named: absolute paths outside the root stay out of tool results.
            raise PermissionError(
                f'`{path}` leads outside {self._root_name(scope)} through a symlink; the file tools only work '
                f'inside it. Use a path inside {self._root_place(scope)}, or a shell tool if you have one.'
            )
        return resolved, real

    def _root_name(self, scope: _Scope) -> str:
        """The boundary as an error names it: the project itself unless `root_dir` was set."""
        return f'the project root `{scope.root}`' if self._root_spelling is None else f'root_dir `{scope.root}`'

    def _root_place(self, scope: _Scope) -> str:
        return 'the project' if self._root_spelling is None else f'`{scope.root}`'

    async def _real_path_inside(self, scope: _Scope, path: str) -> bool:
        """Whether a path a walk reached still leads inside the root once symlinks are resolved."""
        return not scope.checks_realpath or _contains(scope.root, await scope.workspace.realpath(path))

    def _check_access(self, path: str, *, write: bool = False, check_allowed: bool = True) -> None:
        """Validate path against allow/deny/read-only patterns.

        `check_allowed=False` skips the `allowed_patterns` gate. Walkers
        (`list_directory`, `search_files`, `find_files`) pass it so their root
        directory isn't required to match `allowed_patterns` itself -- `.` or
        `src` would never match a file pattern like `src/*.py`. The walk's
        entries are still filtered against `allowed_patterns` per-entry via
        `_is_accessible`. Denied patterns continue to gate the root.
        """
        if write and self._read_only_patterns:
            matched = self._first_matching_pattern(path, self._read_only_patterns, ancestors=True)
            if matched:
                raise PermissionError(f'Path {path!r} is protected (matches {matched!r}).')

        if self._denied_patterns:
            matched = self._first_matching_pattern(path, self._denied_patterns, ancestors=True)
            if matched:
                raise PermissionError(f'Path {path!r} is denied by pattern {matched!r}.')

        if check_allowed and self._allowed_patterns:
            if not any(self._matches(path, p) for p in self._allowed_patterns):
                raise PermissionError(f'Path {path!r} does not match any allowed pattern.')

    def _is_accessible(self, path: str) -> bool:
        """Predicate form of the read-level `_check_access` checks.

        Read-only patterns are not consulted: they gate writes, and the walkers
        only read.
        """
        if self._denied_patterns:
            if self._first_matching_pattern(path, self._denied_patterns, ancestors=True) is not None:
                return False
        if self._allowed_patterns and not any(self._matches(path, p) for p in self._allowed_patterns):
            return False
        return True

    def _walk_entry(self, scope: _Scope, path: str, start: str, *, include_hidden: bool = False) -> str | None:
        """Authorize one entry of a directory walk: its root-relative path, or `None` to skip it.

        Hidden entries are skipped, matching `list_directory`, `search_files`,
        and `find_files`, and entries are matched against the patterns by their
        root-relative spelling.
        """
        if not _contains(scope.root, path):  # pragma: no cover -- walks and `rg` output stay below the root
            # Walks start inside the root and `rg` prints paths below its cwd; this guards a
            # backend or `rg` build that reports an entry elsewhere.
            return None
        relative = posixpath.relpath(path, scope.root)
        # The caller explicitly named `start`; only hidden components below it are omitted.
        if (not include_hidden and _is_hidden(posixpath.relpath(path, start))) or not self._is_accessible(relative):
            return None
        return relative

    async def _readable_entry(self, scope: _Scope, path: str) -> bool:
        """Whether a walked file's real path is inside the root and passes the read-level patterns."""
        if not scope.checks_realpath:
            return True
        real = await scope.workspace.realpath(path)
        return _contains(scope.root, real) and self._is_accessible(posixpath.relpath(real, scope.root))

    def _event_location(self, scope: _Scope, resolved: str) -> _EventLocation:
        """Path fields for an event about `resolved`.

        `path` is relative to the root and `root_dir` is that root, as a
        workspace path, so a subscriber rooted elsewhere can rebuild the
        location instead of assuming the event came from its own root.
        """
        return _EventLocation(path=_model_safe_filename(resolved, scope.root), root_dir=scope.root)

    async def _safe_resolve(self, scope: _Scope, path: str, *, write: bool = False, check_allowed: bool = True) -> str:
        """Resolve and access-check a path in one step.

        Resolution happens first so the access check matches patterns against
        the canonical path relative to the root, collapsing `.`/`..`/`//`
        segments that would otherwise slip past a literal pattern (e.g.
        `config/./secret.txt` evading a `config/secret.txt` deny rule). The
        patterns are matched against the real path too, so a symlink to a
        protected file (`envlink -> .env`) is protected as well.
        """
        return (await self._safe_resolve_real(scope, path, write=write, check_allowed=check_allowed))[0]

    async def _safe_resolve_real(
        self, scope: _Scope, path: str, *, write: bool = False, check_allowed: bool = True
    ) -> tuple[str, str]:
        """`_safe_resolve`, also returning the real path (the same spelling when real paths go unchecked)."""
        resolved, real = await self._resolve_path(scope, path)
        for spelling in dict.fromkeys((resolved, real)):
            self._check_access(posixpath.relpath(spelling, scope.root), write=write, check_allowed=check_allowed)
        return resolved, real

    async def _stat(self, scope: _Scope, resolved: str) -> FileEntry | None:
        """The entry at `resolved`, or `None` when nothing is there (including below a file)."""
        try:
            return await scope.workspace.stat(resolved)
        except (FileNotFoundError, NotADirectoryError):
            return None

    async def _walk(
        self, scope: _Scope, directory: str, *, max_depth: int | None = None, include_hidden: bool = False
    ) -> tuple[list[FileEntry], bool]:
        """Entries below `directory`, walked iteratively with `list_dir`, and whether the walk was cut short.

        Hidden directories are not descended into, since everything under them
        is hidden, unless `include_hidden` (the caller named a hidden path), nor is a directory whose real path is outside the root, and a
        subdirectory that cannot be listed (removed mid-walk, unreadable, a symlink
        loop the backend reports) is skipped. `max_depth` bounds how many levels
        below `directory` are listed. The walk stops at `_MAX_WALK_DIRECTORIES`
        listings or `_MAX_WALK_ENTRIES` entries, which bounds a symlink loop inside
        the root.
        """
        entries: list[FileEntry] = []
        pending: list[tuple[str, int]] = [(directory, 1)]
        seen_dirs = {await scope.workspace.realpath(directory)}
        listed = 0
        while pending:
            if listed >= _MAX_WALK_DIRECTORIES or len(entries) >= _MAX_WALK_ENTRIES:
                return entries[:_MAX_WALK_ENTRIES], True
            listed += 1
            current, depth = pending.pop()
            try:
                children = await scope.workspace.list_dir(current)
            except WorkspaceError:
                raise
            except OSError:
                if current == directory:
                    raise
                continue
            entries.extend(children)
            if max_depth is None or depth < max_depth:
                # Prefer a directory's own spelling over aliases, and do not revisit a real
                # directory through a loop or another symlink inside the same walk.
                directories = [
                    child for child in children if child.is_dir and (include_hidden or not child.name.startswith('.'))
                ]
                resolved_dirs: list[tuple[FileEntry, str]] = []
                for child in directories:
                    try:
                        resolved_dirs.append((child, await scope.workspace.realpath(child.path)))
                    except WorkspaceError:
                        raise
                    except OSError:
                        continue  # like an unlistable directory, skip only this child
                for child, real in sorted(resolved_dirs, key=lambda pair: pair[0].path != pair[1]):
                    if (
                        real not in seen_dirs
                        and _contains(scope.root, real)
                        and not self._first_matching_pattern(
                            posixpath.relpath(child.path, scope.root), self._denied_patterns, ancestors=True
                        )
                    ):
                        seen_dirs.add(real)
                        pending.append((child.path, depth + 1))
        return entries, False

    async def read_file(
        self, path: str, *, offset: int = 0, limit: int | None = None, workspace: WorkspaceBackend
    ) -> str:
        """Read a text file in `workspace` directly, outside an agent run."""
        return await self._read_file(await self._scope(workspace), None, path, offset=offset, limit=limit)

    async def _read_file_tool(
        self, ctx: RunContext[AgentDepsT], path: str, *, offset: int = 0, limit: int | None = None
    ) -> str:
        """Read a text file with line numbers.

        Args:
            ctx: The current agent run context.
            path: File path relative to the working directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            File content with line numbers, plus metadata header.
        """
        return await self._read_file(
            await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), path, offset=offset, limit=limit
        )

    @_recoverable
    async def _read_file(
        self, scope: _Scope, ctx: RunContext[AgentDepsT] | None, path: str, *, offset: int = 0, limit: int | None = None
    ) -> str:
        if offset < 0:
            raise ValueError('offset must be non-negative.')
        if limit is not None and limit < 1:
            raise ValueError('limit must be at least 1.')
        if limit is None:
            limit = self._max_read_lines
        try:
            resolved = await self._safe_resolve(scope, path)
            entry = await self._stat(scope, resolved)
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return f'Path not found: {path}'
        if entry is None or entry.is_dir:
            return f'Path not found: {path}'

        # The whole file: the header reports its line count and the hash write_file and
        # edit_file verify against, and both need every byte.
        try:
            raw = await scope.workspace.read_bytes(resolved)
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return f'Path not found: {path}'
        content_hash = _bytes_hash(raw)
        if _is_binary(raw):
            if ctx is not None:
                await ctx.emit(FileReadEvent(**self._event_location(scope, resolved), content_hash=content_hash))
            return f'[Binary file: {len(raw)} bytes. Use a binary-aware tool to inspect.]'

        text = raw.decode('utf-8', errors='replace')
        lines = text.splitlines(keepends=True)
        header = self._read_header(path, len(lines), content_hash)
        # Format before emitting: an out-of-range offset is a failed read, and
        # a failed read must not look like a successful one to subscribers.
        body = _format_lines(lines, offset, limit, self._body_budget(header))
        if ctx is not None:
            await ctx.emit(FileReadEvent(**self._event_location(scope, resolved), content_hash=content_hash))
        return header + body

    def _read_header(self, path: str, total: int, content_hash: str) -> str:
        label = path
        if self._max_read_chars is not None and len(path) > self._max_read_chars // 4:
            # The label echoes what the model passed; keep a long one from eating the window.
            label = '...' + path[-(self._max_read_chars // 4) :]
        return f'[{label} | {total} lines{" | hash:" + content_hash if self._content_hashes else ""}]\n'

    def _body_budget(self, header: str) -> int | None:
        """Characters left for numbered lines once the header and the longest notice are counted."""
        if self._max_read_chars is None:
            return None
        return max(self._max_read_chars - len(header) - _NOTICE_CHARS, 0)

    def _hash_suffix(self, content_hash: str) -> str:
        """The hash a write or edit result shows the model, or nothing when `content_hashes` is off."""
        return f' [hash:{content_hash}]' if self._content_hashes else ''

    async def write_file(
        self, path: str, content: str, *, expected_hash: str | None = None, workspace: WorkspaceBackend
    ) -> str:
        """Write a text file in `workspace` directly, outside an agent run."""
        return await self._write_file(await self._scope(workspace), None, path, content, expected_hash=expected_hash)

    async def _write_file_tool(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        content: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        """Create or overwrite a file with conflict detection.

        Args:
            ctx: The current agent run context.
            path: File path relative to the working directory.
            content: The text content to write.
            expected_hash: If provided, the write is rejected when the file exists
                and its current hash doesn't match (optimistic concurrency).

        Returns:
            Confirmation message with new hash.
        """
        return await self._write_file(
            await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), path, content, expected_hash=expected_hash
        )

    async def _write_file_tool_unhashed(self, ctx: RunContext[AgentDepsT], path: str, content: str) -> str:
        """Create a file or replace its whole content.

        Args:
            ctx: The current agent run context.
            path: File path relative to the working directory.
            content: The text content to write.
        """
        return await self._write_file(await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), path, content)

    async def _announced_state(
        self, scope: _Scope, resolved: str, path: str, *, exists: bool, expected_hash: str | None
    ) -> tuple[str | None, str | None]:
        """The text a listener is shown a write replacing, and the hash the write is then guarded with.

        A stale `expected_hash` for a file that exists is rejected here first, so
        a listener only sees a write that would go ahead; for a missing file the
        hash is ignored, as documented, and the write is announced as a create.
        A new file diffs from empty and is guarded as absent, so one that appears
        while the write is announced is a conflict whether or not it is empty. The
        text is `None` when it cannot be shown: a file the workspace refuses to
        read is announced as headers alone, marked as cut, and written unguarded,
        while an `expected_hash` it cannot check propagates the error; past
        `MAX_DIFF_SOURCE_CHARS` it would not be diffed.
        """
        if not exists:
            return '', _ABSENT_HASH
        try:
            raw = await scope.workspace.read_bytes(resolved)
        except WorkspaceReadOnlyError:
            raise
        except PermissionError:
            if expected_hash is not None:
                raise
            return None, None
        current_hash = _bytes_hash(raw)
        if expected_hash is not None:
            _check_expected_hash(path, current_hash, expected_hash)
        old = raw.decode('utf-8', errors='replace') if len(raw) <= _DIFF_SOURCE_BYTES else None
        return old, current_hash

    async def _check_guard(self, scope: _Scope, resolved: str, path: str, guard: str) -> None:
        """Refuse a write whose target no longer holds the content it was checked or announced against.

        A file that appeared where none was announced fails against `_ABSENT_HASH`;
        a file that vanished meanwhile is written as a create, as with no guard at all.
        """
        try:
            current = _bytes_hash(await scope.workspace.read_bytes(resolved))
        except FileNotFoundError:
            return
        _check_expected_hash(path, current, guard)

    @_recoverable
    async def _write_file(
        self,
        scope: _Scope,
        ctx: RunContext[AgentDepsT] | None,
        path: str,
        content: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        resolved, real = await self._safe_resolve_real(scope, path, write=True)
        async with self._changing(scope, real):
            entry = await self._stat(scope, resolved)
            if entry is not None and entry.is_dir:
                raise ModelRetry(f'Path {path!r} exists and is not a regular file.')

            parent = posixpath.dirname(resolved)
            try:
                parent_entry = await scope.workspace.stat(parent)
            except FileNotFoundError as e:
                parent_rel = posixpath.relpath(parent, scope.root)
                raise FileNotFoundError(
                    f"Parent directory '{parent_rel}' does not exist. Create the parent directory first."
                ) from e
            except NotADirectoryError as e:
                raise ModelRetry(f'Path {path!r} has a parent that is not a directory.') from e
            # Checked before the announcement, like `create_directory` does, so a
            # listener is only asked about a write the filesystem would accept.
            if not parent_entry.is_dir:
                raise ModelRetry(f'Path {path!r} has a parent that is not a directory.')

            # Outside a run nothing is announced: the diff is for listeners, and the
            # check of `expected_hash` just before the write is the whole contract.
            guard = expected_hash if entry is not None else None
            if ctx is not None:
                old, guard = await self._announced_state(
                    scope, resolved, path, exists=entry is not None, expected_hash=expected_hash
                )
                change = Change.propose(
                    **self._event_location(scope, resolved), operation='write', old=old, new=content
                )
                if (refusal := await self._request(scope, ctx, change, path=path, resolved=resolved)) is not None:
                    return refusal

            if guard is not None:
                await self._check_guard(scope, resolved, path, guard)
            await scope.workspace.write_bytes(resolved, content.encode('utf-8'))

        new_hash = _content_hash(content)
        lines = len(content.splitlines())
        if ctx is not None:
            await ctx.emit(FileWrittenEvent(**self._event_location(scope, resolved), content_hash=new_hash))
        return f'Wrote {len(content)} chars ({lines} lines) to {path}.{self._hash_suffix(new_hash)}'

    async def _request(
        self, scope: _Scope, ctx: RunContext[AgentDepsT] | None, change: Change, *, path: str, resolved: str
    ) -> str | None:
        """Announce `change` and confirm that `path` still passes the access checks afterwards.

        A listener can hold the request for as long as a human takes, and the
        change is then applied by name, so the path is resolved and checked
        again before the write. Outside a run there is nobody to ask.
        """
        if ctx is None:
            return None
        if (refusal := await change.request(ctx)) is not None:
            return refusal
        if await self._safe_resolve(scope, path, write=True) != resolved:  # pragma: no cover
            # Resolution is textual, so the same path resolves the same way unless the
            # workspace's working directory moved while the change was held.
            raise ModelRetry(f'Path {path!r} was replaced while the change was announced. Retry.')
        return None

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: str | None = None,
        workspace: WorkspaceBackend,
    ) -> str:
        """Edit a text file in `workspace` directly, outside an agent run."""
        replacements = [Replacement(old_text=old_text, new_text=new_text)]
        return await self._edit_file(
            await self._scope(workspace), None, path, replacements, expected_hash=expected_hash
        )

    async def _edit_file_tool(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        old_text: str | None = None,
        new_text: str | None = None,
        *,
        replacements: list[Replacement] | None = None,
        expected_hash: str | None = None,
    ) -> str:
        """Edit a file by exact string replacement with conflict detection.

        Pass one `old_text`/`new_text` pair, or several as `replacements`. Each
        old_text must appear exactly once in the file as edited by the previous
        replacements; include surrounding context lines to ensure uniqueness.
        The file is only written once every replacement has matched.

        Args:
            ctx: The current agent run context.
            path: File path relative to the working directory.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            replacements: Replacements to apply in order, instead of a single pair.
            expected_hash: If provided, rejects the edit when the file's
                current hash doesn't match (optimistic concurrency).

        Returns:
            Summary with new hash for subsequent operations.
        """
        edits = _replacements(old_text, new_text, replacements)
        return await self._edit_file(
            await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), path, edits, expected_hash=expected_hash
        )

    async def _edit_file_tool_unhashed(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        old_text: str | None = None,
        new_text: str | None = None,
        *,
        replacements: list[Replacement] | None = None,
    ) -> str:
        """Edit a file by exact string replacement.

        Pass one `old_text`/`new_text` pair, or several as `replacements`. Each
        old_text must appear exactly once in the file as edited by the previous
        replacements; include surrounding context lines to ensure uniqueness.
        The file is only written once every replacement has matched.

        Args:
            ctx: The current agent run context.
            path: File path relative to the working directory.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            replacements: Replacements to apply in order, instead of a single pair.
        """
        edits = _replacements(old_text, new_text, replacements)
        return await self._edit_file(await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), path, edits)

    @_recoverable
    async def _edit_file(
        self,
        scope: _Scope,
        ctx: RunContext[AgentDepsT] | None,
        path: str,
        replacements: Sequence[Replacement],
        *,
        expected_hash: str | None = None,
    ) -> str:
        resolved, real = await self._safe_resolve_real(scope, path, write=True)
        async with self._changing(scope, real):
            entry = await self._stat(scope, resolved)
            if entry is None or entry.is_dir:
                raise FileNotFoundError(f'File not found: {path}')
            raw = await scope.workspace.read_bytes(resolved)
            if _is_binary(raw):
                raise ValueError(f'{path} is a binary file; edit_file only edits text files.')
            # Strict decoding: an edit writes the whole file back, so undecodable bytes are refused
            # rather than replaced. No newline translation, so CRLF and the hash are preserved.
            text = raw.decode('utf-8')
            current_hash = _bytes_hash(raw)

            if expected_hash is not None:
                _check_expected_hash(path, current_hash, expected_hash)

            new_content = _apply_replacements(text, replacements, path)
            change = Change.propose(
                **self._event_location(scope, resolved), operation='edit', old=text, new=new_content
            )
            if (refusal := await self._request(scope, ctx, change, path=path, resolved=resolved)) is not None:
                return refusal
            if ctx is not None:
                # A listener may take a while (a human approving the diff, say). The
                # edit was computed from `text`, so the write checks that the file
                # still holds it, and reports a file deleted in the meantime as missing.
                try:
                    now = await scope.workspace.read_bytes(resolved)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f'File not found: {path}') from e
                _check_expected_hash(path, _bytes_hash(now), current_hash)
            await scope.workspace.write_bytes(resolved, new_content.encode('utf-8'))
        new_hash = _content_hash(new_content)
        if ctx is not None:
            await ctx.emit(change.edited(content_hash=new_hash))
        return f'Edited {path}.{self._hash_suffix(new_hash)}'

    async def list_directory(self, path: str = '.', *, workspace: WorkspaceBackend) -> str:
        """List a directory in `workspace` directly, outside an agent run."""
        return await self._list_directory(await self._scope(workspace), None, path)

    async def _list_directory_tool(self, ctx: RunContext[AgentDepsT], path: str = '.') -> str:
        """List the contents of a directory.

        Args:
            ctx: The current agent run context.
            path: Directory path relative to the working directory.

        Returns:
            Paths relative to the working directory, with type indicators and sizes.
        """
        return await self._list_directory(await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), path)

    @_recoverable
    async def _list_directory(self, scope: _Scope, ctx: RunContext[AgentDepsT] | None, path: str = '.') -> str:
        # The listing root is gated by denied patterns but not by
        # allowed_patterns: a directory like '.' never matches a file pattern.
        # Entries are filtered per-entry against allowed_patterns below.
        resolved = await self._safe_resolve(scope, path, check_allowed=False)
        try:
            children = await scope.workspace.list_dir(resolved)
        except (FileNotFoundError, NotADirectoryError) as e:
            raise NotADirectoryError(f'Not a directory: {path}') from e

        entries: list[str] = []
        entry_count = 0
        hidden_count = 0
        for entry in sorted(children, key=lambda child: child.name):
            if entry.name.startswith('.'):
                hidden_count += 1
            # Skip dotfiles and dot-directories, matching search_files and
            # find_files so the three walkers agree on what exists.
            if self._walk_entry(scope, entry.path, resolved) is None:
                continue
            rel = posixpath.relpath(entry.path, scope.cwd)
            if entry.is_dir:
                line = f'{rel}/'
            else:
                size = entry.size
                if size is None:
                    stat = await self._stat(scope, entry.path)
                    if stat is None:
                        # A dangling symlink or an entry deleted mid-walk: leave it out.
                        continue
                    size = stat.size
                line = f'{rel}  (size unknown)' if size is None else f'{rel}  ({size} bytes)'
            # Only a listing that actually dropped an entry is marked truncated,
            # so one that merely fills the cap reads as complete.
            if len(entries) >= self._max_list_results:
                entries.append(f'[... truncated at {self._max_list_results} entries]')
                break
            entries.append(line)
            entry_count += 1
        if ctx is not None:
            await ctx.emit(DirectoryListedEvent(**self._event_location(scope, resolved), entry_count=entry_count))
        if hidden_count:
            entries.append(f'[{hidden_count} hidden entries omitted; name a hidden path explicitly to include it]')
        return '\n'.join(entries) if entries else '(empty directory)'

    async def search_files(
        self, pattern: str, *, path: str = '.', include_glob: str | None = None, workspace: WorkspaceBackend
    ) -> str:
        """Search file contents in `workspace` directly, outside an agent run."""
        return await self._search_files(
            await self._scope(workspace), None, pattern, path=path, include_glob=include_glob
        )

    async def _search_files_tool(
        self, ctx: RunContext[AgentDepsT], pattern: str, *, path: str = '.', include_glob: str | None = None
    ) -> str:
        """Search file contents using a regular expression.

        Args:
            ctx: The current agent run context.
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the working directory.
            include_glob: If provided, match this glob against root-relative paths (e.g. '*.py').

        Returns:
            str: Matching lines formatted as file:line_number:text, with paths relative to the working directory.
        """
        return await self._search_files(
            await self._scope(ctx.workspace),
            event_ctx(ctx, 'FileSystem'),
            pattern,
            path=path,
            include_glob=include_glob,
        )

    @_recoverable
    async def _search_files(  # noqa: C901  -- filesystem-only and command paths share output bookkeeping
        self,
        scope: _Scope,
        ctx: RunContext[AgentDepsT] | None,
        pattern: str,
        *,
        path: str = '.',
        include_glob: str | None = None,
    ) -> str:
        # See list_directory: the search root isn't gated by allowed_patterns;
        # matched files are filtered per-entry below.
        resolved = await self._safe_resolve(scope, path, check_allowed=False)
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern: {e}') from e

        entry = await self._stat(scope, resolved)
        # Probe rg on the first call too: walking remote files is only for backends
        # without commands, or when the command path itself fails.
        if entry is not None and supports_commands(scope.workspace):
            return await self._command_search_files(scope, ctx, entry, resolved, pattern, include_glob)
        include_hidden = bool(include_glob and _explicit_hidden(include_glob))
        walk_cut = False
        hidden_count = 0
        if entry is None:
            files: list[FileEntry] = []
        elif not entry.is_dir:
            files = [entry]
        else:
            walked, walk_cut = await self._walk(scope, resolved, include_hidden=include_hidden)
            # Hidden directories are pruned; count their directory entry, not unseen descendants.
            if not include_hidden:
                hidden_count = sum(_is_hidden(posixpath.relpath(child.path, resolved)) for child in walked)
            files = [child for child in walked if not child.is_dir]

        results: list[str] = []
        skipped: list[str] = []
        capped = False
        for file in sorted(files, key=lambda child: _sort_key(child.path)):
            file_path = file.path
            rel_str = self._walk_entry(scope, file_path, resolved, include_hidden=include_hidden)
            if rel_str is None:
                continue
            if include_glob and not fnmatch.fnmatch(rel_str, include_glob):
                continue
            # Contents are read, so a file that links outside the root, or to a denied file, is skipped.
            if file_path != resolved and not await self._readable_entry(scope, file_path):
                continue
            if file.size is not None and file.size > _MAX_SEARCH_FILE_BYTES:
                skipped.append(posixpath.relpath(file_path, scope.cwd))
                continue
            try:
                raw = await scope.workspace.read_bytes(file_path)
            except WorkspaceUnavailableError:
                # The environment is gone, so the rest of the tree is unreadable too.
                raise
            except (WorkspaceError, OSError):
                # A single inaccessible file should not hide matches in the rest of the tree.
                skipped.append(posixpath.relpath(file_path, scope.cwd))
                continue
            if _is_binary(raw):
                continue
            text = raw.decode('utf-8', errors='replace')
            matches, capped = _matching_lines(
                text, compiled, posixpath.relpath(file_path, scope.cwd), self._max_search_results - len(results)
            )
            results.extend(matches)
            if capped:
                break

        if ctx is not None:
            await ctx.emit(
                self._searched(
                    scope, resolved, pattern, search='grep', match_count=len(results), truncated=capped or walk_cut
                )
            )
        if capped:
            results.append(f'[... truncated at {self._max_search_results} matches]')
        if skipped:
            results.append(
                f'[{len(skipped)} files skipped (too large or unreadable): {", ".join(skipped[:10])}'
                f'{", ..." if len(skipped) > 10 else ""}]'
            )
        return _with_walk_notice(results, walk_cut, hidden_count)

    async def _command_search_files(
        self,
        scope: _Scope,
        ctx: RunContext[AgentDepsT] | None,
        entry: FileEntry,
        resolved: str,
        pattern: str,
        include_glob: str | None,
    ) -> str:
        cwd = resolved if entry.is_dir else posixpath.dirname(resolved)
        target = '.' if entry.is_dir else posixpath.basename(resolved)
        prepare, permitted = self._batch_authorizer(scope, cwd)
        prefix = posixpath.relpath(cwd, scope.root)
        command_glob = include_glob.removeprefix(prefix + '/') if include_glob and prefix != '.' else include_glob
        include_hidden = bool(include_glob and _explicit_hidden(include_glob))

        async def accept(record: Record) -> str | None:
            if include_glob and not (
                fnmatch.fnmatch(record.path, include_glob)
                or fnmatch.fnmatch(posixpath.join(prefix, record.path), include_glob)
            ):
                return None
            return await self._authorized_match(
                scope,
                cwd,
                record,
                include_hidden=include_hidden,
                permitted=permitted,
            )

        try:
            if scope.lacks_ripgrep:
                raise RipgrepMissing
            results, capped, unreadable = await run_ripgrep(
                scope.workspace,
                [
                    '--line-number',
                    '--with-filename',
                    '--sort',
                    'path',
                    '--max-columns',
                    str(_MAX_MATCH_COLUMNS),
                    '--max-columns-preview',
                    *(['--glob', command_glob] if command_glob else []),
                    '--regexp',
                    pattern,
                    '--',
                    target,
                ],
                cwd=cwd,
                limit=self._max_search_results,
                accept=lambda record: (
                    self._match_line(scope, cwd, record, include_hidden=include_hidden) if permitted(record) else None
                ),
                prepare=prepare,
            )
        except RipgrepMissing:
            _fall_back_from_ripgrep(scope)
            results, capped, unreadable = await run_posix_search(
                scope.workspace,
                cwd=cwd,
                target=target,
                explicit_file=not entry.is_dir,
                pattern=pattern,
                include_hidden=include_hidden,
                limit=self._max_search_results,
                accept=accept,
                prepare=prepare,
            )
        if ctx is not None:
            await ctx.emit(
                self._searched(scope, resolved, pattern, search='grep', match_count=len(results), truncated=capped)
            )
        if capped:
            results.append(f'[... truncated at {self._max_search_results} lines or search output byte limit]')
        return self._with_unreadable(scope, cwd, results or ['No matches found.'], unreadable)

    def _with_unreadable(self, scope: _Scope, cwd: str, lines: list[str], unreadable: list[Unreadable]) -> str:
        """A search result, ending with a note naming the paths it could not read, grouped by reason.

        Paths the patterns hide stay unnamed but are still counted, so the note never reveals them.
        """
        by_reason: dict[str, list[str | None]] = {}
        for entry in unreadable:
            path = posixpath.normpath(posixpath.join(cwd, entry.path))
            by_reason.setdefault(entry.reason, []).append(self._walk_entry(scope, path, cwd, include_hidden=True))
        for reason, paths in by_reason.items():
            named = [path for path in paths if path is not None]
            shown = ', '.join(named[:_MAX_UNREADABLE_NAMED]) + (', ...' if len(named) > _MAX_UNREADABLE_NAMED else '')
            noun = 'path' if len(paths) == 1 else 'paths'
            lines.append(f'[{len(paths)} {noun} could not be read ({reason}){": " + shown if shown else ""}]')
        return '\n'.join(lines)

    def _searched(
        self, scope: _Scope, resolved: str, pattern: str, *, search: SearchKind, match_count: int, truncated: bool
    ) -> FilesSearchedEvent:
        return FilesSearchedEvent(
            **self._event_location(scope, resolved),
            pattern=pattern,
            search=search,
            match_count=match_count,
            truncated=truncated,
        )

    async def find_files(self, pattern: str, *, path: str = '.', workspace: WorkspaceBackend) -> str:
        """Find files by glob pattern in `workspace` directly, outside an agent run."""
        return await self._find_files(await self._scope(workspace), None, pattern, path=path)

    async def _find_files_tool(self, ctx: RunContext[AgentDepsT], pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            ctx: The current agent run context.
            pattern: Glob pattern to match, relative to `path` (e.g. '*.py',
                '**/*.json'). Absolute patterns are rejected.
            path: Directory to search in, relative to the working directory.

        Returns:
            Newline-separated list of matching file paths relative to the working directory.
        """
        return await self._find_files(
            await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), pattern, path=path
        )

    @_recoverable
    async def _find_files(
        self,
        scope: _Scope,
        ctx: RunContext[AgentDepsT] | None,
        pattern: str,
        *,
        path: str = '.',
        files_only: bool = False,
    ) -> str:
        if posixpath.isabs(pattern):
            raise ValueError(f'Pattern {pattern!r} must be relative to the search path, not absolute.')
        parts, directories_only = _glob_parts(pattern)

        # See list_directory: the find root isn't gated by allowed_patterns;
        # matched entries are filtered per-entry below.
        resolved = await self._safe_resolve(scope, path, check_allowed=False)
        entry = await self._stat(scope, resolved)
        if entry is None or not entry.is_dir:
            raise NotADirectoryError(f'Not a directory: {path}')

        # Without `**`, nothing deeper than the pattern's own components can match.
        max_depth = None if '**' in parts else len(parts)
        include_hidden = _explicit_hidden(pattern)
        walked, walk_cut = await self._walk(scope, resolved, max_depth=max_depth, include_hidden=include_hidden)
        # Report only the entries observed: the walk prunes hidden subdirectories.
        hidden_count = (
            0
            if include_hidden
            else sum(
                _is_hidden(relative := posixpath.relpath(child.path, resolved))
                and (child.is_dir and '**' in parts or _glob_match(parts, relative.split('/')))
                for child in walked
            )
        )
        found = [
            child
            for child in walked
            if (not child.is_dir if files_only else child.is_dir or not directories_only)
            and _glob_match(parts, posixpath.relpath(child.path, resolved).split('/'))
        ]

        matches: list[str] = []
        capped = False
        for match in sorted(found, key=lambda child: _sort_key(child.path)):
            if self._walk_entry(scope, match.path, resolved, include_hidden=include_hidden) is None:
                continue
            if not match.is_dir and match.size is None and not await scope.workspace.exists(match.path):
                # A dangling symlink is inside the root but names nothing.
                continue
            if len(matches) >= self._max_find_results:
                capped = True
                break
            rel = posixpath.relpath(match.path, scope.cwd)
            suffix = '/' if match.is_dir else ''
            matches.append(f'{rel}{suffix}')

        if ctx is not None:
            await ctx.emit(
                self._searched(
                    scope, resolved, pattern, search='find', match_count=len(matches), truncated=capped or walk_cut
                )
            )
        if capped:
            matches.append(f'[... truncated at {self._max_find_results} matches]')
        return _with_walk_notice(matches, walk_cut, hidden_count)

    async def list_files(self, path: str = '.', *, glob: str | None = None, workspace: WorkspaceBackend) -> str:
        """List files with ripgrep in `workspace` directly, outside an agent run."""
        return await self._list_files(await self._scope(workspace), None, path, glob=glob)

    async def _list_files_tool(self, ctx: RunContext[AgentDepsT], path: str = '.', *, glob: str | None = None) -> str:
        """List files under a directory, recursively, sorted by path, respecting ignore files and skipping hidden files.

        Args:
            ctx: The current agent run context.
            path: Directory to list, relative to the working directory.
            glob: If provided, only list files matching this glob (e.g. '*.py' or 'src/**').

        Returns:
            One file path per line, relative to the working directory.
        """
        return await self._list_files(await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), path, glob=glob)

    @_recoverable
    async def _list_files(
        self, scope: _Scope, ctx: RunContext[AgentDepsT] | None, path: str = '.', *, glob: str | None
    ) -> str:
        if not supports_commands(scope.workspace):
            # Read-only and filesystem-only workspaces refuse `rg`; walk the files instead.
            pattern = '**' if glob is None else glob if '/' in glob else f'**/{glob}'
            return await self._find_files(scope, ctx, pattern, path=path, files_only=True)
        resolved = await self._safe_resolve(scope, path, check_allowed=False)
        entry = await self._stat(scope, resolved)
        if entry is None or not entry.is_dir:
            raise NotADirectoryError(f'Path {path!r} is not a directory.')
        prepare, permitted = self._batch_authorizer(scope, resolved)
        include_hidden = bool(glob and _explicit_hidden(glob))
        arguments = ['--files', '--sort', 'path', *(['--glob', glob] if glob is not None else [])]
        try:
            if scope.lacks_ripgrep:
                raise RipgrepMissing
            results, capped, unreadable = await run_ripgrep(
                scope.workspace,
                arguments,
                cwd=resolved,
                limit=self._max_find_results,
                listing=True,
                accept=lambda record: (
                    self._ripgrep_entry(scope, resolved, record, include_hidden=include_hidden)
                    if permitted(record)
                    else None
                ),
                prepare=prepare,
            )
        except RipgrepMissing:
            _fall_back_from_ripgrep(scope)

            async def accept(record: Record) -> str | None:
                if glob is not None and not fnmatch.fnmatch(posixpath.normpath(record.path), glob):
                    return None
                return await self._authorized_entry(
                    scope, resolved, record, include_hidden=include_hidden, permitted=permitted
                )

            results, capped, unreadable = await run_posix_search(
                scope.workspace,
                cwd=resolved,
                include_hidden=include_hidden,
                limit=self._max_find_results,
                accept=accept,
                prepare=prepare,
            )
        if ctx is not None:
            await ctx.emit(
                self._searched(scope, resolved, glob or '', search='find', match_count=len(results), truncated=capped)
            )
        if capped:
            results.append(f'[... truncated at {self._max_find_results} files]')
        return self._with_unreadable(scope, resolved, results or ['No files found.'], unreadable)

    async def grep(
        self,
        pattern: str,
        *,
        path: str = '.',
        glob: str | None = None,
        file_type: str | None = None,
        ignore_case: bool = False,
        literal: bool = False,
        context: int = 0,
        workspace: WorkspaceBackend,
    ) -> str:
        """Search file contents with ripgrep in `workspace` directly, outside an agent run."""
        return await self._grep(
            await self._scope(workspace),
            None,
            pattern,
            path=path,
            glob=glob,
            file_type=file_type,
            ignore_case=ignore_case,
            literal=literal,
            context=context,
        )

    async def _grep_tool(
        self,
        ctx: RunContext[AgentDepsT],
        pattern: str,
        *,
        path: str = '.',
        glob: str | None = None,
        file_type: str | None = None,
        ignore_case: bool = False,
        literal: bool = False,
        context: int = 0,
    ) -> str:
        """Search file contents with ripgrep, respecting ignore files and skipping hidden files.

        Args:
            ctx: The current agent run context.
            pattern: Regular expression (ripgrep syntax), or exact text when `literal` is set.
            path: Directory or file to search, relative to the working directory.
            glob: If provided, only search files matching this glob (e.g. '*.py').
            file_type: If provided, only search this ripgrep file type (e.g. 'py', 'rust').
            ignore_case: Match case-insensitively.
            literal: Treat `pattern` as exact text rather than a regular expression.
            context: Lines of context to show around each match (0 to 20).

        Returns:
            Matches as `file:line:text`; context lines as `file-line-text`. Paths are relative to the working directory.
        """
        return await self._grep(
            await self._scope(ctx.workspace),
            event_ctx(ctx, 'FileSystem'),
            pattern,
            path=path,
            glob=glob,
            file_type=file_type,
            ignore_case=ignore_case,
            literal=literal,
            context=context,
        )

    @_recoverable
    async def _grep(  # noqa: C901  -- ripgrep and POSIX fallback share output bookkeeping
        self,
        scope: _Scope,
        ctx: RunContext[AgentDepsT] | None,
        pattern: str,
        *,
        path: str,
        glob: str | None,
        file_type: str | None,
        ignore_case: bool,
        literal: bool,
        context: int,
    ) -> str:
        if not 0 <= context <= 20:
            raise ValueError('context must be between 0 and 20.')
        if not supports_commands(scope.workspace):
            # Read-only and filesystem-only workspaces refuse `rg`; read the files instead.
            if file_type is not None:
                raise ValueError(_FILE_TYPE_NEEDS_RIPGREP)
            if context:
                raise ValueError(
                    '`context` needs a search command, which this workspace cannot run, so context lines are '
                    'unavailable; search with `context=0`, then call `read_file` with an `offset` near the match.'
                )
            regex = re.escape(pattern) if literal else pattern
            return await self._search_files(
                scope, ctx, f'(?i){regex}' if ignore_case else regex, path=path, include_glob=glob
            )
        resolved = await self._safe_resolve(scope, path, check_allowed=False)
        entry = await self._stat(scope, resolved)
        if entry is None:
            raise FileNotFoundError(f'Path {path!r} is not a file or directory.')
        if entry.is_dir:
            cwd, target = resolved, '.'
        else:
            cwd, target = posixpath.dirname(resolved), posixpath.join('.', posixpath.basename(resolved))
        prepare, permitted = self._batch_authorizer(scope, cwd)
        include_hidden = bool(glob and _explicit_hidden(glob))
        arguments = [
            '--line-number',
            '--with-filename',
            '--sort',
            'path',
            '--max-columns',
            str(_MAX_MATCH_COLUMNS),
            '--max-columns-preview',
            '--context',
            str(context),
        ]
        if glob is not None:
            arguments.extend(['--glob', glob])
        if file_type is not None:
            arguments.extend(['--type', file_type])
        if ignore_case:
            arguments.append('--ignore-case')
        if literal:
            arguments.append('--fixed-strings')
        arguments.extend(['--regexp', pattern, '--', target])
        try:
            if scope.lacks_ripgrep:
                raise RipgrepMissing
            results, capped, unreadable = await run_ripgrep(
                scope.workspace,
                arguments,
                cwd=cwd,
                limit=self._max_search_results,
                accept=lambda record: (
                    self._match_line(scope, cwd, record, include_hidden=include_hidden) if permitted(record) else None
                ),
                prepare=prepare,
            )
        except RipgrepMissing:
            _fall_back_from_ripgrep(scope)
            if file_type is not None:
                raise ValueError(_FILE_TYPE_NEEDS_RIPGREP)

            async def accept(record: Record) -> str | None:
                if glob is not None and not fnmatch.fnmatch(posixpath.normpath(record.path), glob):
                    return None
                return await self._authorized_match(
                    scope, cwd, record, include_hidden=include_hidden, permitted=permitted
                )

            results, capped, unreadable = await run_posix_search(
                scope.workspace,
                cwd=cwd,
                target=target,
                explicit_file=not entry.is_dir,
                pattern=pattern,
                literal=literal,
                ignore_case=ignore_case,
                context=context,
                include_hidden=include_hidden,
                limit=self._max_search_results,
                accept=accept,
                prepare=prepare,
            )
        if ctx is not None:
            await ctx.emit(
                self._searched(scope, resolved, pattern, search='grep', match_count=len(results), truncated=capped)
            )
        if capped:
            results.append(f'[... truncated at {self._max_search_results} lines]')
        return self._with_unreadable(scope, cwd, results or ['No matches found.'], unreadable)

    def _batch_authorizer(
        self, scope: _Scope, cwd: str
    ) -> tuple[Callable[[list[Record]], Awaitable[None]], Callable[[Record], bool]]:
        allowed: set[str] = set()

        async def prepare(records: list[Record]) -> None:
            paths = list(dict.fromkeys(posixpath.normpath(posixpath.join(cwd, record.path)) for record in records))
            if not scope.checks_realpath:
                allowed.update(paths)
                return
            if all(record.real_path is not None for record in records):
                allowed.update(
                    posixpath.normpath(posixpath.join(cwd, record.path))
                    for record in records
                    if record.real_path is not None
                    and _contains(scope.root, record.real_path)
                    and self._is_accessible(posixpath.relpath(record.real_path, scope.root))
                )
                return
            # Resolve all candidates in the workspace in one command; never trust a lexical
            # path alone, since an in-root symlink can expose an out-of-root target.
            for offset in range(0, len(paths), 500):
                chunk = paths[offset : offset + 500]
                result = await scope.workspace.run(
                    'sh -c \'for file do real=$(realpath -- "$file" && printf .) || exit 2; '
                    'real=${real%.}; real=${real%?}; printf "%s\\\\0" "$real"; done\' sh ' + shlex.join(chunk),
                    shell=True,
                    timeout=120,
                )
                if result.exit_code != 0:
                    raise ModelRetry(f'Could not resolve search paths: {result.stderr.strip()}')
                realpaths = result.stdout.split('\0')
                if len(realpaths) != len(chunk) + 1 or realpaths[-1]:
                    raise ModelRetry('Could not resolve search paths.')
                allowed.update(
                    path
                    for path, real in zip(chunk, realpaths)
                    if _contains(scope.root, real) and self._is_accessible(posixpath.relpath(real, scope.root))
                )

        def permitted(record: Record) -> bool:
            return posixpath.normpath(posixpath.join(cwd, record.path)) in allowed

        return prepare, permitted

    async def _authorized_entry(
        self,
        scope: _Scope,
        cwd: str,
        record: Record,
        *,
        include_hidden: bool = False,
        permitted: Callable[[Record], bool] | None = None,
    ) -> str | None:
        path = posixpath.normpath(posixpath.join(cwd, record.path))
        if not (permitted(record) if permitted is not None else await self._readable_entry(scope, path)):
            return None
        return self._ripgrep_entry(scope, cwd, record, include_hidden=include_hidden)

    async def _authorized_match(
        self,
        scope: _Scope,
        cwd: str,
        record: Record,
        *,
        include_hidden: bool = False,
        permitted: Callable[[Record], bool] | None = None,
    ) -> str | None:
        if await self._authorized_entry(scope, cwd, record, include_hidden=include_hidden, permitted=permitted) is None:
            return None
        return self._match_line(scope, cwd, record, include_hidden=include_hidden)

    def _ripgrep_entry(self, scope: _Scope, cwd: str, record: Record, *, include_hidden: bool = False) -> str | None:
        """Authorize a path `rg` printed and return it relative to the working directory, or `None` to drop it.

        Hidden entries are kept only when the caller explicitly names a dotfile.
        """
        target = posixpath.normpath(posixpath.join(cwd, record.path))
        if self._walk_entry(scope, target, cwd, include_hidden=include_hidden) is None:
            return None
        return posixpath.relpath(target, scope.cwd)

    def _match_line(self, scope: _Scope, cwd: str, record: Record, *, include_hidden: bool = False) -> str | None:
        """Rebuild ripgrep's `path:line:text` (match) or `path-line-text` (context) line for an authorized path."""
        entry = self._ripgrep_entry(scope, cwd, record, include_hidden=include_hidden)
        if entry is None:
            return None
        digits = len(record.text) - len(record.text.lstrip('0123456789'))
        return f'{entry}{record.text[digits : digits + 1]}{record.text}'

    async def create_directory(self, path: str, *, workspace: WorkspaceBackend) -> str:
        """Create a directory in `workspace` directly, outside an agent run."""
        return await self._create_directory(await self._scope(workspace), None, path)

    async def _create_directory_tool(self, ctx: RunContext[AgentDepsT], path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            ctx: The current agent run context.
            path: Directory path relative to the working directory.

        Returns:
            Confirmation message.
        """
        return await self._create_directory(await self._scope(ctx.workspace), event_ctx(ctx, 'FileSystem'), path)

    async def _nearest_existing_is_dir(self, scope: _Scope, path: str) -> bool:
        """Whether the closest existing ancestor of `path` (or the filesystem root) is a directory."""
        current = posixpath.dirname(path)
        while True:
            try:
                return (await scope.workspace.stat(current)).is_dir
            except FileNotFoundError:
                parent = posixpath.dirname(current)
                if parent == current:  # pragma: no cover -- the workspace root always exists
                    return True
                current = parent
            except NotADirectoryError:
                return False

    @_recoverable
    async def _create_directory(self, scope: _Scope, ctx: RunContext[AgentDepsT] | None, path: str) -> str:
        resolved = await self._safe_resolve(scope, path, write=True)
        entry = await self._stat(scope, resolved)
        if entry is not None:
            if entry.is_dir:
                # Nothing changes, so there is nothing to announce or report.
                return f'Created directory: {path}'
            # The same collisions `make_dir` reports below, checked first so a
            # listener is only asked about a directory that can be created.
            raise ModelRetry(f'Path {path!r} exists and is not a directory.')
        if not await self._nearest_existing_is_dir(scope, resolved):
            raise ModelRetry(f'Path {path!r} has a parent that is not a directory.')
        change = Change.propose(**self._event_location(scope, resolved), operation='create_directory')
        if (refusal := await self._request(scope, ctx, change, path=path, resolved=resolved)) is not None:
            return refusal
        # The checks above already named these collisions; here they mean the
        # path changed while the change was announced. A directory that
        # appeared in the meantime was not created here, so it is not reported.
        if (appeared := await self._stat(scope, resolved)) is not None:
            if appeared.is_dir:
                return f'Created directory: {path}'
            raise ModelRetry(f'Path {path!r} exists and is not a directory.')
        try:
            await scope.workspace.make_dir(resolved)
        except FileExistsError as e:
            raise ModelRetry(f'Path {path!r} exists and is not a directory.') from e
        except NotADirectoryError as e:
            raise ModelRetry(f'Path {path!r} has a parent that is not a directory.') from e
        if ctx is not None:
            await ctx.emit(DirectoryCreatedEvent(**self._event_location(scope, resolved)))
        return f'Created directory: {path}'

    async def file_info(self, path: str, *, workspace: WorkspaceBackend) -> str:
        """Get metadata about a file or directory in `workspace` directly, outside an agent run."""
        return await self._file_info(await self._scope(workspace), path)

    async def _file_info_tool(self, ctx: RunContext[AgentDepsT], path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            ctx: The current agent run context.
            path: File or directory path relative to the working directory.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        return await self._file_info(await self._scope(ctx.workspace), path)

    @_recoverable
    async def _file_info(self, scope: _Scope, path: str) -> str:
        try:
            resolved = await self._safe_resolve(scope, path)
            entry = await self._stat(scope, resolved)
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return f'Path not found: {path}'
        if entry is None:
            return f'Path not found: {path}'

        kind = 'directory' if entry.is_dir else 'file'
        parts = [f'path: {path}', f'type: {kind}']
        if entry.size is not None:
            parts.append(f'size: {entry.size} bytes')

        if not entry.is_dir:
            try:
                raw = await scope.workspace.read_bytes(resolved)
            except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
                return f'Path not found: {path}'
            is_bin = _is_binary(raw)
            parts.append(f'binary: {is_bin}')
            if not is_bin:
                text = raw.decode('utf-8', errors='replace')
                parts.append(f'lines: {len(text.splitlines())}')
                parts.append(f'hash: {_bytes_hash(raw)}')

        if (target := await self._symlink_target(scope, resolved)) is not None:
            parts.append(f'symlink_target: {_model_safe_filename(target, scope.root)}')

        return '\n'.join(parts)

    async def _symlink_target(self, scope: _Scope, resolved: str) -> str | None:
        """What `resolved` points to when it is a symlink, or `None`.

        The workspace filesystem API follows symlinks and has no way to report
        one, so this asks `readlink` inside the workspace; a workspace that
        cannot run commands reports no symlink.
        """
        if not supports_commands(scope.workspace):
            return None
        result = await scope.workspace.run(['readlink', resolved], timeout=_READLINK_TIMEOUT)
        if result.exit_code != 0:
            return None
        return result.stdout.removesuffix('\n')
