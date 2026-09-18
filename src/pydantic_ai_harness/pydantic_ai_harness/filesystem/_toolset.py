"""Filesystem toolset providing sandboxed file operations."""

from __future__ import annotations

import codecs
import errno
import fnmatch
import functools
import hashlib
import itertools
import os
import re
import stat
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import BinaryIO, Concatenate, ParamSpec, TypedDict

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.filesystem._changes import Change
from pydantic_ai_harness.filesystem._events import (
    MAX_DIFF_SOURCE_CHARS,
    DirectoryCreatedEvent,
    DirectoryListedEvent,
    FileReadEvent,
    FilesSearchedEvent,
    FileWrittenEvent,
    SearchKind,
)
from pydantic_ai_harness.filesystem._ripgrep import Record, run_ripgrep

_P = ParamSpec('_P')

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
"""The tools `FileSystem` registers by default; all are pure Python."""

RIPGREP_TOOL_NAMES: tuple[str, ...] = ('list_files', 'grep')
"""Opt-in tools backed by the `rg` executable, which respects `.gitignore` and skips hidden files."""

FILE_SYSTEM_TOOL_NAMES: tuple[str, ...] = (*DEFAULT_TOOL_NAMES, *RIPGREP_TOOL_NAMES)
"""Every tool `FileSystem` can register, in registration order."""

_MAX_MATCH_COLUMNS = 4096
"""Bytes of a matching or context line `grep` shows before ripgrep cuts it with an omission marker."""

READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info', *RIPGREP_TOOL_NAMES}
)
"""Names of filesystem tools that do not modify the workspace."""


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

# The same idea one level down, for failures Python raises as a bare `OSError`
# with no dedicated subclass for `_RECOVERABLE_ERRORS` to name. Entries are
# explicit so other errors keep aborting the run; for example, retrying cannot
# fix `ENOSPC` or `EROFS`.
#
# Which operations reach these depends on the Python version. `Path.is_file`
# and friends stopped propagating `ENAMETOOLONG` in 3.14, so on 3.10 through
# 3.13 the read operations surface it too, not just the write path.
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
announced fails the descriptor check even when it is empty -- which the empty
file's own hash, and so `_disk_hash([b''])`, would have matched.
"""


class _EventLocation(TypedDict):
    """The `path` and `root_dir` fields shared by every filesystem event."""

    path: str
    root_dir: str


def _model_safe_filename(filename: str | bytes, real_root: Path) -> str:
    """Return the path relative to the workspace root.

    Paths not inside the root become `_OUTSIDE_WORKSPACE`; values that are
    not paths at all become `_NOT_A_PATH`.
    """
    try:
        raw = os.fsdecode(filename)
    except TypeError:
        return _NOT_A_PATH
    path = Path(raw)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(real_root).as_posix()
    except ValueError:
        pass
    try:
        return Path(os.path.realpath(path)).relative_to(real_root).as_posix()
    except (ValueError, OSError):
        return _OUTSIDE_WORKSPACE


def _sanitize_recoverable_error(error: BaseException, real_root: Path) -> str:
    """Render a recoverable error without exposing absolute host paths.

    Errors without an OS-supplied `filename` keep their original message.
    OS errors keep `errno` and `strerror`, with the path rewritten relative
    to `real_root` (see `_model_safe_filename` for the fallback placeholders).
    """
    if not isinstance(error, OSError) or error.filename is None:
        return str(error)

    filename = _model_safe_filename(error.filename, real_root)
    return f'[Errno {error.errno}] {error.strerror}: {filename!r}'


def _recoverable(
    fn: Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]]:
    """Surface model-correctable tool errors as `ModelRetry`."""

    @functools.wraps(fn)
    async def wrapper(self: FileSystemToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except _RECOVERABLE_ERRORS as e:
            real_root = self._real_root  # pyright: ignore[reportPrivateUsage]
            raise ModelRetry(_sanitize_recoverable_error(e, real_root)) from e
        except OSError as e:
            reason = _RECOVERABLE_ERRNOS.get(e.errno)
            if reason is None and getattr(e, 'winerror', None) == _WINDOWS_ERROR_INVALID_NAME:
                reason = 'The path name is invalid.'
            if reason is None:
                raise
            # The full error may embed the absolute host path; the reason is path-free.
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


def _content_hash(content: str) -> str:
    """Compute a short content hash for conflict detection.

    The hash is defined over the file's text as decoded from its bytes with
    no newline translation, the view `read_file` returns. Every tool that
    reports a hash (`read_file`, `write_file`, `edit_file`, `file_info`)
    computes it over that same view, so a hash from any of them identifies
    the same bytes on disk and the optimistic-concurrency handshake holds
    regardless of line endings.
    """
    return _bytes_hash(content.encode('utf-8'))


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


_HASH_CHUNK_BYTES = 1 << 20

_DIFF_SOURCE_BYTES = 4 * MAX_DIFF_SOURCE_CHARS
"""Bytes that can hold `MAX_DIFF_SOURCE_CHARS` of UTF-8; a longer file is past the bound whatever it holds."""

_SPECIAL_FILE_FLAGS = os.O_BINARY if os.name == 'nt' else os.O_NONBLOCK | os.O_NOFOLLOW
"""Open flags that keep a special file swapped onto a checked path from stalling or redirecting the open.

POSIX non-blocking mode stops a FIFO from waiting for the other end, and
O_NOFOLLOW stops a symlink swap from redirecting the descriptor. Windows has
neither hazard; O_BINARY keeps its text I/O from translating the bytes.
"""


def _chunks(source: BinaryIO) -> Iterator[bytes]:
    """`source` in `_HASH_CHUNK_BYTES` pieces, so hashing a file never holds it whole."""
    return iter(functools.partial(source.read, _HASH_CHUNK_BYTES), b'')


def _disk_hash(chunks: Iterable[bytes]) -> str:
    """The hash `read_file` reports for a file: of the bytes for a binary file, of the decoded text otherwise.

    Decoding is lenient, as `read_file` decodes, so a text file holding an
    invalid byte hashes to what the model was told and its `expected_hash`
    handshake holds. Every check against the disk uses this one rule. The
    first chunk decides whether the file is binary, so it must cover the
    `_is_binary` sample: a whole file, or at least that sample's 8192 bytes.
    """
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    binary = False
    for index, chunk in enumerate(chunks):
        if index == 0:
            binary = _is_binary(chunk)
        digest.update(chunk if binary else decoder.decode(chunk).encode('utf-8'))
    if not binary:
        digest.update(decoder.decode(b'', final=True).encode('utf-8'))
    return digest.hexdigest()[:12]


def _read_canonical_text(path: Path) -> str:
    """Read a text file as the canonical hash view, without newline translation.

    `Path.open` is used because `Path.read_text` only accepts `newline` on
    Python 3.13+, while `open` has had it since 3.10. Keep `errors` strict,
    matching `read_text`'s default, so invalid UTF-8 surfaces the same way
    it did before the `newline` handling was made explicit.
    """
    with path.open(encoding='utf-8', newline='') as f:
        return f.read()


def _announced_state(resolved: Path, path: str, *, expected_hash: str | None) -> tuple[str | None, str | None]:
    """The text a listener is shown a write replacing, and the hash the write is then guarded with.

    A stale `expected_hash` for a file that exists is rejected here first, so
    a listener only sees a write that would go ahead; for a missing file the
    hash is ignored, as documented, and the write is announced as a create.
    A new file diffs from empty and is guarded as absent, so one that appears
    while the write is announced is a conflict whether or not it is empty: the
    guard cannot be `_disk_hash([b''])`, which an empty file does hash to. The
    text is `None` when it cannot be shown: a file the process cannot read (a
    write-only mode, say) is announced as headers alone, marked as cut, and
    written unguarded, while an `expected_hash` it cannot check propagates the
    error; past `MAX_DIFF_SOURCE_CHARS` it would not be diffed, so only as
    many bytes as can hold that many characters are read into memory and the
    rest is hashed in chunks.
    """
    if not resolved.is_file():
        return '', _ABSENT_HASH
    try:
        # Opened like `_open_for_write`, so a special file swapped onto the
        # path after the `is_file` check cannot stall this read.
        with os.fdopen(os.open(resolved, os.O_RDONLY | _SPECIAL_FILE_FLAGS), 'rb') as source:
            head = source.read(_DIFF_SOURCE_BYTES + 1)
            current_hash = _disk_hash(itertools.chain([head], _chunks(source)))
    except OSError:
        if expected_hash is not None:
            raise
        return None, None
    if expected_hash is not None:
        _check_expected_hash(path, current_hash, expected_hash)
    old = head.decode('utf-8', errors='replace') if len(head) <= _DIFF_SOURCE_BYTES else None
    return old, current_hash


def _check_expected_hash(path: str, current_hash: str, expected_hash: str) -> None:
    """Reject a write or edit whose `expected_hash` no longer matches the file."""
    if current_hash != expected_hash:
        raise ValueError(
            f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
            f'got hash:{current_hash}). Re-read the file and retry.'
        )


def _nearest_existing(path: Path) -> Path:
    """The path itself or its closest ancestor that exists, or its anchor when nothing on the chain does."""
    # The anchor is its own parent, so an absent one (a Windows drive that
    # went away) would otherwise never end the walk.
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _open_for_write(resolved: Path, path: str, *, read_back: bool, create: bool) -> tuple[int, bool]:
    """Open `resolved` for writing without truncating it; returns the descriptor and whether it was created.

    Opening without O_TRUNC lets the caller classify the descriptor and check
    the expected hash before changing the file (`read_back` opens it
    read-write for that). An edit passes `create=False`: a file that vanished
    while its change was announced is reported missing, not recreated.
    `_SPECIAL_FILE_FLAGS` keeps a swapped-in special file from stalling or
    redirecting the open, and binary I/O on Windows means the written bytes
    are exactly the encoded content, so the reported hash matches the bytes a
    later `read_file` hashes.
    """
    access_flags = os.O_RDWR if read_back else os.O_WRONLY
    try:
        if not create:
            return os.open(resolved, access_flags | _SPECIAL_FILE_FLAGS), False
        # The target can disappear after O_EXCL reports that it exists. Retry
        # the complete atomic classification so an ordinary write still
        # recreates it, while bounding churn from a concurrently replaced path.
        for _ in range(3):
            try:
                descriptor = os.open(resolved, access_flags | _SPECIAL_FILE_FLAGS | os.O_CREAT | os.O_EXCL, 0o666)
            except FileExistsError:
                try:
                    return os.open(resolved, access_flags | _SPECIAL_FILE_FLAGS), False
                except FileNotFoundError:
                    continue
            return descriptor, True
        raise ModelRetry(f'Path {path!r} changed repeatedly while opening. Retry the write.')
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise ModelRetry(f'Path {path!r} encountered a symlink loop or changed to a symlink before opening.') from e
        if e.errno in (errno.EISDIR, errno.ENODEV, errno.ENXIO):
            raise ModelRetry(f'Path {path!r} exists and is not a regular file.') from e
        raise


def _write_content(resolved: Path, path: str, content: str, *, expected_hash: str | None, create: bool) -> None:
    """Replace the file's content, checking that an existing file hashes to `expected_hash` under the open descriptor.

    Checking under the descriptor is what guards the write: the file cannot
    change between the check and the write the way it can while a change is
    announced. The bytes are hashed as `read_file` hashes them, so a file that
    is not valid UTF-8 is compared instead of failing to decode.
    """
    descriptor, created = _open_for_write(resolved, path, read_back=expected_hash is not None, create=create)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ModelRetry(f'Path {path!r} exists and is not a regular file.')

        binary_file = os.fdopen(descriptor, 'rb+' if expected_hash is not None else 'wb')
        descriptor = -1
        with binary_file:
            if expected_hash is not None and not created:
                _check_expected_hash(path, _disk_hash(_chunks(binary_file)), expected_hash)

            binary_file.seek(0)
            binary_file.truncate(0)
            binary_file.write(content.encode('utf-8'))
    finally:
        if descriptor >= 0:
            os.close(descriptor)


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


class FileSystemToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing filesystem operations scoped to a root directory.

    Security model:
    - Relative paths resolved from `cwd` and checked for containment in `root_dir`
    - Symlinks resolved before authorization, and again after a listener has
      held a change request; a rename on the path between that check and the
      by-name I/O remains possible, as the documented security model says
    - Glob-based allow/deny filtering
    - Protected path patterns (e.g. `.git/`, `.env`)
    - Binary file detection blocks text operations
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        allowed_patterns: Sequence[str],
        denied_patterns: Sequence[str],
        protected_patterns: Sequence[str],
        max_read_lines: int,
        max_read_chars: int | None = None,
        max_list_results: int,
        max_search_results: int,
        max_find_results: int,
        id: str | None = None,
        cwd: Path | None = None,
        content_hashes: bool = True,
        tools: Sequence[str] = DEFAULT_TOOL_NAMES,
    ) -> None:
        super().__init__(id=id)
        self._root = root_dir.resolve()
        self._real_root = Path(os.path.realpath(self._root))
        self._cwd = self._root if cwd is None else cwd.resolve()
        if not Path(os.path.realpath(self._cwd)).is_relative_to(self._real_root):
            raise ValueError(f'cwd {os.fspath(self._cwd)!r} is outside root_dir {os.fspath(root_dir)!r}.')
        self._allowed_patterns = list(allowed_patterns)
        self._denied_patterns = list(denied_patterns)
        self._protected_patterns = list(protected_patterns)
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
            'file_info': self.file_info,
            'list_files': self._list_files_tool,
            'grep': self._grep_tool,
        }
        for name in FILE_SYSTEM_TOOL_NAMES:
            if name in self._tools:
                self.add_function(registrations[name], name=name)

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

    def _first_matching_pattern(self, path: str, patterns: list[str]) -> str | None:
        """Return the first pattern that matches path, or None."""
        return next((p for p in patterns if self._matches(path, p)), None)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to `cwd`, rejecting traversal outside the root.

        Uses os.path.realpath for symlink resolution before checking containment.
        """
        try:
            candidate = (self._cwd / path).resolve()
        except RuntimeError as e:
            # Python 3.10-3.12 signal a symlink loop this way.
            raise ModelRetry(f'Path {path!r} resolves through a symlink loop.') from e

        if not candidate.exists():
            try:
                candidate.stat()
            except OSError as e:
                # Python 3.13+ suppresses `ELOOP` in `resolve` and `exists`, so
                # probe the path before treating it as missing.
                if e.errno == errno.ELOOP:
                    raise ModelRetry(f'Path {path!r} resolves through a symlink loop.') from e
        real = Path(os.path.realpath(candidate))
        if not real.is_relative_to(self._real_root):
            raise PermissionError(f'Path {path!r} resolves outside the root directory.')

        return real

    def _check_access(self, path: str, *, write: bool = False, check_allowed: bool = True) -> None:
        """Validate path against allow/deny/protected patterns.

        `check_allowed=False` skips the `allowed_patterns` gate. Walkers
        (`list_directory`, `search_files`, `find_files`) pass it so their root
        directory isn't required to match `allowed_patterns` itself -- `.` or
        `src` would never match a file pattern like `src/*.py`. The walk's
        entries are still filtered against `allowed_patterns` per-entry via
        `_resolve_walk_entry`. Denied patterns continue to gate the root.
        """
        if write and self._protected_patterns:
            matched = self._first_matching_pattern(path, self._protected_patterns)
            if matched:
                raise PermissionError(f'Path {path!r} is protected (matches {matched!r}).')

        if self._denied_patterns:
            matched = self._first_matching_pattern(path, self._denied_patterns)
            if matched:
                raise PermissionError(f'Path {path!r} is denied by pattern {matched!r}.')

        if check_allowed and self._allowed_patterns:
            if not any(self._matches(path, p) for p in self._allowed_patterns):
                raise PermissionError(f'Path {path!r} does not match any allowed pattern.')

    def _is_accessible(self, path: str) -> bool:
        """Predicate form of the read-level `_check_access` checks.

        Protected patterns are not consulted: they gate writes, and the walkers
        only read.
        """
        if self._denied_patterns:
            if self._first_matching_pattern(path, self._denied_patterns) is not None:
                return False
        if self._allowed_patterns and not any(self._matches(path, p) for p in self._allowed_patterns):
            return False
        return True

    def _resolve_walk_entry(self, entry: Path) -> Path | None:
        """Authorize one entry of a directory walk, or return `None` to skip it.

        Callers must do their I/O on the returned path. Resolving once means the
        path that was authorized is the path that gets read, and matching the
        patterns against the resolved location keeps the walkers in step with
        direct access: a symlink can neither escape the root nor alias a file
        past a rule its own name would trip.
        """
        target = Path(os.path.realpath(entry))
        if not target.is_relative_to(self._real_root):
            return None
        if not self._is_accessible(self._relative_to_root(target)):
            return None
        return target

    def _relative_to_root(self, resolved: Path) -> str:
        """Canonical path of a resolved location relative to the real root."""
        return str(resolved.relative_to(self._real_root))

    def _relative_to_cwd(self, path: Path) -> str:
        """Format an authorized discovery path for reuse as a tool input."""
        return os.path.relpath(path, self._cwd)

    def _event_location(self, resolved: Path) -> _EventLocation:
        """Path fields for an event about `resolved`.

        `path` is relative to the real root and `root_dir` is that root, so a
        subscriber rooted elsewhere can rebuild the absolute location instead
        of assuming the event came from its own root.
        """
        return _EventLocation(
            path=_model_safe_filename(os.fspath(resolved), self._real_root),
            root_dir=os.fspath(self._real_root),
        )

    def _safe_resolve(self, path: str, *, write: bool = False, check_allowed: bool = True) -> Path:
        """Resolve and access-check a path in one step.

        Resolution happens first so the access check matches patterns against
        the canonical path relative to the root, collapsing `.`/`..`/`//`
        segments that would otherwise slip past a literal pattern (e.g.
        `config/./secret.txt` evading a `config/secret.txt` deny rule).
        """
        resolved = self._resolve_path(path)
        self._check_access(self._relative_to_root(resolved), write=write, check_allowed=check_allowed)
        return resolved

    async def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> str:
        """Read a text file directly, outside an agent run."""
        return await self._read_file(None, path, offset=offset, limit=limit)

    async def _read_file_tool(
        self, ctx: RunContext[AgentDepsT], path: str, *, offset: int = 0, limit: int | None = None
    ) -> str:
        """Read a text file with line numbers.

        Args:
            ctx: The current agent run context.
            path: File path relative to `cwd`.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            File content with line numbers, plus metadata header.
        """
        return await self._read_file(ctx, path, offset=offset, limit=limit)

    @_recoverable
    async def _read_file(
        self, ctx: RunContext[AgentDepsT] | None, path: str, *, offset: int = 0, limit: int | None = None
    ) -> str:
        if limit is None:
            limit = self._max_read_lines
        resolved = self._safe_resolve(path)
        if not resolved.is_file():
            if resolved.is_dir():
                raise FileNotFoundError(f"'{path}' is a directory, not a file.")
            raise FileNotFoundError(f'File not found: {path}')

        raw = resolved.read_bytes()
        content_hash = _disk_hash([raw])
        if _is_binary(raw):
            size = len(raw)
            if ctx is not None:
                await ctx.emit(FileReadEvent(**self._event_location(resolved), content_hash=content_hash))
            return f'[Binary file: {size} bytes. Use a binary-aware tool to inspect.]'

        text = raw.decode('utf-8', errors='replace')
        lines = text.splitlines(keepends=True)
        header = self._read_header(path, len(lines), content_hash)
        # Format before emitting: an out-of-range offset is a failed read, and
        # a failed read must not look like a successful one to subscribers.
        body = _format_lines(lines, offset, limit, self._body_budget(header))
        if ctx is not None:
            await ctx.emit(FileReadEvent(**self._event_location(resolved), content_hash=content_hash))
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

    async def write_file(self, path: str, content: str, *, expected_hash: str | None = None) -> str:
        """Write a text file directly, outside an agent run."""
        return await self._write_file(None, path, content, expected_hash=expected_hash)

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
            path: File path relative to `cwd`.
            content: The text content to write.
            expected_hash: If provided, the write is rejected when the file exists
                and its current hash doesn't match (optimistic concurrency).

        Returns:
            Confirmation message with new hash.
        """
        return await self._write_file(ctx, path, content, expected_hash=expected_hash)

    async def _write_file_tool_unhashed(self, ctx: RunContext[AgentDepsT], path: str, content: str) -> str:
        """Create a file or replace its whole content.

        Args:
            ctx: The current agent run context.
            path: File path relative to `cwd`.
            content: The text content to write.
        """
        return await self._write_file(ctx, path, content)

    @_recoverable
    async def _write_file(
        self,
        ctx: RunContext[AgentDepsT] | None,
        path: str,
        content: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        resolved = self._safe_resolve(path, write=True)

        if resolved.exists() and not resolved.is_file():
            raise ModelRetry(f'Path {path!r} exists and is not a regular file.')

        if not resolved.parent.exists():
            parent_rel = str(resolved.parent.relative_to(self._root))
            raise FileNotFoundError(f"Parent directory '{parent_rel}' does not exist. Use create_directory first.")
        # Checked before the announcement, like `create_directory` does, so a
        # listener is only asked about a write the filesystem would accept.
        if not resolved.parent.is_dir():
            raise ModelRetry(f'Path {path!r} has a parent that is not a directory.')

        # `O_CREAT` in `_open_for_write` would already have created the file
        # the listener is about to refuse, so the request cannot wait for the
        # descriptor. Outside a run nothing is read: the diff is for listeners,
        # and the descriptor check of `expected_hash` is the whole contract.
        guard = expected_hash
        if ctx is not None:
            old, guard = _announced_state(resolved, path, expected_hash=expected_hash)
            change = Change.propose(**self._event_location(resolved), operation='write', old=old, new=content)
            if (refusal := await self._request(ctx, change, path=path, resolved=resolved)) is not None:
                return refusal

        _write_content(resolved, path, content, expected_hash=guard, create=True)

        new_hash = _content_hash(content)
        lines = len(content.splitlines())
        if ctx is not None:
            await ctx.emit(FileWrittenEvent(**self._event_location(resolved), content_hash=new_hash))
        return f'Wrote {len(content)} chars ({lines} lines) to {path}.{self._hash_suffix(new_hash)}'

    async def _request(
        self, ctx: RunContext[AgentDepsT] | None, change: Change, *, path: str, resolved: Path
    ) -> str | None:
        """Announce `change` and confirm that `path` still names `resolved` afterwards.

        A listener can hold the request for as long as a human takes, and the
        change is then applied by name, so the path is resolved and checked
        again: the announcement must not widen the window between the
        containment check and the I/O. Outside a run there is nobody to ask.
        """
        if ctx is None:
            return None
        if (refusal := await change.request(ctx)) is not None:
            return refusal
        if self._safe_resolve(path, write=True) != resolved:
            raise ModelRetry(f'Path {path!r} was replaced while the change was announced. Retry.')
        return None

    async def edit_file(self, path: str, old_text: str, new_text: str, *, expected_hash: str | None = None) -> str:
        """Edit a text file directly, outside an agent run."""
        replacements = [Replacement(old_text=old_text, new_text=new_text)]
        return await self._edit_file(None, path, replacements, expected_hash=expected_hash)

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
            path: File path relative to `cwd`.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            replacements: Replacements to apply in order, instead of a single pair.
            expected_hash: If provided, rejects the edit when the file's
                current hash doesn't match (optimistic concurrency).

        Returns:
            Summary with new hash for subsequent operations.
        """
        edits = _replacements(old_text, new_text, replacements)
        return await self._edit_file(ctx, path, edits, expected_hash=expected_hash)

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
            path: File path relative to `cwd`.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            replacements: Replacements to apply in order, instead of a single pair.
        """
        return await self._edit_file(ctx, path, _replacements(old_text, new_text, replacements))

    @_recoverable
    async def _edit_file(
        self,
        ctx: RunContext[AgentDepsT] | None,
        path: str,
        replacements: Sequence[Replacement],
        *,
        expected_hash: str | None = None,
    ) -> str:
        resolved = self._safe_resolve(path, write=True)
        if not resolved.is_file():
            raise FileNotFoundError(f'File not found: {path}')
        with resolved.open('rb') as f:
            if _is_binary(f.read(8192)):
                raise ValueError(f'{path} is a binary file; edit_file only edits text files.')

        # Reading and writing with `newline=''` disables universal-newline
        # translation, so the text is the canonical bytes-on-disk view that
        # `read_file` hashes, and the replacement preserves `\r\n` exactly
        # instead of writing `\r\r\n` through a translating writer on Windows.
        text = _read_canonical_text(resolved)
        current_hash = _content_hash(text)

        if expected_hash is not None:
            _check_expected_hash(path, current_hash, expected_hash)

        new_content = _apply_replacements(text, replacements, path)
        change = Change.propose(**self._event_location(resolved), operation='edit', old=text, new=new_content)
        if (refusal := await self._request(ctx, change, path=path, resolved=resolved)) is not None:
            return refusal
        # A listener may take a while (a human approving the diff, say). The
        # edit was computed from `text`, so the write checks that the file
        # still holds it, and reports a file deleted in the meantime as missing.
        _write_content(resolved, path, new_content, expected_hash=current_hash, create=False)
        new_hash = _content_hash(new_content)
        if ctx is not None:
            await ctx.emit(change.edited(content_hash=new_hash))
        return f'Edited {path}.{self._hash_suffix(new_hash)}'

    async def list_directory(self, path: str = '.') -> str:
        """List a directory directly, outside an agent run."""
        return await self._list_directory(None, path)

    async def _list_directory_tool(self, ctx: RunContext[AgentDepsT], path: str = '.') -> str:
        """List the contents of a directory.

        Args:
            ctx: The current agent run context.
            path: Directory path relative to `cwd`.

        Returns:
            Paths relative to `cwd`, with type indicators and sizes.
        """
        return await self._list_directory(ctx, path)

    @_recoverable
    async def _list_directory(self, ctx: RunContext[AgentDepsT] | None, path: str = '.') -> str:
        # The listing root is gated by denied patterns but not by
        # allowed_patterns: a directory like '.' never matches a file pattern.
        # Entries are filtered per-entry against allowed_patterns below.
        resolved = self._safe_resolve(path, check_allowed=False)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Not a directory: {path}')

        entries: list[str] = []
        entry_count = 0
        for entry in sorted(resolved.iterdir()):
            try:
                rel_path = entry.relative_to(self._real_root)
            except ValueError:  # pragma: no cover
                continue
            # Skip dotfiles and dot-directories, matching search_files and
            # find_files so the three walkers agree on what exists.
            if any(part.startswith('.') for part in rel_path.parts):
                continue
            target = self._resolve_walk_entry(entry)
            if target is None:
                continue
            rel = self._relative_to_cwd(entry)
            if target.is_dir():
                line = f'{rel}/'
            else:
                try:
                    size = target.stat().st_size
                except OSError:
                    # A dangling symlink, or an entry deleted mid-walk: it has
                    # no size to report, so leave it out of the listing.
                    continue
                line = f'{rel}  ({size} bytes)'
            # Only a listing that actually dropped an entry is marked truncated,
            # so one that merely fills the cap reads as complete.
            if len(entries) >= self._max_list_results:
                entries.append(f'[... truncated at {self._max_list_results} entries]')
                break
            entries.append(line)
            entry_count += 1
        if ctx is not None:
            await ctx.emit(DirectoryListedEvent(**self._event_location(resolved), entry_count=entry_count))
        return '\n'.join(entries) if entries else '(empty directory)'

    async def search_files(self, pattern: str, *, path: str = '.', include_glob: str | None = None) -> str:
        """Search file contents directly, outside an agent run."""
        return await self._search_files(None, pattern, path=path, include_glob=include_glob)

    async def _search_files_tool(
        self, ctx: RunContext[AgentDepsT], pattern: str, *, path: str = '.', include_glob: str | None = None
    ) -> str:
        """Search file contents using a regular expression.

        Args:
            ctx: The current agent run context.
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to `cwd`.
            include_glob: If provided, match this glob against root-relative paths (e.g. '*.py').

        Returns:
            str: Matching lines formatted as file:line_number:text, with paths relative to `cwd`.
        """
        return await self._search_files(ctx, pattern, path=path, include_glob=include_glob)

    @_recoverable
    async def _search_files(
        self,
        ctx: RunContext[AgentDepsT] | None,
        pattern: str,
        *,
        path: str = '.',
        include_glob: str | None = None,
    ) -> str:
        # See list_directory: the search root isn't gated by allowed_patterns;
        # matched files are filtered per-entry below.
        resolved = self._safe_resolve(path, check_allowed=False)
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern: {e}') from e

        results: list[str] = []
        capped = False

        if resolved.is_file():
            files = [resolved]
        else:
            files = sorted(resolved.rglob('*'))

        for file_path in files:
            try:
                rel_path = file_path.relative_to(self._real_root)
            except ValueError:  # pragma: no cover
                continue
            if any(part.startswith('.') for part in rel_path.parts):
                continue
            rel_str = str(rel_path)
            if include_glob and not fnmatch.fnmatch(rel_str, include_glob):
                continue
            target = self._resolve_walk_entry(file_path)
            if target is None:
                continue
            if not target.is_file():
                continue
            try:
                raw = target.read_bytes()
            except OSError:  # pragma: no cover
                continue
            if _is_binary(raw):
                continue
            text = raw.decode('utf-8', errors='replace')
            matches, capped = _matching_lines(
                text, compiled, self._relative_to_cwd(file_path), self._max_search_results - len(results)
            )
            results.extend(matches)
            if capped:
                break

        if ctx is not None:
            await ctx.emit(self._searched(resolved, pattern, search='grep', match_count=len(results), truncated=capped))
        if capped:
            results.append(f'[... truncated at {self._max_search_results} matches]')
        return '\n'.join(results) if results else 'No matches found.'

    def _searched(
        self, resolved: Path, pattern: str, *, search: SearchKind, match_count: int, truncated: bool
    ) -> FilesSearchedEvent:
        return FilesSearchedEvent(
            **self._event_location(resolved),
            pattern=pattern,
            search=search,
            match_count=match_count,
            truncated=truncated,
        )

    async def find_files(self, pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern directly, outside an agent run."""
        return await self._find_files(None, pattern, path=path)

    async def _find_files_tool(self, ctx: RunContext[AgentDepsT], pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            ctx: The current agent run context.
            pattern: Glob pattern to match, relative to `path` (e.g. '*.py',
                '**/*.json'). Absolute patterns are rejected.
            path: Directory to search in, relative to `cwd`.

        Returns:
            Newline-separated list of matching file paths relative to `cwd`.
        """
        return await self._find_files(ctx, pattern, path=path)

    @_recoverable
    async def _find_files(self, ctx: RunContext[AgentDepsT] | None, pattern: str, *, path: str = '.') -> str:
        if os.path.isabs(pattern):
            raise ValueError(f'Pattern {pattern!r} must be relative to the search path, not absolute.')

        # See list_directory: the find root isn't gated by allowed_patterns;
        # matched entries are filtered per-entry below.
        resolved = self._safe_resolve(path, check_allowed=False)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Not a directory: {path}')

        try:
            found = sorted(resolved.glob(pattern))
        except NotImplementedError as e:
            # The `isabs` guard above takes a rooted pattern first on POSIX. On
            # Windows it does not: since 3.13 `os.path.isabs` reports a single
            # leading slash as relative, so `/etc/*.conf` reaches `glob`, which
            # rejects any rooted pattern. `NotImplementedError` is not an
            # `OSError`, so neither the recoverable tuple nor the errno table
            # can reach it.
            raise ModelRetry(f'Pattern {pattern!r} must be relative to {path!r}, not an absolute path.') from e
        except IndexError as e:
            # Python 3.10 through 3.12 raise this for a pattern whose last
            # component is a bare `.`. On 3.13+ the same pattern raises
            # `ValueError`, which the recoverable tuple already covers.
            raise ModelRetry(f'Pattern {pattern!r} is not a valid glob pattern.') from e

        matches: list[str] = []
        capped = False
        for match in found:
            try:
                rel_path = match.relative_to(self._real_root)
            except ValueError:  # pragma: no cover
                continue
            if any(part.startswith('.') for part in rel_path.parts):
                continue
            target = self._resolve_walk_entry(match)
            if target is None:
                continue
            if not target.exists():
                # A dangling symlink resolves inside the root but names nothing.
                continue
            if len(matches) >= self._max_find_results:
                capped = True
                break
            rel = self._relative_to_cwd(match)
            suffix = '/' if target.is_dir() else ''
            matches.append(f'{rel}{suffix}')

        if ctx is not None:
            await ctx.emit(self._searched(resolved, pattern, search='find', match_count=len(matches), truncated=capped))
        if capped:
            matches.append(f'[... truncated at {self._max_find_results} matches]')
        return '\n'.join(matches) if matches else 'No matches found.'

    async def list_files(self, path: str = '.', *, glob: str | None = None) -> str:
        """List files with ripgrep directly, outside an agent run."""
        return await self._list_files(None, path, glob=glob)

    async def _list_files_tool(self, ctx: RunContext[AgentDepsT], path: str = '.', *, glob: str | None = None) -> str:
        """List files under a directory, recursively, sorted by path, respecting ignore files and skipping hidden files.

        Args:
            ctx: The current agent run context.
            path: Directory to list, relative to `cwd`.
            glob: If provided, only list files matching this glob (e.g. '*.py' or 'src/**').

        Returns:
            One file path per line, relative to `cwd`.
        """
        return await self._list_files(ctx, path, glob=glob)

    @_recoverable
    async def _list_files(self, ctx: RunContext[AgentDepsT] | None, path: str = '.', *, glob: str | None) -> str:
        resolved = self._safe_resolve(path, check_allowed=False)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Path {path!r} is not a directory.')
        arguments = ['--files', '--sort', 'path', *(['--glob', glob] if glob is not None else [])]
        results, capped = await run_ripgrep(
            arguments,
            cwd=resolved,
            limit=self._max_find_results,
            listing=True,
            accept=lambda record: self._ripgrep_entry(resolved, record),
        )
        if ctx is not None:
            await ctx.emit(
                self._searched(resolved, glob or '', search='find', match_count=len(results), truncated=capped)
            )
        if capped:
            results.append(f'[... truncated at {self._max_find_results} files]')
        return '\n'.join(results) if results else 'No files found.'

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
    ) -> str:
        """Search file contents with ripgrep directly, outside an agent run."""
        return await self._grep(
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
            path: Directory or file to search, relative to `cwd`.
            glob: If provided, only search files matching this glob (e.g. '*.py').
            file_type: If provided, only search this ripgrep file type (e.g. 'py', 'rust').
            ignore_case: Match case-insensitively.
            literal: Treat `pattern` as exact text rather than a regular expression.
            context: Lines of context to show around each match (0 to 20).

        Returns:
            Matches as `file:line:text`; context lines as `file-line-text`. Paths are relative to `cwd`.
        """
        return await self._grep(
            ctx,
            pattern,
            path=path,
            glob=glob,
            file_type=file_type,
            ignore_case=ignore_case,
            literal=literal,
            context=context,
        )

    @_recoverable
    async def _grep(
        self,
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
        resolved = self._safe_resolve(path, check_allowed=False)
        if resolved.is_dir():
            cwd, target = resolved, '.'
        elif resolved.is_file():
            cwd, target = resolved.parent, os.path.join('.', resolved.name)
        else:
            raise FileNotFoundError(f'Path {path!r} is not a file or directory.')
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
        results, capped = await run_ripgrep(
            arguments,
            cwd=cwd,
            limit=self._max_search_results,
            accept=lambda record: self._match_line(cwd, record),
        )
        if ctx is not None:
            await ctx.emit(self._searched(resolved, pattern, search='grep', match_count=len(results), truncated=capped))
        if capped:
            results.append(f'[... truncated at {self._max_search_results} lines]')
        return '\n'.join(results) if results else 'No matches found.'

    def _ripgrep_entry(self, cwd: Path, record: Record) -> str | None:
        """Authorize a path `rg` printed and return it relative to the configured `cwd`, or `None` to drop it.

        A `glob` makes ripgrep surface hidden files it would otherwise skip;
        dropping dot-prefixed entries here keeps these walkers in step with the
        pure-Python ones, which skip dotfiles regardless of patterns.
        """
        if any(part.startswith('.') for part in Path(record.path).parts if part != '.'):
            return None
        target = self._resolve_walk_entry(cwd / record.path)
        return None if target is None else self._relative_to_cwd(target)

    def _match_line(self, cwd: Path, record: Record) -> str | None:
        """Rebuild ripgrep's `path:line:text` (match) or `path-line-text` (context) line for an authorized path."""
        entry = self._ripgrep_entry(cwd, record)
        if entry is None:
            return None
        digits = len(record.text) - len(record.text.lstrip('0123456789'))
        return f'{entry}{record.text[digits : digits + 1]}{record.text}'

    async def create_directory(self, path: str) -> str:
        """Create a directory directly, outside an agent run."""
        return await self._create_directory(None, path)

    async def _create_directory_tool(self, ctx: RunContext[AgentDepsT], path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            ctx: The current agent run context.
            path: Directory path relative to `cwd`.

        Returns:
            Confirmation message.
        """
        return await self._create_directory(ctx, path)

    @_recoverable
    async def _create_directory(self, ctx: RunContext[AgentDepsT] | None, path: str) -> str:
        resolved = self._safe_resolve(path, write=True)
        if resolved.is_dir():
            # Nothing changes, so there is nothing to announce or report.
            return f'Created directory: {path}'
        # The same collisions `mkdir` reports below, checked first so a listener
        # is only asked about a directory that can be created.
        if resolved.exists():
            raise ModelRetry(f'Path {path!r} exists and is not a directory.')
        if not _nearest_existing(resolved.parent).is_dir():
            raise ModelRetry(f'Path {path!r} has a parent that is not a directory.')
        change = Change.propose(**self._event_location(resolved), operation='create_directory')
        if (refusal := await self._request(ctx, change, path=path, resolved=resolved)) is not None:
            return refusal
        # The checks above already named these collisions; here they mean the
        # path changed while the change was announced. A directory that
        # appeared in the meantime was not created here, so it is not reported.
        try:
            resolved.mkdir(parents=True)
        except FileExistsError as e:
            if resolved.is_dir():
                return f'Created directory: {path}'
            raise ModelRetry(f'Path {path!r} exists and is not a directory.') from e
        except NotADirectoryError as e:
            raise ModelRetry(f'Path {path!r} has a parent that is not a directory.') from e
        if ctx is not None:
            await ctx.emit(DirectoryCreatedEvent(**self._event_location(resolved)))
        return f'Created directory: {path}'

    @_recoverable
    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to `cwd`.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        resolved = self._safe_resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f'Path not found: {path}')

        # Check if the original (pre-resolve) path is a symlink
        original = self._cwd / path
        is_link = original.is_symlink()

        stat = resolved.stat()
        kind = 'directory' if resolved.is_dir() else 'file'
        size = stat.st_size

        parts = [f'path: {path}', f'type: {kind}', f'size: {size} bytes']

        if resolved.is_file():
            raw = resolved.read_bytes()
            is_bin = _is_binary(raw)
            parts.append(f'binary: {is_bin}')
            if not is_bin:
                text = raw.decode('utf-8', errors='replace')
                parts.append(f'lines: {len(text.splitlines())}')
                parts.append(f'hash: {_disk_hash([raw])}')

        if is_link:
            target = _model_safe_filename(os.readlink(original), self._real_root)
            parts.append(f'symlink_target: {target}')

        return '\n'.join(parts)
