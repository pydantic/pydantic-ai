"""Filesystem toolset providing sandboxed file operations."""

from __future__ import annotations

import errno
import fnmatch
import functools
import hashlib
import os
import re
import stat
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Concatenate, ParamSpec

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

_P = ParamSpec('_P')

READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'}
)
"""Names of filesystem tools that do not modify the workspace."""

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


def _format_lines(lines: Sequence[str], offset: int, limit: int) -> str:
    """Format pre-split lines with line numbers and continuation hint."""
    total = len(lines)

    if total == 0:
        return '(empty file)\n'

    if offset >= total:
        raise ValueError(f'Offset {offset} exceeds file length ({total} lines).')

    selected = lines[offset : offset + limit]
    numbered = [f'{i:>6}\t{line}' for i, line in enumerate(selected, start=offset + 1)]
    result = ''.join(numbered)
    if not result.endswith('\n'):
        result += '\n'

    remaining = total - (offset + len(selected))
    if remaining > 0:
        next_offset = offset + len(selected)
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
    """Compute a short content hash for conflict detection."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]


class FileSystemToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing filesystem operations scoped to a root directory.

    Security model:
    - All paths resolved relative to root with canonical path checks
    - Symlinks resolved before authorization (prevents TOCTTOU)
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
        max_list_results: int,
        max_search_results: int,
        max_find_results: int,
    ) -> None:
        super().__init__()
        self._root = root_dir.resolve()
        self._real_root = Path(os.path.realpath(self._root))
        self._allowed_patterns = list(allowed_patterns)
        self._denied_patterns = list(denied_patterns)
        self._protected_patterns = list(protected_patterns)
        self._max_read_lines = max_read_lines
        self._max_list_results = max_list_results
        self._max_search_results = max_search_results
        self._max_find_results = max_find_results

        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.edit_file, name='edit_file')
        self.add_function(self.list_directory, name='list_directory')
        self.add_function(self.search_files, name='search_files')
        self.add_function(self.find_files, name='find_files')
        self.add_function(self.create_directory, name='create_directory')
        self.add_function(self.file_info, name='file_info')

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
        """Resolve path relative to root, rejecting traversal.

        Uses os.path.realpath for symlink resolution before checking containment.
        """
        try:
            candidate = (self._root / path).resolve()
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

    @_recoverable
    async def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path relative to the root directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            File content with line numbers, plus metadata header.
        """
        if limit is None:
            limit = self._max_read_lines
        resolved = self._safe_resolve(path)
        if not resolved.is_file():
            if resolved.is_dir():
                raise FileNotFoundError(f"'{path}' is a directory, not a file.")
            raise FileNotFoundError(f'File not found: {path}')

        raw = resolved.read_bytes()
        if _is_binary(raw):
            size = len(raw)
            return f'[Binary file: {size} bytes. Use a binary-aware tool to inspect.]'

        text = raw.decode('utf-8', errors='replace')
        lines = text.splitlines(keepends=True)
        content_hash = _content_hash(text)

        header = f'[{path} | {len(lines)} lines | hash:{content_hash}]\n'
        return header + _format_lines(lines, offset, limit)

    @_recoverable
    async def write_file(self, path: str, content: str, *, expected_hash: str | None = None) -> str:
        """Create or overwrite a file with conflict detection.

        Args:
            path: File path relative to the root directory.
            content: The text content to write.
            expected_hash: If provided, the write is rejected when the file exists
                and its current hash doesn't match (optimistic concurrency).

        Returns:
            Confirmation message with new hash.
        """
        resolved = self._safe_resolve(path, write=True)

        if resolved.exists() and not resolved.is_file():
            raise ModelRetry(f'Path {path!r} exists and is not a regular file.')

        if not resolved.parent.exists():
            parent_rel = str(resolved.parent.relative_to(self._root))
            raise FileNotFoundError(f"Parent directory '{parent_rel}' does not exist. Use create_directory first.")

        # Opening without O_TRUNC lets us classify the descriptor and check the
        # expected hash before changing the file. POSIX non-blocking mode keeps
        # a FIFO swapped into place from waiting for a reader; O_NOFOLLOW keeps
        # a final-component symlink swap from redirecting the descriptor. Windows
        # has no filesystem FIFO equivalent, and O_BINARY leaves newline handling
        # to the text wrapper just as Path.write_text does.
        platform_flags = os.O_BINARY if os.name == 'nt' else os.O_NONBLOCK | os.O_NOFOLLOW
        access_flags = os.O_RDWR if expected_hash is not None else os.O_WRONLY
        created = False
        descriptor = -1
        try:
            # The target can disappear after O_EXCL reports that it exists. Retry
            # the complete atomic classification so an ordinary write still
            # recreates it, while bounding churn from a concurrently replaced path.
            for _ in range(3):
                try:
                    descriptor = os.open(resolved, access_flags | platform_flags | os.O_CREAT | os.O_EXCL, 0o666)
                except FileExistsError:
                    try:
                        descriptor = os.open(resolved, access_flags | platform_flags)
                    except FileNotFoundError:
                        continue
                else:
                    created = True
                break
            else:
                raise ModelRetry(f'Path {path!r} changed repeatedly while opening. Retry the write.')
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise ModelRetry(
                    f'Path {path!r} encountered a symlink loop or changed to a symlink before opening.'
                ) from e
            if e.errno in (errno.EISDIR, errno.ENODEV, errno.ENXIO):
                raise ModelRetry(f'Path {path!r} exists and is not a regular file.') from e
            raise

        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ModelRetry(f'Path {path!r} exists and is not a regular file.')

            mode = 'r+' if expected_hash is not None else 'w'
            text_file = os.fdopen(descriptor, mode, encoding='utf-8', newline=None)
            descriptor = -1
            with text_file:
                if expected_hash is not None and not created:
                    current_hash = _content_hash(text_file.read())
                    if current_hash != expected_hash:
                        raise ValueError(
                            f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
                            f'got hash:{current_hash}). Re-read the file and retry.'
                        )

                text_file.seek(0)
                text_file.truncate(0)
                text_file.write(content)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        new_hash = _content_hash(content)
        lines = len(content.splitlines())
        return f'Wrote {len(content)} chars ({lines} lines) to {path}. [hash:{new_hash}]'

    @_recoverable
    async def edit_file(self, path: str, old_text: str, new_text: str, *, expected_hash: str | None = None) -> str:
        """Edit a file by exact string replacement with conflict detection.

        The old_text must appear exactly once in the file. Include surrounding
        context lines to ensure uniqueness.

        Args:
            path: File path relative to the root directory.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            expected_hash: If provided, rejects the edit when the file's
                current hash doesn't match (optimistic concurrency).

        Returns:
            Summary with new hash for subsequent operations.
        """
        resolved = self._safe_resolve(path, write=True)
        if not resolved.is_file():
            raise FileNotFoundError(f'File not found: {path}')

        text = resolved.read_text(encoding='utf-8')
        current_hash = _content_hash(text)

        # Optimistic concurrency check
        if expected_hash is not None and current_hash != expected_hash:
            raise ValueError(
                f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
                f'got hash:{current_hash}). Re-read the file and retry.'
            )

        count = text.count(old_text)
        if count == 0:
            raise ValueError(f'old_text not found in {path}.')
        if count > 1:
            raise ValueError(
                f'old_text found {count} times in {path}. Include more surrounding context to make the match unique.'
            )

        new_content = text.replace(old_text, new_text, 1)
        resolved.write_text(new_content, encoding='utf-8')
        new_hash = _content_hash(new_content)
        return f'Edited {path}. [hash:{new_hash}]'

    @_recoverable
    async def list_directory(self, path: str = '.') -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        # The listing root is gated by denied patterns but not by
        # allowed_patterns: a directory like '.' never matches a file pattern.
        # Entries are filtered per-entry against allowed_patterns below.
        resolved = self._safe_resolve(path, check_allowed=False)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Not a directory: {path}')

        entries: list[str] = []
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
            rel = str(rel_path)
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
        return '\n'.join(entries) if entries else '(empty directory)'

    @_recoverable
    async def search_files(self, pattern: str, *, path: str = '.', include_glob: str | None = None) -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.
            include_glob: If provided, only search files matching this glob (e.g. '*.py').

        Returns:
            str: Matching lines formatted as file:line_number:text.
        """
        # See list_directory: the search root isn't gated by allowed_patterns;
        # matched files are filtered per-entry below.
        resolved = self._safe_resolve(path, check_allowed=False)
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern: {e}') from e

        results: list[str] = []

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
            matches, truncated = _matching_lines(text, compiled, rel_str, self._max_search_results - len(results))
            results.extend(matches)
            if truncated:
                results.append(f'[... truncated at {self._max_search_results} matches]')
                break

        return '\n'.join(results) if results else 'No matches found.'

    @_recoverable
    async def find_files(self, pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match, relative to `path` (e.g. '*.py',
                '**/*.json'). Absolute patterns are rejected.
            path: Directory to search in, relative to the root directory.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
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
                matches.append(f'[... truncated at {self._max_find_results} matches]')
                break
            rel = str(rel_path)
            suffix = '/' if target.is_dir() else ''
            matches.append(f'{rel}{suffix}')

        return '\n'.join(matches) if matches else 'No matches found.'

    @_recoverable
    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            Confirmation message.
        """
        resolved = self._safe_resolve(path, write=True)
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except FileExistsError as e:
            # `exist_ok` only suppresses the error when the existing path is a
            # directory; name the conflicting model-supplied path directly.
            raise ModelRetry(f'Path {path!r} exists and is not a directory.') from e
        except NotADirectoryError as e:
            # Distinguish a parent collision from a collision at the leaf.
            raise ModelRetry(f'Path {path!r} has a parent that is not a directory.') from e
        return f'Created directory: {path}'

    @_recoverable
    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to the root directory.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        resolved = self._safe_resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f'Path not found: {path}')

        # Check if the original (pre-resolve) path is a symlink
        original = self._root / path
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
                parts.append(f'hash: {_content_hash(text)}')

        if is_link:
            target = _model_safe_filename(os.readlink(original), self._real_root)
            parts.append(f'symlink_target: {target}')

        return '\n'.join(parts)
