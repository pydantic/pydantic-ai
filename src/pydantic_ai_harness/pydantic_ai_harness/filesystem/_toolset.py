"""Filesystem toolset providing sandboxed file operations."""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Concatenate, ParamSpec

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

_P = ParamSpec('_P')

# Errors that mean "the model asked for something the tool couldn't do" — a
# missing file, a denied path, a stale edit. pyai only feeds `ModelRetry` back
# to the model; any other exception aborts the whole run. `_recoverable`
# converts these so the agent can correct itself and continue.
_RECOVERABLE_ERRORS = (PermissionError, FileNotFoundError, NotADirectoryError, IsADirectoryError, ValueError)


def _recoverable(
    fn: Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]]:
    """Surface model-correctable tool errors as `ModelRetry`."""

    @functools.wraps(fn)
    async def wrapper(self: FileSystemToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except _RECOVERABLE_ERRORS as e:
            raise ModelRetry(str(e)) from e

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
        root-level `secrets.yaml` — there's no leading directory to match.
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
        candidate = (self._root / path).resolve()
        real = Path(os.path.realpath(candidate))
        if not real.is_relative_to(self._real_root):
            raise PermissionError(f'Path {path!r} resolves outside the root directory.')

        return real

    def _check_access(self, path: str, *, write: bool = False, check_allowed: bool = True) -> None:
        """Validate path against allow/deny/protected patterns.

        `check_allowed=False` skips the `allowed_patterns` gate. Walkers
        (`list_directory`, `search_files`, `find_files`) pass it so their root
        directory isn't required to match `allowed_patterns` itself — `.` or
        `src` would never match a file pattern like `src/*.py`. The walk's
        entries are still filtered against `allowed_patterns` per-entry via
        `_is_accessible`. Denied and protected patterns continue to gate the
        root.
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

    def _is_accessible(self, path: str, *, write: bool = False) -> bool:
        """Predicate form of `_check_access` for filtering recursive walkers.

        Used by `list_directory`, `search_files`, and `find_files` to skip
        children that would be rejected if accessed directly. Note this only
        checks the relative path against patterns; it does not resolve symlinks.
        """
        if write and self._protected_patterns:
            if self._first_matching_pattern(path, self._protected_patterns) is not None:
                return False
        if self._denied_patterns:
            if self._first_matching_pattern(path, self._denied_patterns) is not None:
                return False
        if self._allowed_patterns and not any(self._matches(path, p) for p in self._allowed_patterns):
            return False
        return True

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

        # Optimistic concurrency: reject stale writes
        if expected_hash is not None and resolved.is_file():
            current = resolved.read_text(encoding='utf-8')
            current_hash = _content_hash(current)
            if current_hash != expected_hash:
                raise ValueError(
                    f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
                    f'got hash:{current_hash}). Re-read the file and retry.'
                )

        if not resolved.parent.exists():
            parent_rel = str(resolved.parent.relative_to(self._root))
            raise FileNotFoundError(f"Parent directory '{parent_rel}' does not exist. Use create_directory first.")
        resolved.write_text(content, encoding='utf-8')
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
        # The listing root is gated by denied/protected patterns but not by
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
            rel = str(rel_path)
            # Apply the same allow/deny/protected filtering used for direct
            # access so a directory listing can't leak patterns the agent
            # couldn't otherwise read or write.
            if not self._is_accessible(rel, write=True):
                continue
            if entry.is_dir():
                entries.append(f'{rel}/')
            else:
                try:
                    size = entry.stat().st_size
                except OSError:  # pragma: no cover  # file deleted between iterdir and stat
                    size = 0
                entries.append(f'{rel}  ({size} bytes)')
        return '\n'.join(entries) if entries else '(empty directory)'

    @_recoverable
    async def search_files(self, pattern: str, *, path: str = '.', include_glob: str | None = None) -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.
            include_glob: If provided, only search files matching this glob (e.g. '*.py').

        Returns:
            Matching lines formatted as file:line_number:text.
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

        real_root = Path(os.path.realpath(self._root))
        for file_path in files:
            if not file_path.is_file():
                continue
            try:
                rel_parts = file_path.relative_to(real_root).parts
            except ValueError:  # pragma: no cover
                continue
            if any(part.startswith('.') for part in rel_parts):
                continue
            rel_str = str(file_path.relative_to(real_root))
            # Apply the same allow/deny/protected filtering used for direct
            # access so a recursive search can't read patterns the agent
            # couldn't otherwise read.
            if not self._is_accessible(rel_str, write=True):
                continue
            if include_glob and not fnmatch.fnmatch(rel_str, include_glob):
                continue
            try:
                raw = file_path.read_bytes()
            except OSError:  # pragma: no cover
                continue
            if _is_binary(raw):
                continue
            text = raw.decode('utf-8', errors='replace')
            for line_num, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    results.append(f'{rel_str}:{line_num}:{line}')
            if len(results) >= self._max_search_results:
                results.append(f'[... truncated at {self._max_search_results} matches]')
                break

        return '\n'.join(results) if results else 'No matches found.'

    @_recoverable
    async def find_files(self, pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match (e.g. '*.py', '**/*.json').
            path: Directory to search in, relative to the root directory.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
        # See list_directory: the find root isn't gated by allowed_patterns;
        # matched entries are filtered per-entry below.
        resolved = self._safe_resolve(path, check_allowed=False)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Not a directory: {path}')

        matches: list[str] = []
        real_root = Path(os.path.realpath(self._root))
        for match in sorted(resolved.glob(pattern)):
            try:
                rel_parts = match.relative_to(real_root).parts
            except ValueError:  # pragma: no cover
                continue
            if any(part.startswith('.') for part in rel_parts):
                continue
            rel = str(match.relative_to(real_root))
            # Apply the same allow/deny/protected filtering used for direct
            # access so a glob find can't surface patterns the agent
            # couldn't otherwise see.
            if not self._is_accessible(rel, write=True):
                continue
            suffix = '/' if match.is_dir() else ''
            matches.append(f'{rel}{suffix}')
            if len(matches) >= self._max_find_results:
                matches.append(f'[... truncated at {self._max_find_results} matches]')
                break

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
        resolved.mkdir(parents=True, exist_ok=True)
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
            parts.append(f'symlink_target: {os.readlink(original)}')

        return '\n'.join(parts)
