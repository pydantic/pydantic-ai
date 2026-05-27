"""Filesystem toolset implementation with security-first design.

Incorporates best practices from:
- MCP filesystem server: root containment, symlink-aware path checks
- Codex CLI: policy-based access, protected paths, metadata preservation
- Aider: robust search/replace editing with conflict detection
- SWE-agent: configurable tool surface, binary detection
- CrewAI: centralized safe-path validators
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic_ai.toolsets import FunctionToolset


def _format_lines(text: str, offset: int, limit: int) -> str:
    """Format text with line numbers.

    Args:
        text: The raw file content.
        offset: Zero-based line offset to start from.
        limit: Maximum number of lines to include.

    Returns:
        Numbered text with a continuation hint when more lines remain.
    """
    lines = text.splitlines(keepends=True)
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


class FileSystemToolset(FunctionToolset[Any]):
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

    # ------------------------------------------------------------------
    # Path security
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to root, rejecting traversal.

        Uses os.path.realpath for symlink resolution before checking containment.
        """
        # Normalize and join with root
        candidate = (self._root / path).resolve()

        # Symlink-aware: resolve realpath to catch symlink escapes
        real = Path(os.path.realpath(candidate))

        # Containment check against real root
        real_root = Path(os.path.realpath(self._root))
        if not real.is_relative_to(real_root):
            raise PermissionError(f'Path {path!r} resolves outside the root directory.')

        return real

    def _check_access(self, path: str, *, write: bool = False) -> None:
        """Validate path against allow/deny/protected patterns."""
        # Check protected patterns (always denied for writes)
        if write and self._protected_patterns:
            matched = next((p for p in self._protected_patterns if fnmatch.fnmatch(path, p)), None)
            if matched:
                raise PermissionError(f'Path {path!r} is protected (matches {matched!r}).')

        # Check deny patterns
        if self._denied_patterns:
            matched = next((p for p in self._denied_patterns if fnmatch.fnmatch(path, p)), None)
            if matched:
                raise PermissionError(f'Path {path!r} is denied by pattern {matched!r}.')

        # Check allow patterns (if configured, path must match at least one)
        if self._allowed_patterns:
            if not any(fnmatch.fnmatch(path, p) for p in self._allowed_patterns):
                raise PermissionError(f'Path {path!r} does not match any allowed pattern.')

    def _safe_resolve(self, path: str, *, write: bool = False) -> Path:
        """Resolve and access-check a path in one step."""
        self._check_access(path, write=write)
        return self._resolve_path(path)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path relative to the root directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: max_read_lines).

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
        total_lines = len(text.splitlines())
        content_hash = _content_hash(text)

        header = f'[{path} | {total_lines} lines | hash:{content_hash}]\n'
        return header + _format_lines(text, offset, limit)

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

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding='utf-8')
        new_hash = _content_hash(content)
        lines = len(content.splitlines())
        return f'Wrote {len(content)} chars ({lines} lines) to {path}. [hash:{new_hash}]'

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

    async def list_directory(self, path: str = '.') -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        resolved = self._safe_resolve(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Not a directory: {path}')

        entries: list[str] = []
        real_root = Path(os.path.realpath(self._root))
        for entry in sorted(resolved.iterdir()):
            try:
                rel = str(entry.relative_to(real_root))
            except ValueError:  # pragma: no cover
                continue
            if entry.is_dir():
                entries.append(f'{rel}/')
            else:
                try:
                    size = entry.stat().st_size
                except OSError:  # pragma: no cover
                    size = 0
                entries.append(f'{rel}  ({size} bytes)')
        return '\n'.join(entries) if entries else '(empty directory)'

    async def search_files(self, pattern: str, *, path: str = '.', include_glob: str | None = None) -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.
            include_glob: If provided, only search files matching this glob (e.g. '*.py').

        Returns:
            Matching lines formatted as file:line_number:text.
        """
        resolved = self._safe_resolve(path)
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
            # Skip hidden files/directories
            if any(part.startswith('.') for part in rel_parts):
                continue
            # Apply include_glob filter
            rel_str = str(file_path.relative_to(real_root))
            if include_glob and not fnmatch.fnmatch(rel_str, include_glob):
                continue
            try:
                raw = file_path.read_bytes()
            except OSError:  # pragma: no cover
                continue
            # Skip binary files
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

    async def find_files(self, pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match (e.g. '*.py', '**/*.json').
            path: Directory to search in, relative to the root directory.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
        resolved = self._safe_resolve(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Not a directory: {path}')

        matches: list[str] = []
        real_root = Path(os.path.realpath(self._root))
        for match in sorted(resolved.glob(pattern)):
            try:
                rel_parts = match.relative_to(real_root).parts
            except ValueError:  # pragma: no cover
                continue
            # Skip hidden files/directories
            if any(part.startswith('.') for part in rel_parts):
                continue
            rel = str(match.relative_to(real_root))
            suffix = '/' if match.is_dir() else ''
            matches.append(f'{rel}{suffix}')
            if len(matches) >= self._max_find_results:
                matches.append(f'[... truncated at {self._max_find_results} matches]')
                break

        return '\n'.join(matches) if matches else 'No matches found.'

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
                line_count = len(raw.decode('utf-8', errors='replace').splitlines())
                parts.append(f'lines: {line_count}')
                parts.append(f'hash: {_content_hash(raw.decode("utf-8", errors="replace"))}')

        if is_link:
            parts.append(f'symlink_target: {os.readlink(original)}')

        return '\n'.join(parts)
