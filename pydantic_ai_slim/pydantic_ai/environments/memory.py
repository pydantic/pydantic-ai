"""In-memory execution environment for testing.

All file operations use an in-memory dictionary. Shell commands are handled
by an optional callback — if not provided, `shell()` raises `RuntimeError`.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

from ._base import (
    IMAGE_EXTENSIONS,
    ExecutionEnvironment,
    ExecutionResult,
    FileInfo,
    apply_edit,
    collect_grep_matches,
    format_lines,
    glob_match,
)

if TYPE_CHECKING:
    from ._base import EnvToolName


class MemoryEnvironment(ExecutionEnvironment):
    """In-memory execution environment for testing.

    File operations use an in-memory dictionary, making tests fast and
    isolated with no filesystem access. Shell commands can optionally be
    handled by a user-provided callback.

    This is the testing counterpart to `LocalEnvironment`, analogous to
    how `TestModel` and `FunctionModel` relate to real model classes.

    Usage:
        ```python {test="skip" lint="skip"}
        from pydantic_ai.environments.memory import MemoryEnvironment

        env = MemoryEnvironment(files={'main.py': 'print("hello")'})
        async with env:
            content = await env.read_file('main.py')
            assert 'hello' in content
        ```
    """

    def __init__(
        self,
        files: dict[str, str | bytes] | None = None,
        *,
        command_handler: Callable[[str], ExecutionResult] | None = None,
    ) -> None:
        """Create an in-memory execution environment.

        Args:
            files: Initial files to populate the environment with.
                Keys are file paths, values are file contents (str or bytes).
            command_handler: Optional callback for `shell()` calls.
                Receives the command string and returns an `ExecutionResult`.
                If not provided, `shell()` raises `RuntimeError`.
        """
        self._files: dict[str, str | bytes] = {}
        if files:
            for path, content in files.items():
                self._files[self._normalize(path)] = content
        self._command_handler = command_handler

    @property
    def capabilities(self) -> frozenset[EnvToolName]:
        caps: set[EnvToolName] = {'ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep'}
        if self._command_handler is not None:
            caps.add('shell')
        return frozenset(caps)

    @property
    def files(self) -> Mapping[str, str | bytes]:
        """Read-only view of the in-memory file system.

        Keys are normalized file paths, values are file contents.
        Useful for test assertions against raw file content without the
        line-number formatting that [`read_file()`][pydantic_ai.environments.memory.MemoryEnvironment.read_file] adds.
        """
        return self._files

    @staticmethod
    def _normalize(path: str) -> str:
        """Normalize a path for consistent storage."""
        normalized = posixpath.normpath(path)
        # Strip leading './' or '/'
        if normalized.startswith('./'):  # pragma: no cover
            normalized = normalized[2:]
        elif normalized.startswith('/'):
            normalized = normalized[1:]
        return normalized

    async def shell(
        self,
        command: str,
        *,
        timeout: float | None = 120,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a command using the configured handler.

        Args:
            command: The shell command to execute.
            timeout: Ignored for MemoryEnvironment.
            env: Ignored for MemoryEnvironment.

        Returns:
            The result from the command handler.

        Raises:
            RuntimeError: If no command_handler was provided.
        """
        if self._command_handler is None:
            raise RuntimeError(
                'MemoryEnvironment has no command_handler configured. '
                'Pass command_handler= to the constructor to handle shell() calls.'
            )
        return self._command_handler(command)

    async def read_file(self, path: str, *, offset: int = 0, limit: int = 2000) -> str | bytes:
        normalized = self._normalize(path)

        # Check if path is a "directory" (any file starts with path/)
        if any(k.startswith(normalized + '/') for k in self._files):
            if normalized not in self._files:
                raise FileNotFoundError(f"'{path}' is a directory, not a file.")

        if normalized not in self._files:
            raise FileNotFoundError(f'File not found: {path}')

        content = self._files[normalized]

        # Return raw bytes for image files
        ext = posixpath.splitext(normalized)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            if isinstance(content, bytes):
                return content
            return content.encode('utf-8')

        # Text mode
        if isinstance(content, bytes):
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                return content
        else:
            text = content

        return format_lines(text, offset, limit)

    async def write_file(self, path: str, content: str | bytes) -> None:
        self._files[self._normalize(path)] = content

    async def replace_str(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        normalized = self._normalize(path)
        if normalized not in self._files:
            raise FileNotFoundError(f'File not found: {path}')

        content = self._files[normalized]
        text = content.decode('utf-8') if isinstance(content, bytes) else content
        new_text, count = apply_edit(text, old, new, path, replace_all=replace_all)
        self._files[normalized] = new_text
        return count

    async def ls(self, path: str = '.') -> list[FileInfo]:
        normalized = self._normalize(path)

        # Collect direct children
        entries: dict[str, FileInfo] = {}
        for file_path in sorted(self._files):
            if normalized == '.':
                rel = file_path
            elif file_path.startswith(normalized + '/'):
                rel = file_path[len(normalized) + 1 :]
            else:
                continue

            # Get the first component (direct child)
            parts = rel.split('/', 1)
            name = parts[0]
            if name in entries:
                continue

            is_dir = len(parts) > 1
            if is_dir:
                entries[name] = FileInfo(
                    name=name,
                    path=f'{normalized}/{name}' if normalized != '.' else name,
                    is_dir=True,
                )
            else:
                content = self._files[file_path]
                size = len(content) if isinstance(content, bytes) else len(content.encode('utf-8'))
                entries[name] = FileInfo(
                    name=name,
                    path=f'{normalized}/{name}' if normalized != '.' else name,
                    is_dir=False,
                    size=size,
                )

        if not entries and normalized != '.':
            raise NotADirectoryError(f'Not a directory: {path}')

        return list(entries.values())

    async def glob(self, pattern: str, *, path: str = '.') -> list[str]:
        normalized = self._normalize(path)
        matches: list[str] = []
        for file_path in sorted(self._files):
            if normalized != '.':
                if not file_path.startswith(normalized + '/'):
                    continue
                rel = file_path[len(normalized) + 1 :]
            else:
                rel = file_path

            if glob_match(rel, pattern):
                matches.append(file_path)

        return matches

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob_pattern: str | None = None,
        output_mode: Literal['content', 'files_with_matches', 'count'] = 'content',
    ) -> str:
        """Search file contents using a regex pattern (Python `re` module syntax)."""
        normalized = self._normalize(path or '.')
        compiled = re.compile(pattern)

        is_exact_file = normalized != '.' and normalized in self._files

        results: list[str] = []
        for file_path in sorted(self._files):
            # Path filtering
            if normalized != '.':
                if normalized == file_path:
                    pass  # exact file match
                elif not file_path.startswith(normalized + '/'):
                    continue

            # Glob filtering (skip for exact file matches, matching LocalEnvironment behavior)
            if not is_exact_file and glob_pattern and not fnmatch.fnmatch(posixpath.basename(file_path), glob_pattern):
                continue

            # Skip hidden files unless explicitly specified
            if not is_exact_file and any(part.startswith('.') for part in file_path.split('/')):
                continue

            content = self._files[file_path]

            # Skip binary files
            if isinstance(content, bytes):
                if b'\x00' in content[:8192]:
                    continue
                text = content.decode('utf-8', errors='replace')
            else:
                text = content

            collect_grep_matches(file_path, text, compiled, output_mode, results)

            if len(results) > 1000:
                results.append('[... truncated at 1000 matches]')
                break

        return '\n'.join(results)
