"""In-memory execution environment for testing.

All file operations use an in-memory dictionary. Shell commands are handled
by an optional callback — if not provided, `shell()` raises `RuntimeError`.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from ._base import (
    IMAGE_EXTENSIONS,
    ExecutionEnvironment,
    ExecutionResult,
    apply_edit,
    format_lines,
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
        caps: set[EnvToolName] = {'read_file', 'write_file', 'edit_file'}
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
