"""Base abstractions for execution environments.

This module defines the core types, the `ExecutionEnvironment` ABC, and the
`ExecutionProcess` ABC for interactive execution with bidirectional streaming I/O.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import Self

# --- Type aliases ---

EnvCapability = Literal[
    'shell',
    'read_file',
    'write_file',
    'replace_str',
    'create_process',
]
"""Execution environment capability.

Used in `capabilities` to declare which operations an environment implements.
`ExecutionEnvironmentToolset` maps its tool names onto these capabilities.
"""


# --- Data types ---


@dataclass
class ExecutionResult:
    """Result of a completed command execution."""

    output: str
    """The combined stdout/stderr output of the command."""

    exit_code: int
    """The exit code of the command."""


@dataclass
class TextFileReadResult:
    """A paginated UTF-8 text file read."""

    text: str
    """The selected text content, without line numbers."""

    offset: int
    """The zero-based line offset used for this read."""

    total_lines: int
    """The total number of lines in the file."""


class ExecutionProcess(ABC):
    """Handle to a running process with bidirectional streaming I/O."""

    @abstractmethod
    async def send(self, data: bytes) -> None:
        """Write data to the process's stdin."""

    @abstractmethod
    async def recv(self, timeout: float | None = None) -> bytes:
        """Read available output from stdout.

        Args:
            timeout: Maximum seconds to wait. `None` waits indefinitely.
        """

    @abstractmethod
    async def recv_stderr(self, timeout: float | None = None) -> bytes:
        """Read available output from stderr.

        Args:
            timeout: Maximum seconds to wait. `None` waits indefinitely.
        """

    @property
    @abstractmethod
    def returncode(self) -> int | None:
        """Return code if the process has exited, `None` if still running."""

    @abstractmethod
    async def wait(self, timeout: float | None = None) -> int:
        """Wait for the process to exit.

        Args:
            timeout: Maximum seconds to wait. `None` waits indefinitely.

        Returns:
            The process exit code.
        """

    @abstractmethod
    async def kill(self) -> None:
        """Kill the process."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.returncode is None:
            await self.kill()


# --- ExecutionEnvironment ---


class ExecutionEnvironment(ABC):
    """Abstract base class for execution environments.

    An execution environment provides a place where agents can execute
    commands and read/write files. Implementations include local subprocess,
    Docker containers, and cloud-hosted VMs.

    Subclasses implement `capabilities`, `read_file`, `write_file`,
    and `shell` as needed. `replace_str` and `read_text_file` have
    defaults built on `read_file` and `write_file`.
    """

    # --- Capability introspection ---

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[EnvCapability]:
        """Tool capabilities this environment supports."""
        ...

    # --- File and shell operations ---

    async def shell(
        self,
        command: str,
        *,
        timeout: float | None = 120,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a shell command and return the result.

        Args:
            command: The shell command to execute.
            timeout: Maximum seconds to wait. `None` disables the timeout.
            env: Additional environment variables, merged with any baseline vars.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support shell.')

    async def read_file(self, path: str) -> bytes:
        """Read the raw content of a file.

        Args:
            path: The file path within the environment.

        Returns:
            The raw bytes of the file.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support read_file.')

    async def read_text_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
    ) -> TextFileReadResult:
        """Read a UTF-8 text file, optionally paginated by line number.

        Implementations can override this to efficiently fetch just the
        requested line range from remote environments. The default
        implementation falls back to `read_file`.

        Args:
            path: The file path within the environment.
            offset: The zero-based line number to start reading from.
            limit: The maximum number of lines to read.
        """
        raw = await self.read_file(path)
        text = raw.decode('utf-8')
        return _slice_text_file(text, offset=offset, limit=limit)

    async def write_file(self, path: str, content: str | bytes) -> None:
        """Create or overwrite a file in the environment."""
        raise NotImplementedError(f'{type(self).__name__} does not support write_file.')

    async def replace_str(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        """Edit a file by exact string replacement.

        Args:
            path: The file path within the environment.
            old: The exact text to find.
            new: The replacement text.
            replace_all: If `True`, replace all occurrences. If `False`,
                `old` must appear exactly once or a `ValueError` is raised.

        Returns:
            The number of replacements made.
        """
        content = await self.read_file(path)
        text = content.decode('utf-8')
        new_text, count = apply_replace_str(text, old, new, path, replace_all=replace_all)
        await self.write_file(path, new_text)
        return count

    # --- Internal helpers (not tools) ---

    async def create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ) -> ExecutionProcess:
        """Create an interactive process with streaming stdin/stdout.

        The returned process must be used as an async context manager::

            async with await env.create_process('cmd') as proc:
                await proc.send(b'input')
                output = await proc.recv()

        Args:
            command: The shell command to run.
            env: Additional environment variables for this process.

        Returns:
            An `ExecutionProcess` handle for bidirectional I/O.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support interactive processes.')

    # --- Lifecycle ---

    async def __aenter__(self) -> Self:
        """Start the environment (e.g., create a Docker container)."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop the environment and clean up resources."""


# --- Helper functions ---


def apply_replace_str(text: str, old_string: str, new_string: str, path: str, *, replace_all: bool) -> tuple[str, int]:
    """Apply a string replacement edit, returning the new text and the number of replacements.

    Raises:
        ValueError: If old_string is not found, or appears multiple times
            when replace_all is False.
    """
    count = text.count(old_string)

    if count == 0:
        raise ValueError(f'old_string not found in {path}.')
    if not replace_all and count > 1:
        raise ValueError(f'old_string found {count} times in {path}. Use replace_all=True or provide more context.')

    if replace_all:
        new_text = text.replace(old_string, new_string)
    else:
        new_text = text.replace(old_string, new_string, 1)

    return new_text, count if replace_all else 1


def _slice_text_file(text: str, *, offset: int, limit: int) -> TextFileReadResult:
    """Slice UTF-8 text by line number."""
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    validate_text_read_range(offset=offset, limit=limit, total_lines=total_lines)
    return TextFileReadResult(
        text=''.join(lines[offset : offset + limit]),
        offset=offset,
        total_lines=total_lines,
    )


def validate_text_read_range(*, offset: int, limit: int, total_lines: int) -> None:
    """Validate a line-based text read."""
    if offset < 0:
        raise ValueError(f'offset must be non-negative, got {offset}.')
    if limit <= 0:
        raise ValueError(f'limit must be positive, got {limit}.')
    if total_lines == 0:
        if offset != 0:
            raise ValueError(f'Offset {offset} exceeds file length ({total_lines} lines).')
    elif offset >= total_lines:
        raise ValueError(f'Offset {offset} exceeds file length ({total_lines} lines).')
