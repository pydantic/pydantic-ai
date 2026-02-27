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

EnvToolName = Literal[
    'shell',
    'read_file',
    'write_file',
    'edit_file',
]
"""Tool name for an environment capability.

Used in `capabilities` to declare which methods an environment implements,
and by `ExecutionEnvironmentToolset` for `include`/`exclude` filtering.
"""


# --- Data types ---


@dataclass
class ExecutionResult:
    """Result of a completed command execution."""

    output: str
    """The combined stdout/stderr output of the command."""

    exit_code: int
    """The exit code of the command."""


class ExecutionProcess(ABC):
    """Handle to a running process with bidirectional streaming I/O.

    Used for interactive execution where a script outputs data,
    waits for input, processes it, and outputs more data.
    """

    @abstractmethod
    async def send(self, data: bytes) -> None:
        """Write data to the process's stdin.

        Args:
            data: The bytes to write to stdin.
        """

    @abstractmethod
    async def recv(self, timeout: float | None = None) -> bytes:
        """Read available output from stdout.

        Blocks until data is available, the process exits, or the timeout expires.

        Args:
            timeout: Maximum seconds to wait for data. None means wait indefinitely.

        Raises:
            TimeoutError: If the timeout expires with no data available.
        """

    @abstractmethod
    async def recv_stderr(self, timeout: float | None = None) -> bytes:
        """Read available output from stderr.

        Args:
            timeout: Maximum seconds to wait for data. None means wait indefinitely.

        Raises:
            TimeoutError: If the timeout expires with no data available.
        """

    @property
    @abstractmethod
    def returncode(self) -> int | None:
        """Return code if the process has exited, None if still running."""

    @abstractmethod
    async def wait(self, timeout: float | None = None) -> int:
        """Wait for the process to exit.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.

        Returns:
            The process exit code.

        Raises:
            TimeoutError: If the timeout expires before the process exits.
        """

    @abstractmethod
    async def kill(self) -> None:
        """Kill the process."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.returncode is None:
            await self.kill()


# --- Constants ---

IMAGE_EXTENSIONS = frozenset(
    {
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
        '.webp',
        '.bmp',
        '.svg',
    }
)

IMAGE_MEDIA_TYPES: dict[str, str] = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.bmp': 'image/bmp',
    '.svg': 'image/svg+xml',
}


# --- ExecutionEnvironment ---


class ExecutionEnvironment(ABC):
    """Abstract base class for execution environments.

    An execution environment provides a place where agents can execute
    commands, read/write files, and search the filesystem.

    Implementations range from in-memory (for testing) to local subprocess,
    Docker containers, and cloud-hosted VMs.

    The only abstract member is `capabilities`; all tool methods raise
    `NotImplementedError` by default. Concrete subclasses override the
    methods that match their declared capabilities.
    """

    # --- Capability introspection ---

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[EnvToolName]:
        """Capabilities this environment supports (high-level).

        Used by toolsets to decide which tools to register. Only methods
        corresponding to declared capabilities need to be implemented.
        """
        ...

    # --- Tool methods ---
    # All raise NotImplementedError by default. Concrete subclasses override
    # the methods that match their declared capabilities.

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
            timeout: Maximum seconds to wait for completion.
                Pass `None` to disable the timeout.
            env: Additional environment variables for this command.
                Merged with (and overrides) any baseline environment variables.

        Returns:
            An `ExecutionResult` with the command output and exit code.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support shell.')

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
    ) -> str | bytes:
        """Read a file from the environment.

        For text files, returns a string with `cat -n` style line numbers.
        For binary files (images), returns raw bytes.

        Args:
            path: The file path within the environment.
            offset: The line number to start reading from (0-indexed).
                Ignored for binary files.
            limit: Maximum number of lines to read.
                Ignored for binary files.

        Returns:
            Text content with line numbers (`str`), or raw bytes for binary files.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support read_file.')

    async def write_file(self, path: str, content: str | bytes) -> None:
        """Create or overwrite a file in the environment.

        Args:
            path: The file path within the environment.
            content: The file content (text or binary).
        """
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
            replace_all: If True, replace all occurrences. If False, the
                old string must appear exactly once or an error is raised.

        Returns:
            The number of replacements made.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If `old` is not found, or appears multiple times
                when `replace_all` is False.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support replace_str.')

    # --- Internal helpers (not tools) ---

    async def create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ) -> ExecutionProcess:
        r"""Create an interactive process with streaming stdin/stdout.

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


def format_lines(text: str, offset: int, limit: int) -> str:
    """Format text with line numbers and continuation hints.

    Shared helper used by `LocalEnvironment` and `MemoryEnvironment`
    to produce consistent `cat -n` style output.
    """
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    if offset >= total_lines and total_lines > 0:
        raise ValueError(f'Offset {offset} exceeds file length ({total_lines} lines).')

    selected = lines[offset : offset + limit]

    numbered = [f'{i:>6}\t{line}' for i, line in enumerate(selected, start=offset + 1)]
    result = ''.join(numbered)
    if not result.endswith('\n'):
        result += '\n'

    remaining = total_lines - (offset + len(selected))
    if remaining > 0:
        next_offset = offset + len(selected)
        result += f'... ({remaining} more lines. Use offset={next_offset} to continue reading.)\n'

    return result


def apply_edit(text: str, old_string: str, new_string: str, path: str, *, replace_all: bool) -> tuple[str, int]:
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
