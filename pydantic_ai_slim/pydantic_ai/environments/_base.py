"""Base abstractions for code execution environments.

This module defines the core types, the `ExecutionEnvironment` ABC, and the
`ExecutionProcess` ABC for interactive execution with bidirectional streaming I/O.
"""

from __future__ import annotations

import fnmatch
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import Self


@dataclass
class ExecuteResult:
    """Result of a completed command execution."""

    output: str
    """The combined stdout/stderr output of the command."""

    exit_code: int
    """The exit code of the command."""

    truncated: bool = False
    """Whether the output was truncated due to length limits."""


@dataclass
class FileInfo:
    """Metadata about a file or directory."""

    name: str
    """The file or directory name."""

    path: str
    """The full path."""

    is_dir: bool
    """Whether this entry is a directory."""

    size: int | None = None
    """The file size in bytes, or None for directories."""


class ExecutionProcess(ABC):
    r"""Handle to a running process with bidirectional streaming I/O.

    Used for interactive execution where a script outputs data,
    waits for input, processes it, and outputs more data.

    This is the lower-level building block for "code mode" where a
    running script exchanges data via pipes.
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

MAX_OUTPUT_CHARS = 100_000


def format_lines(text: str, offset: int, limit: int) -> str:
    """Format text with line numbers and continuation hints.

    Shared helper used by `LocalEnvironment` and `MemoryEnvironment`
    to produce consistent ``cat -n`` style output.
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


def collect_grep_matches(
    rel_path: str,
    text: str,
    compiled: re.Pattern[str],
    output_mode: Literal['content', 'files_with_matches', 'count'],
    results: list[str],
) -> None:
    """Collect grep matches from a single file into ``results``.

    Shared helper used by `LocalEnvironment` and `MemoryEnvironment`.
    """
    if output_mode == 'files_with_matches':
        if any(compiled.search(line) for line in text.splitlines()):
            results.append(rel_path)
    elif output_mode == 'count':
        match_count = sum(1 for line in text.splitlines() if compiled.search(line))
        if match_count > 0:
            results.append(f'{rel_path}:{match_count}')
    else:
        for line_num, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                results.append(f'{rel_path}:{line_num}:{line}')


def glob_match(path: str, pattern: str) -> bool:
    """Match a path against a glob pattern with ``**`` support.

    ``fnmatch`` does not support ``**`` for recursive matching.
    This helper converts glob patterns to regex so that ``**``
    matches zero or more path segments (including ``/``).
    """
    if '**' not in pattern:
        return fnmatch.fnmatch(path, pattern)

    # Convert glob pattern to regex:
    # ** matches zero or more path segments (including /)
    # * matches within a single segment (no /)
    # ? matches any single char except /
    regex = ''
    i = 0
    while i < len(pattern):
        if pattern[i : i + 3] == '**/':
            regex += '(.*/)?'
            i += 3
        elif pattern[i : i + 2] == '**':
            regex += '.*'
            i += 2
        elif pattern[i] == '*':
            regex += '[^/]*'
            i += 1
        elif pattern[i] == '?':
            regex += '[^/]'
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return bool(re.fullmatch(regex, path))


class ExecutionEnvironment(ABC):
    """Abstract base class for code execution environments.

    An execution environment provides a place where agents can execute
    commands, read/write files, and search the filesystem.

    Implementations range from local subprocess (no isolation) to Docker
    containers and cloud-hosted VMs.

    Implementations must provide `execute` for command execution and
    file/search operations. Override `create_process` for interactive use.
    """

    async def create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ) -> ExecutionProcess:
        r"""Create an interactive process with streaming stdin/stdout.

        Override this for interactive use cases. The default raises
        `NotImplementedError`.

        The returned process is **not yet started** — use it as an async
        context manager to start it and ensure cleanup::

            async with await env.create_process('python worker.py') as proc:
                await proc.send(b'hello\n')

        Args:
            command: The shell command to run.
            env: Additional environment variables for this process.
                Merged with (and overrides) any baseline environment variables.

        Returns:
            An `ExecutionProcess` handle for bidirectional I/O.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not support interactive processes. Use execute() instead.'
        )

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = 120,
        env: dict[str, str] | None = None,
    ) -> ExecuteResult:
        """Execute a shell command and return the result.

        This is the simple fire-and-forget API. For interactive I/O,
        use `create_process()` directly.

        Args:
            command: The shell command to execute.
            timeout: Maximum seconds to wait for completion.
                Pass `None` to disable the timeout. Must be a positive number
                when set; behavior for zero or negative values is undefined.
            env: Additional environment variables for this command.
                Merged with (and overrides) any baseline environment variables.

        Returns:
            An `ExecuteResult` with the command output and exit code.
        """

    # --- File Operations ---

    @abstractmethod
    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read a text file from the environment with line numbers.

        Returns a string with ``cat -n`` style line numbers.
        Use `read_file_bytes` for raw binary reads (e.g. images).

        Args:
            path: The file path within the environment.
            offset: The line number to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Text content with line numbers.
        """

    @abstractmethod
    async def read_file_bytes(self, path: str) -> bytes:
        """Read raw file content as bytes.

        Used for binary files (images, etc.) where line-number
        formatting is not appropriate.

        Args:
            path: The file path within the environment.

        Returns:
            The raw file content as bytes.
        """

    @abstractmethod
    async def write_file(self, path: str, content: str | bytes) -> None:
        """Create or overwrite a file in the environment.

        Args:
            path: The file path within the environment.
            content: The file content (text or binary).
        """

    @abstractmethod
    async def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> int:
        """Edit a file by exact string replacement.

        Args:
            path: The file path within the environment.
            old_string: The exact text to find.
            new_string: The replacement text.
            replace_all: If True, replace all occurrences. If False, the
                old_string must appear exactly once or an error is raised.

        Returns:
            The number of replacements made.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If old_string is not found, or appears multiple times
                when replace_all is False.
        """

    # --- Search ---

    @abstractmethod
    async def ls(self, path: str = '.') -> list[FileInfo]:
        """List directory contents.

        Args:
            path: The directory path within the environment.

        Returns:
            A list of `FileInfo` entries.
        """

    @abstractmethod
    async def glob(self, pattern: str, *, path: str = '.') -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: The glob pattern (e.g. '**/*.py').
            path: The directory to search in.

        Returns:
            A list of matching file paths.
        """

    @abstractmethod
    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob_pattern: str | None = None,
        output_mode: Literal['content', 'files_with_matches', 'count'] = 'content',
    ) -> str:
        """Search file contents with a regex pattern.

        Args:
            pattern: The regex pattern to search for.
            path: The file or directory to search in.
            glob_pattern: Optional glob to filter which files are searched.
            output_mode: Controls output format:
                - ``'content'`` (default): matching lines as ``file:line_number:text``
                - ``'files_with_matches'``: only file paths containing matches
                - ``'count'``: ``file:count`` pairs

        Returns:
            Matching lines formatted as text.
        """

    # --- System prompt ---

    @property
    def system_prompt(self) -> str | None:
        """Optional system prompt additions describing environment-specific capabilities or limitations.

        Override this to provide environment-specific instructions that will be
        included in the toolset's system prompt. For example, a restricted sandbox
        might describe which commands are unavailable.
        """
        return None

    # --- Lifecycle ---

    async def __aenter__(self) -> Self:
        """Start the environment (e.g., create a Docker container)."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop the environment and clean up resources."""


def shell_escape(s: str) -> str:
    """Escape a string for safe use in shell commands."""
    return "'" + s.replace("'", "'\\''") + "'"


# --- Shell command builders for Docker/E2B environments ---


def build_read_file_cmd(path: str, *, offset: int = 0, limit: int = 2000) -> str:
    """Build a shell command that reads a file with line numbers.

    Uses `awk` for reliable line numbering that handles tabs correctly.
    Includes a continuation hint when more lines remain, consistent
    with the `format_lines` helper used by Local/Memory environments.
    """
    escaped = shell_escape(path)
    start = offset + 1
    end = offset + limit
    return (
        f'awk \'NR>={start} && NR<={end} {{printf "%6d\\t%s\\n", NR, $0}}'
        f' END {{if(NR>{end}) printf "... (%d more lines. Use offset={end} to continue reading.)\\n", NR-{end}}}\''
        f' {escaped}'
    )


def build_grep_cmd(
    pattern: str,
    *,
    path: str | None = None,
    glob_pattern: str | None = None,
    output_mode: Literal['content', 'files_with_matches', 'count'] = 'content',
) -> str:
    """Build a shell `grep` command from structured arguments."""
    parts = ['grep', '-rI']  # -I skips binary files
    if output_mode == 'files_with_matches':
        parts.append('-l')
    elif output_mode == 'count':
        parts.append('-c')
    else:
        parts.append('-n')
    if glob_pattern:
        parts.extend(['--include', shell_escape(glob_pattern)])
    parts.append(shell_escape(pattern))
    parts.append(shell_escape(path or '.'))
    return ' '.join(parts)


def filter_grep_count_output(text: str) -> str:
    """Filter grep -c output to remove files with 0 matches."""
    return '\n'.join(line for line in text.splitlines() if not line.endswith(':0'))


def build_glob_cmd(pattern: str, *, path: str = '.') -> str:
    """Build a shell `find` command to match files by pattern."""
    return f'find {shell_escape(path)} -path {shell_escape(pattern)} -o -name {shell_escape(pattern)} 2>/dev/null | head -100'


def parse_glob_output(text: str) -> list[str]:
    """Parse output of a find/glob command into a list of paths."""
    text = text.strip()
    if not text:
        return []
    return [line for line in text.splitlines() if line]


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
