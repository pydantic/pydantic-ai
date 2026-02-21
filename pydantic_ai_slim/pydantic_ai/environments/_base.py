"""Base abstractions for execution environments.

This module defines the core types, the `ExecutionEnvironment` ABC, and the
`ExecutionProcess` ABC for interactive execution with bidirectional streaming I/O.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import Self

# --- Capability type alias ---

Capability = Literal[
    'ls',
    'shell',
    'read_file',
    'write_file',
    'edit_file:replace_str',
    'edit_file:apply_patch',
    'glob',
    'grep',
]
"""Capability identifier for environment methods.

Used in `capabilities` to declare which methods an environment implements.
"""

ToolName = Literal[
    'ls',
    'shell',
    'read_file',
    'write_file',
    'edit_file',
    'glob',
    'grep',
]
"""Tool name exposed to the model by `ExecutionEnvironmentToolset`.

Most match `Capability` 1:1, except `edit_file` which maps to either
`edit_file:replace_str` or `edit_file:apply_patch` depending on environment support.
Used for `include`/`exclude` filtering on the toolset.
"""


# --- Data types ---


@dataclass
class ExecutionResult:
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

MAX_OUTPUT_CHARS = 100_000


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
    def capabilities(self) -> frozenset[Capability]:
        """Capabilities this environment supports (high-level).

        Used by toolsets to decide which tools to register. Only methods
        corresponding to declared capabilities need to be implemented.
        """
        ...

    def instructions(self, capability: Capability) -> str | None:
        """Per-capability instructions for the LLM.

        Override to provide environment-specific hints that toolsets include
        in the tool description shown to the model, e.g.::

            def instructions(self, capability):
                if capability == 'shell':
                    return 'Bash in Docker container, numpy/pandas installed'
                if capability == 'grep':
                    return 'Uses POSIX basic regex, not Python re syntax'
                return None

        Args:
            capability: The capability name (e.g. `'shell'`).

        Returns:
            Instruction text for the LLM, or None for no extra instructions.
        """
        return None

    # --- Tool methods ---
    # All raise NotImplementedError by default. Concrete subclasses override
    # the methods that match their declared capabilities.

    async def ls(self, path: str = '.') -> list[FileInfo]:
        """List directory contents.

        Args:
            path: The directory path within the environment.

        Returns:
            A list of `FileInfo` entries.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support ls.')

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

    async def apply_patch(self, path: str, patch: str) -> str:
        """Apply a unified diff patch to a file.

        Args:
            path: The file path within the environment.
            patch: The unified diff patch content.

        Returns:
            The resulting file content after applying the patch.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support apply_patch.')

    async def glob(self, pattern: str, *, path: str = '.') -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: The glob pattern (e.g. `'**/*.py'`).
            path: The directory to search in.

        Returns:
            A list of matching file paths.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support glob.')

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
                - `'content'` (default): matching lines as `file:line_number:text`
                - `'files_with_matches'`: only file paths containing matches
                - `'count'`: `file:count` pairs

        Returns:
            Matching lines formatted as text.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support grep.')

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


def shell_escape(s: str) -> str:
    """Escape a string for safe use in shell commands."""
    return "'" + s.replace("'", "'\\''") + "'"


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


def collect_grep_matches(
    rel_path: str,
    text: str,
    compiled: re.Pattern[str],
    output_mode: Literal['content', 'files_with_matches', 'count'],
    results: list[str],
) -> None:
    """Collect grep matches from a single file into `results`.

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
    """Match a path against a glob pattern with `**` support.

    This helper converts glob patterns to regex where `*` matches
    within a single path segment and `**` matches zero or more
    path segments (including `/`).
    """
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


# --- Shell command builders for Docker environments ---


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
    """Filter `grep -c` output to remove files with 0 matches."""
    return '\n'.join(line for line in text.splitlines() if not line.endswith(':0'))


def build_glob_cmd(pattern: str, *, path: str = '.') -> str:
    """Build a shell `find` command to match files by pattern."""
    # For -path, prepend the search path since find outputs full paths relative to the starting point
    path_pattern = f'{path}/{pattern}' if '/' in pattern else pattern
    return (
        f'find {shell_escape(path)}'
        f' \\( -path {shell_escape(path_pattern)} -o -name {shell_escape(pattern)} \\)'
        f' 2>/dev/null | head -100'
    )


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
