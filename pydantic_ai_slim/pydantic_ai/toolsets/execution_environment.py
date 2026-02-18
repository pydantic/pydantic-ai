"""ExecutionEnvironmentToolset — exposes coding-agent-style tools backed by an ExecutionEnvironment."""

from __future__ import annotations

import posixpath
import re
from asyncio import Lock
from collections.abc import Iterator
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from typing_extensions import Self

from ..environments._base import (
    IMAGE_MEDIA_TYPES,
    ExecutionEnvironment,
)
from ..exceptions import ModelRetry
from ..messages import BinaryContent
from ..toolsets.function import FunctionToolset

Capability = Literal[
    'ls', 'shell', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'run_code', 'run_code_with_functions'
]
"""Toolset-level capability used in ``include``/``exclude``.

These are higher-level than the environment's fine-grained capabilities.
The toolset maps these to the appropriate environment capabilities.
"""

EditStrategy = Literal['replace_str', 'apply_patch']
"""Specific edit tool strategy. Expanded from the ``edit_file`` capability."""

CodeLanguage = Literal['python', 'typescript']
"""Code execution language. Expanded from the ``run_code`` capability."""

# Capabilities that are excluded by default (handled by CodeExecutionToolset)
_DEFAULT_EXCLUDE: frozenset[Capability] = frozenset({'run_code'})

# Mapping from toolset-level code capabilities to per-language env capabilities
_CODE_CAPABILITY_MAP: dict[str, dict[CodeLanguage, str]] = {
    'run_code': {
        'python': 'run_python',
        'typescript': 'run_typescript',
    },
    'run_code_with_functions': {
        'python': 'run_python_with_functions',
        'typescript': 'run_typescript_with_functions',
    },
}


class ExecutionEnvironmentToolset(FunctionToolset[Any]):
    """Toolset providing coding-agent-style tools backed by an `ExecutionEnvironment`.

    Tool names and schemas are designed to match what popular coding agents
    expose, so models are well-trained on them.

    Tools are dynamically registered based on the environment's ``capabilities``,
    filtered by ``include``/``exclude``. The ``run_code`` capability is excluded
    by default (use ``CodeExecutionToolset`` for code execution).

    The environment can be:
    - Passed directly at construction time (most common)
    - Set/overridden via context var using ``use_environment()`` (for testing or per-call-site config)

    Usage:
        ```python {test="skip" lint="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.environments import ExecutionEnvironmentToolset
        from pydantic_ai.environments.docker import DockerEnvironment

        env = DockerEnvironment(image='python:3.12-slim')
        toolset = ExecutionEnvironmentToolset(env)

        agent = Agent('openai:gpt-5.2', toolsets=[toolset])

        async with env:
            result = await agent.run('Write a script that prints hello')
        ```
    """

    def __init__(
        self,
        environment: ExecutionEnvironment | None = None,
        *,
        include: frozenset[Capability] | None = None,
        exclude: frozenset[Capability] | None = None,
        edit_strategy: EditStrategy | None = None,
        code_language: CodeLanguage | None = None,
        require_shell_approval: bool = False,
        require_write_approval: bool = False,
        image_support: bool = True,
        max_image_bytes: int = 50 * 1024 * 1024,
        max_retries: int = 1,
        id: str | None = None,
    ):
        """Create a new execution environment toolset.

        Args:
            environment: The execution environment to use for tool execution.
                Can also be set later via ``use_environment()``.
            include: Capabilities to include. ``None`` means all capabilities
                from the environment (minus ``run_code``). Pass an explicit set
                to restrict to specific capabilities.
            exclude: Capabilities to exclude. ``None`` defaults to ``{'run_code'}``.
                Use ``frozenset()`` to include all capabilities including ``run_code``.
            edit_strategy: Which edit strategy to use. ``None`` auto-selects
                ``'replace_str'`` if supported by the environment.
            code_language: Code execution language. ``None`` auto-detects
                from the environment's capabilities (defaults to ``'python'``).
            require_shell_approval: Whether the ``shell`` tool requires human-in-the-loop
                approval before execution. Recommended for ``LocalEnvironment`` where
                commands run directly on the host.
            require_write_approval: Whether ``write_file`` and edit tools require
                human-in-the-loop approval before execution.
            image_support: Whether ``read_file`` should return images as ``BinaryContent``
                for multimodal models (otherwise returns a placeholder message).
            max_image_bytes: Maximum image file size to return as BinaryContent.
            max_retries: Maximum retries per tool call.
            id: Optional unique ID for the toolset (required for durable execution).
        """
        super().__init__(max_retries=max_retries, id=id)
        self._default_environment = environment
        self._environment_override: ContextVar[ExecutionEnvironment | None] = ContextVar(
            f'_environment_override_{id or "environment"}', default=None
        )
        self._include = include
        self._exclude = exclude if exclude is not None else _DEFAULT_EXCLUDE
        self._edit_strategy: EditStrategy | None = edit_strategy
        self._code_language: CodeLanguage | None = code_language
        self._image_support = image_support
        self._max_image_bytes = max_image_bytes
        self._require_shell_approval = require_shell_approval
        self._require_write_approval = require_write_approval
        self._enter_lock: Lock = Lock()
        self._running_count: int = 0
        self._exit_stack: AsyncExitStack | None = None

        # Register tools based on what we know at init time.
        # If no environment is provided, we register a full set of tools and
        # let runtime errors catch unsupported capabilities.
        self._register_tools(environment)

    def _resolve_capabilities(self, env: ExecutionEnvironment | None) -> set[Capability]:
        """Determine which toolset-level capabilities to register as tools."""
        if env is not None:
            env_caps = env.capabilities
            available: set[Capability] = set()
            # Map env capabilities back to toolset capabilities
            for cap in ('ls', 'shell', 'read_file', 'write_file', 'glob', 'grep'):
                if cap in env_caps:
                    available.add(cap)
            # Check for edit_file: env has replace_str or apply_patch
            if 'replace_str' in env_caps or 'apply_patch' in env_caps:
                available.add('edit_file')
            # Check for run_code: env has run_python or run_typescript
            if 'run_python' in env_caps or 'run_typescript' in env_caps:
                available.add('run_code')
            # Check for run_code_with_functions
            if 'run_python_with_functions' in env_caps or 'run_typescript_with_functions' in env_caps:
                available.add('run_code_with_functions')
        else:
            # No environment yet — register everything (runtime will error on unsupported)
            available = {'ls', 'shell', 'read_file', 'write_file', 'edit_file', 'glob', 'grep'}

        if self._include is not None:
            available &= self._include

        available -= self._exclude
        return available

    def _resolve_edit_tool(self, env: ExecutionEnvironment | None) -> EditStrategy | None:
        """Determine which edit strategy to use."""
        if self._edit_strategy is not None:
            return self._edit_strategy
        if env is not None:
            env_caps = env.capabilities
            if 'replace_str' in env_caps:
                return 'replace_str'
            if 'apply_patch' in env_caps:
                return 'apply_patch'
            return None
        # Default when no environment is available
        return 'replace_str'

    def _register_tools(self, env: ExecutionEnvironment | None) -> None:
        """Register tools dynamically based on capabilities."""
        caps = self._resolve_capabilities(env)

        if 'ls' in caps:
            self._register_ls()
        if 'shell' in caps:
            self._register_shell()
        if 'read_file' in caps:
            self._register_read_file()
        if 'write_file' in caps:
            self._register_write_file()
        if 'edit_file' in caps:
            edit_strategy = self._resolve_edit_tool(env)
            if edit_strategy == 'replace_str':
                self._register_replace_str()
        if 'glob' in caps:
            self._register_glob()
        if 'grep' in caps:
            self._register_grep()

    def _register_ls(self) -> None:
        async def ls(path: str = '.') -> str:
            """List directory contents.

            Args:
                path: The directory path to list. Defaults to the working directory.
            """
            try:
                entries = await self.required_environment.ls(path)
            except (NotADirectoryError, PermissionError, OSError) as e:
                return f'Error: {e}'
            if not entries:
                return 'Empty directory.'
            lines: list[str] = []
            for entry in entries:
                if entry.is_dir:
                    lines.append(f'{entry.name}/')
                elif entry.size is not None:
                    lines.append(f'{entry.name} ({entry.size} bytes)')
                else:
                    lines.append(entry.name)
            return '\n'.join(lines)

        self.tool(ls)

    def _register_shell(self) -> None:
        async def shell(command: str, timeout: int = 120) -> str:
            """Execute a shell command and return its output.

            Use this for running scripts, installing packages, and other terminal operations.

            Args:
                command: The shell command to execute.
                timeout: Maximum seconds to wait for the command to complete.
            """
            result = await self.required_environment.shell(command, timeout=timeout)
            parts: list[str] = []
            if result.output:
                parts.append(result.output)
            if result.truncated:
                parts.append('[output truncated]')
            parts.append(f'Exit code: {result.exit_code}')
            return '\n'.join(parts)

        self.tool(requires_approval=self._require_shell_approval)(shell)

    def _register_read_file(self) -> None:
        async def read_file(path: str, offset: int = 0, limit: int = 2000) -> Any:
            """Read a file from the filesystem.

            Returns text files with line numbers, or renders image files for visual inspection.
            Use offset and limit to read specific sections of large files.

            Args:
                path: The file path to read.
                offset: The line number to start reading from (0-indexed).
                limit: Maximum number of lines to read.
            """
            try:
                content = await self.required_environment.read_file(path, offset=offset, limit=limit)
                if isinstance(content, bytes):
                    # Image file — return as BinaryContent or placeholder
                    if self._image_support:
                        if len(content) > self._max_image_bytes:
                            return f'Error: Image too large ({len(content)} bytes, max {self._max_image_bytes} bytes).'
                        ext = posixpath.splitext(path)[1].lower()
                        media_type = IMAGE_MEDIA_TYPES.get(ext, 'application/octet-stream')
                        return BinaryContent(data=content, media_type=media_type)
                    else:
                        return f'[Image file: {path} — image_support is disabled on this toolset]'
                return content
            except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
                return f'Error: {e}'

        self.tool(read_file)

    def _register_write_file(self) -> None:
        async def write_file(path: str, content: str) -> str:
            """Create or overwrite a file.

            The file and any parent directories will be created if they do not exist.

            Args:
                path: The file path to write.
                content: The content to write to the file.
            """
            try:
                await self.required_environment.write_file(path, content)
                return f'File written: {path}'
            except (PermissionError, OSError) as e:
                return f'Error: {e}'

        self.tool(requires_approval=self._require_write_approval)(write_file)

    def _register_replace_str(self) -> None:
        async def replace_str(path: str, old: str, new: str, replace_all: bool = False) -> str:
            """Edit a file by exact string replacement.

            The old string must match exactly (including whitespace and indentation).
            For uniqueness, include surrounding context lines.
            Only use this after reading the file first.

            Args:
                path: The file path to edit.
                old: The exact text to find and replace.
                new: The replacement text.
                replace_all: Replace all occurrences. Defaults to false (old must be unique).
            """
            try:
                count = await self.required_environment.replace_str(path, old, new, replace_all=replace_all)
                return f'Replaced {count} occurrence{"s" if count != 1 else ""} in {path}.'
            except (FileNotFoundError, ValueError) as e:
                raise ModelRetry(str(e))

        self.tool(requires_approval=self._require_write_approval)(replace_str)

    def _register_glob(self) -> None:
        async def glob_tool(pattern: str, path: str = '.') -> str:
            """Find files matching a glob pattern.

            Supports patterns like ``**/*.py``, ``src/**/*.ts``.
            Returns up to 100 matching file paths.

            Args:
                pattern: The glob pattern to match files against.
                path: The directory to search in. Defaults to the working directory.
            """
            try:
                matches = await self.required_environment.glob(pattern, path=path)
            except (PermissionError, OSError) as e:
                return f'Error: {e}'
            if not matches:
                return 'No files found.'
            truncated = len(matches) > 100
            matches = matches[:100]
            result = '\n'.join(matches)
            if truncated:
                result += '\n[... truncated, showing first 100 matches]'
            return result

        self.tool(name='glob')(glob_tool)

    def _register_grep(self) -> None:
        async def grep_tool(
            pattern: str,
            path: str | None = None,
            glob: str | None = None,
            output_mode: Literal['content', 'files_with_matches', 'count'] = 'content',
        ) -> str:
            """Search file contents with a regex pattern.

            Args:
                pattern: The regex pattern to search for.
                path: The file or directory to search in.
                glob: Glob pattern to filter which files are searched (e.g. ``*.py``).
                output_mode: Controls output format:
                    ``content`` (default) shows matching lines with file paths and line numbers,
                    ``files_with_matches`` shows only file paths,
                    ``count`` shows match counts per file.
            """
            try:
                result = await self.required_environment.grep(
                    pattern, path=path, glob_pattern=glob, output_mode=output_mode
                )
            except (PermissionError, OSError, re.error) as e:
                return f'Error: {e}'
            if not result.strip():
                return 'No matches found.'
            return result

        self.tool(name='grep')(grep_tool)

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Wrap the ExecutionEnvironmentToolset in a PrefixedToolset to avoid name conflicts.'

    @property
    def environment(self) -> ExecutionEnvironment | None:
        """The active execution environment, or None if not configured.

        Checks the context var override first, then falls back to the default.
        """
        override = self._environment_override.get()
        if override is not None:
            return override
        return self._default_environment

    @property
    def required_environment(self) -> ExecutionEnvironment:
        """The active execution environment, raising if not configured.

        Raises:
            RuntimeError: If no environment is available.
        """
        env = self.environment
        if env is not None:
            return env
        raise RuntimeError(
            'No execution environment configured. Pass one to ExecutionEnvironmentToolset() or use .use_environment().'
        )

    @contextmanager
    def use_environment(self, environment: ExecutionEnvironment) -> Iterator[None]:
        """Override the execution environment for the current context.

        Useful for testing or using different environments at different call sites.

        Usage:
            ```python {test="skip" lint="skip"}
            with toolset.use_environment(test_env):
                result = await agent.run('test prompt', toolsets=[toolset])
            ```

        Args:
            environment: The execution environment to use within this context.
        """
        token = self._environment_override.set(environment)
        try:
            yield
        finally:
            self._environment_override.reset(token)

    # --- Lifecycle ---

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            self._running_count += 1
            if self._running_count == 1:
                self._exit_stack = AsyncExitStack()
                try:
                    await self._exit_stack.enter_async_context(self.required_environment)
                except Exception:
                    self._running_count -= 1
                    raise
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None
        return None
