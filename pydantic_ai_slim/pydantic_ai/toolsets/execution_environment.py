"""ExecutionEnvironmentToolset — exposes coding-agent-style tools backed by an ExecutionEnvironment."""

from __future__ import annotations

import posixpath
import re
from asyncio import Lock
from collections.abc import Callable, Iterator, Sequence
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Self

from ..environments._base import (
    IMAGE_EXTENSIONS,
    IMAGE_MEDIA_TYPES,
    ExecutionEnvironment,
    ToolName,
)
from ..exceptions import ModelRetry
from ..messages import BinaryContent
from ..toolsets.function import FunctionToolset

if TYPE_CHECKING:
    from .._run_context import AgentDepsT, RunContext
    from ..toolsets.abstract import ToolsetTool


class ExecutionEnvironmentToolset(FunctionToolset[Any]):
    """Toolset providing coding-agent-style tools backed by an `ExecutionEnvironment`.

    Tool names and schemas are designed to match what popular coding agents
    expose, so models are well-trained on them.

    Tools are dynamically registered based on the environment's `capabilities`,
    filtered by `include`/`exclude`.

    The environment can be:
    - Passed directly at construction time via `shared_environment` (shared across concurrent runs)
    - Created per-run via `environment_factory` (isolated concurrent runs)
    - Set/overridden via context var using `use_environment()` (for testing or per-call-site config)

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
        shared_environment: ExecutionEnvironment | None = None,
        *,
        environment_factory: Callable[[], ExecutionEnvironment] | None = None,
        include: Sequence[ToolName] | None = None,
        exclude: Sequence[ToolName] | None = None,
        edit_strategy: Literal['edit_file:replace_str', 'edit_file:apply_patch'] | None = None,
        require_shell_approval: bool = False,
        require_write_approval: bool = False,
        image_support: bool = True,
        max_image_bytes: int = 50 * 1024 * 1024,
        max_retries: int = 1,
        id: str | None = None,
    ):
        """Create a new execution environment toolset.

        Args:
            shared_environment: A shared execution environment for tool execution.
                All concurrent runs share this single environment instance.
                Can also be set later via `use_environment()`.
            environment_factory: A callable that creates a fresh environment per
                `async with toolset:` entry. Use this for concurrent runs that need
                isolation (e.g. separate Docker containers). Mutually exclusive with
                `shared_environment`.
            include: Tool names to include. `None` means all tools supported
                by the environment. Pass an explicit sequence to restrict to
                specific tools.
            exclude: Tool names to exclude. `None` defaults to no exclusions.
                Pass an explicit sequence to exclude specific tools.
            edit_strategy: Which edit strategy to use. `None` auto-selects
                `'edit_file:replace_str'` if supported by the environment.
            require_shell_approval: Whether the `shell` tool requires human-in-the-loop
                approval before execution. Recommended for `LocalEnvironment` where
                commands run directly on the host.
            require_write_approval: Whether `write_file` and edit tools require
                human-in-the-loop approval before execution.
            image_support: Whether `read_file` should return images as `BinaryContent`
                for multimodal models (otherwise returns a placeholder message).
            max_image_bytes: Maximum image file size to return as BinaryContent.
            max_retries: Maximum retries per tool call.
            id: Optional unique ID for the toolset (required for durable execution).
        """
        if shared_environment is not None and environment_factory is not None:
            raise ValueError('Cannot provide both shared_environment and environment_factory.')

        super().__init__(max_retries=max_retries, id=id)
        self._shared_environment = shared_environment
        self._environment_factory = environment_factory
        self._environment_override: ContextVar[ExecutionEnvironment | None] = ContextVar(
            f'_environment_override_{id or "environment"}', default=None
        )
        self._per_run_state: ContextVar[tuple[AsyncExitStack, Token[ExecutionEnvironment | None]] | None] = ContextVar(
            f'_per_run_state_{id or "environment"}', default=None
        )
        self._include: frozenset[ToolName] | None = frozenset(include) if include is not None else None
        self._exclude: frozenset[ToolName] = frozenset(exclude) if exclude else frozenset()
        self._edit_strategy: Literal['edit_file:replace_str', 'edit_file:apply_patch'] | None = edit_strategy
        self._image_support = image_support
        self._max_image_bytes = max_image_bytes
        self._require_shell_approval = require_shell_approval
        self._require_write_approval = require_write_approval
        self._enter_lock: Lock = Lock()
        self._running_count: int = 0
        self._exit_stack: AsyncExitStack | None = None

        # Register all tools unconditionally so schemas are built eagerly.
        # get_tools() filters at runtime based on the current environment's capabilities.
        self._register_tools()

    def _resolve_tool_names(self, env: ExecutionEnvironment) -> frozenset[str]:
        """Determine which tool names to expose, based on the environment's capabilities and include/exclude."""
        # Map env capabilities → tool names (most 1:1, but edit_file:* → edit_file)
        tool_names: set[str] = set()
        for cap in env.capabilities:
            if cap.startswith('edit_file:'):
                continue  # handled below
            tool_names.add(cap)

        # Add edit_file if the resolved strategy's capability is available
        if self._resolve_edit_tool(env) is not None:
            tool_names.add('edit_file')

        # Apply include/exclude at the tool-name level
        if self._include is not None:
            tool_names &= self._include
        tool_names -= self._exclude

        return frozenset(tool_names)

    def _resolve_edit_tool(
        self, env: ExecutionEnvironment
    ) -> Literal['edit_file:replace_str', 'edit_file:apply_patch'] | None:
        """Determine which edit strategy to use.

        If ``edit_strategy`` was explicitly set and the environment supports it,
        that strategy is used. Otherwise falls back to auto-detection
        (preferring ``replace_str`` over ``apply_patch``).
        """
        env_caps = env.capabilities
        if self._edit_strategy is not None and self._edit_strategy in env_caps:
            return self._edit_strategy
        if 'edit_file:replace_str' in env_caps:
            return 'edit_file:replace_str'
        if 'edit_file:apply_patch' in env_caps:
            return 'edit_file:apply_patch'
        return None

    def _register_tools(self) -> None:
        """Register all tools unconditionally.

        Filtering based on the environment's capabilities and include/exclude
        is deferred to ``get_tools()``, which runs at request time when the
        active environment is known.
        """
        self._register_ls()
        self._register_shell()
        self._register_read_file()
        self._register_write_file()
        self._register_edit_file()
        self._register_glob()
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
                    ext = posixpath.splitext(path)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        # Image file — return as BinaryContent or placeholder
                        if self._image_support:
                            if len(content) > self._max_image_bytes:
                                return (
                                    f'Error: Image too large ({len(content)} bytes, max {self._max_image_bytes} bytes).'
                                )
                            media_type = IMAGE_MEDIA_TYPES.get(ext, 'application/octet-stream')
                            return BinaryContent(data=content, media_type=media_type)
                        else:
                            return f'[Image file: {path} — image_support is disabled on this toolset]'
                    else:
                        return f'[Binary file: {path} — cannot display as text]'
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

    def _register_edit_file(self) -> None:
        async def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
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

        self.tool(requires_approval=self._require_write_approval)(edit_file)

    def _register_glob(self) -> None:
        async def glob_tool(pattern: str, path: str = '.') -> str:
            """Find files matching a glob pattern.

            Supports patterns like `**/*.py`, `src/**/*.ts`.
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
                glob: Glob pattern to filter which files are searched (e.g. `*.py`).
                output_mode: Controls output format:
                    `content` (default) shows matching lines with file paths and line numbers,
                    `files_with_matches` shows only file paths,
                    `count` shows match counts per file.
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

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await super().get_tools(ctx)
        tool_names = self._resolve_tool_names(self.required_environment)
        return {name: tool for name, tool in all_tools.items() if name in tool_names}

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Wrap the ExecutionEnvironmentToolset in a PrefixedToolset to avoid name conflicts.'

    @property
    def environment(self) -> ExecutionEnvironment | None:
        """The active execution environment, or None if not configured.

        Checks the context var override first (which includes per-run factory
        environments), then falls back to the shared environment.
        """
        override = self._environment_override.get()
        if override is not None:
            return override
        return self._shared_environment

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
        if self._environment_factory is not None:
            env = self._environment_factory()
            stack = AsyncExitStack()
            await stack.enter_async_context(env)
            token = self._environment_override.set(env)
            self._per_run_state.set((stack, token))
        else:
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
        if self._environment_factory is not None:
            state = self._per_run_state.get()
            if state is not None:  # pragma: no branch
                stack, token = state
                await stack.aclose()
                self._environment_override.reset(token)
                self._per_run_state.set(None)
        else:
            async with self._enter_lock:
                self._running_count -= 1
                if self._running_count == 0 and self._exit_stack is not None:
                    await self._exit_stack.aclose()
                    self._exit_stack = None
        return None
