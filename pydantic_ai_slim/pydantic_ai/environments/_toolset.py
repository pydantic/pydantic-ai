"""ExecutionToolset — exposes coding-agent-style tools backed by an ExecutionEnvironment."""

from __future__ import annotations

import posixpath
import re
from asyncio import Lock
from collections.abc import Iterator
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from typing_extensions import Self

from ..exceptions import ModelRetry
from ..messages import BinaryContent
from ..toolsets.function import FunctionToolset
from ._base import IMAGE_EXTENSIONS, IMAGE_MEDIA_TYPES, ExecutionEnvironment

_SYSTEM_PROMPT = """\
You have access to an execution environment with the following tools:

- **bash**: Execute shell commands (installing packages, running scripts, etc.)
- **read_file**: Read file contents with line numbers (images are rendered visually)
- **write_file**: Create or overwrite files
- **edit_file**: Edit files by exact string replacement (read the file first!)
- **glob**: Find files by pattern (e.g. "**/*.py")
- **grep**: Search file contents with regex

Best practices:
- Always read a file before editing it
- Use glob to find files before reading them
- Use grep to search for specific patterns across files
- Write scripts to disk and execute them for complex operations
- Check command output and iterate if something fails
"""


class ExecutionToolset(FunctionToolset[Any]):
    """Toolset providing coding-agent-style tools backed by an `ExecutionEnvironment`.

    Tool names and schemas are designed to match what popular coding agents
    expose, so models are well-trained on them.

    The environment can be:
    - Passed directly at construction time (most common)
    - Set/overridden via context var using `use_environment()` (for testing or per-call-site config)

    Usage:
        ```python {test="skip" lint="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.environments import ExecutionToolset
        from pydantic_ai.environments.docker import DockerSandbox

        sandbox = DockerSandbox(image='python:3.12-slim', packages=['numpy'])
        toolset = ExecutionToolset(sandbox)

        agent = Agent('openai:gpt-5.2', toolsets=[toolset])

        async with sandbox:
            result = await agent.run('Write a script that prints hello')
        ```
    """

    def __init__(
        self,
        environment: ExecutionEnvironment | None = None,
        *,
        include_bash: bool = True,
        include_file_tools: bool = True,
        include_search_tools: bool = True,
        require_bash_approval: bool = False,
        require_write_approval: bool = False,
        image_support: bool = True,
        max_image_bytes: int = 50 * 1024 * 1024,
        max_retries: int = 1,
        id: str | None = None,
    ):
        """Create a new execution toolset.

        Args:
            environment: The execution environment to use for tool execution.
                Can also be set later via `use_environment()`.
            include_bash: Whether to include the `bash` tool for shell command execution.
            include_file_tools: Whether to include file tools (read_file, write_file, edit_file).
            include_search_tools: Whether to include search tools (glob, grep).
            require_bash_approval: Whether the `bash` tool requires human-in-the-loop
                approval before execution. Recommended for `LocalEnvironment` where
                commands run directly on the host.
            require_write_approval: Whether `write_file` and `edit_file` require
                human-in-the-loop approval before execution.
            image_support: Whether read_file should return images as `BinaryContent`
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
        self._image_support = image_support
        self._max_image_bytes = max_image_bytes
        self._require_bash_approval = require_bash_approval
        self._require_write_approval = require_write_approval
        self._enter_lock: Lock = Lock()
        self._running_count: int = 0
        self._exit_stack: AsyncExitStack | None = None

        # Register tools as plain async functions that close over `self`
        # to access self.required_environment. Schemas and descriptions are
        # auto-generated from the function signatures and docstrings.

        if include_bash:
            self._register_bash()
        if include_file_tools:
            self._register_file_tools()
        if include_search_tools:
            self._register_search_tools()

    def _register_bash(self) -> None:
        async def bash(command: str, timeout: int = 120) -> str:
            """Execute a shell command and return its output.

            Use this for running scripts, installing packages, and other terminal operations.

            Args:
                command: The shell command to execute.
                timeout: Maximum seconds to wait for the command to complete.
            """
            result = await self.required_environment.execute(command, timeout=timeout)
            parts: list[str] = []
            if result.output:
                parts.append(result.output)
            if result.truncated:
                parts.append('[output truncated]')
            parts.append(f'Exit code: {result.exit_code}')
            return '\n'.join(parts)

        self.tool(requires_approval=self._require_bash_approval)(bash)

    def _register_file_tools(self) -> None:
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
                ext = posixpath.splitext(path)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    if self._image_support:
                        raw = await self.required_environment.read_file_bytes(path)
                        if len(raw) > self._max_image_bytes:
                            return f'Error: Image too large ({len(raw)} bytes, max {self._max_image_bytes} bytes).'
                        media_type = IMAGE_MEDIA_TYPES.get(ext, 'application/octet-stream')
                        return BinaryContent(data=raw, media_type=media_type)
                    else:
                        return f'[Image file: {path} — image_support is disabled on this toolset]'

                return await self.required_environment.read_file(path, offset=offset, limit=limit)
            except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
                return f'Error: {e}'

        self.tool(read_file)

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

        async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
            """Edit a file by exact string replacement.

            The old_string must match exactly (including whitespace and indentation).
            For uniqueness, include surrounding context lines.
            Only use this after reading the file first.

            Args:
                path: The file path to edit.
                old_string: The exact text to find and replace.
                new_string: The replacement text.
                replace_all: Replace all occurrences. Defaults to false (old_string must be unique).
            """
            try:
                count = await self.required_environment.edit_file(path, old_string, new_string, replace_all=replace_all)
                return f'Replaced {count} occurrence{"s" if count != 1 else ""} in {path}.'
            except (FileNotFoundError, ValueError) as e:
                raise ModelRetry(str(e))

        self.tool(requires_approval=self._require_write_approval)(edit_file)

    def _register_search_tools(self) -> None:
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

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Wrap the ExecutionToolset in a PrefixedToolset to avoid name conflicts.'

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
            'No execution environment configured. Pass one to ExecutionToolset() or use .use_environment().'
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

    # TODO: Once https://github.com/pydantic/pydantic-ai/pull/4123 is merged,
    # override `async def instructions(self, ctx) -> str | None` to return
    # self.system_prompt so it's automatically injected into the agent's prompt.

    @property
    def system_prompt(self) -> str:
        """Suggested system prompt describing available tools and best practices.

        Includes the base toolset prompt and any environment-specific additions
        from the environment's `system_prompt` property.
        """
        env = self.environment
        if env is not None:
            env_prompt = env.system_prompt
            if env_prompt:
                return _SYSTEM_PROMPT + '\n' + env_prompt
        return _SYSTEM_PROMPT

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
