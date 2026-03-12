"""ExecutionEnvironmentToolset — exposes coding-agent-style tools backed by an ExecutionEnvironment."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import anyio
from typing_extensions import Protocol, Self

from ..environments._base import (
    EnvCapability,
    ExecutionEnvironment,
    TextFileReadResult,
)
from ..exceptions import ModelRetry
from ..messages import BinaryContent, infer_binary_media_type_from_path
from ..toolsets.function import FunctionToolset

if TYPE_CHECKING:
    from .._run_context import AgentDepsT, RunContext
    from ..toolsets.abstract import ToolsetTool


ExecutionEnvironmentToolName = Literal['shell', 'read_file', 'write_file', 'replace_str']
"""Tool names exposed by `ExecutionEnvironmentToolset`."""

_TOOL_ENV_CAPABILITIES: dict[ExecutionEnvironmentToolName, EnvCapability] = {
    'shell': 'shell',
    'read_file': 'read_file',
    'write_file': 'write_file',
    'replace_str': 'replace_str',
}


@dataclass(frozen=True)
class _EnterState:
    kind: Literal['external_override', 'factory', 'shared']
    stack: AsyncExitStack | None = None
    token: Token[ExecutionEnvironment | None] | None = None


class ExecutionEnvironmentToolset(FunctionToolset[Any]):
    """Toolset providing coding-agent-style tools backed by an `ExecutionEnvironment`.

    Tool names and schemas are designed to match what popular coding agents
    expose, so models are well-trained on them.

    Tools are dynamically registered based on the environment's `capabilities`,
    filtered by `include`/`exclude`.

    The environment can be:
    - Passed directly at construction time as an instance (shared across concurrent runs)
    - Created per-run via a callable factory (isolated concurrent runs)
    - Set/overridden via context var using `use_environment()` (for testing or per-call-site config)

    Usage:
        ```python {test="skip" lint="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.environments.docker import DockerEnvironment
        from pydantic_ai.toolsets.execution_environment import ExecutionEnvironmentToolset

        toolset = ExecutionEnvironmentToolset(DockerEnvironment(image='python:3.12-slim'))

        agent = Agent('openai:gpt-5.2', toolsets=[toolset])
        result = await agent.run('Write a script that prints hello')
        ```
    """

    def __init__(
        self,
        environment: ExecutionEnvironment | Callable[[], ExecutionEnvironment] | None = None,
        *,
        include: Sequence[ExecutionEnvironmentToolName] | None = None,
        exclude: Sequence[ExecutionEnvironmentToolName] | None = None,
        require_shell_approval: bool = False,
        require_write_approval: bool = False,
        max_binary_content_bytes: int = 50 * 1024 * 1024,
        max_output_chars: int = 200_000,
        max_retries: int = 1,
        id: str | None = None,
    ):
        """Create a new execution environment toolset.

        Args:
            environment: The execution environment for tool execution. Can be:

                - An `ExecutionEnvironment` instance — shared across all concurrent runs.
                - A callable returning an `ExecutionEnvironment` — called once per
                  `async with toolset:` entry, creating isolated environments for
                  concurrent runs (e.g. separate Docker containers).
                - `None` — set later via `use_environment()`.
            include: Tool names to include. `None` means all tools supported
                by the environment. Pass an explicit sequence to restrict to
                specific tools.
            exclude: Tool names to exclude. `None` defaults to no exclusions.
                Pass an explicit sequence to exclude specific tools.
            require_shell_approval: Whether the `shell` tool requires human-in-the-loop
                approval before execution. Recommended for `LocalEnvironment` where
                commands run directly on the host.
            require_write_approval: Whether `write_file` and edit tools require
                human-in-the-loop approval before execution.
            max_binary_content_bytes: Maximum file size to return as BinaryContent (for images, etc.).
            max_output_chars: Maximum characters of tool output before truncation.
            max_retries: Maximum retries per tool call.
            id: Optional unique ID for the toolset (required for durable execution).
        """
        super().__init__(max_retries=max_retries, id=id)
        if callable(environment) and not isinstance(environment, ExecutionEnvironment):
            self._shared_environment: ExecutionEnvironment | None = None
            self._environment_factory: Callable[[], ExecutionEnvironment] | None = environment
        else:
            self._shared_environment = environment
            self._environment_factory = None
        self._environment_override: ContextVar[ExecutionEnvironment | None] = ContextVar(
            f'_environment_override_{id or "environment"}', default=None
        )
        self._enter_states: ContextVar[tuple[_EnterState, ...]] = ContextVar(
            f'_enter_states_{id or "environment"}', default=()
        )
        self._include: frozenset[ExecutionEnvironmentToolName] | None = (
            frozenset(include) if include is not None else None
        )
        self._exclude: frozenset[ExecutionEnvironmentToolName] = frozenset(exclude) if exclude else frozenset()
        self._max_binary_content_bytes = max_binary_content_bytes
        self._require_shell_approval = require_shell_approval
        self._require_write_approval = require_write_approval
        self._enter_lock = anyio.Lock()
        self._running_count: int = 0
        self._exit_stack: AsyncExitStack | None = None
        self._max_output_chars = max_output_chars

        # Register all tools unconditionally so schemas are built eagerly.
        # get_tools() filters at runtime based on the current environment's capabilities.
        self._register_tools()

    def _resolve_tool_names(self, env: ExecutionEnvironment) -> frozenset[ExecutionEnvironmentToolName]:
        """Determine which tool names to expose, based on the environment's capabilities and include/exclude."""
        tool_names: set[ExecutionEnvironmentToolName] = {
            tool_name for tool_name, capability in _TOOL_ENV_CAPABILITIES.items() if capability in env.capabilities
        }

        if self._include is not None:
            tool_names &= self._include
        tool_names -= self._exclude

        return frozenset(tool_names)

    def _register_tools(self) -> None:
        """Register all tools unconditionally.

        Filtering based on the environment's capabilities and include/exclude
        is deferred to ``get_tools()``, which runs at request time when the
        active environment is known.
        """
        self._register_shell()
        self._register_read_file()
        self._register_write_file()
        self._register_replace_str()

    def _register_shell(self) -> None:
        async def shell(command: str, timeout: int = 120) -> str:
            """Execute a shell command and return its output.

            Use this for running scripts, installing packages, and other terminal operations.

            Args:
                command: The shell command to execute.
                timeout: Maximum seconds to wait for the command to complete.
            """
            result = await self.required_environment.shell(command, timeout=timeout)
            exit_line = f'\nExit code: {result.exit_code}'
            output = result.output or ''
            # Truncate command output first, reserving space for the exit code line
            max_output = max(self._max_output_chars - len(exit_line), 0)
            if len(output) > max_output:
                output = output[:max_output] + '\n... (truncated)'
            return (output + exit_line).strip()

        _extend_docstring(shell, self.required_environment.capability_details().get('shell'))

        self.tool(requires_approval=self._require_shell_approval)(shell)

    def _register_read_file(self) -> None:
        async def read_file(path: str, offset: int = 0, limit: int = 2000) -> Any:
            """Read a file from the filesystem.

            Returns text files with line numbers, or supported binary files as `BinaryContent`.
            Use offset and limit to read specific sections of large text files.

            Args:
                path: The file path to read.
                offset: The line number to start reading from (0-indexed).
                limit: Maximum number of lines to read.
            """
            try:
                binary_media_type = infer_binary_media_type_from_path(path)
                if binary_media_type is not None:
                    raw = await self.required_environment.read_file(path)
                    if len(raw) > self._max_binary_content_bytes:
                        return (
                            f'Error: Binary content too large ({len(raw)} bytes, '
                            f'max {self._max_binary_content_bytes} bytes).'
                        )
                    return BinaryContent(data=raw, media_type=binary_media_type)

                try:
                    text_result = await self.required_environment.read_text_file(path, offset=offset, limit=limit)
                except UnicodeDecodeError:
                    return f'[Binary file: {path} — cannot display as text]'

                content = _format_lines(text_result)
                if len(content) > self._max_output_chars:
                    content = content[: self._max_output_chars] + '\n... (truncated)'
                return content
            except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
                return f'Error: {e}'

        _extend_docstring(read_file, self.required_environment.capability_details().get('read_file'))

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

        _extend_docstring(write_file, self.required_environment.capability_details().get('write_file'))

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
            except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
                raise ModelRetry(str(e))

        _extend_docstring(replace_str, self.required_environment.capability_details().get('replace_str'))

        self.tool(requires_approval=self._require_write_approval)(replace_str)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await super().get_tools(ctx)
        env = self.required_environment
        tool_names = self._resolve_tool_names(env)
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
        The override environment's lifecycle is managed externally.

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
        if self._environment_override.get() is not None:
            self._push_enter_state(_EnterState(kind='external_override'))
            return self

        if self._environment_factory is not None:
            env = self._environment_factory()
            stack = AsyncExitStack()
            await stack.enter_async_context(env)
            token = self._environment_override.set(env)
            self._push_enter_state(_EnterState(kind='factory', stack=stack, token=token))
        else:
            async with self._enter_lock:
                self._running_count += 1
                if self._running_count == 1:
                    # Use _shared_environment directly (not required_environment) to avoid
                    # entering a use_environment() override into the shared exit stack.
                    env = self._shared_environment
                    if env is None:
                        self._running_count -= 1
                        raise RuntimeError(
                            'No execution environment configured. Pass one to ExecutionEnvironmentToolset() or use .use_environment().'
                        )
                    self._exit_stack = AsyncExitStack()
                    try:
                        await self._exit_stack.enter_async_context(env)
                    except Exception:
                        self._running_count -= 1
                        raise
            self._push_enter_state(_EnterState(kind='shared'))
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        state = self._pop_enter_state()
        if state.kind == 'external_override':
            return None

        if state.kind == 'factory':
            assert state.stack is not None
            assert state.token is not None
            await state.stack.aclose()
            self._environment_override.reset(state.token)
        else:
            async with self._enter_lock:
                self._running_count -= 1
                if self._running_count == 0 and self._exit_stack is not None:
                    await self._exit_stack.aclose()
                    self._exit_stack = None
        return None

    def _push_enter_state(self, state: _EnterState) -> None:
        states = self._enter_states.get()
        self._enter_states.set(states + (state,))

    def _pop_enter_state(self) -> _EnterState:
        states = self._enter_states.get()
        assert states
        state = states[-1]
        self._enter_states.set(states[:-1])
        return state


def _format_lines(read_result: TextFileReadResult) -> str:
    """Format a paginated text file read with line numbers and pagination hints."""
    lines = read_result.text.splitlines(keepends=True)
    numbered = [f'{i}\t{line}' for i, line in enumerate(lines, start=read_result.offset + 1)]
    result = ''.join(numbered)
    if not result.endswith('\n'):
        result += '\n'

    remaining = read_result.total_lines - (read_result.offset + len(lines))
    if remaining > 0:
        next_offset = read_result.offset + len(lines)
        result += f'... ({remaining} more lines. Use offset={next_offset} to continue reading.)\n'

    return result


class _HasDocs(Protocol):
    __doc__: str


def _extend_docstring(fn: _HasDocs, details: str | None) -> None:
    # Insert the details above the args
    if details is not None:
        docstring_parts = fn.__doc__.split('Args:')
        docstring_parts[0] = docstring_parts[0] + f'{details}\n\n'
        fn.__doc__ += 'Args:'.join(docstring_parts)
