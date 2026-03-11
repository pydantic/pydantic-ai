"""Shell toolset for local command execution with provider-native format support.

Uses provider-native format when supported (Anthropic ``bash_20250124``, OpenAI
``shell`` with ``local`` environment), falls back to a standard function tool otherwise.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anyio
from pydantic_core import SchemaValidator, core_schema

from .._run_context import AgentDepsT, RunContext
from ..exceptions import ModelRetry
from ..tools import ShellNativeDefinition, ToolDefinition
from .abstract import AbstractToolset, ToolsetTool

__all__ = (
    'ShellExecutor',
    'ShellOutput',
    'ShellToolset',
)

_SHELL_TOOL_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'command': {'type': 'string', 'description': 'The shell command to execute.'},
        'restart': {'type': 'boolean', 'description': 'Restart the shell session.', 'default': False},
    },
    'required': ['command'],
}

_SHELL_ARGS_VALIDATOR = SchemaValidator(
    core_schema.typed_dict_schema(
        {
            'command': core_schema.typed_dict_field(core_schema.str_schema()),
            'restart': core_schema.typed_dict_field(
                core_schema.with_default_schema(core_schema.bool_schema(), default=False),
                required=False,
            ),
        }
    )
)


@dataclass(kw_only=True)
class ShellOutput:
    """Result of a shell command execution."""

    output: str
    exit_code: int


class ShellExecutor(Protocol):
    """Protocol for shell command execution backends."""

    async def execute(self, command: str) -> ShellOutput:
        """Execute a shell command and return the output."""
        ...

    async def restart(self) -> ShellOutput:
        """Restart the shell session and return a confirmation message."""
        ...

    async def close(self) -> None:
        """Clean up any resources (e.g. subprocess)."""
        ...


class _LocalShellExecutor:
    """Subprocess-based shell executor with a persistent session.

    Maintains a running shell process so that state (``cd``, ``export``, etc.)
    persists across consecutive ``execute()`` calls, matching the behaviour that
    models expect from Anthropic's ``bash`` tool.
    """

    def __init__(self, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
        self._cwd = str(cwd) if cwd is not None else None
        self._env = env
        self._process: asyncio.subprocess.Process | None = None

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._process is None or self._process.returncode is not None:
            self._process = await asyncio.create_subprocess_exec(
                '/bin/bash',
                '--noediting',
                '--norc',
                '--noprofile',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._cwd,
                env=self._env,
            )
        return self._process

    async def execute(self, command: str) -> ShellOutput:
        process = await self._ensure_process()
        assert process.stdin is not None
        assert process.stdout is not None

        # Use a unique sentinel to detect end of output
        sentinel = f'__PYDANTIC_AI_EXIT_{id(self)}__'
        wrapped = f'{command}\n__exit_code=$?\necho ""\necho "{sentinel} $__exit_code"\n'
        process.stdin.write(wrapped.encode())
        await process.stdin.drain()

        output_lines: list[str] = []
        exit_code = 0
        while True:
            line_bytes = await process.stdout.readline()
            if not line_bytes:
                # Process ended unexpectedly
                exit_code = process.returncode or 1
                break
            line = line_bytes.decode(errors='replace')
            if sentinel in line:
                # Parse exit code from sentinel line
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        exit_code = int(parts[-1])
                    except ValueError:
                        exit_code = 1
                break
            output_lines.append(line)

        output = ''.join(output_lines)
        # Strip trailing newline added by our echo
        if output.endswith('\n'):
            output = output[:-1]
        return ShellOutput(output=output, exit_code=exit_code)

    async def restart(self) -> ShellOutput:
        await self.close()
        return ShellOutput(output='Shell session restarted.', exit_code=0)

    async def close(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()  # pragma: no cover
            self._process = None


@dataclass
class ShellToolset(AbstractToolset[AgentDepsT]):
    """Toolset for local shell command execution with provider-native format.

    Uses provider-native format when supported (Anthropic ``bash_20250124``,
    OpenAI ``shell`` with ``local`` environment), falls back to a standard
    function tool otherwise.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai.toolsets import ShellToolset

        agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[ShellToolset.local(cwd='/workspace')])
        result = agent.run_sync('List all Python files')
    """

    executor: ShellExecutor
    """The shell executor. Use :meth:`ShellToolset.local` for a subprocess-based executor."""

    _: Any = field(init=False, repr=False, default=None)  # KW_ONLY sentinel

    tool_name: str = 'shell'
    """The name of the tool exposed to the model."""

    description: str = 'Execute a shell command and return the output.'
    """Description of the tool."""

    max_output_chars: int = 200_000
    """Maximum output characters. Exceeding output is truncated with a marker."""

    timeout: float | None = None
    """Timeout in seconds per command. Raises :class:`~pydantic_ai.exceptions.ModelRetry` on timeout."""

    max_retries: int = 1
    """Maximum number of retries for failed tool calls."""

    _id: str | None = field(default=None, repr=False)

    @classmethod
    def local(
        cls,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ShellToolset[AgentDepsT]:
        """Create a ``ShellToolset`` with a subprocess-based executor.

        Args:
            cwd: Working directory for the shell session.
            env: Environment variables for the shell session.
            **kwargs: Additional arguments passed to ``ShellToolset``.
        """
        return cls(executor=_LocalShellExecutor(cwd=cwd, env=env), **kwargs)

    @property
    def id(self) -> str | None:
        return self._id

    async def __aexit__(self, *args: Any) -> bool | None:
        await self.executor.close()
        return None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tool_def = ToolDefinition(
            name=self.tool_name,
            description=self.description,
            parameters_json_schema=_SHELL_TOOL_SCHEMA,
            sequential=True,
            native_definition=ShellNativeDefinition(),
        )
        return {
            self.tool_name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self.max_retries,
                args_validator=_SHELL_ARGS_VALIDATOR,
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> str:
        restart = tool_args.get('restart', False)
        command = tool_args.get('command', '')

        if restart:
            result = await self._run_with_timeout(self.executor.restart)
        else:
            result = await self._run_with_timeout(lambda: self.executor.execute(command))

        output = result.output
        if len(output) > self.max_output_chars:
            total = len(output)
            output = output[: self.max_output_chars]
            output += f'\n[output truncated — {self.max_output_chars} chars of {total} total]'

        # Format output for the model
        return json.dumps({'output': output, 'exit_code': result.exit_code})

    async def _run_with_timeout(self, func: Any) -> ShellOutput:
        if self.timeout is not None:
            try:
                with anyio.fail_after(self.timeout):
                    return await func()
            except TimeoutError:
                raise ModelRetry(f'Command timed out after {self.timeout}s.') from None
        else:
            return await func()
