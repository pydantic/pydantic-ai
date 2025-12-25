"""Script execution implementations and LocalSkill for skills.

This module provides implementations for executing skill scripts in different environments
and the LocalSkill class that implements filesystem-based skill operations.

Implementations:
- [`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill]: Filesystem-based skill implementation
- [`LocalSkillScriptExecutor`][pydantic_ai.toolsets.skills.LocalSkillScriptExecutor]: Execute scripts using local Python subprocess
- [`CallableSkillScriptExecutor`][pydantic_ai.toolsets.skills.CallableSkillScriptExecutor]: Wrap a callable in the executor interface

Executors handle script execution with timeout support, proper error handling,
and async-compatible subprocess management using anyio.
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anyio

from ..._run_context import AgentDepsT, RunContext
from ..._utils import is_async_callable, run_in_executor
from ._exceptions import SkillResourceLoadError, SkillResourceNotFoundError, SkillScriptExecutionError
from ._types import Skill, SkillScript


@dataclass
class LocalSkill(Skill):
    """Filesystem-based skill implementation.

    This class implements the abstract Skill interface for skills stored on the local filesystem.
    It provides concrete implementations for reading resources and running scripts.

    Attributes:
        name: Skill name (from metadata).
        uri: URI string for skill's base location (filesystem path).
        metadata: Parsed metadata from SKILL.md.
        content: Main content from SKILL.md (without frontmatter).
        resources: Resource files (FORMS.md, etc.). None if no resources.
        scripts: Available scripts in the skill directory or scripts/ subdirectory.
            None if no scripts.
        script_executor: Executor for running scripts. Defaults to LocalSkillScriptExecutor
            if not provided. Can be None for skills without scripts.
    """

    async def read_resource(self, ctx: RunContext[AgentDepsT], resource_uri: str) -> str:
        """Read a resource file from this skill.

        For filesystem-based skills, reads the resource directly from the filesystem
        with proper path validation and security checks.

        Args:
            ctx: The run context for this agent run.
            resource_uri: URI or name of the resource file.

        Returns:
            Resource file content.

        Raises:
            SkillResourceNotFoundError: If resource is not found.
            SkillResourceLoadError: If resource path is unsafe or cannot be read.
        """
        # Find the resource - match by URI or by name
        resource = None
        if self.resources:
            for r in self.resources:
                if r.uri == resource_uri or r.name == resource_uri:
                    resource = r
                    break

        if resource is None:
            available_resources = [r.name for r in self.resources] if self.resources else []
            raise SkillResourceNotFoundError(
                f"Resource '{resource_uri}' not found in skill '{self.name}'. "
                f'Available resources: {available_resources}'
            )

        # For filesystem resources, read directly
        resource_path = Path(resource.uri)

        # Security check - only if skill has a URI (filesystem-based)
        if self.uri:
            try:
                resource_path.resolve().relative_to(Path(self.uri).resolve())
            except ValueError as exc:
                raise SkillResourceLoadError('Resource path escapes skill directory.') from exc

        try:
            return resource_path.read_text(encoding='utf-8')
        except OSError as e:
            raise SkillResourceLoadError(f"Failed to read resource '{resource_uri}': {e}") from e

    async def run_script(
        self,
        ctx: RunContext[AgentDepsT],
        script_uri: str,
        args: list[str] | None = None,
    ) -> str:
        """Execute a script from this skill.

        Args:
            ctx: The run context for this agent run.
            script_uri: URI or name of the script (without .py extension).
            args: Optional command-line arguments.

        Returns:
            Script output (stdout and stderr combined).

        Raises:
            SkillResourceNotFoundError: If script is not found.
            SkillResourceLoadError: If script path is unsafe.
            SkillScriptExecutionError: If script execution fails.
        """
        # Find the script - match by URI or by name
        script = None
        if self.scripts:
            for s in self.scripts:
                if s.uri == script_uri or s.name == script_uri:
                    script = s
                    break

        if script is None:
            available = [s.name for s in self.scripts] if self.scripts else []
            raise SkillResourceNotFoundError(
                f"Script '{script_uri}' not found in skill '{self.name}'. Available scripts: {available}"
            )

        # Security check - only if skill has a URI (filesystem-based)
        if self.uri:
            script_path = Path(script.uri)
            try:
                script_path.resolve().relative_to(Path(self.uri).resolve())
            except ValueError as exc:
                raise SkillResourceLoadError('Script path escapes skill directory.') from exc

        # Get or create executor
        if self.script_executor is None:
            self.script_executor = LocalSkillScriptExecutor()

        # Execute using executor (timeout is configured on the executor)
        result = await self.script_executor.run(self, script, args)
        # Ensure we always return a string
        return str(result) if result is not None else 'Script executed with no output.'


class LocalSkillScriptExecutor:
    """Execute skill scripts using local Python interpreter.

    Uses anyio.run_process for async-compatible subprocess execution.
    Scripts are executed in the skill's directory with the specified
    Python executable.

    Attributes:
        timeout: Execution timeout in seconds (default: 30).
    """

    def __init__(
        self,
        python_executable: str | Path | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialize the local script executor.

        Args:
            python_executable: Path to Python executable. If None, uses sys.executable.
            timeout: Execution timeout in seconds (default: 30).
        """
        self._python_executable = str(python_executable) if python_executable else sys.executable
        self.timeout = timeout

    async def run(
        self,
        skill: Skill,
        script: SkillScript,
        args: list[str] | None = None,
    ) -> Any:
        """Run a skill script locally using subprocess.

        Args:
            skill: The skill containing the script.
            script: The script to run.
            args: Optional command-line arguments.

        Returns:
            Combined stdout and stderr output.

        Raises:
            SkillScriptExecutionError: If execution fails or times out.
        """
        # Convert URI to path for filesystem execution
        script_path = Path(script.uri)

        # Build command
        cmd = [self._python_executable, str(script_path)]
        if args:
            cmd.extend(args)

        try:
            # Use anyio.run_process for async-compatible execution
            # cwd is the skill's directory - use uri if available, otherwise None
            cwd = str(skill.uri) if skill.uri else None

            result = None
            with anyio.move_on_after(self.timeout) as scope:
                result = await anyio.run_process(
                    cmd,
                    check=False,  # We handle return codes manually
                    cwd=cwd,
                )

            # Check if timeout was reached (result would be None if cancelled)
            if scope.cancelled_caught or result is None:
                raise SkillScriptExecutionError(f"Script '{script.name}' timed out after {self.timeout} seconds")

            # Decode output from bytes to string
            output = result.stdout.decode('utf-8', errors='replace')
            if result.stderr:
                stderr = result.stderr.decode('utf-8', errors='replace')
                output += f'\n\nStderr:\n{stderr}'

            if result.returncode != 0:
                output += f'\n\nScript exited with code {result.returncode}'

            return output.strip() or '(no output)'

        except OSError as e:
            raise SkillScriptExecutionError(f"Failed to execute script '{script.name}': {e}") from e


class CallableSkillScriptExecutor:
    """Wraps a callable in a SkillScriptExecutor interface.

    Similar to FunctionSchema, handles both sync and async callables properly.
    """

    def __init__(
        self,
        func: Callable[..., Any],
    ) -> None:
        """Initialize the callable executor.

        Args:
            func: Callable that executes scripts. Can be sync or async.
        """
        self._func = func
        self._is_async = is_async_callable(func)

    async def run(
        self,
        skill: Skill,
        script: SkillScript,
        args: list[str] | None = None,
    ) -> Any:
        """Run using the wrapped callable.

        Args:
            skill: The skill containing the script.
            script: The script to run.
            args: Optional command-line arguments.

        Returns:
            Script output.
        """
        # Call the wrapped function - handle both sync and async similar to FunctionSchema
        if self._is_async:
            function = cast(Callable[..., Awaitable[Any]], self._func)
            return await function(skill=skill, script=script, args=args)
        else:
            return await run_in_executor(self._func, skill=skill, script=script, args=args)
