"""Filesystem-based skill resources, scripts, and executors.

This module provides:
- FileBasedSkillResource: File-based skill resource implementation
- FileBasedSkillScript: File-based skill script implementation
- LocalSkillScriptExecutor: Execute scripts using local Python subprocess
- CallableSkillScriptExecutor: Wrap a callable in the executor interface
- Factory functions for creating file-based resources and scripts

Implementations:
- [`LocalSkillScriptExecutor`][pydantic_ai.toolsets.skills.LocalSkillScriptExecutor]: Execute scripts using local Python subprocess
- [`CallableSkillScriptExecutor`][pydantic_ai.toolsets.skills.CallableSkillScriptExecutor]: Wrap a callable in the executor interface
- [`FileBasedSkillResource`][pydantic_ai.toolsets.skills.FileBasedSkillResource]: File-based resource with disk loading
- [`FileBasedSkillScript`][pydantic_ai.toolsets.skills.FileBasedSkillScript]: File-based script with subprocess execution
"""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anyio
import yaml

from .._utils import is_async_callable, run_in_executor
from ._exceptions import SkillResourceLoadError, SkillScriptExecutionError
from ._types import SkillResource, SkillScript


@dataclass
class FileBasedSkillResource(SkillResource):
    """A file-based skill resource that loads content from disk.

    This subclass extends SkillResource to add filesystem support.
    The uri attribute points to the file location and serves as the unique identifier.
    """

    async def load(self, ctx: Any, args: dict[str, Any] | None = None) -> Any:
        """Load resource content from file.

        JSON and YAML files are parsed; falls back to text if parsing fails.
        Other file types are returned as UTF-8 text.

        Args:
            ctx: RunContext for accessing dependencies (unused for file-based resources).
            args: Named arguments (unused for file-based resources).

        Returns:
            Parsed dict (JSON/YAML) or UTF-8 text string.

        Raises:
            SkillResourceLoadError: If file cannot be read or path is invalid.
        """
        if not self.uri:
            raise SkillResourceLoadError(f"Resource '{self.name}' has no URI")

        resource_path = Path(self.uri)

        try:
            content = resource_path.read_text(encoding='utf-8')
        except OSError as e:
            raise SkillResourceLoadError(f"Failed to read resource '{self.name}': {e}") from e

        file_extension = Path(self.name).suffix.lower()

        if file_extension == '.json':
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content

        elif file_extension in ('.yaml', '.yml'):
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError:
                return content

        return content


class LocalSkillScriptExecutor:
    """Execute skill scripts using local Python interpreter via subprocess.

    Executes file-based scripts as subprocesses with args passed as command-line named arguments.
    Dictionary keys are used exactly as provided (e.g., {"max-papers": 5} becomes --max-papers 5).
    Uses anyio.run_process for async-compatible subprocess execution.

    Note:
        All scripts must accept named arguments. Positional arguments are not supported.

    Attributes:
        timeout: Execution timeout in seconds.
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
        script: SkillScript,
        args: dict[str, Any] | None = None,
    ) -> Any:
        """Run a skill script locally using subprocess.

        Args:
            script: The script to run.
            args: Named arguments as a dictionary.
                Boolean True emits flag only, False/None omits it,
                lists repeat the flag for each item, other types convert to string.

        Returns:
            Combined stdout and stderr output.

        Raises:
            SkillScriptExecutionError: If execution fails or times out.
        """
        if script.uri is None:
            raise SkillScriptExecutionError(f"Script '{script.name}' has no URI for subprocess execution")

        script_path = Path(script.uri)
        cmd = [self._python_executable, str(script_path)]

        if args:
            for key, value in args.items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f'--{key}')
                elif isinstance(value, list):
                    for item in cast(list[Any], value):
                        cmd.append(f'--{key}')
                        cmd.append(str(item))
                elif value is not None:
                    cmd.append(f'--{key}')
                    cmd.append(str(value))

        stdin_data: bytes | None = None
        cwd = str(script_path.parent)

        try:
            result = None
            with anyio.move_on_after(self.timeout) as scope:
                result = await anyio.run_process(
                    cmd,
                    check=False,
                    cwd=cwd,
                    input=stdin_data,
                )

            if scope.cancelled_caught or result is None:
                raise SkillScriptExecutionError(f"Script '{script.name}' timed out after {self.timeout} seconds")

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
    """Wraps a callable in a script executor interface.

    Allows users to provide custom execution logic for file-based scripts
    instead of using subprocess execution. Useful for remote execution, sandboxed
    execution, or other custom scenarios.

    Example:
        ```python
        from pydantic_ai.toolsets.skills import CallableSkillScriptExecutor, SkillsDirectory

        async def my_executor(script, args=None):
            # Custom execution logic - script.uri contains the file path
            return f"Executed {script.name} at {script.uri} with {args}"

        executor = CallableSkillScriptExecutor(func=my_executor)
        directory = SkillsDirectory(path="./skills", script_executor=executor)
        ```
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        """Initialize the callable executor.

        Args:
            func: Callable that executes scripts. Can be sync or async.
                Should accept keyword arguments: script (SkillScript) and args (dict[str, Any] | None),
                and return the script output as a string. The script's uri attribute contains the file path.
        """
        self._func = func
        self._is_async = is_async_callable(func)

    async def run(
        self,
        script: SkillScript,
        args: dict[str, Any] | None = None,
    ) -> Any:
        """Run using the wrapped callable.

        Args:
            script: The script to run.
            args: Named arguments as a dictionary.

        Returns:
            Script output (can be any type like str, dict, etc.).
        """
        if self._is_async:
            function = cast(Callable[..., Awaitable[Any]], self._func)
            return await function(script=script, args=args)
        else:
            return await run_in_executor(self._func, script=script, args=args)


def create_file_based_resource(
    name: str,
    uri: str,
    description: str | None = None,
) -> FileBasedSkillResource:
    """Create a file-based resource.

    Args:
        name: Resource name (e.g., "FORMS.md", "data.json").
        uri: Path to the resource file.
        description: Optional resource description.

    Returns:
        FileBasedSkillResource instance.
    """
    return FileBasedSkillResource(
        name=name,
        uri=uri,
        description=description,
    )


@dataclass
class FileBasedSkillScript(SkillScript):
    """A file-based skill script that executes via subprocess.

    This subclass extends SkillScript to add subprocess execution support.
    The uri attribute points to the Python script file and serves as the unique identifier.

    Attributes:
        executor: Executor for running the script.
    """

    executor: LocalSkillScriptExecutor | CallableSkillScriptExecutor = LocalSkillScriptExecutor()

    async def run(self, ctx: Any, args: dict[str, Any] | None = None) -> Any:
        """Execute script file via subprocess.

        Args:
            ctx: RunContext for accessing dependencies (unused for file-based scripts).
            args: Named arguments passed as command-line arguments.
                Argument conversion rules:
                - Boolean True: emits flag only (e.g., --verbose)
                - Boolean False or None: omits the flag
                - List: repeats flag for each item (e.g., --item a --item b)
                - Other: converts to string (e.g., --query test)

        Returns:
            Script output (stdout + stderr).

        Raises:
            SkillScriptExecutionError: If execution fails.
        """
        if not self.uri:
            raise SkillScriptExecutionError(f"Script '{self.name}' has no URI")

        return await self.executor.run(self, args)


def create_file_based_script(
    name: str,
    uri: str,
    skill_name: str,
    executor: LocalSkillScriptExecutor | CallableSkillScriptExecutor,
    description: str | None = None,
) -> FileBasedSkillScript:
    """Create a file-based script with executor.

    Args:
        name: Script name (includes .py extension).
        uri: Path to the script file.
        skill_name: Name of the parent skill.
        executor: Executor for running the script.
        description: Optional script description.

    Returns:
        FileBasedSkillScript instance.
    """
    return FileBasedSkillScript(
        name=name,
        uri=uri,
        skill_name=skill_name,
        description=description,
        executor=executor,
    )
