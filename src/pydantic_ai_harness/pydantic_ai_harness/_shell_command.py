"""Recognize the shell command a tool call runs."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic_ai.tools import ToolDefinition


def shell_command_arg(tool_def: ToolDefinition) -> str | None:
    """The argument holding the shell command a tool runs, when its metadata declares `code_arg_language='shell'`.

    `Shell`'s `run_command`, `start_command`, and `shell` tools declare it, with `code_arg_name='command'`.
    """
    metadata = tool_def.metadata or {}
    name = metadata.get('code_arg_name')
    return name if metadata.get('code_arg_language') == 'shell' and isinstance(name, str) else None


def shell_command(tool_def: ToolDefinition, args: Mapping[str, object]) -> str | None:
    """The shell command a call runs, when its tool declares one."""
    name = shell_command_arg(tool_def)
    command = None if name is None else args.get(name)
    return command if isinstance(command, str) else None
