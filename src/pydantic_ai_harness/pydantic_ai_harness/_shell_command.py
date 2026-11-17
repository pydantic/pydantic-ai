"""Recognize the shell command a tool call runs."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic_ai.tools import ToolDefinition


def shell_command(tool_def: ToolDefinition, args: Mapping[str, object]) -> str | None:
    """The shell command a call runs, when its tool's metadata declares `code_arg_language='shell'` and a `code_arg_name`.

    `Shell`'s `run_command`, `start_command`, and `shell` tools declare both.
    """
    metadata = tool_def.metadata or {}
    name = metadata.get('code_arg_name')
    if metadata.get('code_arg_language') != 'shell' or not isinstance(name, str):
        return None
    command = args.get(name)
    return command if isinstance(command, str) else None
