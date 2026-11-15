"""Tool kind and typed tool part for `CodeMode`'s `run_code` tool.

`CodeModeToolset` gives the `run_code` tool definition the `'code_mode.run_code'` kind. Its call parts
are then promoted to `RunCodeCallPart`, so hooks and history processors can recognize `run_code` with
`isinstance` instead of matching its name.
"""

from __future__ import annotations

from typing import Annotated, NotRequired

from pydantic import Field
from typing_extensions import TypedDict

from pydantic_ai.messages import ToolCallPart, ToolKind


class RunCodeArgs(TypedDict):
    """Arguments of a `run_code` tool call."""

    code: Annotated[str, Field(description='The Python code to execute in the sandbox.')]
    """The Python code to execute in the sandbox."""

    restart: NotRequired[
        Annotated[
            bool,
            Field(
                description='Set to true to reset REPL state. When false (default), state is preserved between calls.'
            ),
        ]
    ]
    """Whether to reset the REPL state before running `code`."""


RUN_CODE_TOOL_NAME = 'run_code'
"""The name `CodeMode` gives its tool; a wrapped tool with this name is rejected as a conflict."""

RUN_CODE_KIND = ToolKind('code_mode', RUN_CODE_TOOL_NAME, args=RunCodeArgs)
"""The kind of `CodeMode`'s `run_code` tool."""


class RunCodeCallPart(ToolCallPart[RunCodeArgs], kind=RUN_CODE_KIND):
    """Typed [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] for `CodeMode`'s `run_code` tool.

    A `run_code` call is promoted to this class through its `'code_mode.run_code'` kind, whatever
    the tool is called (for example after [`PrefixTools`][pydantic_ai.capabilities.PrefixTools]),
    so `isinstance(part, RunCodeCallPart)` recognizes it in hooks and history processors.
    """

    @property
    def code(self) -> str | None:
        """The submitted Python code, or `None` if the arguments are incomplete (still streaming) or malformed."""
        typed = self.typed_args
        return typed['code'] if typed is not None else None
