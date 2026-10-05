"""Typed tool part for `CodeMode`'s `run_code` tool.

Defining the part registers the `'code_mode.run_code'` tool kind, which `CodeModeToolset` sets on
the `run_code` tool definition. Its call parts are then promoted to `RunCodeCallPart`, so hooks and
history processors can recognize `run_code` with `isinstance` instead of matching its name.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict

from pydantic_ai.messages import ToolCallPart

RUN_CODE_TOOL_KIND = 'code_mode.run_code'
"""The [`tool_kind`][pydantic_ai.tools.ToolDefinition.tool_kind] of `CodeMode`'s `run_code` tool."""


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


_RUN_CODE_ARGS_TA = TypeAdapter(RunCodeArgs)


@dataclass(repr=False)
class RunCodeCallPart(ToolCallPart, namespace='code_mode', tool_kind='run_code'):
    """Typed [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] for `CodeMode`'s `run_code` tool.

    A `run_code` call is promoted to this class through its `'code_mode.run_code'` kind, whatever
    the tool is called (for example after [`PrefixTools`][pydantic_ai.capabilities.PrefixTools]),
    so `isinstance(part, RunCodeCallPart)` recognizes it in hooks and history processors.
    """

    _: KW_ONLY

    args: str | RunCodeArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    """The call's arguments: a JSON string while they stream (or as some providers return them), else a `RunCodeArgs`."""

    @property
    def typed_args(self) -> RunCodeArgs | None:
        """The parsed arguments, or `None` while a streamed JSON string is still incomplete."""
        if self.args is None or isinstance(self.args, dict):
            return self.args
        try:
            return _RUN_CODE_ARGS_TA.validate_json(self.args)
        except ValidationError:
            return None

    @property
    def code(self) -> str | None:
        """The submitted Python code, or `None` while the arguments are still streaming."""
        typed = self.typed_args
        return typed['code'] if typed is not None else None
