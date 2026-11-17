"""Send an unhandled tool exception back to the model as a failed tool result instead of ending the run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    RunCancelled,
    SkipToolExecution,
    ToolFailed,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

__all__ = ['ReportToolErrors']

_PASS_THROUGH = (ModelRetry, ToolFailed, ApprovalRequired, CallDeferred, SkipToolExecution, RunCancelled)
"""Exceptions with their own meaning to core, which a tool raises on purpose or core already reports."""


def _describe(tool_name: str, error: Exception) -> str:
    return f'`{tool_name}` raised {type(error).__name__}: {error}'


@dataclass
class ReportToolErrors(AbstractCapability[AgentDepsT]):
    """Report an exception a tool raises to the model as a failed tool result, so the run continues.

    The exception becomes a `ToolFailed`, as if the tool had raised one itself. The model sees the tool name, the
    exception type and its message, and decides whether to call the tool again, try another way, or give up. There
    is no limit of its own: the run's `UsageLimits` bound a model that keeps calling a failing tool. `ModelRetry`,
    `ToolFailed`, `ApprovalRequired`, `CallDeferred`, `SkipToolExecution` and `RunCancelled` keep their usual meaning.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import ReportToolErrors

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[ReportToolErrors()])
    ```
    """

    format_error: Callable[[str, Exception], str] = _describe
    """Build the text the model sees from the tool name and the exception, for example to hide internals.

    The default names the tool, the exception type and its message.
    """

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return _ReportErrorsToolset(toolset, self.format_error)


@dataclass
class _ReportErrorsToolset(WrapperToolset[AgentDepsT]):
    format_error: Callable[[str, Exception], str]

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        try:
            return await super().call_tool(name, tool_args, ctx, tool)
        except _PASS_THROUGH:
            raise
        except Exception as error:
            raise ToolFailed(self.format_error(name, error)) from error
