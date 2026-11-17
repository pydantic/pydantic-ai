"""Send an unhandled tool exception back to the model as a failed tool result instead of ending the run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    ToolFailed,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

__all__ = ['ReportToolErrors']

_PASS_THROUGH = (ToolFailed, ModelRetry, ApprovalRequired, CallDeferred, SkipToolExecution)
"""Signals an `on_tool_execute_error` hook that ran before this one may raise, which keep their meaning."""


def _describe(call: ToolCallPart, error: Exception) -> str:
    return f'`{call.tool_name}` raised {type(error).__name__}: {error}'


@dataclass
class ReportToolErrors(AbstractCapability[AgentDepsT]):
    """Report an exception a tool raises to the model as a failed tool result, so the run continues.

    The model sees the tool name, the exception type and its message, and decides whether to call the tool again,
    try another way, or give up. There is no limit of its own: the run's `UsageLimits` bound a model that keeps
    calling a failing tool. `ModelRetry`, `ToolFailed`, `ApprovalRequired`, `CallDeferred` and `SkipToolExecution`
    keep their usual meaning, and a tool that runs out of retries still ends the run.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import ReportToolErrors

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[ReportToolErrors()])
    ```
    """

    format_error: Callable[[ToolCallPart, Exception], str] = _describe
    """Build the text the model sees from the failed call and the exception, for example to hide internals.

    The default names the tool, the exception type and its message.
    """

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        """Raise `ToolFailed` with the formatted error, unless it is a signal or the tool ran out of retries."""
        if isinstance(error, _PASS_THROUGH) or (
            isinstance(error, UnexpectedModelBehavior) and isinstance(error.__cause__, ModelRetry)
        ):
            raise error
        raise ToolFailed(self.format_error(call, error)) from error
