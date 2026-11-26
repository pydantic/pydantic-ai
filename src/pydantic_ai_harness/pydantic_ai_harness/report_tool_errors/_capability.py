"""Send an unhandled tool exception back to the model as a failed tool result instead of ending the run."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
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
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

__all__ = ['ReportToolErrors', 'ToolErrorInfo']


_PASS_THROUGH = (
    ModelRetry,
    ToolFailed,
    ApprovalRequired,
    CallDeferred,
    SkipToolExecution,
    RunCancelled,
    UsageLimitExceeded,
    UserError,
)
"""Signals a tool raises on purpose, a cancellation core already reports, and errors no retry can fix."""


def _is_durable_failure(error: Exception) -> bool:
    """Whether `error` is a failed Temporal activity, checked without importing `temporalio`.

    It names the tool's exception only by its type, after the activity's retry policy gave up, so it can't be
    matched against `exceptions` or the signals above and stays with the workflow.
    """
    activity_error: object = getattr(sys.modules.get('temporalio.exceptions'), 'ActivityError', None)
    return isinstance(activity_error, type) and isinstance(error, activity_error)


@dataclass(frozen=True, kw_only=True)
class ToolErrorInfo:
    """The failed tool call and the exception it raised, as `ReportToolErrors.format_error` sees them."""

    name: str
    """Name of the tool the model called."""

    args: Mapping[str, Any]
    """Validated arguments the tool was called with, as a copy."""

    tool_call_id: str
    """Identifier of this call, for correlating with the run's messages."""

    error: Exception
    """The exception the tool raised."""


def _describe(ctx: RunContext[Any], info: ToolErrorInfo) -> str:
    return f'`{info.name}` raised {type(info.error).__name__}'


@dataclass
class ReportToolErrors(AbstractCapability[AgentDepsT]):
    """Report an exception a tool raises to the model as a failed tool result, so the run continues.

    The exception becomes a `ToolFailed`, as if the tool had raised one itself. The model sees the tool name and the
    exception type, and decides whether to call the tool again, try another way, or give up. There
    is no limit of its own: `UsageLimits.request_limit` stops a model that keeps calling a failing tool. `ModelRetry`,
    `ToolFailed`, `ApprovalRequired`, `CallDeferred`, `SkipToolExecution`, `RunCancelled`, `UsageLimitExceeded` and
    `UserError` keep their usual meaning.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import ReportToolErrors

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[ReportToolErrors()])
    ```
    """

    format_error: Callable[[RunContext[AgentDepsT], ToolErrorInfo], str] = _describe
    """Build the text the model sees from the run context and the failed call.

    The default names the tool and the exception type. It leaves out the exception's message, which can carry
    file paths, hostnames or connection strings; pass a formatter to include it.
    """

    exceptions: tuple[type[Exception], ...] = (Exception,)
    """The exception types to report. Any other exception propagates and ends the run, as without the capability."""

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return _ReportErrorsToolset(toolset, self)


@dataclass
class _ReportErrorsToolset(WrapperToolset[AgentDepsT]):
    report: ReportToolErrors[AgentDepsT]

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        try:
            return await super().call_tool(name, tool_args, ctx, tool)
        except _PASS_THROUGH:
            raise
        except self.report.exceptions as error:
            if _is_durable_failure(error):
                raise
            assert ctx.tool_call_id is not None
            info = ToolErrorInfo(name=name, args=dict(tool_args), tool_call_id=ctx.tool_call_id, error=error)
            raise ToolFailed(self.report.format_error(ctx, info)) from error
