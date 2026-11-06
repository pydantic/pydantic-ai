"""Workflow policies layered onto the harness Coder composition."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import ClassVar

from mcp.shared.exceptions import McpError

from pydantic_ai import ModelRequestContext, ModelRetry, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from ._budgets import RUN_TIME_RESERVE_SECS, run_seconds_left, time_budget_notice
from ._config import is_ci_review_run, shell_disabled
from ._safe_outputs import safe_output_type

SHELL_TOOL_NAMES = frozenset({'shell'})
CI_REVIEW_RESERVE_SAFE_OUTPUTS = frozenset({'submit_pull_request_review', 'create_pull_request_review_comment'})
CI_REVIEW_WITHHELD_MESSAGE = (
    'error: this tool is not available to CI Review now. Finish by submitting your review with '
    '`submit_pull_request_review`, as a COMMENT review if coverage is incomplete.'
)
SHELL_DISABLED_MESSAGE = 'error: this workflow does not allow shell commands; use the file tools instead.'


def _mcp_protocol_error_message(error: Exception) -> str | None:
    """The message of a bare `McpError` that `MCPToolset` left uncaught, else `None`."""
    if isinstance(error, McpError):
        return str(error)
    return None


@dataclass
class AnswerSafeOutputLimit(WrapperToolset[object]):
    """Answer a safe output past its per-run limit where the tool manager has not charged it yet.

    `MCPToolset` raises the server's rejection as a `ModelRetry`, which the tool manager
    counts against the tool's retries before any tool-execute hook runs.
    """

    # gh-aw's safe-outputs server rejects a call past a type's `max` with this code.
    _LIMIT_ERROR: ClassVar[str] = 'E002: '

    async def call_tool(
        self, name: str, tool_args: dict[str, object], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> object:
        try:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        except ModelRetry as retry:
            limit_error = retry.__cause__
            if (
                safe_output_type(name) is None
                or not isinstance(limit_error, McpError)
                or self._LIMIT_ERROR not in str(limit_error)
            ):
                raise
            return f'error: {limit_error}. The earlier calls are already recorded, so do not call this tool again.'


@dataclass
class RecoverMCPToolErrors(AbstractCapability[object]):
    """Hand escaped MCP protocol errors back to the model as recoverable `error:` results.

    Without this, a bare `McpError` from an MCP server crashes the whole run (an
    unhandled `ExceptionGroup`), which gh-aw can only recover from by re-invoking the
    agent wholesale.
    """

    def get_wrapper_toolset(self, toolset: AbstractToolset[object]) -> AbstractToolset[object]:
        return AnswerSafeOutputLimit(toolset)

    async def on_tool_execute_error(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: object,
        error: Exception,
    ) -> str:
        message = _mcp_protocol_error_message(error)
        if message is None:
            raise error
        return f'error: {message}'


def _withheld_from_ci_review(tool_name: str) -> bool:
    """Whether CI Review's model must not be offered this tool now."""
    safe_output = safe_output_type(tool_name)
    if safe_output is None:
        return time_budget_notice() is not None
    if safe_output == 'noop':
        return True
    return time_budget_notice() is not None and safe_output not in CI_REVIEW_RESERVE_SAFE_OUTPUTS


def _offered(tool_name: str) -> bool:
    if shell_disabled() and tool_name in SHELL_TOOL_NAMES:
        return False
    if is_ci_review_run():
        return not _withheld_from_ci_review(tool_name)
    if time_budget_notice() is not None:
        return safe_output_type(tool_name) is not None
    return True


@dataclass
class ToolPolicy(AbstractCapability[object]):
    """Apply the workflow's shell exposure and CI Review's deadline policy.

    - `tools.bash: []` withholds the shell tool.
    - CI Review is not offered `noop`; its stale-PR skip step writes the run's only one.
    - Inside the time reserve, CI Review is offered only the safe outputs that publish
      its review, and each `shell` command stops when the reserve starts.
    - Outside CI Review, only safe outputs are offered inside the reserve.

    Withheld tools leave the request, not the run's tools: a call to a tool the run does
    not have counts against the model's single tool retry, so a second one ends the run.
    A call the model makes anyway is answered here instead.
    """

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        del ctx
        parameters = request_context.model_request_parameters
        offered = [tool_def for tool_def in parameters.function_tools if _offered(tool_def.name)]
        return dataclasses.replace(
            request_context, model_request_parameters=dataclasses.replace(parameters, function_tools=offered)
        )

    async def wrap_tool_execute(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        del ctx, call
        if shell_disabled() and tool_def.name in SHELL_TOOL_NAMES:
            return SHELL_DISABLED_MESSAGE
        if is_ci_review_run():
            if _withheld_from_ci_review(tool_def.name):
                return CI_REVIEW_WITHHELD_MESSAGE
            seconds_left = run_seconds_left()
            if tool_def.name in SHELL_TOOL_NAMES and seconds_left is not None:
                before_reserve = math.floor(seconds_left - RUN_TIME_RESERVE_SECS)
                if before_reserve <= 0:
                    return 'error: the time budget is nearly spent; finish the task from what you already have'
                requested = args.get('timeout')
                timeout = requested if isinstance(requested, int) and requested > 0 else 120
                return await handler({**args, 'timeout': min(timeout, before_reserve)})
        return await handler(args)
