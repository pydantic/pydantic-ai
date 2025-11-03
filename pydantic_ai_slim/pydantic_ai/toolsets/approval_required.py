from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.exceptions import ApprovalRequired

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    """A toolset that requires (some) calls to tools it contains to be approved.

    See [toolset docs](../toolsets.md#requiring-tool-approval) for more information.
    """

    approval_required_func: Callable[[RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool] = (
        lambda ctx, tool_def, tool_args: True
    )

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not ctx.tool_call_approved and self.approval_required_func(ctx, tool.tool_def, tool_args):
            raise ApprovalRequired

        return await super().call_tool(name, tool_args, ctx, tool)
