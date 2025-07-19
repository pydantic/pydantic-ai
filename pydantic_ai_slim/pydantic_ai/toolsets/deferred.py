from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_core import SchemaValidator, core_schema

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import AbstractToolset, ToolsetTool

TOOL_SCHEMA_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())


@dataclass
class DeferredToolset(AbstractToolset[AgentDepsT]):
    """A toolset that holds deferred tools whose results will be produced outside of the Pydantic AI agent run in which they were called.

    See [toolset docs](../toolsets.md#deferred-toolset), [`ToolDefinition.kind`][pydantic_ai.tools.ToolDefinition.kind], and [`DeferredToolCalls`][pydantic_ai.output.DeferredToolCalls] for more information.
    """

    tool_defs: list[ToolDefinition]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=replace(tool_def, kind='deferred'),
                max_retries=0,
                args_validator=TOOL_SCHEMA_VALIDATOR,
            )
            for tool_def in self.tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        raise NotImplementedError('Deferred tools cannot be called')
