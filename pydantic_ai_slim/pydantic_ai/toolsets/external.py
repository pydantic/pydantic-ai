from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_core import SchemaValidator, core_schema
from typing_extensions import deprecated

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import AbstractToolset, ToolsetTool

TOOL_SCHEMA_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())


class ExternalToolset(AbstractToolset[AgentDepsT]):
    """A toolset that holds tools whose results will be produced outside of the Pydantic AI agent run in which they were called.

    See [toolset docs](../toolsets.md#external-toolset) for more information.
    """

    tool_defs: list[ToolDefinition]
    _id: str | None

    def __init__(self, tool_defs: list[ToolDefinition], *, id: str | None = None):
        self.tool_defs = tool_defs
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=replace(tool_def, kind='external'),
                max_retries=0,
                args_validator=TOOL_SCHEMA_VALIDATOR,
            )
            for tool_def in self.tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        raise NotImplementedError('External tools cannot be called directly')


@deprecated('`DeferredToolset` is deprecated, use `ExternalToolset` instead')
class DeferredToolset(ExternalToolset):
    """Deprecated alias for `ExternalToolset`."""
