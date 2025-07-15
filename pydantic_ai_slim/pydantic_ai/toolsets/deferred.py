from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import AbstractToolset


@dataclass
class DeferredToolset(AbstractToolset[AgentDepsT]):
    """A toolset that holds deferred tools.

    See [`ToolDefinition.kind`][pydantic_ai.tools.ToolDefinition.kind] for more information about deferred tools.
    """

    tool_defs: list[ToolDefinition]

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return [replace(tool_def, kind='deferred') for tool_def in self.tool_defs]

    def max_retries_for_tool(self, name: str) -> int:
        raise NotImplementedError('Deferred tools cannot be retried')

    def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        raise NotImplementedError('Deferred tools cannot be validated')

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        raise NotImplementedError('Deferred tools cannot be called')
