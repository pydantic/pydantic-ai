from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .wrapper import WrapperToolset


@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that maps renamed tool names to original tool names."""

    name_map: dict[str, str]

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        original_to_new_name_map = {v: k for k, v in self.name_map.items()}
        return [
            replace(tool_def, name=new_name)
            if (new_name := original_to_new_name_map.get(tool_def.name, None))
            else tool_def
            for tool_def in await super().list_tool_defs(ctx)
        ]

    def max_retries_for_tool(self, name: str) -> int:
        return super().max_retries_for_tool(self._map_name(name))

    def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        original_name = self._map_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return super().get_tool_args_validator(ctx, original_name)

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        original_name = self._map_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return await super().call_tool(ctx, self._map_name(name), tool_args)

    def _map_name(self, name: str) -> str:
        return self.name_map.get(name, name)
