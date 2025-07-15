from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from ._tool_defs import ToolDefsToolset
from .abstract import AbstractToolset
from .wrapper import WrapperToolset


@dataclass(init=False)
class RenamedToolset(ToolDefsToolset[AgentDepsT]):
    """A toolset that maps renamed tool names to original tool names."""

    name_map: dict[str, str]

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        name_map: dict[str, str],
        tool_defs: list[ToolDefinition] | None = None,
    ):
        self.name_map = name_map

        if tool_defs is None:
            original_to_new_name_map = {v: k for k, v in name_map.items()}
            tool_defs = [
                replace(tool_def, name=new_name)
                if (new_name := original_to_new_name_map.get(tool_def.name, None))
                else tool_def
                for tool_def in wrapped.tool_defs
            ]

        super().__init__(wrapped, tool_defs)

    async def rewrap_for_run(
        self, wrapped: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
    ) -> WrapperToolset[AgentDepsT]:
        return RenamedToolset(wrapped, self.name_map, self._tool_defs)

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
