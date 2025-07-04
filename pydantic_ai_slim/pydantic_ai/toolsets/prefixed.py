from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..messages import ToolCallPart
from ..tools import ToolDefinition
from ._run import RunToolset
from .wrapper import WrapperToolset


@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains."""

    prefix: str

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        prefixed_for_run = PrefixedToolset(wrapped_for_run, self.prefix)
        return RunToolset(prefixed_for_run, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [replace(tool_def, name=self._prefixed_tool_name(tool_def.name)) for tool_def in super().tool_defs]

    def _max_retries_for_tool(self, name: str) -> int:
        return super()._max_retries_for_tool(self._unprefixed_tool_name(name))

    async def call_tool(self, call: ToolCallPart, ctx: RunContext[AgentDepsT], allow_partial: bool = False) -> Any:
        call = replace(call, tool_name=self._unprefixed_tool_name(call.tool_name))
        return await super().call_tool(call, ctx, allow_partial=allow_partial)

    def _prefixed_tool_name(self, tool_name: str) -> str:
        return f'{self.prefix}_{tool_name}'

    def _unprefixed_tool_name(self, tool_name: str) -> str:
        full_prefix = f'{self.prefix}_'
        if not tool_name.startswith(full_prefix):
            raise ValueError(f"Tool name '{tool_name}' does not start with prefix '{full_prefix}'")
        return tool_name[len(full_prefix) :]
