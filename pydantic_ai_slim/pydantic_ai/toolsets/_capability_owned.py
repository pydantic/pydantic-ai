from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .._run_context import AgentDepsT, RunContext
from ..messages import InstructionPart
from .abstract import AbstractToolset, ToolsetTool
from .wrapper import WrapperToolset

if TYPE_CHECKING:
    from ..tools import ToolDefinition


@dataclass
class CapabilityOwnedToolset(WrapperToolset[AgentDepsT]):
    """Binds a contributed toolset to the capability that owns it."""

    capability_id: str

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await self.wrapped.get_tools(ctx)
        cap = ctx.capabilities.get(self.capability_id)
        # Keep the declaration stable; model-facing visibility is resolved later.
        defer_loading = cap.defer_loading is True if cap is not None else False
        return {
            name: replace(
                tool,
                tool_def=replace(
                    tool.tool_def,
                    capability_id=tool.tool_def.capability_id
                    if tool.tool_def.capability_id is not None
                    else self.capability_id,
                    defer_loading=defer_loading or tool.tool_def.defer_loading,
                ),
            )
            for name, tool in tools.items()
        }

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        cap = ctx.capabilities.get(self.capability_id)
        if cap is not None and cap.defer_loading is True:
            return None
        return await self.wrapped.get_instructions(ctx)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        # Visit self because the capability id binding lives on this wrapper.
        visitor(self)
        self.wrapped.apply(visitor)


def tools_for_loaded_capabilities(ctx: RunContext[Any], tool_defs: Iterable[ToolDefinition]) -> set[str]:
    """Return resolved function-tool names owned by loaded deferred capabilities."""
    return {
        tool_def.name
        for tool_def in tool_defs
        if (capability_id := tool_def.capability_id) is not None
        and capability_id in ctx.loaded_capability_ids
        and (cap := ctx.capabilities.get(capability_id)) is not None
        and cap.defer_loading is True
    }
