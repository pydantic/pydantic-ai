from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .._run_context import AgentDepsT, RunContext
from ..messages import InstructionPart
from .abstract import AbstractToolset, ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class CapabilityScopedToolset(WrapperToolset[AgentDepsT]):
    """Wraps a toolset contributed by a capability, binding it to that capability.

    Stamps `capability_id` onto every emitted tool definition (so downstream
    filters like Tool Search can hide tools whose owning capability isn't loaded
    yet), and silences `get_instructions` while the owning capability is
    `defer_loading=True` and not yet in `RunContext.loaded_capability_ids`. The
    suppressed instructions are re-emitted via the `load_capability` tool
    return when the model loads the capability.
    """

    capability_id: str

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await self.wrapped.get_tools(ctx)
        return {
            name: replace(
                tool,
                tool_def=replace(
                    tool.tool_def,
                    capability_id=tool.tool_def.capability_id
                    if tool.tool_def.capability_id is not None
                    else self.capability_id,
                    defer_loading=ctx.capabilities.get(self.capability_id) is not None
                    and ctx.capabilities.get(self.capability_id).defer_loading is True,
                    # It doesn't matter if the capability is loaded or not because we don't want that to change anything on the tool_def anyway
                    # So it should stay deferred based on this so that it remains stable for the cache
                ),
            )
            for name, tool in tools.items()
        }

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        cap = ctx.capabilities.get(self.capability_id)
        if cap is not None and cap.defer_loading is True and self.capability_id not in ctx.loaded_capability_ids:
            return None
        return await self.wrapped.get_instructions(ctx)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        # Visit self so capability-aware walks (e.g. `_load_capability`) can find
        # us. The standard `WrapperToolset.apply` skips wrappers and visits only
        # leaves; we make an exception here because the cap-id binding lives on
        # this node.
        visitor(self)
        self.wrapped.apply(visitor)
