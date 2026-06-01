from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .._deferred_capabilities import DEFERRED_CAPABILITY_TOOL_METADATA_KEY
from .._run_context import AgentDepsT, RunContext
from ..messages import InstructionPart
from .abstract import AbstractToolset, ToolsetTool
from .wrapper import WrapperToolset

if TYPE_CHECKING:
    from ..capabilities import AbstractCapability
    from ..tools import ToolDefinition


@dataclass
class CapabilityOwnedToolset(WrapperToolset[AgentDepsT]):
    """Binds a contributed toolset to the capability that owns it."""

    capability: AbstractCapability[AgentDepsT]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await self.wrapped.get_tools(ctx)
        capability_id = resolve_capability_id(ctx, self.capability)
        defer_loading = self.capability.defer_loading is True
        result: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in tools.items():
            tool_def = tool.tool_def
            metadata = tool_def.metadata
            if defer_loading:
                metadata = {**(metadata or {}), DEFERRED_CAPABILITY_TOOL_METADATA_KEY: True}
            result[name] = replace(
                tool,
                tool_def=replace(
                    tool_def,
                    capability_id=tool_def.capability_id if tool_def.capability_id is not None else capability_id,
                    defer_loading=defer_loading or tool_def.defer_loading,
                    metadata=metadata,
                ),
            )
        return result

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        if self.capability.defer_loading is True:
            return None
        return await self.wrapped.get_instructions(ctx)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        visitor(self)
        self.wrapped.apply(visitor)


def resolve_capability_id(ctx: RunContext[AgentDepsT], capability: AbstractCapability[AgentDepsT]) -> str:
    """Recover the id a capability was registered under in `ctx.capabilities` for the current run.

    A capability with no explicit `id` is registered under a derived id (see
    `_build_run_capabilities`), so the resolved id only exists as a registry key.
    """
    for capability_id, registered_capability in ctx.capabilities.items():
        if registered_capability is capability:
            return capability_id
    raise RuntimeError(  # pragma: no cover
        f'Capability {capability!r} is not registered in this run; this is an internal error in Pydantic AI.'
    )


# This is the wire-side resolver: `ToolSearchToolset.get_tools` calls it to decide which
# capability-owned deferred tools to actually surface in the request this turn. It is deliberately
# separate from `RunContext.available_tool_names` (the read-side resolver hooks query) — the two
# answer different questions (what to send vs. what the user can observe) over different inputs
# (the toolset's tool defs vs. the run context), so they aren't duplicated logic to fold together.
def tool_defs_for_loaded_capabilities(
    ctx: RunContext[Any], tool_defs: Iterable[ToolDefinition]
) -> dict[str, ToolDefinition]:
    """Return resolved function-tool definitions owned by loaded deferred capabilities, keyed by name."""
    return {
        tool_def.name: tool_def
        for tool_def in tool_defs
        if (capability_id := tool_def.capability_id) is not None
        and capability_id in ctx.available_capability_ids
        and (cap := ctx.capabilities.get(capability_id)) is not None
        and cap.defer_loading is True
    }
