from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

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
            result[name] = replace(
                tool,
                tool_def=replace(
                    tool_def,
                    capability_id=tool_def.capability_id if tool_def.capability_id is not None else capability_id,
                    defer_loading=defer_loading or tool_def.defer_loading,
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


def is_gated_by_deferred_capability(ctx: RunContext[Any], tool_def: ToolDefinition) -> bool:
    """Whether an on-demand capability decides when this tool becomes available.

    Such a tool is hidden until its owning capability loads, and it is never searchable: no query
    should surface it, because the model isn't meant to reach it by asking. That's the line between
    the two things a deferred tool can be — hidden until something reveals it, which every deferred
    tool is, and a member of the searchable corpus, which only the ungated ones are. Which side a
    tool falls on depends on how the run is configured, not on the model serving it, so it's settled
    here rather than in `Model.prepare_request`.
    """
    return (
        (capability_id := tool_def.capability_id) is not None
        and (cap := ctx.capabilities.get(capability_id)) is not None
        and cap.defer_loading is True
    )


def tool_defs_from_pre_definition_load_returns(
    ctx: RunContext[Any], tool_defs: Iterable[ToolDefinition]
) -> dict[str, ToolDefinition]:
    """Reconstruct definitions for histories serialized before load returns carried tool definitions."""
    result: dict[str, ToolDefinition] = {}
    for tool_def in tool_defs:
        capability_id = tool_def.capability_id
        if capability_id is None or capability_id not in ctx.loaded_capability_ids:
            continue
        capability = ctx.capabilities.get(capability_id)
        if capability is not None and capability.defer_loading is True:
            result[tool_def.name] = tool_def
    return result
