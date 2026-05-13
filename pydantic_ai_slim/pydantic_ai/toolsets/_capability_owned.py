from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..exceptions import ModelRetry
from ..messages import InstructionPart
from .abstract import AbstractToolset, ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class CapabilityOwnedToolset(WrapperToolset[AgentDepsT]):
    """Wraps a toolset contributed by a capability, binding it to that capability.

    Stamps `capability_id` onto every emitted tool definition (so downstream
    filters like Tool Search can hide tools whose owning capability isn't loaded
    yet), and silences `get_instructions` while the owning capability is
    `defer_loading=True` and not yet in `RunContext.loaded_capability_ids`. The
    suppressed instructions are re-emitted via the `load_capability` tool
    return when the model loads the capability.

    When the owning capability is `defer_loading=True`, each tool's description
    also gets a hint appended naming the capability and the load step, so the
    model learns the prerequisite up-front rather than discovering it by
    failing on an execution-time `ModelRetry`. The hint is appended on every
    turn (loaded or not) so the description bytes stay stable across the load
    boundary — the wording hedges with "if you haven't" to remain accurate
    after load. Trading a small description annotation for cache stability is
    the load-bearing choice here.
    """

    capability_id: str

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await self.wrapped.get_tools(ctx)
        cap = ctx.capabilities.get(self.capability_id)
        hint = (
            f" (This tool belongs to capability '{self.capability_id}'. "
            f"Call load_capability(id='{self.capability_id}') first if you haven't.)"
            if cap is not None and cap.defer_loading is True
            else None
        )
        return {
            name: replace(
                tool,
                tool_def=replace(
                    tool.tool_def,
                    capability_id=tool.tool_def.capability_id
                    if tool.tool_def.capability_id is not None
                    else self.capability_id,
                    description=(f'{tool.tool_def.description}{hint}' if tool.tool_def.description else hint.lstrip())
                    if hint is not None
                    else tool.tool_def.description,
                ),
            )
            for name, tool in tools.items()
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        cap = ctx.capabilities.get(self.capability_id)
        if cap is not None and cap.defer_loading is True and self.capability_id not in ctx.loaded_capability_ids:
            raise ModelRetry(
                f"Tool '{name}' belongs to capability '{self.capability_id}', which has not been loaded yet. "
                f"Call load_capability(id='{self.capability_id}') first."
            )
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

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
