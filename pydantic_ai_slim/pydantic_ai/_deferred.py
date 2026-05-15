from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

# `messages.py` re-exports `LoadCapabilityReturn` from `_load_capability.py` (canonical home,
# alongside the typed `LoadCapabilityReturnPart` that carries it). Re-export it here too
# so existing `from pydantic_ai._deferred import LoadCapabilityReturn` paths keep working.
from pydantic_ai.messages import (
    LoadCapabilityReturn as LoadCapabilityReturn,
    LoadCapabilityReturnPart,
    ModelRequest,
)

if TYPE_CHECKING:
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.toolsets.abstract import AbstractToolset


LOAD_CAPABILITY_TOOL_NAME = 'load_capability'


def parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse message history to find capabilities loaded via `load_capability`.

    Relies on the agent loop's typed-promotion path (`_agent_graph` reading
    `ToolDefinition.tool_kind` and calling `ToolReturnPart.narrow_type`) to surface
    every `load_capability` return as a `LoadCapabilityReturnPart` — see the
    `tool_kind='capability-load'` stamp on `DeferredCapabilityToolset`'s tool def.
    """
    return {
        part.loaded_capability
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, LoadCapabilityReturnPart)
    }


async def tools_for_loaded_capabilities(ctx: RunContext[Any], root: AbstractToolset[Any]) -> set[str]:
    """Collect tool names exposed by every loaded deferred capability.

    Walks the toolset tree for `CapabilityScopedToolset` nodes whose `capability_id`
    is in `ctx.loaded_capability_ids` and whose owning capability has
    `defer_loading=True`, then asks each wrapped toolset for its tools.

    Source-of-truth pairs with `parse_discovered_tools` (which reads
    `ToolSearchReturnPart` history): this helper is what tells the rest of the system
    "these names are unlocked because their owning capability was loaded," while
    `parse_discovered_tools` reports "these names are unlocked because tool search
    discovered them." Together they form `ctx.discovered_tools`.
    """
    # Local import to avoid a module-level cycle: `_capability_scoped` lives under
    # `toolsets/` which imports from `_run_context`, which is itself referenced via
    # `TYPE_CHECKING` in this module.
    from pydantic_ai.toolsets._capability_scoped import CapabilityScopedToolset

    scoped_by_cap: dict[str, list[CapabilityScopedToolset[Any]]] = {}

    def collect(ts: AbstractToolset[Any]) -> None:
        if isinstance(ts, CapabilityScopedToolset):
            scoped_by_cap.setdefault(ts.capability_id, []).append(ts)

    root.apply(collect)

    out: set[str] = set()
    for cap_id in ctx.loaded_capability_ids:  # We only check for loaded capabilities here
        cap = ctx.capabilities.get(cap_id)
        if cap is None or cap.defer_loading is not True:  # Only for capabilities which are deferred_loading
            continue
        for ts in scoped_by_cap.get(cap_id, []):
            tools = await ts.wrapped.get_tools(ctx)
            out.update(tools.keys())
    return out
