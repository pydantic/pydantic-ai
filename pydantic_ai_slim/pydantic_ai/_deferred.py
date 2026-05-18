from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

# `messages.py` re-exports `LoadCapabilityReturn` from `_load_capability.py` (canonical home,
# alongside the typed `LoadCapabilityReturnPart` that carries it). Re-export it here too
# so existing `from pydantic_ai._deferred import LoadCapabilityReturn` paths keep working.
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturn as LoadCapabilityReturn,
    LoadCapabilityReturnPart,
)

if TYPE_CHECKING:
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.toolsets.abstract import AbstractToolset


LOAD_CAPABILITY_TOOL_NAME = 'load_capability'


def parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse message history to find capabilities loaded via `load_capability`.

    Relies on the agent loop's typed-promotion path (`_agent_graph` reading
    `ToolDefinition.tool_kind` and calling `ToolReturnPart.narrow_type` /
    `ToolCallPart.narrow_type`) to surface load_capability call/return pairs as
    the typed subclasses — see the `tool_kind='capability-load'` stamp on
    `DeferredCapabilityToolset`'s tool def.
    """
    call_id_by_tool_call_id: dict[str, str] = {}
    loaded: set[str] = set()
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, LoadCapabilityCallPart):
                if part.capability_id is not None:
                    call_id_by_tool_call_id[part.tool_call_id] = part.capability_id
            elif isinstance(part, LoadCapabilityReturnPart):
                cap_id = call_id_by_tool_call_id.get(part.tool_call_id)
                if cap_id is not None:
                    loaded.add(cap_id)
    return loaded


async def tools_for_loaded_capabilities(ctx: RunContext[Any], root: AbstractToolset[Any]) -> set[str]:
    """Collect tool names exposed by every loaded deferred capability.

    Walks the toolset tree for `CapabilityScopedToolset` nodes whose `capability_id`
    is in `ctx.loaded_capability_ids` and whose owning capability has
    `defer_loading=True`, then asks each wrapped toolset for its tools.

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
