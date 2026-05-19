from __future__ import annotations

from collections.abc import Iterable, Sequence
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
    from pydantic_ai.tools import ToolDefinition


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
