from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

# `messages.py` re-exports `LoadCapabilityReturn` from `_load_capability.py` (canonical home,
# alongside the typed `LoadCapabilityReturnPart` that carries it). Re-export it here too
# so existing `from pydantic_ai._deferred import LoadCapabilityReturn` paths keep working.
from pydantic_ai.messages import (
    LoadCapabilityReturn as LoadCapabilityReturn,
    LoadCapabilityReturnPart,
    ModelRequest,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage


LOAD_CAPABILITY_TOOL_NAME = 'load_capability'


def parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse message history to find capabilities loaded via ``load_capability``.

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
