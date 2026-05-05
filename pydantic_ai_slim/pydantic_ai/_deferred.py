from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import TypedDict

from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, ToolDefinition, ToolsPrepareFunc

if TYPE_CHECKING:
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.messages import ModelMessage


LOAD_CAPABILITY_TOOL_NAME = 'load_capability'


class LoadCapabilityReturn(TypedDict):
    capability_id: str
    instructions: str | None


def extract_load_capability_return(content: Any) -> LoadCapabilityReturn | None:
    if not isinstance(content, dict):
        return None

    content = cast(dict[str, Any], content)
    capability_id = content.get('capability_id')
    if not isinstance(capability_id, str):
        return None

    instructions = content.get('instructions')
    if instructions is not None and not isinstance(instructions, str):
        return None

    return LoadCapabilityReturn(capability_id=capability_id, instructions=instructions)


def parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse message history to find capabilities loaded via ``load_capability``."""
    loaded: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == LOAD_CAPABILITY_TOOL_NAME:
                    parsed = extract_load_capability_return(part.content)
                    if parsed is not None:
                        loaded.add(parsed['capability_id'])

    return loaded


def prepare_capability_tool_definitions(
    *,
    capability_id: str | None,
    capability_defer_loading: bool | None,
) -> ToolsPrepareFunc[AgentDepsT]:
    def prepare(_ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return [
            replace(
                tool_def,
                capability_id=tool_def.capability_id if tool_def.capability_id is not None else capability_id,
                defer_loading=(
                    tool_def.defer_loading if tool_def.defer_loading is not None else capability_defer_loading
                ),
            )
            for tool_def in tool_defs
        ]

    return prepare
