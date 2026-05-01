from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Generic, cast

from typing_extensions import TypedDict

from pydantic_ai._instructions import Instruction
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, ToolDefinition, ToolsPrepareFunc

if TYPE_CHECKING:
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.capabilities.abstract import AbstractCapability
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


@dataclass
class DeferredCapabilityCatalogEntry:
    """Catalog metadata for a capability that can be loaded on demand."""

    capability_id: str
    description: str


@dataclass
class DeferredLoadingRegistry(Generic[AgentDepsT]):
    """Run-local catalog and instruction bodies used by ``load_capability``."""

    catalog: dict[str, DeferredCapabilityCatalogEntry]
    instructions: dict[str, list[Instruction[AgentDepsT]]]


def build_deferred_loading_registry(
    capability: AbstractCapability[AgentDepsT],
    instructions: Sequence[Instruction[AgentDepsT]],
) -> DeferredLoadingRegistry[AgentDepsT] | None:
    catalog: dict[str, DeferredCapabilityCatalogEntry] = {}

    def collect_deferred_capability(cap: AbstractCapability[AgentDepsT]) -> None:
        if cap.defer_loading is not True:
            return

        capability_id = cap.id
        description = cap.get_description()
        if capability_id is None:
            raise ValueError('Capabilities with defer_loading=True must have an id.')
        if description is None:
            raise ValueError('Capabilities with defer_loading=True must have a description.')

        catalog[capability_id] = DeferredCapabilityCatalogEntry(
            capability_id=capability_id,
            description=description,
        )

    capability.apply(collect_deferred_capability)
    if not catalog:
        return None

    deferred_instructions: dict[str, list[Instruction[AgentDepsT]]] = {capability_id: [] for capability_id in catalog}
    for instruction in instructions:
        if instruction.defer_loading is not True:
            continue
        capability_id = instruction.capability_id
        if capability_id in deferred_instructions:
            deferred_instructions[capability_id].append(instruction)

    return DeferredLoadingRegistry(catalog=catalog, instructions=deferred_instructions)
