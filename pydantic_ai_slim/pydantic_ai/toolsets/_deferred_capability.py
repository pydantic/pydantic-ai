from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai._deferred import DeferredLoadingRegistry
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._system_prompt import SystemPromptRunner
from pydantic_ai.messages import ModelRequest, ToolReturn, ToolReturnPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

LOAD_CAPABILITY_TOOL_NAME = 'load_capability'
LOADED_CAPABILITY_METADATA_KEY = 'loaded_capability_id'


class _LoadCapabilityArgs(TypedDict):
    id: Annotated[
        str,
        Field(
            description='The id of the capability to load, as shown in the available capabilities list.',
        ),
    ]


_load_capability_args_ta = TypeAdapter(_LoadCapabilityArgs)
_LOAD_CAPABILITY_SCHEMA = _load_capability_args_ta.json_schema()
_LOAD_CAPABILITY_SCHEMA['title'] = 'LoadCapabilityArgs'


class LoadCapabilityReturn(TypedDict):
    capability_id: str
    instructions: str | None
    # Maybe tools, not sure right now but basically showing that this capability has been revealed
    # I am not entirely sure yet of this structure but let us take it


def extract_load_capability_return(content: Any) -> LoadCapabilityReturn | None:
    if not isinstance(content, dict):
        return None

    # Unfortunate that we are having to do this here, the tool return content is Mapping[str | Any]
    # It is not inferred
    content = cast(dict[str, Any], content)

    capability_id = content.get('capability_id')
    if not isinstance(capability_id, str):
        return None

    instructions = content.get('instructions')
    if instructions is not None and not isinstance(instructions, str):
        return None

    return LoadCapabilityReturn(capability_id=capability_id, instructions=instructions)


@dataclass
class DeferredCapabilityToolset(WrapperToolset[AgentDepsT]):
    """Toolset that wraps an agent's tools and injects a ``load_capability`` discovery tool.

    When unloaded capabilities exist, ``get_tools`` adds a ``load_capability`` tool whose
    description lists the available capabilities as a catalog. When the model calls
    ``load_capability(id)``, the matching :class:`~pydantic_ai.capabilities.deferred.DeferredCapability`
    is loaded and its instructions are returned as the tool result. Once all capabilities
    are loaded, the ``load_capability`` tool is removed.
    """

    registry: DeferredLoadingRegistry[AgentDepsT]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        loaded_ids = parse_loaded_capabilities(ctx.messages)
        unloaded = [entry for entry in self.registry.catalog.values() if entry.capability_id not in loaded_ids]
        if not unloaded:
            return all_tools

        catalog = '\n'.join(f'- {entry.capability_id}: {entry.description}' for entry in unloaded)

        load_tool_def = ToolDefinition(
            name=LOAD_CAPABILITY_TOOL_NAME,
            description=(
                f'Load a capability to access its full instructions and tools. Available capabilities:\n{catalog}'
            ),
            parameters_json_schema=_LOAD_CAPABILITY_SCHEMA,
        )

        load_tool = ToolsetTool(
            toolset=self,
            tool_def=load_tool_def,
            max_retries=1,
            args_validator=_load_capability_args_ta.validator,  # pyright: ignore[reportArgumentType]
        )

        result: dict[str, ToolsetTool[AgentDepsT]] = {LOAD_CAPABILITY_TOOL_NAME: load_tool}
        result.update(all_tools)
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == LOAD_CAPABILITY_TOOL_NAME:
            return await self._load_capability(tool_args, ctx)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _load_capability(self, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> ToolReturn | str:
        capability_id = tool_args['id']
        if capability_id not in self.registry.catalog:
            return f'No capability found with id {capability_id!r}.'

        outputs = self.registry.outputs.get(capability_id)
        instructions = outputs.instructions if outputs is not None else []

        parts: list[str] = []

        for instruction in instructions:
            content = instruction.content
            if isinstance(content, str):
                parts.append(content)
            else:
                resolved = await SystemPromptRunner[AgentDepsT](content).run(ctx)
                if resolved is not None:
                    parts.append(resolved)

        instructions_text = '\n\n'.join(parts) or None
        return ToolReturn(
            return_value=LoadCapabilityReturn(capability_id=capability_id, instructions=instructions_text)
        )


def parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse message history to find capabilities loaded via load_capability."""
    loaded: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == LOAD_CAPABILITY_TOOL_NAME:
                    parsed = extract_load_capability_return(part.content)
                    if parsed is not None:
                        loaded.add(parsed['capability_id'])

    return loaded
