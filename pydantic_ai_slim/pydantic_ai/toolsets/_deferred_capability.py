from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai._deferred import (
    LOAD_CAPABILITY_TOOL_NAME,
    DeferredLoadingRegistry,
    LoadCapabilityReturn,
    parse_loaded_capabilities,
)
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._system_prompt import SystemPromptRunner
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset


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

        instructions = self.registry.instructions.get(capability_id, [])

        parts: list[str] = []

        for instruction in instructions:
            if isinstance(instruction, str):
                parts.append(instruction)
            else:
                resolved = await SystemPromptRunner[AgentDepsT](instruction).run(ctx)
                if resolved is not None:
                    parts.append(resolved)

        instructions_text = '\n\n'.join(parts) or None
        return ToolReturn(
            return_value=LoadCapabilityReturn(capability_id=capability_id, instructions=instructions_text)
        )
