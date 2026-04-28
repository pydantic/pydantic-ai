from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.messages import ModelRequest, ToolReturn, ToolReturnPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

if TYPE_CHECKING:
    from pydantic_ai.capabilities.deferred import DeferredCapability
    from pydantic_ai.messages import ModelMessage

_LOAD_CAPABILITY_NAME = 'load_capability'
_LOADED_CAPABILITY_METADATA_KEY = 'loaded_capability_id'


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

    deferred_capabilities: Sequence[DeferredCapability[AgentDepsT]]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        unloaded = [cap for cap in self.deferred_capabilities if not cap._loaded]  # type: ignore[reportPrivateUsage]
        if not unloaded:
            return all_tools

        catalog = '\n'.join(f'- {cap.wrapped.id}: {cap.wrapped.description}' for cap in unloaded)

        load_tool_def = ToolDefinition(
            name=_LOAD_CAPABILITY_NAME,
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

        result: dict[str, ToolsetTool[AgentDepsT]] = {_LOAD_CAPABILITY_NAME: load_tool}
        result.update(all_tools)
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == _LOAD_CAPABILITY_NAME:
            return self._load_capability(tool_args)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    def _load_capability(self, tool_args: dict[str, Any]) -> ToolReturn | str:
        capability_id = tool_args['id']
        for cap in self.deferred_capabilities:
            if cap.wrapped.id == capability_id:
                cap.load()
                instructions = cap.wrapped.get_instructions()
                if instructions is None:
                    msg = f'Capability {capability_id!r} loaded (no additional instructions).'
                elif isinstance(instructions, str):
                    msg = instructions
                else:
                    msg = f'Capability {capability_id!r} loaded.'
                return ToolReturn(return_value=msg, metadata={_LOADED_CAPABILITY_METADATA_KEY: capability_id})
        return f'No capability found with id {capability_id!r}.'


def parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse message history to find capabilities loaded via load_capability."""
    loaded: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_name == _LOAD_CAPABILITY_NAME
                    and isinstance(metadata := part.metadata, dict)
                    and isinstance(cap_id := metadata.get(_LOADED_CAPABILITY_METADATA_KEY), str)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                ):
                    loaded.add(cap_id)
    return loaded
