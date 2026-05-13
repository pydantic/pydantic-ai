from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai._deferred import (
    LOAD_CAPABILITY_TOOL_NAME,
    LoadCapabilityReturn,
)
from pydantic_ai._instructions import normalize_instructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._system_prompt import SystemPromptRunner
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart, ToolReturn
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
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

_LOAD_CAPABILITY_TOOL_DEF = ToolDefinition(
    name=LOAD_CAPABILITY_TOOL_NAME,
    description='Load a capability to access its full instructions and tools.',
    parameters_json_schema=_LOAD_CAPABILITY_SCHEMA,
    tool_kind='capability-load',
)


@dataclass
class CapabilityLoaderToolset(WrapperToolset[AgentDepsT]):
    """Toolset that wraps an agent's tools and injects a ``load_capability`` discovery tool.

    When unloaded capabilities exist, ``get_tools`` adds a ``load_capability`` tool.
    The catalog of loadable capabilities is provided by
    :class:`~pydantic_ai.capabilities._loader.CapabilityLoader` instructions.
    When the model calls ``load_capability(id)``, the matching capability's
    instructions are returned as the tool result. Once all capabilities are loaded,
    the ``load_capability`` tool is removed.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        loaded_ids = ctx.loaded_capability_ids
        unloaded = [
            entry for entry in ctx.capabilities.values() if entry.id not in loaded_ids and entry.defer_loading is True
        ]
        if not unloaded:
            return all_tools

        if LOAD_CAPABILITY_TOOL_NAME in all_tools:
            raise UserError(
                f"Tool name '{LOAD_CAPABILITY_TOOL_NAME}' is reserved for deferred capability loading. "
                'Rename your tool to avoid conflicts.'
            )

        load_tool = ToolsetTool(
            toolset=self,
            tool_def=_LOAD_CAPABILITY_TOOL_DEF,
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
        if capability_id not in ctx.capabilities:
            return f'No capability found with id {capability_id!r}.'

        parts: list[str] = []

        for instruction in normalize_instructions(ctx.capabilities[capability_id].get_instructions()):
            if isinstance(instruction, str):
                parts.append(instruction)
            else:
                resolved = await SystemPromptRunner[AgentDepsT](instruction).run(ctx)
                if resolved is not None:
                    parts.append(resolved)

        for resolved in await self._collect_owned_toolset_instructions(capability_id, ctx):
            parts.append(resolved)

        instructions_text = '\n\n'.join(parts) or None
        ctx.loaded_capability_ids.add(capability_id)
        return ToolReturn(
            return_value=LoadCapabilityReturn(capability_id=capability_id, instructions=instructions_text)
        )

    async def _collect_owned_toolset_instructions(self, capability_id: str, ctx: RunContext[AgentDepsT]) -> list[str]:
        """Pull instructions from `CapabilityOwnedToolset`s tagged with this cap_id.

        Bypasses each wrapper's own gate (still closed because the cap isn't yet
        in `loaded_capability_ids` — its tool return hasn't been appended to
        history) by calling `get_instructions` on `ts.wrapped` directly.
        """
        owned: list[CapabilityOwnedToolset[AgentDepsT]] = []

        def collect(ts: AbstractToolset[AgentDepsT]) -> None:
            if isinstance(ts, CapabilityOwnedToolset) and ts.capability_id == capability_id:
                owned.append(ts)

        self.apply(collect)

        out: list[str] = []
        for ts in owned:
            result = await ts.wrapped.get_instructions(ctx)
            if result is None:
                continue
            for item in [result] if isinstance(result, (str, InstructionPart)) else result:
                content = item.content if isinstance(item, InstructionPart) else item
                if content and content.strip():
                    out.append(content)
        return out
