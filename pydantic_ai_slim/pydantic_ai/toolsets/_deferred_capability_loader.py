from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from pydantic_ai._deferred_capabilities import LoadCapabilityArgs, LoadCapabilityReturn
from pydantic_ai._instructions import normalize_toolset_instructions, resolve_instructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

LOAD_CAPABILITY_TOOL_NAME = 'load_capability'
LOAD_CAPABILITY_TOOL_DESCRIPTION = 'Load a capability to access its full instructions and tools.'
LOAD_CAPABILITY_ALREADY_AVAILABLE_MESSAGE_TEMPLATE = (
    'Capability {capability_id!r} is already available. '
    'Use its existing instructions and tools; do not call `load_capability` for it again.'
)

_load_capability_args_ta = TypeAdapter(LoadCapabilityArgs)
_LOAD_CAPABILITY_SCHEMA = _load_capability_args_ta.json_schema()
_LOAD_CAPABILITY_SCHEMA['title'] = 'LoadCapabilityArgs'


@dataclass
class DeferredCapabilityLoaderToolset(WrapperToolset[AgentDepsT]):
    """Adds the framework-managed `load_capability` tool."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        if LOAD_CAPABILITY_TOOL_NAME in all_tools:
            raise UserError(
                f"Tool name '{LOAD_CAPABILITY_TOOL_NAME}' is reserved for deferred capability loading. "
                'Rename your tool to avoid conflicts.'
            )

        load_tool_def = ToolDefinition(
            name=LOAD_CAPABILITY_TOOL_NAME,
            description=LOAD_CAPABILITY_TOOL_DESCRIPTION,
            parameters_json_schema=_LOAD_CAPABILITY_SCHEMA,
            tool_kind='capability-load',
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
        if tool.tool_def.tool_kind == 'capability-load':
            return await self._load_capability(tool_args, ctx)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _load_capability(self, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> LoadCapabilityReturn:
        capability_id = tool_args['id']
        capability = ctx.capabilities.get(capability_id)
        if capability is None:
            raise ModelRetry(f'No capability found with id {capability_id!r}.')
        if capability_id in ctx.available_capability_ids:
            raise ModelRetry(LOAD_CAPABILITY_ALREADY_AVAILABLE_MESSAGE_TEMPLATE.format(capability_id=capability_id))

        parts = [
            InstructionPart(content=instruction, dynamic=True)
            for instruction in await resolve_instructions(capability.get_instructions(), ctx)
        ]

        parts.extend(await self._collect_owned_toolset_instructions(capability_id, ctx))

        instructions_text = InstructionPart.join(parts)

        ctx.loaded_capability_ids.add(capability_id)
        return {'instructions': instructions_text} if instructions_text is not None else {}

    async def _collect_owned_toolset_instructions(
        self, capability_id: str, ctx: RunContext[AgentDepsT]
    ) -> list[InstructionPart]:
        owned: list[CapabilityOwnedToolset[AgentDepsT]] = []

        def collect(ts: AbstractToolset[AgentDepsT]) -> None:
            if isinstance(ts, CapabilityOwnedToolset) and ctx.capabilities[capability_id] is ts.capability:
                owned.append(ts)

        self.apply(collect)

        parts: list[InstructionPart] = []
        for ts in owned:
            parts.extend(normalize_toolset_instructions(await ts.wrapped.get_instructions(ctx)))
        return parts
