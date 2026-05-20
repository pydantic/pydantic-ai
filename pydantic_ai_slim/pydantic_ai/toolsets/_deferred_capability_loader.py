from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from pydantic_ai._instructions import resolve_instructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import InstructionPart, LoadCapabilityArgs, LoadCapabilityReturn, ToolReturn
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

LOAD_CAPABILITY_TOOL_NAME = 'load_capability'

_load_capability_args_ta = TypeAdapter(LoadCapabilityArgs)
_LOAD_CAPABILITY_SCHEMA = _load_capability_args_ta.json_schema()
_LOAD_CAPABILITY_SCHEMA['title'] = 'LoadCapabilityArgs'


@dataclass
class DeferredCapabilityLoaderToolset(WrapperToolset[AgentDepsT]):
    """Adds the framework-managed `load_capability` tool."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        assert any(entry.defer_loading is True for entry in ctx.capabilities.values()), (
            'DeferredCapabilityLoaderToolset should only be installed when deferred capabilities exist.'
        )

        if LOAD_CAPABILITY_TOOL_NAME in all_tools:
            raise UserError(
                f"Tool name '{LOAD_CAPABILITY_TOOL_NAME}' is reserved for deferred capability loading. "
                'Rename your tool to avoid conflicts.'
            )

        load_tool_def = ToolDefinition(
            name=LOAD_CAPABILITY_TOOL_NAME,
            description=('Load a capability to access its full instructions and tools.'),
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
        if name == LOAD_CAPABILITY_TOOL_NAME:
            return await self._load_capability(tool_args, ctx)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _load_capability(self, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> ToolReturn:
        capability_id = tool_args['id']
        if capability_id not in ctx.capabilities:
            raise ModelRetry(f'No capability found with id {capability_id!r}.')

        parts: list[str] = []

        parts.extend(await resolve_instructions(ctx.capabilities[capability_id].get_instructions(), ctx))

        if toolset_instructions := await self._collect_owned_toolset_instructions(capability_id, ctx):
            parts.append(toolset_instructions)

        instructions_text = '\n\n'.join(parts) or None

        ctx.loaded_capability_ids.add(capability_id)
        content: LoadCapabilityReturn = {'instructions': instructions_text} if instructions_text is not None else {}
        return ToolReturn(return_value=content)

    async def _collect_owned_toolset_instructions(self, capability_id: str, ctx: RunContext[AgentDepsT]) -> str | None:
        owned: list[CapabilityOwnedToolset[AgentDepsT]] = []

        def collect(ts: AbstractToolset[AgentDepsT]) -> None:
            if isinstance(ts, CapabilityOwnedToolset) and ts.capability_id == capability_id:
                owned.append(ts)

        self.apply(collect)

        parts: list[InstructionPart] = []
        for ts in owned:
            result = await ts.wrapped.get_instructions(ctx)
            if result is None:
                continue
            for item in [result] if isinstance(result, (str, InstructionPart)) else result:
                part = item if isinstance(item, InstructionPart) else InstructionPart(content=item, dynamic=True)
                if part.content.strip():
                    parts.append(part)
        return InstructionPart.join(parts)
