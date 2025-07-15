from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .wrapper import WrapperToolset


@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains."""

    prefix: str

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return [
            replace(tool_def, name=self._prefixed_tool_name(tool_def.name))
            for tool_def in await super().list_tool_defs(ctx)
        ]

    def max_retries_for_tool(self, name: str) -> int:
        return super().max_retries_for_tool(self._unprefixed_tool_name(name))  # pragma: no cover

    def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        original_name = self._unprefixed_tool_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return super().get_tool_args_validator(ctx, original_name)

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        original_name = self._unprefixed_tool_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return await super().call_tool(ctx, original_name, tool_args)

    def _prefixed_tool_name(self, tool_name: str) -> str:
        return f'{self.prefix}_{tool_name}'

    def _unprefixed_tool_name(self, tool_name: str) -> str:
        full_prefix = f'{self.prefix}_'
        if not tool_name.startswith(full_prefix):
            raise ValueError(f"Tool name '{tool_name}' does not start with prefix '{full_prefix}'")
        return tool_name[len(full_prefix) :]
