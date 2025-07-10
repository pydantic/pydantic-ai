from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .base import AbstractToolset
from .wrapper import WrapperToolset


@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains."""

    prefix: str

    async def _rewrap_for_run(
        self, wrapped: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
    ) -> WrapperToolset[AgentDepsT]:
        return PrefixedToolset(wrapped, self.prefix)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [replace(tool_def, name=self._prefixed_tool_name(tool_def.name)) for tool_def in super().tool_defs]

    def _max_retries_for_tool(self, name: str) -> int:
        return super()._max_retries_for_tool(self._unprefixed_tool_name(name))  # pragma: no cover

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        original_name = self._unprefixed_tool_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return super()._get_tool_args_validator(ctx, original_name)

    def _call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        original_name = self._unprefixed_tool_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return super()._call_tool(ctx, original_name, tool_args)

    def _prefixed_tool_name(self, tool_name: str) -> str:
        return f'{self.prefix}_{tool_name}'

    def _unprefixed_tool_name(self, tool_name: str) -> str:
        full_prefix = f'{self.prefix}_'
        if not tool_name.startswith(full_prefix):
            raise ValueError(f"Tool name '{tool_name}' does not start with prefix '{full_prefix}'")
        return tool_name[len(full_prefix) :]
