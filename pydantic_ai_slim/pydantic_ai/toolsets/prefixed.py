from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from ._run import RunToolset
from .wrapper import WrapperToolset


@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains."""

    prefix: str

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        prefixed_for_run = PrefixedToolset(wrapped_for_run, self.prefix)
        return RunToolset(prefixed_for_run, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [replace(tool_def, name=self._prefixed_tool_name(tool_def.name)) for tool_def in super().tool_defs]

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return super()._get_tool_args_validator(ctx, self._unprefixed_tool_name(name))

    def _max_retries_for_tool(self, name: str) -> int:
        return super()._max_retries_for_tool(self._unprefixed_tool_name(name))

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await super().call_tool(ctx, self._unprefixed_tool_name(name), tool_args, *args, **kwargs)

    def _prefixed_tool_name(self, tool_name: str) -> str:
        return f'{self.prefix}_{tool_name}'

    def _unprefixed_tool_name(self, tool_name: str) -> str:
        full_prefix = f'{self.prefix}_'
        if not tool_name.startswith(full_prefix):
            raise ValueError(f"Tool name '{tool_name}' does not start with prefix '{full_prefix}'")
        return tool_name[len(full_prefix) :]
