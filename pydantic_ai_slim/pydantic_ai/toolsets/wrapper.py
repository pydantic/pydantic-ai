from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import AbstractToolset


@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    """A toolset that wraps another toolset and delegates to it."""

    wrapped: AbstractToolset[AgentDepsT]

    @property
    def name(self) -> str:
        return self.wrapped.name

    @property
    def tool_name_conflict_hint(self) -> str:
        return self.wrapped.tool_name_conflict_hint

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return replace(self, wrapped=await self.wrapped.for_run(ctx))

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return await self.wrapped.list_tool_defs(ctx)

    def max_retries_for_tool(self, name: str) -> int:
        return self.wrapped.max_retries_for_tool(name)

    def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self.wrapped.get_tool_args_validator(ctx, name)

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        return await self.wrapped.call_tool(ctx, name, tool_args)

    def accept(self, visitor: Callable[[AbstractToolset[AgentDepsT]], Any]) -> Any:
        return self.wrapped.accept(visitor)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)  # pragma: no cover
