from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from . import AbstractToolset

if TYPE_CHECKING:
    from ._run import RunToolset


@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT], ABC):
    """A toolset that wraps another toolset and delegates to it."""

    wrapped: AbstractToolset[AgentDepsT]

    @property
    def name(self) -> str:
        return self.wrapped.name

    @property
    def _tool_name_conflict_hint(self) -> str:
        return self.wrapped._tool_name_conflict_hint

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    @abstractmethod
    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        raise NotImplementedError()

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self.wrapped.tool_defs

    def _max_retries_for_tool(self, name: str) -> int:
        return self.wrapped._max_retries_for_tool(name)

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self.wrapped._get_tool_args_validator(ctx, name)

    def _call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        return self.wrapped._call_tool(ctx, name, tool_args)

    def accept(self, visitor: Callable[[AbstractToolset[AgentDepsT]], Any]) -> Any:
        return self.wrapped.accept(visitor)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)  # pragma: no cover
