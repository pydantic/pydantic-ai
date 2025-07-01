from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from . import AbstractToolset

if TYPE_CHECKING:
    from ..models import Model
    from .run import RunToolset


@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT], ABC):
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

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return await self.wrapped.__aexit__(exc_type, exc_value, traceback)

    @abstractmethod
    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        raise NotImplementedError()

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self.wrapped.tool_defs

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self.wrapped._get_tool_args_validator(ctx, name)

    def _max_retries_for_tool(self, name: str) -> int:
        return self.wrapped._max_retries_for_tool(name)

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await self.wrapped.call_tool(ctx, name, tool_args, *args, **kwargs)

    def set_mcp_sampling_model(self, model: Model) -> None:
        self.wrapped.set_mcp_sampling_model(model)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)  # pragma: no cover
