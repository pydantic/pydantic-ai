from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..messages import ToolCallPart
from ..tools import ToolDefinition

if TYPE_CHECKING:
    from ._run import RunToolset


class AbstractToolset(ABC, Generic[AgentDepsT]):
    """A toolset is a collection of tools that can be used by an agent.

    It is responsible for:
    - Listing the tools it contains
    - Validating the arguments of the tools
    - Calling the tools
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace('Toolset', ' toolset')

    @property
    def _tool_name_conflict_hint(self) -> str:
        return 'Consider renaming the tool or wrapping the toolset in a `PrefixedToolset` to avoid name conflicts.'

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    @abstractmethod
    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def tool_defs(self) -> list[ToolDefinition]:
        raise NotImplementedError()

    @property
    def tool_names(self) -> list[str]:
        return [tool_def.name for tool_def in self.tool_defs]

    def get_tool_def(self, name: str) -> ToolDefinition | None:
        return next((tool_def for tool_def in self.tool_defs if tool_def.name == name), None)

    @abstractmethod
    def _max_retries_for_tool(self, name: str) -> int:
        raise NotImplementedError()

    @abstractmethod
    async def call_tool(self, call: ToolCallPart, ctx: RunContext[AgentDepsT], allow_partial: bool = False) -> Any:
        raise NotImplementedError()

    def accept(self, visitor: Callable[[AbstractToolset[AgentDepsT]], Any]) -> Any:
        return visitor(self)
