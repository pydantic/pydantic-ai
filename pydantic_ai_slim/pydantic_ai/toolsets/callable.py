from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic_ai.toolsets._run import RunToolset

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from . import AbstractToolset


class CallableToolset(AbstractToolset[AgentDepsT], ABC):
    """A toolset that implements tool listing, tool args validation, and tool calling."""

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        return RunToolset(self, ctx)


class AsyncCallableToolset(CallableToolset[AgentDepsT], ABC):
    """A toolset that implements asynchronous tool listing, tool args validation, and tool calling."""

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        return RunToolset(self, ctx, await self.async_tool_defs())

    @abstractmethod
    async def async_tool_defs(self) -> list[ToolDefinition]:
        raise NotImplementedError()

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return []
