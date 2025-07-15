from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic_ai.toolsets._run import RunToolset

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from ._tool_defs import ToolDefsToolset
from .abstract import AbstractToolset


class BaseToolset(AbstractToolset[AgentDepsT], ABC):
    """A toolset that implements tool listing, tool args validation, and tool calling."""

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        return RunToolset(self, ctx)


class AsyncBaseToolset(BaseToolset[AgentDepsT], ABC):
    """A toolset that implements asynchronous tool listing, tool args validation, and tool calling."""

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        frozen_defs = ToolDefsToolset(self, await self.async_tool_defs())
        return RunToolset(frozen_defs, ctx, original=self)

    @abstractmethod
    async def async_tool_defs(self) -> list[ToolDefinition]:
        """The tool definitions that are available in this toolset.

        Here you can perform an async request to fetch available tool definitions from a remote source.
        """
        raise NotImplementedError()

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return []
