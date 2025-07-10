from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .._run_context import AgentDepsT, RunContext
from . import AbstractToolset
from ._run import RunToolset
from ._wrapper import AbstractWrapperToolset


@dataclass
class WrapperToolset(AbstractWrapperToolset[AgentDepsT], ABC):
    """A toolset that wraps another toolset and delegates to it."""

    @abstractmethod
    async def _rewrap_for_run(
        self, wrapped: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
    ) -> WrapperToolset[AgentDepsT]:
        raise NotImplementedError()

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped = await self.wrapped.prepare_for_run(ctx)
        wrapper_for_run = await self._rewrap_for_run(wrapped, ctx)
        return RunToolset(wrapper_for_run, ctx, original=self)
