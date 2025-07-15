from __future__ import annotations

import inspect
from dataclasses import dataclass

from .._run_context import AgentDepsT, RunContext
from ._run import RunToolset
from ._wrapper import AbstractWrapperToolset
from .abstract import AbstractToolset


@dataclass
class WrapperToolset(AbstractWrapperToolset[AgentDepsT]):
    """A toolset that wraps another toolset and delegates to it."""

    async def rewrap_for_run(
        self, wrapped: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
    ) -> WrapperToolset[AgentDepsT]:
        """Return a new instance that wraps the provided stand-in for the original wrapped toolset that's been prepared for a specific run step.

        `WrapperToolset` subclasses with fields other than `wrapped` must implement this function to pass the appropriate extra arguments to the constructor.
        """
        if list(inspect.signature(self.__class__.__init__).parameters.keys()) == ['self', 'wrapped']:
            return self.__class__(wrapped)
        else:
            raise NotImplementedError(  # pragma: no cover
                '`WrapperToolset` subclasses with fields other than `wrapped` must implement `rewrap_for_run` to return a new instance wrapping the provided toolset.'
            )

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped = await self.wrapped.prepare_for_run(ctx)
        wrapper_for_run = await self.rewrap_for_run(wrapped, ctx)
        return RunToolset(wrapper_for_run, ctx, original=self)
