from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool

ToolsetFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    AbstractToolset[AgentDepsT] | None | Awaitable[AbstractToolset[AgentDepsT] | None],
]
"""A sync/async function which takes a run context and returns a toolset."""


@dataclass
class DynamicToolset(AbstractToolset[AgentDepsT]):
    """A toolset that dynamically builds a toolset using a function that takes the run context.

    It should only be used during a single agent run as it stores the generated toolset.
    To use it multiple times, copy it using `dataclasses.replace`.
    """

    toolset_func: ToolsetFunc[AgentDepsT]
    per_run_step: bool = True

    _toolset: AbstractToolset[AgentDepsT] | None = None
    _run_step: int | None = None

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            if self._toolset is not None:
                return await self._toolset.__aexit__(*args)
        finally:
            self._toolset = None
            self._run_step = None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if self._toolset is None or (self.per_run_step and ctx.run_step != self._run_step):
            if self._toolset is not None:
                await self._toolset.__aexit__()

            toolset = self.toolset_func(ctx)
            if inspect.isawaitable(toolset):
                toolset = await toolset

            if toolset is not None:
                await toolset.__aenter__()

            self._toolset = toolset
            self._run_step = ctx.run_step

        if self._toolset is None:
            return {}

        return await self._toolset.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert self._toolset is not None
        return await self._toolset.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        if self._toolset is None:
            super().apply(visitor)
        else:
            self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        if self._toolset is None:
            return super().visit_and_replace(visitor)
        else:
            return replace(self, _toolset=self._toolset.visit_and_replace(visitor))
