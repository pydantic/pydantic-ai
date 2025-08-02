from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Union

from typing_extensions import Self, TypeAlias

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool

ToolsetFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    Union[AbstractToolset[AgentDepsT], None, Awaitable[Union[AbstractToolset[AgentDepsT], None]]],
]
"""An sync/async function which takes a run context and returns a toolset."""


@dataclass()
class _DynamicToolset(AbstractToolset[AgentDepsT]):
    """A toolset that dynamically builds a toolset using a function that takes the run context.

    The DynamicToolset should only be provided to a single Agent run and provides a convenient copy method to ensure
    each Agent run uses its own instance of the dynamic toolset.
    """

    toolset_func: ToolsetFunc[AgentDepsT]
    per_run_step: bool = True

    _toolset: AbstractToolset[AgentDepsT] | None = None
    _run_step: int | None = None

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
        if self._toolset is not None:
            self._toolset.apply(visitor)

    def copy(self) -> _DynamicToolset[AgentDepsT]:
        return _DynamicToolset(toolset_func=self.toolset_func, per_run_step=self.per_run_step)
