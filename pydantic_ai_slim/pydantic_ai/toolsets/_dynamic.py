from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool

ToolsetFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    AbstractToolset[AgentDepsT] | None | Awaitable[AbstractToolset[AgentDepsT] | None],
]
"""A sync/async function which takes a run context and returns a toolset."""


class DynamicToolset(AbstractToolset[AgentDepsT]):
    """A toolset that dynamically builds a toolset using a function that takes the run context."""

    def __init__(
        self,
        toolset_func: ToolsetFunc[AgentDepsT],
        *,
        per_run_step: bool = True,
        id: str | None = None,
    ):
        """Build a new dynamic toolset.

        Args:
            toolset_func: A function that takes the run context and returns a toolset or None.
            per_run_step: Whether to re-evaluate the toolset for each run step.
            id: An optional unique ID for the toolset. Required for durable execution environments like Temporal.
        """
        self.toolset_func = toolset_func
        self.per_run_step = per_run_step
        self._id = id
        self._toolset: AbstractToolset[AgentDepsT] | None = None
        self._run_step: int | None = None

    @property
    def id(self) -> str | None:
        return self._id

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DynamicToolset)
            and self.toolset_func is other.toolset_func  # pyright: ignore[reportUnknownMemberType]
            and self.per_run_step == other.per_run_step
            and self._id == other._id
        )

    def copy(self) -> DynamicToolset[AgentDepsT]:
        """Create a copy of this toolset for use in a new agent run."""
        return DynamicToolset(
            self.toolset_func,
            per_run_step=self.per_run_step,
            id=self._id,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            result = None
            if self._toolset is not None:
                result = await self._toolset.__aexit__(*args)
        finally:
            self._toolset = None
            self._run_step = None
        return result

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
            new_toolset = self.copy()
            new_toolset._toolset = self._toolset.visit_and_replace(visitor)
            new_toolset._run_step = self._run_step
            return new_toolset
