from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeAlias

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..messages import InstructionPart
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

    async def _evaluate_factory(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Evaluate the toolset factory function."""
        toolset = self.toolset_func(ctx)
        if inspect.isawaitable(toolset):
            toolset = await toolset
        return toolset

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Create a per-run copy with the factory evaluated.

        For `per_run_step=False`, evaluates the factory now (only chance).
        For `per_run_step=True`, defers factory evaluation to `for_run_step`.
        """
        new = DynamicToolset(
            self.toolset_func,
            per_run_step=self.per_run_step,
            id=self._id,
        )
        if not self.per_run_step:
            new._toolset = await new._evaluate_factory(ctx)
        return new

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """If per_run_step, re-evaluate factory and manage transitions in-place.

        Handles the inner toolset lifecycle (exiting old, entering new) and returns self.
        """
        if not self.per_run_step:
            return self

        new_toolset = await self._evaluate_factory(ctx)
        if new_toolset is self._toolset:
            return self

        # Detach the old toolset first so that if either the exit-old or enter-new step
        # raises, `__aexit__` does not try to exit a toolset that was never entered
        # (or has already been exited).
        old_toolset = self._toolset
        self._toolset = None
        if old_toolset is not None:
            await old_toolset.__aexit__(None, None, None)
        await self._enter_inner_toolset(new_toolset)
        return self

    async def __aenter__(self) -> Self:
        await self._enter_inner_toolset(self._toolset)
        return self

    async def _enter_inner_toolset(self, toolset: AbstractToolset[AgentDepsT] | None) -> None:
        # Only register `toolset` as the active inner toolset after a successful
        # `__aenter__`, so `__aexit__` cannot be called on a toolset that was
        # never entered.
        self._toolset = None
        if toolset is None:
            return
        await toolset.__aenter__()
        self._toolset = toolset

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            result = None
            if self._toolset is not None:
                result = await self._toolset.__aexit__(*args)
            return result
        finally:
            self._toolset = None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if self._toolset is None:
            return {}
        return await self._toolset.get_tools(ctx)

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        if self._toolset is None:
            return None
        return await self._toolset.get_instructions(ctx)

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
            new = DynamicToolset(
                self.toolset_func,
                per_run_step=self.per_run_step,
                id=self._id,
            )
            new._toolset = self._toolset.visit_and_replace(visitor)
            return new
