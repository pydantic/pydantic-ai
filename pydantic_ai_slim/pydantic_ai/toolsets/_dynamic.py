from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool

ToolsetFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    AbstractToolset[AgentDepsT] | None | Awaitable[AbstractToolset[AgentDepsT] | None],
]
"""A sync/async function which takes a run context and returns a toolset."""


@dataclass
class _RunState(Generic[AgentDepsT]):
    """Per-run state for a DynamicToolset."""

    toolset: AbstractToolset[AgentDepsT] | None = None
    run_step: int | None = None


class DynamicToolset(AbstractToolset[AgentDepsT]):
    """A toolset that dynamically builds a toolset using a function that takes the run context.

    State is isolated per run using `ctx.run_id` as a key, allowing the same instance
    to be safely reused across multiple agent runs.
    """

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
        self._run_state: dict[str, _RunState[AgentDepsT]] = defaultdict(_RunState)

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

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            result = None
            for run_state in self._run_state.values():
                if run_state.toolset is not None:
                    result = await run_state.toolset.__aexit__(*args)
        finally:
            self._run_state.clear()
        return result

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        assert ctx.run_id is not None
        run_state = self._run_state[ctx.run_id]

        if run_state.toolset is None or (self.per_run_step and ctx.run_step != run_state.run_step):
            if run_state.toolset is not None:
                await run_state.toolset.__aexit__()

            toolset = self.toolset_func(ctx)
            if inspect.isawaitable(toolset):
                toolset = await toolset

            if toolset is not None:
                await toolset.__aenter__()

            run_state.toolset = toolset
            run_state.run_step = ctx.run_step

        if run_state.toolset is None:
            return {}

        return await run_state.toolset.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert ctx.run_id is not None
        run_state = self._run_state.get(ctx.run_id)
        assert run_state is not None and run_state.toolset is not None
        return await run_state.toolset.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        wrapped_toolsets = [rs.toolset for rs in self._run_state.values() if rs.toolset is not None]
        if not wrapped_toolsets:
            super().apply(visitor)
        else:
            for toolset in wrapped_toolsets:
                toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        wrapped_items = {run_id: rs for run_id, rs in self._run_state.items() if rs.toolset is not None}
        if not wrapped_items:
            return super().visit_and_replace(visitor)
        else:
            new_run_state: dict[str, _RunState[AgentDepsT]] = defaultdict(_RunState)
            for run_id, run_state in wrapped_items.items():
                assert run_state.toolset is not None
                new_run_state[run_id] = _RunState(
                    toolset=run_state.toolset.visit_and_replace(visitor),
                    run_step=run_state.run_step,
                )
            new_toolset = DynamicToolset(
                toolset_func=self.toolset_func,
                per_run_step=self.per_run_step,
                id=self._id,
            )
            new_toolset._run_state = new_run_state
            return new_toolset
