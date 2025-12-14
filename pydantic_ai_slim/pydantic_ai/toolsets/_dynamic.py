from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeAlias

from typing_extensions import Self, TypedDict

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool

ToolsetFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    AbstractToolset[AgentDepsT] | None | Awaitable[AbstractToolset[AgentDepsT] | None],
]
"""A sync/async function which takes a run context and returns a toolset."""


class ToolsetRunStep(TypedDict, Generic[AgentDepsT]):
    """State for a DynamicToolset for a specific run."""

    toolset: AbstractToolset[AgentDepsT] | None
    run_step: int | None


class DynamicToolset(AbstractToolset[AgentDepsT]):
    """A toolset that dynamically builds a toolset using a function that takes the run context.

    State is isolated per run using `ctx.run_id` as a key, allowing the same instance
    to be safely reused across multiple agent runs.
    """

    toolset_func: ToolsetFunc[AgentDepsT]
    per_run_step: bool
    _id: str | None
    _toolset_runstep: dict[str, ToolsetRunStep[AgentDepsT]]

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
            per_run_step: Whether to re-evaluate the toolset for each run step. Defaults to True.
            id: An optional unique ID for the toolset. A toolset needs to have an ID in order to be used
                in a durable execution environment like Temporal, in which case the ID will be used to
                identify the toolset's activities within the workflow.
        """
        self.toolset_func = toolset_func
        self.per_run_step = per_run_step
        self._id = id
        self._toolset_runstep = {}

    @property
    def id(self) -> str | None:
        return self._id

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
            for run_state in self._toolset_runstep.values():
                if run_state['toolset'] is not None:
                    result = await run_state['toolset'].__aexit__(*args)
        finally:
            self._toolset_runstep.clear()
        return result

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        run_id = ctx.run_id or '__default__'

        if run_id not in self._toolset_runstep:
            self._toolset_runstep[run_id] = {'toolset': None, 'run_step': None}

        run_state = self._toolset_runstep[run_id]

        if run_state['toolset'] is None or (self.per_run_step and ctx.run_step != run_state['run_step']):
            if run_state['toolset'] is not None:
                await run_state['toolset'].__aexit__()

            toolset = self.toolset_func(ctx)
            if inspect.isawaitable(toolset):
                toolset = await toolset

            if toolset is not None:
                await toolset.__aenter__()

            run_state['toolset'] = toolset
            run_state['run_step'] = ctx.run_step

        if run_state['toolset'] is None:
            return {}

        return await run_state['toolset'].get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        run_id = ctx.run_id or '__default__'
        run_state = self._toolset_runstep.get(run_id)
        assert run_state is not None and run_state['toolset'] is not None
        return await run_state['toolset'].call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        wrapped_toolsets = [rs['toolset'] for rs in self._toolset_runstep.values() if rs['toolset'] is not None]
        if not wrapped_toolsets:
            super().apply(visitor)
        else:
            for toolset in wrapped_toolsets:
                toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        wrapped_items = [(run_id, rs) for run_id, rs in self._toolset_runstep.items() if rs['toolset'] is not None]
        if not wrapped_items:
            return super().visit_and_replace(visitor)
        else:
            new_copy = self.copy()
            for run_id, run_state in wrapped_items:
                toolset = run_state['toolset']
                assert toolset is not None
                new_copy._toolset_runstep[run_id] = {
                    'toolset': toolset.visit_and_replace(visitor),
                    'run_step': run_state['run_step'],
                }
            return new_copy
