from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self
from weakref import WeakKeyDictionary

from dbos import DBOS
from dbos._error import DBOSPatchNondeterminismError

from pydantic_ai.durable_exec._toolset import holds_tool_instance_defaults, listing_tools_before_tool_instance_defaults
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset

from ._utils import in_workflow_code

if TYPE_CHECKING:
    from dbos._context import DBOSContext

TOOL_INSTANCE_DEFAULTS_PATCH = 'pydantic_ai:function_toolset_defaults_on_tool_instances'
"""`DBOS.patch_async()` name marking workflows in which `FunctionToolset` applied its `sequential` and `defer_loading` to `Tool` instances."""

_tool_instance_defaults_patched: WeakKeyDictionary[DBOSContext, asyncio.Task[bool] | bool] = WeakKeyDictionary()
"""Each workflow execution's `DBOS.patch_async()` result, shared by the agent runs in it like `workflow.patched()`'s."""


@dataclass
class ToolInstanceDefaultsGateToolset(WrapperToolset[AgentDepsT]):
    """Lists `function_toolset`'s `Tool` instances without the `sequential` and `defer_loading` a `FunctionToolset` set on them, in workflows started before it did.

    DBOS numbers steps as they start, and a sequential or deferred tool changes which steps the workflow
    starts, so those workflows can only recover without them.
    """

    function_toolset: FunctionToolset[AgentDepsT]

    async def __aenter__(self) -> Self:
        # A run enters its toolsets one at a time before it starts any step of its own, and
        # `DBOS.patch_async` must not run concurrently with other steps in the workflow.
        if in_workflow_code() and holds_tool_instance_defaults(self.function_toolset):
            context = workflow_context()
            patched = _tool_instance_defaults_patched.get(context)
            if not isinstance(patched, bool):
                if patched is None:
                    patched = asyncio.ensure_future(_patch_tool_instance_defaults())
                    _tool_instance_defaults_patched[context] = patched
                # A task keeps the context it ran in, and with it this key, alive; keep only its result.
                _tool_instance_defaults_patched[context] = await asyncio.shield(patched)
        return await super().__aenter__()

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        patched = _tool_instance_defaults_patched.get(workflow_context()) if in_workflow_code() else None
        if patched is None or patched is True:
            return await super().get_tools(ctx)
        with listing_tools_before_tool_instance_defaults():
            return await super().get_tools(ctx)


def gate_tool_instance_defaults(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
    """Wrap `toolset` in a `ToolInstanceDefaultsGateToolset` if it's a `FunctionToolset` holding a `Tool` a `FunctionToolset` made sequential or deferred."""
    if isinstance(toolset, FunctionToolset) and holds_tool_instance_defaults(toolset):
        return ToolInstanceDefaultsGateToolset(toolset, function_toolset=toolset)
    return toolset


async def _patch_tool_instance_defaults() -> bool:
    try:
        return await DBOS.patch_async(TOOL_INSTANCE_DEFAULTS_PATCH)
    except DBOSPatchNondeterminismError:
        # Another run in the workflow started a step during the probe. `patch_async` reserved no
        # position and recorded nothing, so this workflow lists the tools as it did before the patch.
        return False


def workflow_context() -> DBOSContext:
    """The DBOS context of the workflow execution this code runs in."""
    from dbos._context import get_local_dbos_context

    context = get_local_dbos_context()
    assert context is not None
    return context
