"""Whether a tool's capability events can reach the run's event stream."""

from __future__ import annotations

import sys

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext


def event_ctx(ctx: RunContext[AgentDepsT], capability: str) -> RunContext[AgentDepsT] | None:
    """`ctx`, or `None` where a tool's events cannot reach the run's event stream.

    Under Temporal a tool runs in an activity, whose run context refuses `emit`
    (pydantic/pydantic-ai#7971). A tool given `None` then works as it does outside a run:
    it emits nothing, and a change no listener can be asked about goes ahead.

    Raises `UserError` naming `capability` when the tool came from a bare or renamed toolset:
    core refuses its capability events, and skipping them would let a write go ahead that a
    listener (an approval gate, say) was meant to be asked about.
    """
    temporal = sys.modules.get('pydantic_ai.durable_exec.temporal')
    if temporal is not None and isinstance(ctx, temporal.TemporalRunContext):
        return None
    tools = ctx.tool_manager.tools if ctx.tool_manager is not None else None
    if ctx.tool_name is not None and tools is not None and ctx._capability is None:  # pyright: ignore[reportPrivateUsage]
        # The test core's `emit` applies: the running tool must be stamped with one of this run's
        # capabilities. A bare toolset's tools are not, nor are renamed ones, which core looks up
        # under their original name.
        tool = tools.get(ctx.tool_name)
        if tool is None or tool.tool_def.capability_id not in ctx.capabilities:
            raise UserError(
                f'The `{ctx.tool_name}` tool emits capability events, which Pydantic AI accepts only from '
                f'tools a capability contributes under their own names. Pass `capabilities=[{capability}()]` '
                'rather than its toolset in `toolsets=`, without renaming its tools.'
            )
    return ctx
