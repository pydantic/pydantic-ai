"""Whether a tool's capability events can reach the run's event stream."""

from __future__ import annotations

import sys

from pydantic_ai.durable_exec import DurableRunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext


def event_ctx(ctx: RunContext[AgentDepsT], capability: str, *, asks: bool = False) -> RunContext[AgentDepsT] | None:
    """`ctx`, or `None` where a tool's events cannot reach the run's event stream.

    A durable unit's run context refuses `emit` (pydantic/pydantic-ai#7971), except Render's,
    which carries events back to the run. A tool given `None` then works as it does outside a
    run: it emits nothing. A tool that `asks` listeners before a change raises `UserError` there
    instead, as no listener could answer; durable engines run such tools inline.

    Raises `UserError` naming `capability` when the tool came from a bare toolset: core refuses
    its capability events, and skipping them would let a write go ahead that a listener (an
    approval gate, say) was meant to be asked about.
    """
    if isinstance(ctx, DurableRunContext):
        if asks:
            raise UserError(
                f'{capability} cannot ask listeners about a change from inside a durable unit. '
                'Run the tool inline in the durable workflow instead.'
            )
        render = sys.modules.get('pydantic_ai_harness.render._compat')
        return ctx if render is not None and isinstance(ctx, render.RenderRunContext) else None
    tools = ctx.tool_manager.tools if ctx.tool_manager is not None else None
    if ctx.tool_name is not None and tools is not None and ctx._capability is None:  # pyright: ignore[reportPrivateUsage]
        # The test core's `emit` applies: the running tool must be stamped with one of this run's
        # capabilities. A bare toolset's tools are not. Like core, look the tool up under the name
        # the tool manager dispatched it by, which differs from `tool_name` under `PrefixTools`.
        name = ctx._dispatched_tool_name or ctx.tool_name  # pyright: ignore[reportPrivateUsage]
        tool = tools.get(name)
        if tool is None or tool.tool_def.capability_id not in ctx.capabilities:
            raise UserError(
                f'The {name!r} tool emits capability events, which Pydantic AI accepts only from '
                f'tools a capability contributes. Pass `capabilities=[{capability}()]` rather than its '
                'toolset in `toolsets=`.'
            )
    return ctx
