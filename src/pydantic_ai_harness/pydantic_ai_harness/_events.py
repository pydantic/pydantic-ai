"""Whether a tool's capability events can reach the run's event stream."""

from __future__ import annotations

import sys

from pydantic_ai.tools import AgentDepsT, RunContext


def event_ctx(ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT] | None:
    """`ctx`, or `None` where a tool's events cannot reach the run's event stream.

    Under Temporal a tool runs in an activity, whose run context refuses `emit`
    (pydantic/pydantic-ai#7971). A tool given `None` then works as it does outside a run:
    it emits nothing, and a change no listener can be asked about goes ahead.
    """
    temporal = sys.modules.get('pydantic_ai.durable_exec.temporal')
    if temporal is not None and isinstance(ctx, temporal.TemporalRunContext):
        return None
    return ctx
