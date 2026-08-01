from dbos import DBOS
from typing_extensions import TypedDict

from pydantic_ai.durable_exec._toolset import guard_run_context_enqueue
from pydantic_ai.tools import AgentDepsT, RunContext


class StepConfig(TypedDict, total=False):
    """Configuration for a step in the DBOS workflow."""

    retries_allowed: bool
    interval_seconds: float
    max_attempts: int
    backoff_rate: float


def guard_enqueue_in_workflow(ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
    """Make `ctx.enqueue()` raise inside a workflow's step-wrapped tool call.

    Recovery replays a step's recorded output without re-executing the tool, so in-step
    enqueued messages would be silently dropped. Outside a workflow, steps degrade to
    plain calls and enqueueing keeps working, so the original context is returned.
    """
    if DBOS.workflow_id is None:
        return ctx
    return guard_run_context_enqueue(ctx, unit_noun='step', container_noun='workflow')
