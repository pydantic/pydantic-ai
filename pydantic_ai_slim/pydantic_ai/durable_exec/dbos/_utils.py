from collections.abc import Generator
from contextlib import contextmanager

import anyio
from dbos import DBOS
from pydantic import ConfigDict, with_config
from typing_extensions import TypedDict

from pydantic_ai._agent_graph import set_agent_graph_sleep
from pydantic_ai.durable_exec._toolset import guard_run_context
from pydantic_ai.tools import AgentDepsT, RunContext


class StepConfig(TypedDict, total=False):
    """Configuration for a step in the DBOS workflow."""

    retries_allowed: bool
    interval_seconds: float
    max_attempts: int
    backoff_rate: float


@with_config(ConfigDict(extra='forbid'))
class ChildWorkflowConfig(TypedDict, total=False):
    """Configuration for a tool call run as a DBOS child workflow."""

    workflow_timeout_sec: float
    """Passed to DBOS's `SetWorkflowTimeout` when the child workflow starts.

    When it expires, DBOS cancels the child workflow and the tool call raises `DBOSAwaitedWorkflowCancelledError`.
    """


def guard_enqueue_in_workflow(ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
    """Make `ctx.enqueue()` raise inside a workflow's step-wrapped tool call.

    Recovery replays a step's recorded output without re-executing the tool, so in-step
    enqueued messages would be silently dropped. Outside a workflow, steps degrade to
    plain calls and enqueueing keeps working, so the original context is returned.
    """
    if DBOS.workflow_id is None:
        return ctx
    return guard_run_context(ctx, unit_noun='step', container_noun='workflow')


def in_workflow_code() -> bool:
    """Whether this is a DBOS workflow's own code, rather than a step's or code outside any workflow."""
    return DBOS.workflow_id is not None and DBOS.step_id is None


async def durable_sleep(seconds: float) -> None:
    # DBOS rejects its durable sleep inside a step, which a nested run can reach through the
    # inherited `ContextVar`, so only workflow code sleeps durably.
    if in_workflow_code():
        await DBOS.sleep_async(seconds)
    else:
        await anyio.sleep(max(seconds, 0))


@contextmanager
def using_durable_sleep() -> Generator[None]:
    """Make agent-graph waits (e.g. between background-mode polls) durable `DBOS.sleep_async` steps in workflow code.

    Recovery then skips completed waits instead of sleeping them again. Outside a workflow the
    agent graph's sleep is left as is.
    """
    if not in_workflow_code():
        yield
        return
    with set_agent_graph_sleep(durable_sleep):
        yield
