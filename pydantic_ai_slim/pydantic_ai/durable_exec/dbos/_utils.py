from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from weakref import WeakKeyDictionary

import anyio
from dbos import DBOS
from dbos._context import DBOSContext, get_local_dbos_context
from pydantic import ConfigDict, with_config
from typing_extensions import TypedDict

from pydantic_ai._agent_graph import set_agent_graph_sleep
from pydantic_ai.durable_exec._toolset import guard_run_context
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.tools import AgentDepsT, RunContext

_run_ids_by_workflow: WeakKeyDictionary[DBOSContext, set[str | None]] = WeakKeyDictionary()
"""The `run_id`s of the agent runs each workflow execution has started."""


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


def record_workflow_run(run_id: str | None) -> None:
    """Record that the current workflow execution started the agent run `run_id`."""
    context = get_local_dbos_context()
    assert context is not None
    _run_ids_by_workflow.setdefault(context, set()).add(run_id)


def reject_other_runs_response(response: object, ctx: RunContext[Any]) -> None:
    """Raise if a model-request step returned a response recorded by another run of this workflow execution.

    DBOS replays a recovered workflow's steps by position, checking only their names, so concurrent runs
    of one agent can take each other's recorded steps. A response recorded before steps were stamped
    with their run's `run_id` has none, and isn't checked.
    """
    context = get_local_dbos_context()
    if (
        isinstance(response, ModelResponse)
        and response.run_id is not None
        and response.run_id != ctx.run_id
        and context is not None
        and response.run_id in _run_ids_by_workflow.get(context, ())
    ):
        raise UserError(
            f'Recovering DBOS workflow {context.workflow_id!r} replayed a model response that agent run '
            f'{response.run_id!r} recorded into agent run {ctx.run_id!r}. DBOS replays steps by position, so '
            'concurrent runs of one agent in one workflow can swap their recorded steps on recovery. Give each '
            'concurrent run a workflow of its own: a `@DBOS.workflow` function that runs the agent once, or a '
            'tool with `child_workflow` metadata that runs a nested agent.'
        )


async def _durable_sleep(seconds: float) -> None:
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
    with set_agent_graph_sleep(_durable_sleep):
        yield
