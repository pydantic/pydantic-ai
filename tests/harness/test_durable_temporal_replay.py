"""`DurableEngineSuite` under Temporal, `TestTemporal`: every capability case's calls and every scenario, in a workflow.

Temporal runs a function tool in an activity named after its toolset's `id`, so the tool's I/O is recorded by
construction, and what can go wrong is around it: an id-less toolset, which Temporal refuses, and workflow code
(hooks, instructions, workflow-side tools) that doesn't replay the way it ran. `TemporalEngine` runs each agent in a
workflow, and recovers it by replaying the recorded history with a `Replayer`, which must succeed without a request.
Temporal can't resume a workflow from part of its history, so it recovers only from the whole record.
`pydantic_ai.durable_exec.temporal._conformance` declares the cells that aren't `supported`.
`tests/harness/test_durable_tool_io.py` runs the same suite under DBOS.

One dev server serves the class, and each run starts a worker for its agent. The workflow runs unsandboxed: the
cases' stand-ins (in-process MCP servers, counting clients, temporary directories) live in this process, and
replaying is what checks determinism.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from pydantic_ai import Agent
from pydantic_ai.durable_exec.conformance import EngineRun
from pydantic_ai.exceptions import UserError
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.temporal_utils import temporal_dev_server_cache_dir

from ..capability_cases import DURABLE_TEMPORAL
from ._durable_cases import BuiltInEngineSuite, suite_cases

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec.temporal import (
    AgentPlugin,
    PydanticAIPlugin,
    TemporalDurability,
    _conformance as temporal_conformance,
)

# No 3.14 skip, unlike the sandboxed Temporal suites: the sandbox is what fails there, and this module runs unsandboxed.
pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

_AGENTS: dict[str, Agent[None, str]] = {}
"""The agents `TemporalEngine` runs, by name, which its workflow looks up."""


@workflow.defn(failure_exception_types=[Exception])
class RunAgentWorkflow:
    """Runs the agent it's named; failing on any exception, so a refused or broken run fails fast."""

    @workflow.run
    async def run(self, agent_name: str, prompt: str) -> str:
        return (await _AGENTS[agent_name].run(prompt)).output


def _runner() -> UnsandboxedWorkflowRunner:
    return UnsandboxedWorkflowRunner()


def _failure(error: WorkflowFailureError) -> BaseException:
    """The `UserError` the workflow's failure wraps, if any; otherwise an error naming its cause chain.

    The workflow's failure wraps the activity's, which wraps the error, as `ApplicationError`s that keep its type name.
    """
    messages: list[str] = []
    cause: BaseException | None = error.cause
    while cause is not None:
        if isinstance(cause, ApplicationError) and cause.type == UserError.__name__:
            return UserError(cause.message)
        messages.append(f'{type(cause).__name__}: {cause}')  # pragma: lax no cover - only a broken cell fails so
        cause = cause.__cause__ or getattr(cause, 'cause', None)
    return RuntimeError(f'the workflow fails: {" <- ".join(messages)}')


@dataclass
class _Record:
    agent: Agent[None, str]
    history: WorkflowHistory


@dataclass
class TemporalEngine:
    """A Temporal dev server: a run is a workflow on a worker of its own, and recovery replays its history."""

    client: Client
    name: str = 'temporal'
    records: dict[str, _Record] = field(default_factory=dict[str, _Record])
    """The agent and history of each run, by workflow id."""

    def durability(self) -> TemporalDurability[None]:
        # One attempt, so a failing activity fails the workflow instead of retrying with backoff.
        config = ActivityConfig(
            start_to_close_timeout=timedelta(seconds=READINESS_WAIT_TIMEOUT),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        return TemporalDurability[None](activity_config=config)

    async def run(self, agent: Agent[None, str], prompt: str) -> EngineRun:
        assert agent.name is not None
        _AGENTS[agent.name] = agent
        workflow_id = f'{agent.name}-{uuid.uuid4()}'
        task_queue = f'durable-engine-suite-{workflow_id}'
        async with Worker(
            self.client,
            task_queue=task_queue,
            workflows=[RunAgentWorkflow],
            plugins=[AgentPlugin(agent)],
            workflow_runner=_runner(),
        ):
            try:
                output: str | BaseException = await self.client.execute_workflow(
                    RunAgentWorkflow.run,
                    args=[agent.name, prompt],
                    id=workflow_id,
                    task_queue=task_queue,
                    execution_timeout=timedelta(seconds=READINESS_WAIT_TIMEOUT),
                )
            except WorkflowFailureError as error:
                output = _failure(error)
        history = await self.client.get_workflow_handle(workflow_id).fetch_history()
        self.records[workflow_id] = _Record(agent, history)
        prefix = f'agent__{agent.name}__'
        units = [
            event.activity_task_scheduled_event_attributes.activity_type.name.removeprefix(prefix)
            for event in history.events
            if event.HasField('activity_task_scheduled_event_attributes')
        ]
        return EngineRun(workflow_id, output, units)

    async def recover(self, run: EngineRun, *, after: int | None = None) -> EngineRun:
        assert after is None, 'Temporal replays the whole history'
        record = self.records[run.id]
        replayer = Replayer(
            workflows=[RunAgentWorkflow],
            plugins=[PydanticAIPlugin(), AgentPlugin(record.agent)],
            workflow_runner=_runner(),
        )
        replay = await replayer.replay_workflow(record.history, raise_on_replay_failure=False)
        if (failure := replay.replay_failure) is not None:  # pragma: lax no cover - only a broken cell fails so
            return EngineRun(run.id, RuntimeError(f'the recorded history does not replay: {failure!r}'), run.units)
        return EngineRun(run.id, run.output, run.units)


class TestTemporal(BuiltInEngineSuite):
    support = temporal_conformance.SUPPORT
    recovery = temporal_conformance.RECOVERY
    cases = suite_cases(DURABLE_TEMPORAL)
    durable_steps = 'temporal'
    model_request_unit = 'model_request'

    @pytest.fixture(scope='class')
    @classmethod
    async def engine(cls) -> AsyncIterator[TemporalEngine]:
        async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
            client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
            yield TemporalEngine(client)
        _AGENTS.clear()

    @pytest.fixture
    def can_fork_mid_run(self) -> bool:
        return False  # Temporal replays a workflow's whole history; it can't resume one from part of it.
