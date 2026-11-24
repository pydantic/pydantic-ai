"""Every capability case's calls, run in a Temporal workflow: `DURABLE_TEMPORAL`, half of D1.

Temporal runs a function tool in an activity named after its toolset's `id`, so the tool's I/O is recorded by
construction, and what can go wrong is around it: an id-less toolset, which Temporal refuses, and workflow code
(hooks, instructions, workflow-side tools) that doesn't replay the way it ran. Each case in
`tests/capability_cases.py` and `tests/harness/capability_cases.py` with `calls` makes them in a workflow, one per
response. A `Replayer` must replay the recorded history equal, without a request, and the activities the run
scheduled must be the ones the case's `durable_steps.temporal` lists. `tests/harness/test_durable_tool_io.py` runs
the same calls under DBOS.

One dev server and one worker serve the module. The workflow runs unsandboxed: the cases' stand-ins (in-process MCP
servers, counting clients, temporary directories) live in this process, and replaying is what checks determinism.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

from ..append_only import skip_unless_installed
from ..capability_cases import DURABLE_TEMPORAL
from ._contracts import CASES, assert_holds, contract_cases
from ._counting_clients import REQUESTS
from ._durable_cases import PROMPT, build_case, installed, journal_breaks, one_call_per_response

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]

TASK_QUEUE = 'capability-cases'

_AGENTS: dict[str, Agent[None, str]] = {}
"""The agent that makes each case's calls, which the workflow runs by case name; `temporal` builds them."""


@workflow.defn(failure_exception_types=[Exception])
class CaseWorkflow:
    """Runs the agent of the case it's given; failing on any exception, so a broken case fails fast."""

    @workflow.run
    async def run(self, name: str) -> str:
        return (await _AGENTS[name].run(PROMPT)).output


@dataclass
class _Temporal:
    client: Client
    refused: dict[str, str]
    """Why Temporal refused the agent of each case it did."""


def _activity_config() -> ActivityConfig:
    # One attempt, so a failing activity fails the workflow instead of retrying with backoff.
    return ActivityConfig(
        start_to_close_timeout=timedelta(seconds=READINESS_WAIT_TIMEOUT),
        retry_policy=RetryPolicy(maximum_attempts=1),
    )


def _runner() -> UnsandboxedWorkflowRunner:
    return UnsandboxedWorkflowRunner()


@pytest.fixture(scope='module')
async def temporal(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[_Temporal]:
    """One dev server, and one worker for every case's agent, built here rather than at collection."""
    refused: dict[str, str] = {}
    for name, case in CASES.items():
        if not (DURABLE_TEMPORAL.applies(name, case) and installed(name)):
            continue
        assert case.calls is not None
        try:
            _AGENTS[name] = Agent(
                one_call_per_response(case.calls.made),
                name=f'{name}_agent',
                deps_type=type(None),
                capabilities=[
                    *build_case(name, tmp_path_factory.mktemp(name)),
                    TemporalDurability[None](activity_config=_activity_config()),
                ],
            )
        except UserError as error:
            refused[name] = str(error)
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[CaseWorkflow],
            plugins=[AgentPlugin(agent) for agent in _AGENTS.values()],
            workflow_runner=_runner(),
        ):
            yield _Temporal(client, refused)
    _AGENTS.clear()


async def _breaks(name: str, temporal: _Temporal) -> list[str]:
    """How the case of `name` breaks `DURABLE_TEMPORAL`."""
    if (refusal := temporal.refused.get(name)) is not None:
        return [f'Temporal refuses the agent: {refusal}']
    calls = CASES[name].calls
    assert calls is not None
    workflow_id = f'{name}-{uuid.uuid4()}'
    output = await temporal.client.execute_workflow(
        CaseWorkflow.run,
        name,
        id=workflow_id,
        task_queue=TASK_QUEUE,
        execution_timeout=timedelta(seconds=READINESS_WAIT_TIMEOUT),
    )
    assert output == 'done'
    history = await temporal.client.get_workflow_handle(workflow_id).fetch_history()

    before = Counter(REQUESTS)
    replayer = Replayer(
        workflows=[CaseWorkflow],
        plugins=[PydanticAIPlugin(), AgentPlugin(_AGENTS[name])],
        workflow_runner=_runner(),
    )
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    failure = replay.replay_failure
    after = Counter(REQUESTS)
    breaks = [] if failure is None else [f'the recorded history does not replay: {failure!r}']
    breaks += [] if after == before else [f'replaying made a request again: {before} became {after}']
    activities = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]
    return breaks + journal_breaks(
        activities,
        prefix=f'agent__{_AGENTS[name].name}__',
        model_request='model_request',
        calls=len(calls.made),
        expected=calls.durable_steps.temporal,
        engine='temporal',
    )


@pytest.mark.parametrize('name', contract_cases(DURABLE_TEMPORAL))
async def test_a_case_replays_equal_under_temporal(name: str, temporal: _Temporal) -> None:
    skip_unless_installed(name, CASES[name].requires)
    assert_holds(DURABLE_TEMPORAL, name, await _breaks(name, temporal))
