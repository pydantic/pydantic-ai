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
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.temporal_utils import temporal_dev_server_cache_dir

from ..append_only import skip_unless_installed
from ..capability_cases import DURABLE_TEMPORAL, SCENARIO_RUNS, Scenario
from ._contracts import CASES, assert_cell_holds, contract_cases
from ._durable_cases import (
    OBSERVED,
    PROMPT,
    assert_scenario_cell,
    build_case,
    counting_requests,
    installed,
    journal_breaks,
    observe_run,
    scenario_agent,
    scenario_breaks,
    streaming_one_call_per_response,
)

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability

# No 3.14 skip, unlike the sandboxed Temporal suites: the sandbox is what fails there, and this module runs unsandboxed.
pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

TASK_QUEUE = 'capability-cases'

_AGENTS: dict[str, Agent[None, str]] = {}
"""The agent that makes each case's calls, which the workflow runs by case name; `temporal` builds them."""


@workflow.defn(failure_exception_types=[Exception])
class CaseWorkflow:
    """Runs the agent of the case it's given; failing on any exception, so a broken case fails fast."""

    @workflow.run
    async def run(self, name: str) -> str:
        return (await _AGENTS[name].run(PROMPT)).output


@workflow.defn(failure_exception_types=[Exception])
class ScenarioWorkflow:
    """Runs the agent of the scenario it's given, and returns what the run observed."""

    @workflow.run
    async def run(self, scenario: str) -> list[str]:
        return await observe_run(_AGENTS[scenario])


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
                streaming_one_call_per_response(case.calls.made),
                name=f'{name}_agent',
                deps_type=type(None),
                capabilities=[
                    *build_case(name, tmp_path_factory.mktemp(name)),
                    TemporalDurability[None](activity_config=_activity_config()),
                ],
            )
        except UserError as error:
            refused[name] = str(error)
    for scenario in SCENARIO_RUNS:
        _AGENTS[scenario] = scenario_agent(scenario, TemporalDurability[None](activity_config=_activity_config()))
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[CaseWorkflow, ScenarioWorkflow],
            plugins=[AgentPlugin(agent) for agent in _AGENTS.values()],
            workflow_runner=_runner(),
        ):
            yield _Temporal(client, refused)
    _AGENTS.clear()


def _causes(error: BaseException) -> str:
    """The messages of `error`'s cause chain: the workflow's failure wraps the activity's, which wraps the error."""
    messages: list[str] = []
    cause: BaseException | None = error.__cause__ or getattr(error, 'cause', None)
    while cause is not None:
        messages.append(f'{type(cause).__name__}: {cause}')
        cause = cause.__cause__ or getattr(cause, 'cause', None)
    return ' <- '.join(messages)


async def _breaks(name: str, temporal: _Temporal, count_requests: Callable[[], Counter[str]]) -> list[str]:
    """How the case of `name` breaks `DURABLE_TEMPORAL`."""
    if (refusal := temporal.refused.get(name)) is not None:
        return [f'Temporal refuses the agent: {refusal}']
    calls = CASES[name].calls
    assert calls is not None
    workflow_id = f'{name}-{uuid.uuid4()}'
    try:
        output = await temporal.client.execute_workflow(
            CaseWorkflow.run,
            name,
            id=workflow_id,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=READINESS_WAIT_TIMEOUT),
        )
    except WorkflowFailureError as error:
        return [f'the workflow fails: {_causes(error)}']
    assert output == 'done'
    history = await temporal.client.get_workflow_handle(workflow_id).fetch_history()

    before = count_requests()
    replayer = Replayer(
        workflows=[CaseWorkflow],
        plugins=[PydanticAIPlugin(), AgentPlugin(_AGENTS[name])],
        workflow_runner=_runner(),
    )
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    failure = replay.replay_failure
    after = count_requests()
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
        model_requests=('model_request_stream',) if calls.durable_steps.streams else ('model_request',),
        calls=len(calls.made),
        expected=calls.durable_steps.temporal,
        engine='temporal',
    )


@pytest.mark.parametrize('name', contract_cases(DURABLE_TEMPORAL))
async def test_a_case_replays_equal_under_temporal(
    name: str, temporal: _Temporal, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    skip_unless_installed(name, CASES[name].requires)
    count_requests = counting_requests(monkeypatch, tmp_path)
    assert_cell_holds('temporal', DURABLE_TEMPORAL, name, await _breaks(name, temporal, count_requests))


@pytest.mark.parametrize('scenario', sorted(SCENARIO_RUNS))
async def test_a_scenario_observes_the_same_on_temporal_replay(scenario: Scenario, temporal: _Temporal) -> None:
    """The events and enqueued messages a run observes come back the same, each once, when a `Replayer` replays it."""
    workflow_id = f'{scenario}-{uuid.uuid4()}'
    try:
        first = await temporal.client.execute_workflow(
            ScenarioWorkflow.run,
            scenario,
            id=workflow_id,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=READINESS_WAIT_TIMEOUT),
        )
    except WorkflowFailureError as error:
        breaks = [f'the workflow fails: {_causes(error)}']
    else:
        history = await temporal.client.get_workflow_handle(workflow_id).fetch_history()
        agent = _AGENTS[scenario]
        assert agent.name is not None
        OBSERVED.pop(agent.name, None)
        replayer = Replayer(
            workflows=[ScenarioWorkflow],
            plugins=[PydanticAIPlugin(), AgentPlugin(agent)],
            workflow_runner=_runner(),
        )
        replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
        breaks = [] if replay.replay_failure is None else [f'the history does not replay: {replay.replay_failure!r}']
        breaks += scenario_breaks(scenario, first, OBSERVED.get(agent.name, []))
    assert_scenario_cell('temporal', scenario, breaks)
