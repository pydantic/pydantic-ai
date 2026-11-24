"""How each durable engine runs a `BackgroundTools` background tool, and keeps one running across a mid-run approval.

A pinning table: each cell records what the engine does today, and its test fails once that changes, so the cell
changes with it. One scripted model and one set of tools run under every engine, in that engine's own test harness.
"""

from __future__ import annotations

import re
import uuid
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import ClassVar, Literal, get_args

import anyio
import pytest

from pydantic_ai import Agent, FunctionToolset, Tool
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai_harness import BackgroundTools
from tests.capability_cases import ENGINES, Engine, Support

from ...conftest import try_import

# Each engine's SDK is imported before its test helpers, so a missing SDK is an `ImportError` here, not the
# module-level skip some of those helpers raise.
with try_import() as temporal_imports:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
    from tests.temporal_utils import temporal_dev_server_cache_dir

    @workflow.defn
    class BackgroundToolsWorkflow(PydanticAIWorkflow):
        agent: ClassVar[Agent[None, str]]

        @workflow.run
        async def run(self) -> str:
            return (await self.agent.run('go')).output


with try_import() as dbos_imports:
    from dbos import DBOS, DBOSConfig

    from pydantic_ai.durable_exec.dbos import DBOSDurability

with try_import() as prefect_imports:
    from prefect import flow

    from pydantic_ai.durable_exec.prefect import PrefectDurability

with try_import() as aws_lambda_imports:
    import aws_durable_execution_sdk_python  # noqa: F401  # pyright: ignore[reportUnusedImport]

    from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable

    from ..aws_lambda.conftest import FakeDurableContext

with try_import() as absurd_imports:
    import absurd_sdk

    from pydantic_ai_harness.absurd import AbsurdDurability

    from ..absurd._task import running_task_context
    from ..absurd.conftest import absurd, async_conn, db_dsn, queue_name  # noqa: F401  # pyright: ignore[reportUnusedImport]

with try_import() as pgtask_imports:
    import pgtask  # noqa: F401  # pyright: ignore[reportUnusedImport]

    from pydantic_ai_harness.pgtask import PGTaskDurability

    from ..pgtask._task import PGTaskDatabase, run_in_task
    from ..pgtask.conftest import pgtask_db, pgtask_dsn  # noqa: F401  # pyright: ignore[reportUnusedImport]

with try_import() as azure_imports:
    import durabletask  # noqa: F401  # pyright: ignore[reportUnusedImport]

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

    from ..azure_durable_task.conftest import durable_worker  # noqa: F401  # pyright: ignore[reportUnusedImport]
    from ..azure_durable_task.worker import InMemoryWorker, registered_agent, run_agent

with try_import() as render_imports:
    from render.workflows import Workflows

    from pydantic_ai_harness import RenderWorkflows

    from ..render.conftest import RecordingTaskContext, run_agent_in_task


Scenario = Literal['BackgroundTools.runs_in_background', 'BackgroundTools.survives_approval_wait']
SCENARIOS: tuple[Scenario, ...] = get_args(Scenario)

_FOREGROUND = (
    'the engine runs tool calls sequentially, so the tool runs in the foreground and returns its result inline'
)
_SEQUENTIAL_DEFAULT = f"{_FOREGROUND}; `parallel_execution_mode='sequential'` is the default"
_NO_WAIT = (
    'a mid-run approval wait raises `UserError` under the engine, so no background task can keep running across it'
)
_NO_DURABLE_WAIT = 'the engine has no durable wait for an outside answer, so a mid-run approval wait raises `UserError`'
_REFUSED = 'does not support waiting for external events yet'

# Kept out of `ENGINE_SUPPORT` in `tests/capability_cases.py`: those cells are D1's bar on a case's `calls`, and these
# are scenarios of their own. They may move into `capability_cases.py` later.
SCENARIO_SUPPORT: Mapping[Scenario, Mapping[Engine, Support]] = {
    'BackgroundTools.runs_in_background': {
        'temporal': Support('supported'),
        'dbos': Support('supported'),
        'prefect': Support('supported'),
        'aws_lambda': Support('limited', _FOREGROUND),
        'absurd': Support('limited', _SEQUENTIAL_DEFAULT),
        'pgtask': Support('limited', _SEQUENTIAL_DEFAULT),
        'azure_durable_task': Support('supported'),
        'render': Support('supported'),
    },
    'BackgroundTools.survives_approval_wait': {
        'temporal': Support('broken', _NO_WAIT, issue=14443),
        'dbos': Support('broken', _NO_WAIT, issue=14443),
        'prefect': Support('refused', _NO_DURABLE_WAIT, raises=_REFUSED),
        'aws_lambda': Support('limited', f'{_FOREGROUND}, so no call is in flight across the wait'),
        'absurd': Support('refused', _NO_DURABLE_WAIT, raises=_REFUSED),
        'pgtask': Support('refused', _NO_DURABLE_WAIT, raises=_REFUSED),
        'azure_durable_task': Support('refused', _NO_DURABLE_WAIT, raises=_REFUSED),
        'render': Support('refused', _NO_DURABLE_WAIT, raises=_REFUSED),
    },
}
"""What each engine does with `BackgroundTools` in each scenario.

- `runs_in_background`: a tool marked `background` replies that it is running in the background, and its result
  arrives later as a follow-up message.
- `survives_approval_wait`: while that call is still running, the model calls a tool that needs approval, which a
  `HandleDeferredToolCalls` handler that takes a resume token answers mid-run. The call keeps running across the
  wait, without the run suspending, and its result arrives.
"""

BACKGROUND = 'background result'
"""What the model says when the result arrived as a follow-up message: the bar of every scenario."""
FOREGROUND = 'inline result'
"""What the model says when the call returned its result inline, as a normal tool return."""

RESULT = 'research result'


@dataclass
class _Gate:
    """Holds `research` until the approval is asked for, so a background call is still running across the wait."""

    open: bool = True
    event: anyio.Event | None = None

    async def wait(self) -> None:
        if not self.open:
            # Created on first use, on the event loop of the engine that runs the tool.
            self.event = self.event or anyio.Event()
            await self.event.wait()

    def release(self) -> None:
        self.open = True
        if self.event is not None:
            self.event.set()


_gate = _Gate()


async def research() -> str:
    await _gate.wait()
    return RESULT


def delete() -> str:
    return 'deleted'


async def ask(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> DeferredToolResults:
    """Answer in place, so an engine that supports the wait records the answer and goes on without suspending."""
    _gate.release()
    return requests.build_results(approve_all=True)


def _returns(messages: Sequence[ModelMessage], tool_name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call `research`, then `delete` if the agent has it, then say how `research`'s result arrived."""
    if not _returns(messages, 'research'):
        return ModelResponse(parts=[ToolCallPart('research', {}, tool_call_id='research-call')])
    if any(tool.name == 'delete' for tool in info.function_tools) and not _returns(messages, 'delete'):
        return ModelResponse(parts=[ToolCallPart('delete', {}, tool_call_id='delete-call')])
    if any(part.content == RESULT for part in _returns(messages, 'research')):
        return ModelResponse(parts=[TextPart(FOREGROUND)])
    if any(RESULT in str(part) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart(BACKGROUND)])
    # The call is still running: ending the response lets its result start the next one.
    return ModelResponse(parts=[TextPart('waiting')])


def _agent(durability: AbstractCapability[None], scenario: Scenario, name: str) -> Agent[None, str]:
    approval = scenario == 'BackgroundTools.survives_approval_wait'
    return Agent(
        FunctionModel(_model),
        name=name,
        deps_type=type(None),
        toolsets=[
            FunctionToolset([research], id='research', metadata={'background': True}),
            *([FunctionToolset([Tool(delete, requires_approval=True)], id='gated')] if approval else []),
        ],
        capabilities=[
            BackgroundTools(),
            durability,
            *([HandleDeferredToolCalls[None](ask, timeout=timedelta(minutes=1))] if approval else []),
        ],
    )


def _cell(scenario: Scenario, engine: Engine) -> Support:
    """The cell under test; an `untested` one skips. Closes the gate when the scenario needs `research` held."""
    cell = SCENARIO_SUPPORT[scenario][engine]
    if cell.level == 'untested':  # pragma: no cover - no cell is untested today
        pytest.skip(f'#{cell.issue}: {cell.reason}')
    # A foreground engine runs `research` before the model asks for approval, so the gate can't wait for it. Nor can
    # Azure's in-memory worker, which runs one activity at a time: a held call would block the whole orchestration.
    global _gate
    _gate = _Gate(
        open=scenario != 'BackgroundTools.survives_approval_wait'
        or _foreground(engine)
        or engine == 'azure_durable_task'
    )
    return cell


def _foreground(engine: Engine) -> bool:
    return SCENARIO_SUPPORT['BackgroundTools.runs_in_background'][engine].level == 'limited'


@contextmanager
def _foreground_warning_allowed(engine: Engine) -> Generator[None]:
    """Allow, without requiring, a warning that a sequential run runs a background tool in the foreground."""
    with warnings.catch_warnings():
        if _foreground(engine):
            warnings.filterwarnings('ignore', message=r".*runs '.*' in the foreground", category=UserWarning)
        yield


def _error_text(error: BaseException) -> str:
    """The messages along an error's cause chain, as engines wrap the run's own error."""
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None:
        messages.append(f'{type(current).__name__}: {current}')
        current = current.__cause__
    return ' <- '.join(messages)


def _assert_cell(scenario: Scenario, engine: Engine, cell: Support, observed: str) -> None:
    if cell.level == 'supported':
        assert observed == BACKGROUND
    elif cell.level == 'limited':
        assert observed == FOREGROUND
    elif cell.level == 'refused':
        assert cell.raises is not None and re.search(cell.raises, observed), observed
    else:
        assert observed != BACKGROUND, f'{scenario} now passes under {engine}: update its `SCENARIO_SUPPORT` cell'


async def _check(
    scenario: Scenario,
    engine: Engine,
    run: Callable[[Agent[None, str]], Awaitable[str]],
    durability: Callable[[], AbstractCapability[None]],
) -> None:
    cell = _cell(scenario, engine)
    agent = _agent(durability(), scenario, f'bg_{engine}_{uuid.uuid4().hex[:8]}')
    try:
        with _foreground_warning_allowed(engine):
            observed = await run(agent)
    except Exception as error:
        observed = _error_text(error)
    finally:
        # A tool still holding the gate would keep its engine's worker from shutting down.
        _gate.release()
    _assert_cell(scenario, engine, cell, observed)


def _check_sync(
    scenario: Scenario,
    engine: Engine,
    run: Callable[[Agent[None, str]], str],
    durability: Callable[[], AbstractCapability[None]],
) -> None:
    """`_check` for an engine whose test harness drives the run from a synchronous handler, on a loop of its own."""
    cell = _cell(scenario, engine)
    agent = _agent(durability(), scenario, f'bg_{engine}')
    try:
        with _foreground_warning_allowed(engine):
            observed = run(agent)
    except Exception as error:
        observed = _error_text(error)
    _assert_cell(scenario, engine, cell, observed)


parametrize_scenarios = pytest.mark.parametrize('scenario', SCENARIOS, ids=lambda scenario: scenario.split('.')[1])


def test_every_scenario_has_a_cell_for_every_engine() -> None:
    for scenario in SCENARIOS:
        assert set(SCENARIO_SUPPORT[scenario]) == set(ENGINES), scenario
    assert set(SCENARIO_SUPPORT) == set(SCENARIOS)


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
        download_dest_dir=temporal_dev_server_cache_dir(),
    ) as env:
        yield env


@pytest.mark.temporal
@pytest.mark.xdist_group(name='harness-temporal')
@pytest.mark.skipif(not temporal_imports(), reason='temporalio not installed')
@parametrize_scenarios
async def test_temporal(scenario: Scenario, temporal_env: WorkflowEnvironment) -> None:
    async def run(agent: Agent[None, str]) -> str:
        BackgroundToolsWorkflow.agent = agent
        BackgroundToolsWorkflow.__pydantic_ai_agents__ = [agent]
        client = await Client.connect(
            temporal_env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()]
        )
        queue = f'background-tools-{uuid.uuid4().hex}'
        runner = SandboxedWorkflowRunner(
            restrictions=SandboxRestrictions.default.with_passthrough_modules(
                __name__, 'annotated_types', 'coverage', 'pydantic_graph'
            )
        )
        # `debug_mode` for the reason `test_temporal.py` gives: no deadlock detector on a slow first activation.
        async with Worker(
            client, task_queue=queue, workflows=[BackgroundToolsWorkflow], workflow_runner=runner, debug_mode=True
        ):
            try:
                return await client.execute_workflow(BackgroundToolsWorkflow.run, id=queue, task_queue=queue)
            finally:
                _gate.release()

    activity_config = ActivityConfig(
        start_to_close_timeout=timedelta(seconds=30), retry_policy=RetryPolicy(maximum_attempts=1)
    )
    await _check(scenario, 'temporal', run, lambda: TemporalDurability[None](activity_config=activity_config))


@pytest.mark.skipif(not dbos_imports(), reason='dbos not installed')
@parametrize_scenarios
async def test_dbos(scenario: Scenario, tmp_path: Path) -> None:
    async def run(agent: Agent[None, str]) -> str:
        @DBOS.workflow(name=f'{agent.name}_workflow')
        async def workflow() -> str:
            return (await agent.run('go')).output

        DBOS(config=DBOSConfig(name='background_tools', system_database_url=f'sqlite:///{tmp_path / "db.sqlite"}'))
        DBOS.launch()
        try:
            return await workflow()
        finally:
            DBOS.destroy()

    await _check(scenario, 'dbos', run, DBOSDurability[None])


@pytest.mark.xdist_group(name='prefect')
@pytest.mark.skipif(not prefect_imports(), reason='prefect not installed')
@parametrize_scenarios
async def test_prefect(scenario: Scenario, prefect_test_server: None) -> None:
    async def run(agent: Agent[None, str]) -> str:
        @flow
        async def run_flow() -> str:
            return (await agent.run('go')).output

        return await run_flow()

    await _check(scenario, 'prefect', run, PrefectDurability[None])


@pytest.mark.skipif(not aws_lambda_imports(), reason='aws-durable-execution-sdk-python not installed')
@parametrize_scenarios
def test_aws_lambda(scenario: Scenario) -> None:
    _check_sync(
        scenario,
        'aws_lambda',
        lambda agent: run_durable(lambda: agent.run('go'), context=FakeDurableContext()).output,
        AWSLambdaDurability[None],
    )


@pytest.mark.skipif(not absurd_imports(), reason='absurd-sdk not installed')
@parametrize_scenarios
async def test_absurd(scenario: Scenario, absurd: absurd_sdk.AsyncAbsurd) -> None:
    async def run(agent: Agent[None, str]) -> str:
        async with running_task_context(absurd):
            return (await agent.run('go')).output

    await _check(scenario, 'absurd', run, AbsurdDurability[None])


@pytest.mark.skipif(not pgtask_imports(), reason='pgtask not installed')
@parametrize_scenarios
async def test_pgtask(scenario: Scenario, pgtask_db: PGTaskDatabase) -> None:
    async def run(agent: Agent[None, str]) -> str:
        outputs: list[str] = []

        async def body(attempt: int) -> None:
            outputs.append((await agent.run('go')).output)

        await run_in_task(pgtask_db, body)
        return outputs[0]

    await _check(scenario, 'pgtask', run, PGTaskDurability[None])


@pytest.mark.skipif(not azure_imports(), reason='durabletask not installed')
@parametrize_scenarios
def test_azure_durable_task(scenario: Scenario, durable_worker: InMemoryWorker) -> None:
    def run(agent: Agent[None, str]) -> str:
        with registered_agent(durable_worker, agent):
            return run_agent(durable_worker, agent, 'go', deps=None).output

    _check_sync(scenario, 'azure_durable_task', run, AzureDurableTaskDurability)


@pytest.mark.skipif(not render_imports(), reason='render not installed')
@parametrize_scenarios
async def test_render(scenario: Scenario) -> None:
    runtime = RenderWorkflows[None](Workflows())

    async def run(agent: Agent[None, str]) -> str:
        return await run_agent_in_task(agent, runtime, RecordingTaskContext())

    await _check(scenario, 'render', run, lambda: runtime)
