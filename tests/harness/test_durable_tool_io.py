"""`DurableEngineSuite` under DBOS, `TestDBOS`: every capability case's calls and every scenario, in a DBOS workflow.

DBOS runs function tools in workflow code, so a capability's tool I/O is only recorded when the
capability routes it through a `durable_operation` or a workspace call. `DBOSEngine` runs each agent in
a workflow, and recovers it by forking the workflow from a step, the way recovering a committed run
re-executes it from after its last recorded step. `pydantic_ai.durable_exec.dbos._conformance` declares
the cells that aren't `supported`. `tests/harness/test_durable_temporal_replay.py` runs the same suite
under Temporal.

Planning's tools also run with an event stream handler, whose steps for the events a tool emits must
replay in their recorded order. Shell's `shell` tool names its job in workflow code, so a recovered
workflow must report the job it started.

Prefect runs function tools in their own durable unit, which needs the toolset's `id`; until a Prefect
suite runs the calls, the construction test below checks every case Temporal runs supplies one.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, AgentStreamEvent, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, ValidatedToolArgs, durable_operation
from pydantic_ai.durable_exec.conformance import EngineRun, Support
from pydantic_ai.messages import CapabilityEvent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.conformance import CALLS as HARNESS_CALLS, HarnessDurableEngineSuite
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanCompletedEvent,
    PlanItem,
    Planning,
    PlanStatusChangedEvent,
)
from pydantic_ai_harness.shell import Shell
from tests._timeouts import READINESS_WAIT_TIMEOUT

from ..append_only import skip_unless_installed
from ..capability_cases import DURABLE_DBOS, DURABLE_TEMPORAL, ID_LESS_TOOLSETS, engine_table
from ._contracts import CASES
from ._durable_cases import BuiltInEngineSuite, build_case, suite_cases

# Module scope registers the DBOS workflows, which DBOS requires before launch.
try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
    from pydantic_ai.durable_exec.prefect import PrefectDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos or prefect not installed', allow_module_level=True)


def _launch(directory: Path, name: str) -> DBOS:
    config: DBOSConfig = {'name': name, 'system_database_url': f'sqlite:///{directory / "dbos.sqlite"}'}
    instance = DBOS(config=config)
    DBOS.launch()
    return instance


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    instance = _launch(tmp_path, 'durable_tool_io')
    try:
        yield instance
    finally:
        DBOS.destroy()


_ENGINE_AGENTS: dict[str, Agent[None, str]] = {}
"""The agents `DBOSEngine` runs, by name, which its one workflow looks up."""


@DBOS.workflow(name='durable_engine_suite_run')
async def _run_agent(agent_name: str, prompt: str) -> str:
    return (await _ENGINE_AGENTS[agent_name].run(prompt)).output


@dataclass
class DBOSEngine:
    """DBOS, launched on SQLite: a run is a workflow, and recovery forks it from a step."""

    name: str = 'dbos'
    agent_names: dict[str, str] = field(default_factory=dict[str, str])
    """The name of the agent each workflow runs, by workflow id."""

    def durability(self) -> DBOSDurability[None]:
        return DBOSDurability[None]()

    async def run(self, agent: Agent[None, str], prompt: str) -> EngineRun:
        assert agent.name is not None
        _ENGINE_AGENTS[agent.name] = agent
        workflow_id = str(uuid.uuid4())
        self.agent_names[workflow_id] = agent.name
        try:
            with SetWorkflowID(workflow_id):
                output: str | BaseException = await _run_agent(agent.name, prompt)
        except Exception as error:
            output = error
        units, failed = await self._units(workflow_id)
        return EngineRun(workflow_id, output, units, failed=failed)

    async def recover(self, run: EngineRun, *, after: int | None = None) -> EngineRun:
        steps = await DBOS.list_workflow_steps_async(run.id)
        if after is not None and after < len(steps):
            start = steps[after]['function_id']
        else:
            start = max((step['function_id'] for step in steps), default=0) + 1
        handle = await DBOS.fork_workflow_async(run.id, start)
        self.agent_names[handle.workflow_id] = self.agent_names[run.id]
        try:
            # The default 1s result poll would add up to a second on top of the internal queue's own poll.
            output: str | BaseException = await handle.get_result(polling_interval_sec=0.01)
        except Exception as error:
            output = error
        units, failed = await self._units(handle.workflow_id)
        return EngineRun(handle.workflow_id, output, units, failed=failed)

    async def _units(self, workflow_id: str) -> tuple[list[str], list[str]]:
        """The steps whose result DBOS recorded, and the ones it recorded an error for."""
        prefix = f'{self.agent_names[workflow_id]}__'
        steps = await DBOS.list_workflow_steps_async(workflow_id)
        units = [step['function_name'].removeprefix(prefix) for step in steps if step['error'] is None]
        failed = [step['function_name'].removeprefix(prefix) for step in steps if step['error'] is not None]
        return units, failed


class TestDBOS(BuiltInEngineSuite):
    support = engine_table('dbos').support
    recovery = engine_table('dbos').recovery
    cases = suite_cases(DURABLE_DBOS)
    durable_steps = 'dbos'
    model_request_units = ('model.request', 'model.request_stream')

    @pytest.fixture(scope='class')
    @classmethod
    def engine(cls, tmp_path_factory: pytest.TempPathFactory) -> Generator[DBOSEngine, None, None]:
        _launch(tmp_path_factory.mktemp('dbos'), 'durable_engine_suite')
        try:
            yield DBOSEngine()
        finally:
            DBOS.destroy()
            _ENGINE_AGENTS.clear()


@dataclass
class _RestartingDBOSEngine(DBOSEngine):
    """DBOS, with a recovery that runs the workflow again from its start, repeating every completed step."""

    async def recover(self, run: EngineRun, *, after: int | None = None) -> EngineRun:
        return await super().recover(run, after=0)


class TestTheSuiteCountsTheHarnessStores(HarnessDurableEngineSuite):
    """A recovery that repeats a public case's store writes fails its cell, with no `count_requests` of the engine's.

    So the check a third-party engine relies on, that a recovery repeats no request, isn't vacuous for the Harness
    cases whose tools write to a store.
    """

    cases = {name: HARNESS_CALLS[name] for name in ('Memory', 'Planning')}
    support = dict.fromkeys(
        cases,
        Support('broken', 'its recovery runs the case again', match='recovery made a request again', issue=14514),
    )
    test_scenario = None  # pyright: ignore[reportAssignmentType]

    @pytest.fixture(scope='class')
    @classmethod
    def engine(cls, tmp_path_factory: pytest.TempPathFactory) -> Generator[DBOSEngine, None, None]:
        _launch(tmp_path_factory.mktemp('dbos'), 'durable_engine_suite_counts')
        try:
            yield _RestartingDBOSEngine()
        finally:
            DBOS.destroy()
            _ENGINE_AGENTS.clear()

    @pytest.fixture
    def can_fork_mid_run(self) -> bool:
        return False


def _call_tools(*calls: tuple[str, dict[str, Any]]) -> FunctionModel:
    """A model that calls every tool in `calls` in one response, then answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if len([message for message in messages if isinstance(message, ModelRequest)]) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=name) for name, args in calls])

    return FunctionModel(respond)


def _workflow(agent: Agent[None, str]) -> Callable[[], Awaitable[str]]:
    @DBOS.workflow(name=f'{agent.name}_workflow')
    async def run() -> str:
        return (await agent.run('go')).output

    return run


_SHELL_DIRECTORY = Path(tempfile.mkdtemp(prefix='harness_durable_tool_io_shell_'))
_SHELL_AGENT = Agent(
    _call_tools(('shell', {'command': 'echo one >> side-effects.txt', 'mode': 'background'})),
    name='shell_agent',
    deps_type=type(None),
    capabilities=[LocalWorkspace[None](str(_SHELL_DIRECTORY)), Shell[None](tools=['shell']), DBOSDurability[None]()],
)
_shell_returns: list[str] = []


class _ProcessCrash(BaseException):
    """Stands in for the process dying: DBOS leaves the workflow `PENDING` for recovery."""


@DBOS.workflow(name='shell_agent_workflow')
async def _shell_workflow() -> str:
    result = await _SHELL_AGENT.run('go')
    _shell_returns.extend(
        str(part.content)
        for message in result.new_messages()
        for part in message.parts
        if part.part_kind == 'tool-return'
    )
    if len(_shell_returns) == 1:
        raise _ProcessCrash
    return result.output


async def test_dbos_recovery_reports_the_same_shell_job(dbos: DBOS) -> None:
    """The `shell` tool runs in workflow code, so a recovered workflow must name the job its replayed launch started."""
    _shell_returns.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_ProcessCrash):
        await _shell_workflow()

    # Resuming the pending workflow re-executes it under the same ID, the way recovery does.
    handle = await DBOS.resume_workflow_async(workflow_id)
    assert await handle.get_result(polling_interval_sec=0.01) == 'done'
    first, recovered = _shell_returns
    assert recovered == first
    status = next(line for line in first.splitlines() if line.startswith('Status: '))
    assert Path(status.removeprefix('Status: ')).exists()
    assert (_SHELL_DIRECTORY / 'side-effects.txt').read_text().splitlines() == ['one']


class _SlowReadPlanStore(InMemoryPlanStore):
    """A plan store whose `get_item` waits, once a plan tool has emitted `last_event`, until it's handled.

    It stands in for a read slow enough that the event stream handler's steps for the tool's events
    all start while it is in flight. A replay returns the recorded read without calling the store.
    The status tools read before emitting, so only a regression makes the wait.
    """

    def __init__(self, last_event: tuple[type[PlanStatusChangedEvent | PlanCompletedEvent], str]) -> None:
        super().__init__()
        self.last_event = last_event
        self.released = asyncio.Event()
        """Cleared when the tool emits `last_event` and set again once the handler has handled it."""
        self.released.set()

    def is_last_event(self, event: object) -> bool:
        event_type, task_id = self.last_event
        return isinstance(event, event_type) and event.item.id == task_id

    async def get_item(self, item_id: str) -> PlanItem | None:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await self.released.wait()
        return await super().get_item(item_id)


class _AuditToolCalls(AbstractCapability[None]):
    """Makes one durable call after each tool call, as an auditing capability would."""

    id = 'audit'

    @durable_operation('record')
    async def record(self, tool_name: str) -> None:
        pass

    async def after_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        await self.record(call.tool_name)
        return result


def _plan_event_agent(
    name: str,
    update: tuple[str, dict[str, Any]],
    last_event: tuple[type[PlanStatusChangedEvent | PlanCompletedEvent], str],
    *extra: AbstractCapability[None],
) -> tuple[Agent[None, str], _SlowReadPlanStore]:
    """An agent that writes a two-step plan, makes the `update` plan tool call, then answers."""
    calls = [('write_plan', {'items': [{'id': 'a', 'content': 'Survey'}, {'id': 'b', 'content': 'Draft'}]}), update]

    # An `event_stream_handler` makes the run stream its model responses.
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        del info
        turn = len([message for message in messages if isinstance(message, ModelRequest)]) - 1
        if turn < len(calls):
            tool_name, args = calls[turn]
            yield {0: DeltaToolCall(name=tool_name, json_args=json.dumps(args), tool_call_id=tool_name)}
        else:
            yield 'done'

    store = _SlowReadPlanStore(last_event)

    async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            if store.is_last_event(event):
                store.released.set()

    agent = Agent(
        FunctionModel(stream_function=stream),
        name=name,
        deps_type=type(None),
        capabilities=[Planning[None](store=store), *extra, DBOSDurability[None](event_stream_handler=handle)],
    )
    return agent, store


_PLAN_EVENT_AGENTS: dict[str, tuple[Agent[None, str], _SlowReadPlanStore]] = {
    'update_task_statuses': _plan_event_agent(
        'plan_statuses_agent',
        (
            'update_task_statuses',
            {'updates': [{'task_id': 'a', 'status': 'completed'}, {'task_id': 'b', 'status': 'in_progress'}]},
        ),
        (PlanStatusChangedEvent, 'b'),
    ),
    # A single update reads one step back, so another capability's hook supplies the second durable
    # call that could follow the tool's events.
    'update_task_status': _plan_event_agent(
        'plan_status_agent',
        ('update_task_status', {'task_id': 'a', 'status': 'completed'}),
        (PlanCompletedEvent, 'a'),
        _AuditToolCalls(),
    ),
}
_PLAN_EVENT_WORKFLOWS = {key: _workflow(agent) for key, (agent, _) in _PLAN_EVENT_AGENTS.items()}


@pytest.mark.parametrize('scenario', sorted(_PLAN_EVENT_AGENTS))
async def test_dbos_recovery_replays_plan_tool_events_in_recorded_order(
    scenario: str, dbos: DBOS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan tool emits its events after its last store call, so their handler steps keep their order.

    DBOS runs a function tool in workflow code and starts a handler step for each event the tool
    emits while the tool is still running. Both draw step numbers from one counter. The first
    durable call the tool call makes after emitting is numbered before any handler step, but a
    second one is numbered after however many handler steps fit into the first one's I/O: a store
    round trip live, a journal lookup on replay.
    """
    _, store = _PLAN_EVENT_AGENTS[scenario]
    # A replay re-emits `last_event` without rerunning the handler, so a previous run can leave it closed.
    store.released.set()
    emit = RunContext.emit
    last_event_emits = 0

    async def emit_marking_last_event(self: RunContext[None], event: CapabilityEvent, /) -> CapabilityEvent:
        nonlocal last_event_emits
        if store.is_last_event(event):
            last_event_emits += 1
            store.released.clear()
        return await emit(self, event)

    monkeypatch.setattr(RunContext, 'emit', emit_marking_last_event)
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await _PLAN_EVENT_WORKFLOWS[scenario]() == 'done'
    # The status tools read before emitting, so the wait never engages: pin both sides of the
    # scaffolding before the fork's replay re-emits `last_event` without rerunning the handler.
    assert last_event_emits == 1, 'the tool should emit `last_event` once through the patched `emit`'
    assert store.released.is_set(), 'the handler never handled `last_event`'

    steps = await dbos.list_workflow_steps_async(workflow_id)
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result(polling_interval_sec=0.01) == 'done'


@pytest.mark.parametrize(
    'name',
    [
        name
        for name, case in sorted(CASES.items())
        if DURABLE_TEMPORAL.applies(name, case) and name not in ID_LESS_TOOLSETS
    ],
)
def test_prefect_wraps_every_toolset_temporal_does(name: str, tmp_path: Path) -> None:
    """Prefect, like Temporal, runs each function tool in its own unit, named after the toolset's `id`, so it refuses
    the same id-less toolsets."""
    skip_unless_installed(name, CASES[name].requires)
    Agent(
        _call_tools(),
        name=f'wrapped_{uuid.uuid4().hex}',
        deps_type=type(None),
        capabilities=[*build_case(name, tmp_path), PrefectDurability[None]()],
    )
