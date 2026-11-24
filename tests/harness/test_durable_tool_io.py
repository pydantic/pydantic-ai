"""Every capability case's calls, run in a DBOS workflow: `DURABLE_DBOS`, half of D1.

DBOS runs function tools in workflow code, so a capability's tool I/O is only recorded when the
capability routes it through a `durable_operation` or a workspace call. Each case in
`tests/capability_cases.py` and `tests/harness/capability_cases.py` with `calls` makes them in a DBOS
workflow, one per response, then re-executes the workflow from after its last recorded step, the way
recovering a committed run does. No request may be made again, and the steps the run recorded must be
the ones the case's `durable_steps.dbos` lists. `tests/harness/test_durable_temporal_replay.py` runs
the same calls under Temporal.

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
from collections import Counter
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Generator
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, AgentStreamEvent, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, ValidatedToolArgs, durable_operation
from pydantic_ai.messages import CapabilityEvent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.capability_creation import CapabilityStore
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.memory import InMemoryMemoryStore
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanCompletedEvent,
    PlanItem,
    Planning,
    PlanStatusChangedEvent,
    SqlitePlanStore,
)
from pydantic_ai_harness.shell import Shell
from tests._timeouts import READINESS_WAIT_TIMEOUT

from ..append_only import skip_unless_installed
from ..capability_cases import DURABLE_DBOS, DURABLE_TEMPORAL
from ._counting_clients import AWS_CLI_COUNT_VARIABLE, REQUESTS
from ._durable_cases import (
    CASES,
    assert_holds,
    build_case,
    case_params,
    installed,
    journal_breaks,
    one_call_per_response,
)

# Module scope builds the agents and registers their DBOS workflows, which DBOS requires before launch.
try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
    from pydantic_ai.durable_exec.prefect import PrefectDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos or prefect not installed', allow_module_level=True)


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_tool_io',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


def _call_tools(*calls: tuple[str, dict[str, Any]]) -> FunctionModel:
    """A model that calls every tool in `calls` in one response, then answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if len([message for message in messages if isinstance(message, ModelRequest)]) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=name) for name, args in calls])

    return FunctionModel(respond)


def _case_agent(name: str) -> Agent[None, str]:
    calls = CASES[name].calls
    assert calls is not None
    directory = Path(tempfile.mkdtemp(prefix=f'harness_durable_tool_io_{name}_'))
    return Agent(
        one_call_per_response(calls.made),
        name=f'{name}_agent',
        deps_type=type(None),
        capabilities=[*build_case(name, directory), DBOSDurability[None]()],
    )


_AGENTS: dict[str, Agent[None, str]] = {
    name: _case_agent(name) for name, case in CASES.items() if DURABLE_DBOS.applies(name, case) and installed(name)
}
"""The agent that makes each case's calls, for every case `DURABLE_DBOS` applies to whose dependency is installed."""


def _workflow(agent: Agent[None, str]) -> Callable[[], Awaitable[str]]:
    @DBOS.workflow(name=f'{agent.name}_workflow')
    async def run() -> str:
        return (await agent.run('go')).output

    return run


_WORKFLOWS = {key: _workflow(agent) for key, agent in _AGENTS.items()}


@pytest.fixture
def count_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[], Counter[str]]:
    """Count every request the cases' stand-ins, the `aws` stand-in, and the stores make."""
    REQUESTS.clear()
    aws_count = tmp_path / 'aws_cli_count'
    monkeypatch.setenv(AWS_CLI_COUNT_VARIABLE, str(aws_count))

    for method in ('write', 'list_all', 'disable'):
        original = getattr(CapabilityStore, method)

        def counted(self: CapabilityStore, *args: Any, _original: Any = original, _method: str = method) -> Any:
            REQUESTS[f'store.{_method}'] += 1
            return _original(self, *args)

        monkeypatch.setattr(CapabilityStore, method, counted)

    # A memory write first looks up its idempotency key; the write is what recovery must not repeat.
    original_memory_write = InMemoryMemoryStore.write

    async def counted_memory_write(self: InMemoryMemoryStore, *args: Any, **kwargs: Any) -> Any:
        REQUESTS['memory.write'] += 1
        return await original_memory_write(self, *args, **kwargs)

    monkeypatch.setattr(InMemoryMemoryStore, 'write', counted_memory_write)

    # The plan tools read the store many times; a write is what recovery must not repeat.
    for method in ('set_items', 'add_item'):
        original = getattr(SqlitePlanStore, method)

        async def counted_plan(
            self: SqlitePlanStore, *args: Any, _original: Any = original, _method: str = method
        ) -> Any:
            REQUESTS[f'plan.{_method}'] += 1
            return await _original(self, *args)

        monkeypatch.setattr(SqlitePlanStore, method, counted_plan)

    original_list_runs = SnapshotHistorySource.list_runs

    async def counted_list_runs(self: SnapshotHistorySource) -> Any:
        REQUESTS['history.list_runs'] += 1
        return await original_list_runs(self)

    monkeypatch.setattr(SnapshotHistorySource, 'list_runs', counted_list_runs)

    def snapshot() -> Counter[str]:
        counts = Counter(REQUESTS)
        if aws_count.exists():
            counts['aws_cli'] = len(aws_count.read_text().splitlines())
        return counts

    return snapshot


@pytest.mark.parametrize('name', case_params(DURABLE_DBOS))
async def test_dbos_recovery_does_not_repeat_tool_requests(
    name: str, dbos: DBOS, count_requests: Callable[[], Counter[str]]
) -> None:
    skip_unless_installed(name, CASES[name].requires)
    calls = CASES[name].calls
    assert calls is not None
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await _WORKFLOWS[name]() == 'done'
    first_run = count_requests()
    assert set(first_run.values()) <= {1}, f'a request was made more than once: {first_run}'

    steps = await dbos.list_workflow_steps_async(workflow_id)
    # Re-execute the workflow function from after its last step, the way recovery does. Calling it
    # again under the same ID is no replay: DBOS returns the stored result without running it.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    # The default 1s result poll would add up to a second on top of the internal queue's own poll.
    assert await handle.get_result(polling_interval_sec=0.01) == 'done'
    recovered = count_requests()
    breaks = [] if recovered == first_run else [f'recovery made a request again: {first_run} became {recovered}']
    breaks += journal_breaks(
        [step['function_name'] for step in steps],
        prefix=f'{_AGENTS[name].name}__',
        model_request='model.request',
        calls=len(calls.made),
        expected=calls.durable_steps.dbos,
        engine='dbos',
    )
    assert_holds(DURABLE_DBOS, name, breaks)


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
        if DURABLE_TEMPORAL.applies(name, case) and name not in DURABLE_TEMPORAL.exempt
    ],
)
def test_prefect_wraps_every_toolset_temporal_does(name: str, tmp_path: Path) -> None:
    """Prefect, like Temporal, runs each function tool in its own unit, named after the toolset's `id`."""
    skip_unless_installed(name, CASES[name].requires)
    Agent(
        _call_tools(),
        name=f'wrapped_{uuid.uuid4().hex}',
        deps_type=type(None),
        capabilities=[*build_case(name, tmp_path), PrefectDurability[None]()],
    )
