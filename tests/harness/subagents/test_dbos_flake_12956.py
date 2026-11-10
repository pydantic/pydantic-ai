"""Temporary diagnostic for #12956. Never merge.

Repeats `test_background_task_survives_recovery` with tracing at the three places the
`sub_agents.task_backgrounded` event can drop out, and prints the trace only when an iteration fails:

1. announce: `_delegate_controlled` announces only if the task still runs once `owner.delegate` returns.
2. `_emits_events`: `_announce` emits only when the tool belongs to a capability.
3. stream dispatch: `dispatch_event_stream` skips the listener when it finds an immediate-dispatch marker.

All tracing is test-local monkeypatching; product code is unchanged.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import AgentStreamEvent, CapabilityEvent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.subagents import SubAgent, SubAgents, TaskControl
from tests.conftest import detach_dbos_logging, try_import, without_admin_server
from tests.harness.subagents._background import (
    WORKER_CALLS,
    EventKinds,
    background_parent,
    background_worker,
    reset_worker,
)

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]

# Some repeats stay in the `harness-dbos` group, after the suite's other DBOS tests on the same worker, as in the
# failing jobs; the rest run in groups of their own so other workers repeat them in parallel.
TRACED_CASES = [
    pytest.param(i, marks=pytest.mark.xdist_group(name='harness-dbos' if i < 60 else f'diag-12956-{i % 3}'))
    for i in range(180)
]
UNTRACED_CASES = [
    pytest.param(i, marks=pytest.mark.xdist_group(name='harness-dbos' if i < 20 else 'diag-12956-untraced'))
    for i in range(60)
]

TRACE: list[str] = []
_T0 = [0]


def note(entry: str) -> None:
    TRACE.append(f'{(time.perf_counter_ns() - _T0[0]) // 1000:>8}us [{threading.current_thread().name}] {entry}')


def _is_task_event(event: object) -> bool:
    return isinstance(event, CapabilityEvent) and event.kind.startswith('sub_agents.')


class TracedKinds(list[str]):
    def append(self, kind: str) -> None:
        note(f'LISTENER got {kind}')
        super().append(kind)


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'subagents_dynamic',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=without_admin_server(config))
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


PENDING: list[dict[int, Any]] = []
BUFFERS: list[list[Any]] = []
KIND_BY_ID: dict[int, str] = {}


def _traced_dispatch(original: Callable[..., AsyncIterator[AgentStreamEvent]]) -> Callable[..., Any]:
    async def dispatch(
        ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
    ) -> AsyncIterator[AgentStreamEvent]:
        pending = ctx._pending_immediate_dispatches  # pyright: ignore[reportPrivateUsage]

        async def observed() -> AsyncIterator[AgentStreamEvent]:
            try:
                async for event in stream:
                    if _is_task_event(event):
                        assert isinstance(event, CapabilityEvent)
                        root = ctx.root_capability
                        note(
                            f'STREAM {event.kind} id={id(event):x} marker={id(event) in pending} '
                            f'listens={root is not None and root.listens_to(event)}'
                        )
                    yield event
            finally:
                aclose = getattr(stream, 'aclose', None)
                if aclose is not None:
                    await aclose()

        inner = original(ctx, observed())
        try:
            async for event in inner:
                yield event
        finally:
            await inner.aclose()  # pyright: ignore[reportAttributeAccessIssue]

    return dispatch


@pytest.fixture
def traced(monkeypatch: pytest.MonkeyPatch) -> None:
    import pydantic_ai._agent_graph as agent_graph_module
    import pydantic_ai._run_context as run_context_module
    import pydantic_ai.agent as agent_module
    import pydantic_ai.result as result_module
    import pydantic_ai_harness.subagents._toolset as toolset_module
    from pydantic_ai_harness.subagents._tasks import DelegationTasks

    toolset_cls = toolset_module.SubAgentToolset

    original_emits = toolset_module._emits_events  # pyright: ignore[reportPrivateUsage]

    def emits_events(ctx: RunContext[Any]) -> bool:
        result = original_emits(ctx)
        tool_name = ctx.tool_name
        tool_def = ctx.tools.get(tool_name) if tool_name is not None else None
        note(
            f'EMITS_EVENTS tool={tool_name} has_def={tool_def is not None} '
            f'cap={getattr(tool_def, "capability_id", None)} -> {result}'
        )
        return result

    monkeypatch.setattr(toolset_module, '_emits_events', emits_events)

    original_announce = toolset_cls._announce  # pyright: ignore[reportPrivateUsage]

    async def announce(self: Any, ctx: Any, owner: Any, record: Any, event: Any) -> Any:
        note(f'ANNOUNCE {event.kind} record_status={getattr(record, "status", None)}')
        return await original_announce(self, ctx, owner, record, event)

    monkeypatch.setattr(toolset_cls, '_announce', announce)

    original_recorded = toolset_cls._delegate_recorded  # pyright: ignore[reportPrivateUsage]

    async def delegate_recorded(self: Any, *args: Any, **kwargs: Any) -> Any:
        note('WORKER recorded-step start')
        try:
            return await original_recorded(self, *args, **kwargs)
        finally:
            note('WORKER recorded-step end')

    monkeypatch.setattr(toolset_cls, '_delegate_recorded', delegate_recorded)

    original_task_group = toolset_module.task_group

    @asynccontextmanager
    async def task_group() -> Any:
        async with original_task_group() as group:
            yield group
        owner = DelegationTasks.current()
        statuses = None if owner is None else [record.status for record in owner.records.values()]
        note(f'TASK_GROUP exit (status check follows) statuses={statuses}')

    monkeypatch.setattr(toolset_module, 'task_group', task_group)

    original_delegate = DelegationTasks.delegate

    async def delegate(self: DelegationTasks, **kwargs: Any) -> str:
        out = await original_delegate(self, **kwargs)
        record = self.records.get(kwargs.get('task_id') or '')
        worker = self._workers.get(kwargs.get('task_id') or '')  # pyright: ignore[reportPrivateUsage]
        note(
            f'DELEGATE returned status={getattr(record, "status", None)} '
            f'worker_done={None if worker is None else worker.done()}'
        )
        return out

    monkeypatch.setattr(DelegationTasks, 'delegate', delegate)

    original_run = DelegationTasks._run  # pyright: ignore[reportPrivateUsage]

    async def run(self: DelegationTasks, record: Any, runner: Any) -> str:
        note('WORKER start')
        try:
            return await original_run(self, record, runner)
        finally:
            note(f'WORKER finished status={record.status} outcome={record.outcome}')

    monkeypatch.setattr(DelegationTasks, '_run', run)

    original_emit = RunContext.emit

    async def emit(self: RunContext[Any], event: Any) -> Any:
        if _is_task_event(event):
            pending = self._pending_immediate_dispatches  # pyright: ignore[reportPrivateUsage]
            buffer = self._event_stream_buffer  # pyright: ignore[reportPrivateUsage]
            if not any(p is pending for p in PENDING):
                PENDING.append(pending)
            if buffer is not None and not any(b is buffer for b in BUFFERS):
                BUFFERS.append(buffer)
            KIND_BY_ID[id(event)] = event.kind
            note(
                f'EMIT {event.kind} id={id(event):x} dispatch={event.event_dispatch} '
                f'marker_before={id(event) in pending} pending={[f"{k:x}" for k in pending]} '
                f'buffer={type(buffer).__name__}/{len(buffer) if buffer is not None else None}'
            )
        return await original_emit(self, event)

    monkeypatch.setattr(RunContext, 'emit', emit)

    original_dispatch = run_context_module.dispatch_event_stream
    for module in (agent_graph_module, result_module, agent_module):
        monkeypatch.setattr(module, 'dispatch_event_stream', _traced_dispatch(original_dispatch))


def _leftovers() -> str:
    markers = [f'{KIND_BY_ID.get(key, "?")}@{key:x}' for pending in PENDING for key, value in pending.items() if value]
    buffered = [getattr(event, 'kind', type(event).__name__) for buffer in BUFFERS for event in buffer]
    PENDING.clear()
    BUFFERS.clear()
    return f'stale_markers={markers} unconsumed_buffer={buffered}'


def _order_ok(order: list[str]) -> bool:
    return [kind for kind in order if kind != 'sub_agents.task_backgrounded'] == [
        'sub_agents.task_assigned',
        'sub_agents.task_started',
        'sub_agents.task_completed',
    ] and order.count('sub_agents.task_backgrounded') == 1


async def _scenario(dbos: DBOS, name: str, events: EventKinds) -> dict[str, Any]:
    """`test_background_task_survives_recovery` under a workflow and agent name no other module registers.

    The workflow returns a per-iteration token, so a recovered run that resolved another function by name shows up.
    """
    token = uuid.uuid4().hex
    worker = Agent(FunctionModel(background_worker), name='worker')
    agent = Agent(
        background_parent(),
        name=name,
        deps_type=type(None),
        capabilities=[
            SubAgents[None](agents=[SubAgent(worker)], task_control=TaskControl()),
            events,
            DBOSDurability[None](),
        ],
    )

    @DBOS.workflow(name=name)
    async def workflow() -> list[str]:
        events.kinds.clear()
        return [token, (await agent.run('Delegate it')).output, *events.kinds]

    reset_worker()
    workflow_id = str(uuid.uuid4())
    note('=== FIRST RUN ===')
    with SetWorkflowID(workflow_id):
        first_token, output, *kinds = await workflow()
    note(f'=== FIRST RUN done kinds={kinds} {_leftovers()}')
    steps = await dbos.list_workflow_steps_async(workflow_id)
    note('=== RECOVERY ===')
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    recovered_token, recovered, *recovered_kinds = await handle.get_result(polling_interval_sec=0.01)
    note(f'=== RECOVERY done kinds={recovered_kinds} {_leftovers()}')
    recovered_steps = await dbos.list_workflow_steps_async(handle.get_workflow_id())

    failures = [label for label, order in (('first', kinds), ('recovered', recovered_kinds)) if not _order_ok(order)]
    if WORKER_CALLS != ['worker']:
        failures.append(f'worker_calls={WORKER_CALLS}')
    if 'from the worker' not in output or 'from the worker' not in recovered:
        failures.append(f'outputs={output!r},{recovered!r}')
    identity = 'ok' if first_token == token and recovered_token == token else 'MISMATCH'
    return {
        'failures': failures,
        'identity': identity,
        'kinds': kinds,
        'recovered_kinds': recovered_kinds,
        'steps': [(step['function_id'], step['function_name']) for step in steps],
        'recovered_steps': [(step['function_id'], step['function_name']) for step in recovered_steps],
    }


def _report(i: int, variant: str, result: dict[str, Any]) -> None:
    if result['failures'] or result['identity'] != 'ok' or os.environ.get('DIAG_12956_PRINT'):
        report = '\n'.join(
            [
                f'DIAG-12956 {variant} iteration={i} failing={result["failures"]} identity={result["identity"]}',
                f'first kinds={result["kinds"]}',
                f'recovered kinds={result["recovered_kinds"]}',
                f'first steps={result["steps"]}',
                f'forked steps={result["recovered_steps"]}',
                *TRACE,
            ]
        )
        print(report)
        if result['failures'] or result['identity'] != 'ok':
            pytest.fail(report, pytrace=False)


@pytest.mark.parametrize('i', TRACED_CASES)
async def test_traced_background_recovery(dbos: DBOS, traced: None, i: int) -> None:
    TRACE.clear()
    KIND_BY_ID.clear()
    PENDING.clear()
    BUFFERS.clear()
    _T0[0] = time.perf_counter_ns()
    result = await _scenario(dbos, 'diag_12956_traced', EventKinds(kinds=TracedKinds()))
    _report(i, 'traced', result)


@pytest.mark.parametrize('i', UNTRACED_CASES)
async def test_untraced_background_recovery(dbos: DBOS, i: int) -> None:
    TRACE.clear()
    result = await _scenario(dbos, 'diag_12956_untraced', EventKinds())
    _report(i, 'untraced', result)
