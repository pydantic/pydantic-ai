from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, StepEvent, StepPersistence


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_step_persistence',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


_store = InMemoryStepStore()
_persistence: StepPersistence[None] = StepPersistence(store=_store, agent_name='durable')
_agent: Agent[None, str] = Agent(
    TestModel(call_tools='all', custom_output_text='done'),
    name='durable_steps',
    deps_type=type(None),
    capabilities=[_persistence, DBOSDurability[None]()],
)


@_agent.tool
def add(ctx: RunContext[None], a: int, b: int) -> int:
    del ctx
    return a + b


@DBOS.workflow(name='durable_steps')
async def _workflow() -> str:
    return (await _agent.run('add 1 and 2')).output


def test_default_id_is_stable() -> None:
    persistence = StepPersistence()
    assert persistence.id == 'step_persistence'
    assert persistence.compaction_transcript_handle() is None


def test_agent_constructs_with_default_id() -> None:
    Agent(
        TestModel(),
        name='default_step_persistence',
        deps_type=type(None),
        capabilities=[StepPersistence[None](), DBOSDurability[None]()],
    )


@pytest.mark.anyio
async def test_derive_run_id_requires_context_run_id() -> None:
    ctx = RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        run_id=None,
    )
    with pytest.raises(RuntimeError, match='agent graph'):
        await StepPersistence[None]().for_run(ctx)

    with pytest.raises(RuntimeError, match='not materialized'):
        await StepPersistence[None]().before_run(ctx)


@pytest.mark.anyio
async def test_store_suppresses_keyed_replays_without_collapsing_snapshot_states() -> None:
    store = InMemoryStepStore()
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('hello')])]
    complete = ContinuableSnapshot(
        run_id='run', step_index=3, messages=messages, state='complete', idempotency_key='3:complete'
    )
    interrupted = ContinuableSnapshot(
        run_id='run', step_index=3, messages=messages, state='interrupted', idempotency_key='3:interrupted'
    )

    await store.save_snapshot(complete)
    await store.save_snapshot(interrupted)
    await store.save_snapshot(complete)
    await store.save_snapshot(interrupted)

    assert await store.list_snapshots(run_id='run', include_interrupted=True) == [complete, interrupted]
    assert await store.latest_snapshot(run_id='run') == complete

    keyed_event = StepEvent(run_id='run', kind='run_started', step_index=0, idempotency_key='event:0')
    await store.append_event(keyed_event)
    await store.append_event(keyed_event)
    await store.append_event(StepEvent(run_id='run', kind='run_started', step_index=0))
    await store.append_event(StepEvent(run_id='run', kind='run_started', step_index=0))
    assert len(await store.list_events(run_id='run')) == 3


@pytest.mark.anyio
async def test_same_step_complete_snapshots_keep_newer_history_and_suppress_replays() -> None:
    store = InMemoryStepStore()
    older_messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('v1')])]
    newer_messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('v1'), UserPromptPart('v2')])]
    older = ContinuableSnapshot(
        run_id='run', step_index=3, messages=older_messages, state='complete', idempotency_key='0:3:complete'
    )
    newer = ContinuableSnapshot(
        run_id='run', step_index=3, messages=newer_messages, state='complete', idempotency_key='1:3:complete'
    )

    await store.save_snapshot(older)
    await store.save_snapshot(newer)
    await store.save_snapshot(older)
    await store.save_snapshot(newer)

    assert await store.list_snapshots(run_id='run') == [older, newer]
    assert await store.latest_snapshot(run_id='run') == newer


@pytest.mark.anyio
async def test_pruned_snapshot_key_remains_suppressed() -> None:
    store = InMemoryStepStore(max_snapshots_per_run=1)
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('hello')])]
    older = ContinuableSnapshot(run_id='run', step_index=1, messages=messages, idempotency_key='0:1:complete')
    newer = ContinuableSnapshot(run_id='run', step_index=2, messages=messages, idempotency_key='1:2:complete')

    await store.save_snapshot(older)
    await store.save_snapshot(newer)
    await store.save_snapshot(older)

    assert await store.latest_snapshot(run_id='run') == newer


@pytest.mark.anyio
async def test_opaque_and_invalid_snapshot_keys_use_retained_record_suppression() -> None:
    store = InMemoryStepStore()
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('hello')])]
    for key in ('opaque', 'not-an-int:complete', '-1:complete'):
        snapshot = ContinuableSnapshot(run_id='run', step_index=1, messages=messages, idempotency_key=key)
        await store.save_snapshot(snapshot)
        await store.save_snapshot(snapshot)

    assert len(await store.list_snapshots(run_id='run')) == 3


@pytest.mark.anyio
async def test_dbos_replay_reuses_run_and_journaled_writes(dbos: DBOS) -> None:
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _workflow() == 'done'

    runs = await _store.list_runs()
    assert len(runs) == 1
    run_id = runs[0].run_id
    run_started_at = runs[0].started_at
    events = await _store.list_events(run_id=run_id)
    timestamps = [event.timestamp for event in events]
    snapshots = await _store.list_snapshots(run_id=run_id, include_interrupted=True)
    effect = await _store.get_tool_effect(run_id=run_id, tool_call_id='pyd_ai_tool_call_id__add')
    assert effect is not None

    with SetWorkflowID(workflow_id):
        assert await _workflow() == 'done'

    assert [run.run_id for run in await _store.list_runs()] == [run_id]
    replayed_run = await _store.get_run(run_id=run_id)
    assert replayed_run is not None
    assert replayed_run.started_at == run_started_at
    replayed_events = await _store.list_events(run_id=run_id)
    assert [event.timestamp for event in replayed_events] == timestamps
    assert await _store.list_snapshots(run_id=run_id, include_interrupted=True) == snapshots
    assert await _store.get_tool_effect(run_id=run_id, tool_call_id=effect.tool_call_id) == effect
    replayed_effect = await _store.get_tool_effect(run_id=run_id, tool_call_id=effect.tool_call_id)
    assert replayed_effect is not None
    assert replayed_effect.started_at == effect.started_at

    steps = await dbos.list_workflow_steps_async(workflow_id)
    operation_names = {step['function_name'] for step in steps}
    assert 'durable_steps__capability__step_persistence.append_event' in operation_names
    assert 'durable_steps__capability__step_persistence.save_snapshot' in operation_names
    assert 'durable_steps__capability__step_persistence.record_tool_effect' in operation_names
    assert 'durable_steps__capability__step_persistence.finish_tool_effect' in operation_names
    assert 'durable_steps__capability__step_persistence.register_run' in operation_names
    assert 'durable_steps__capability__step_persistence.registration_id' in operation_names
