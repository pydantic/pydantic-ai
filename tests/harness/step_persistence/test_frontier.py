"""Earlier checkpoints are opt-in and survive a real process kill without unwind hooks."""

import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    InMemoryStepStore,
    SnapshotSaved,
    SqliteStepStore,
    StepEvent,
    StepPersistence,
    ToolEffectRecord,
)
from pydantic_ai_harness.step_persistence.recovery import inspect_recovery


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('stream', [False, True])
async def test_input_checkpoint_on_first_model_failure(stream: bool) -> None:
    store = InMemoryStepStore()
    notices: list[SnapshotSaved] = []
    hooks = Hooks()

    @hooks.on.event(SnapshotSaved)
    async def saved(ctx: RunContext[None], event: SnapshotSaved) -> None:
        # Receipt is observable only after the store can read the checkpoint.
        assert await store.latest_snapshot(run_id=event.persistence_run_id, include_interrupted=True) is not None
        notices.append(event)

    async def fail_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ValueError('provider failed')
        yield ''  # pragma: no cover

    agent = Agent(
        FunctionModel(stream_function=fail_stream),
        capabilities=[StepPersistence(store=store, capture_frontier=True), hooks],
    )
    with pytest.raises(ValueError, match='provider failed'):
        if stream:
            async with agent.run_stream('preserve this prompt', run_id='first') as result:
                await result.get_output()  # pragma: no cover
        else:
            await agent.run('preserve this prompt', run_id='first')
    snapshot = await store.latest_snapshot(run_id='first')
    assert snapshot is not None
    assert any(
        isinstance(p, UserPromptPart) and p.content == 'preserve this prompt'
        for m in snapshot.messages
        for p in m.parts
    )
    assert notices


async def test_same_length_after_run_rewrite_gets_final_checkpoint(tmp_path: Path) -> None:
    hooks = Hooks()

    @hooks.on.after_run
    async def rewrite(ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
        response = result.all_messages()[-1]
        assert isinstance(response, ModelResponse)
        response.parts = [TextPart('rewritten final response')]
        return result

    memory = SqliteStepStore(database=tmp_path / 'steps.db')
    agent = Agent(
        TestModel(custom_output_text='before'),
        capabilities=[StepPersistence(store=memory, capture_frontier=True), hooks],
    )
    result = await agent.run('hello', run_id='rewrite')
    snapshot = await memory.latest_snapshot(run_id='rewrite')
    assert snapshot is not None
    assert snapshot.messages == result.all_messages()


async def test_hard_kill_leaves_model_frontier_and_unknown_effects(tmp_path: Path) -> None:
    database = tmp_path / 'steps.db'
    script = """
import asyncio, sys
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence import StepPersistence, SqliteStepStore
agent = Agent(TestModel(call_tools=['wait']), capabilities=[StepPersistence(
    store=SqliteStepStore(database=sys.argv[1]), capture_frontier=True)])
@agent.tool_plain
async def wait() -> str:
    print('tool entered', flush=True)
    await asyncio.Event().wait()
    return 'unreachable'
asyncio.run(agent.run('go', run_id='killed', conversation_id='conversation'))
"""
    with anyio.fail_after(20):
        async with await anyio.open_process(
            [sys.executable, '-c', script, str(database)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            assert process.stdout is not None
            output = await process.stdout.receive()
            assert b'tool entered' in output
            process.kill()
            assert await process.wait() != 0
    store = SqliteStepStore(database=database)
    snapshot = await store.latest_snapshot(run_id='killed', include_interrupted=True)
    assert snapshot is not None and snapshot.state == 'interrupted'
    assert any(isinstance(p, ToolCallPart) for m in snapshot.messages for p in m.parts)
    effects = await store.list_unresolved_tool_effects(run_id='killed')
    assert len(effects) == 1 and effects[0].tool_name == 'wait'
    assert not any(e.kind == 'run_failed' for e in await store.list_events(run_id='killed'))


async def test_recovery_inspection_reports_facts_not_replay_permission() -> None:
    store = InMemoryStepStore()
    empty = await inspect_recovery(store=store, run_id='absent')
    assert empty.latest is None and empty.settled is None
    assert empty.unresolved == ()
    await store.save_snapshot(ContinuableSnapshot(run_id='r', step_index=1, messages=[]))
    await store.save_snapshot(ContinuableSnapshot(run_id='r', step_index=2, messages=[], state='interrupted'))
    await store.record_tool_effect(ToolEffectRecord(run_id='r', tool_name='write', tool_call_id='w', status='started'))
    await store.append_event(StepEvent(run_id='r', step_index=2, kind='tool_call_completed', tool_name='read'))
    await store.append_event(StepEvent(run_id='r', step_index=2, kind='tool_call_failed', tool_name='push'))
    await store.append_event(StepEvent(run_id='r', step_index=2, kind='run_failed'))
    inspected = await inspect_recovery(store=store, run_id='r')
    assert inspected.latest is not None and inspected.latest.state == 'interrupted'
    assert inspected.settled is not None and inspected.settled.step_index == 1
    assert inspected.unresolved[0].tool_name == 'write'
    assert inspected.completed_tools == ('read',)
    assert inspected.failed_tools == ('push',)
