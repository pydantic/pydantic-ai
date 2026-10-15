"""Managed child lifetime exercised through real agent runs and native queued messages."""

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator
from decimal import Decimal
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    INTERRUPTED_TOOL_RETURN_CONTENT,
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence
from pydantic_ai_harness.subagents import (
    DelegationEndEvent,
    DelegationReports,
    DelegationStartEvent,
    DelegationTask,
    DelegationTaskEvent,
    DelegationTasks,
    SubAgent,
    SubAgents,
)

WAIT = 10


def parent_model(*, background: bool = False, resume: str | None = None) -> FunctionModel:
    step = 0

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        nonlocal step
        step += 1
        if step == 1:
            yield {
                0: DeltaToolCall(
                    name='delegate_task',
                    json_args=json.dumps(
                        {
                            'agent_name': 'worker',
                            'task': 'inspect',
                            'background': background,
                            'resume': resume,
                        }
                    ),
                    tool_call_id='delegate',
                )
            }
        else:
            yield 'parent finished'

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        async for item in stream(messages, info):
            if isinstance(item, str):
                return ModelResponse(parts=[TextPart(item)])
            call = item[0]
            return ModelResponse(parts=[ToolCallPart(call.name or '', call.json_args or '{}', tool_call_id='delegate')])
        raise AssertionError('Model produced no response')  # pragma: no cover

    return FunctionModel(function=respond, stream_function=stream)


async def test_foreground_history_and_resume(tmp_path: Path) -> None:
    owner = DelegationTasks(directory=tmp_path)
    child = Agent(TestModel(custom_output_text='child result'), deps_type=object, name='worker')
    with anyio.fail_after(WAIT):
        async with owner.opened():
            with owner.bind():
                agent = Agent(parent_model(), capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)])
                await agent.run('go', conversation_id='parent')
                (record,) = owner.records.values()
                first_history = list(record.messages)
                assert record.delivered and record.outcome == 'ok'
                assert record.messages and record.id != 'parent'
                resumed = Agent(
                    parent_model(resume=record.id),
                    capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)],
                )
                await resumed.run('continue', conversation_id='parent')
                assert len(owner.records) == 1
                assert record.generation == 2
                assert record.messages[: len(first_history)] == first_history
    async with DelegationTasks(directory=tmp_path).opened() as restored:
        saved = restored.records[record.id]
        assert saved.delivered
        assert saved.messages == record.messages


async def test_owned_child_streams_to_the_event_stream_handler_and_the_observer() -> None:
    handled: list[AgentStreamEvent] = []
    observed: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            handled.append(event)

    async def observe(update: DelegationTaskEvent) -> None:
        if update.event is not None:
            observed.append(update.event)

    owner = DelegationTasks(observer=observe)
    child = Agent(TestModel(custom_output_text='child result'), deps_type=object, name='worker')
    with anyio.fail_after(WAIT):
        async with owner.opened():
            with owner.bind():
                agent: Agent[object, str] = Agent(
                    parent_model(),
                    capabilities=[
                        SubAgents(agents=[SubAgent(child)], agent_folders=None, event_stream_handler=handler)
                    ],
                )
                await agent.run('go', conversation_id='parent')
    assert handled
    assert handled == [event for event in observed if not isinstance(event, (DelegationStartEvent, DelegationEndEvent))]


async def test_child_stream_events_carry_the_child_model_and_window() -> None:
    updates: list[DelegationTaskEvent] = []

    async def observe(update: DelegationTaskEvent) -> None:
        updates.append(update)

    owner = DelegationTasks(observer=observe)
    child_model = TestModel(custom_output_text='child result', model_name='child', profile={'context_window': 4096})
    child = Agent(child_model, deps_type=object, name='worker')
    with anyio.fail_after(WAIT):
        async with owner.opened():
            with owner.bind():
                agent: Agent[object, str] = Agent(
                    parent_model(), capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)]
                )
                await agent.run('go', conversation_id='parent')
    lifecycle = [
        u for u in updates if u.event is None or isinstance(u.event, (DelegationStartEvent, DelegationEndEvent))
    ]
    streamed = [u for u in updates if u not in lifecycle]
    assert streamed and {(u.model_name, u.context_window) for u in streamed} == {('child', 4096)}
    assert lifecycle and {(u.model_name, u.context_window) for u in lifecycle} == {(None, None)}


@pytest.mark.parametrize(('background', 'resume'), [(True, None), (False, 'earlier')])
async def test_background_and_resume_need_an_owner(background: bool, resume: str | None) -> None:
    child = Agent(TestModel(custom_output_text='child result'), deps_type=object, name='worker')
    agent = Agent(
        parent_model(background=background, resume=resume),
        capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)],
    )
    result = await agent.run('go')
    retries = [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, RetryPromptPart)
    ]
    assert retries == ['Background execution and resume require an open `DelegationTasks` owner']


async def test_background_receipt_then_automated_report(tmp_path: Path) -> None:
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def child_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        started.set()
        await release.wait()
        yield 'child evidence'

    child = Agent(FunctionModel(stream_function=child_stream), deps_type=object, name='worker')

    async def observe(update: DelegationTaskEvent) -> None:
        if update.task.status == 'finished':
            finished.set()

    owner = DelegationTasks(directory=tmp_path, observer=observe)
    with anyio.fail_after(WAIT):
        async with owner.opened():
            with owner.bind():
                parent = Agent(
                    parent_model(background=True),
                    capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)],
                )
                result = await parent.run('go', conversation_id='parent')
                await started.wait()
                (record,) = owner.records.values()
                assert record.status == 'running'
                assert 'child evidence' not in str(result.all_messages())
                assert 'not a result' in str(result.all_messages())
                release.set()
                await finished.wait()
            with owner.bind():
                followup = Agent(
                    TestModel(custom_output_text='reviewed'),
                    capabilities=[DelegationReports(owner, conversation_id='parent')],
                )
                result = await followup.run('next turn', conversation_id='parent')
            reports = [
                part
                for message in result.all_messages()
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, SystemPromptPart) and 'Automated subagent' in part.content
            ]
            assert len(reports) == 1
            assert 'untrusted task data' in reports[0].content
            assert 'child evidence' in reports[0].content
            assert record.delivered
    async with DelegationTasks(directory=tmp_path).opened() as restored:
        assert restored.records[record.id].delivered
        assert not restored.reports(conversation_id='parent')


async def test_promotion_and_targeted_cancellation(tmp_path: Path) -> None:
    first_started, second_started = asyncio.Event(), asyncio.Event()
    release = asyncio.Event()
    owner = DelegationTasks(directory=tmp_path)

    async def first(record: DelegationTask) -> str:
        first_started.set()
        await release.wait()
        return 'first'  # pragma: lax no cover

    async def second(record: DelegationTask) -> str:
        second_started.set()
        await release.wait()
        return 'second'  # pragma: lax no cover

    with anyio.fail_after(WAIT):
        async with owner.opened():
            foreground = asyncio.create_task(
                owner.delegate(
                    agent_name='worker',
                    prompt='one',
                    conversation_id='parent',
                    model=None,
                    background=False,
                    resume=None,
                    run=first,
                )
            )
            await first_started.wait()
            (first_record,) = owner.records.values()
            owner.background(first_record.id)
            assert 'not a result' in await foreground
            await owner.delegate(
                agent_name='worker',
                prompt='two',
                conversation_id='parent',
                model=None,
                background=True,
                resume=None,
                run=second,
            )
            await second_started.wait()
            await owner.cancel(first_record.id)
            release.set()
    records = list(owner.records.values())
    assert records[0].outcome == 'cancelled'
    assert records[0].user_stopped
    # Shutdown cancels any still-running second task; it was not marked as user-stopped.
    assert not records[1].user_stopped
    async with DelegationTasks(directory=tmp_path).opened() as restored:
        assert restored.records[first_record.id].user_stopped
        await restored.allow_resume(first_record.id)
        assert not restored.records[first_record.id].user_stopped


async def test_nested_background_joins_and_routes_to_direct_parent() -> None:
    leaf_started, leaf_release = asyncio.Event(), asyncio.Event()

    async def leaf_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        leaf_started.set()
        await leaf_release.wait()
        yield 'leaf evidence'

    leaf = Agent(FunctionModel(stream_function=leaf_stream), deps_type=object, name='worker')
    child = Agent(
        parent_model(background=True),
        deps_type=object,
        name='worker',
        capabilities=[SubAgents(agents=[SubAgent(leaf)], agent_folders=None)],
    )
    owner = DelegationTasks()
    with anyio.fail_after(WAIT):
        async with owner.opened():
            with owner.bind():
                parent = Agent(parent_model(), capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)])
                running = asyncio.create_task(parent.run('go', conversation_id='root'))
                await leaf_started.wait()
                assert not running.done()
                leaf_release.set()
                await running
            records = list(owner.records.values())
            assert len(records) == 2
            outer, inner = records
            assert inner.parent_id == outer.id
            assert inner.conversation_id == outer.conversation_id == 'root'
            assert inner.delivered
            assert 'leaf evidence' in str(outer.messages)
            assert not owner.reports(conversation_id='root')


@pytest.mark.parametrize('identity', ['../escape', '/tmp/escape', 'x' * 32])
async def test_rejects_unsafe_saved_identity(tmp_path: Path, identity: str) -> None:
    (tmp_path / 'record.json').write_text(
        json.dumps(
            {
                'id': identity,
                'agent_name': 'worker',
                'prompt': 'go',
                'conversation_id': 'root',
            }
        )
    )
    with pytest.raises(ValueError, match='Invalid saved task identity'):
        async with DelegationTasks(directory=tmp_path).opened():
            pass


async def immediate(record: DelegationTask) -> str:
    return 'done'


async def delegate(
    owner: DelegationTasks,
    *,
    resume: str | None = None,
    background: bool = False,
    agent_name: str = 'worker',
    conversation_id: str = 'root',
    backgroundable: bool = True,
) -> str:
    return await owner.delegate(
        agent_name=agent_name,
        prompt='go',
        conversation_id=conversation_id,
        model=None,
        background=background,
        resume=resume,
        run=immediate,
        backgroundable=backgroundable,
    )


async def test_owner_guards_and_resume_errors() -> None:
    with pytest.raises(ValueError, match='max_depth'):
        DelegationTasks(max_depth=0)
    owner = DelegationTasks()
    with pytest.raises(RuntimeError, match='closed'):
        await delegate(owner)
    with pytest.raises(RuntimeError, match='before binding'):
        with owner.bind():
            pass
    async with owner.opened():
        with pytest.raises(RuntimeError, match='already open'):
            async with owner.opened():
                pass
        with pytest.raises(ModelRetry, match='workspace'):
            await delegate(owner, background=True, backgroundable=False)
        with pytest.raises(ModelRetry, match='Unknown'):
            await delegate(owner, resume='unknown')
        await delegate(owner)
        (record,) = owner.records.values()
        with pytest.raises(ModelRetry, match='Unknown'):
            await delegate(owner, resume=record.id, conversation_id='other')
        with pytest.raises(ModelRetry, match='original agent'):
            await delegate(owner, resume=record.id, agent_name='other')
        record.status = 'running'
        with pytest.raises(ModelRetry, match='still running'):
            await delegate(owner, resume=record.id)
        with pytest.raises(ValueError, match='settled'):
            await owner.allow_resume(record.id)
        record.status = 'finished'
        record.resumable = False
        with pytest.raises(ModelRetry, match='one-shot'):
            await delegate(owner, resume=record.id)
        with pytest.raises(ValueError, match='settled'):
            await owner.allow_resume(record.id)
        record.resumable = True
        await owner.cancel(record.id)
        with pytest.raises(ModelRetry, match='stopped'):
            await delegate(owner, resume=record.id)
        await owner.allow_resume(record.id)
        await delegate(owner, resume=record.id)
        owner.background(record.id)
        record.backgroundable = False
        with pytest.raises(ValueError, match='workspace'):
            owner.background(record.id)
    invalid = DelegationTask(id='../bad', agent_name='worker', prompt='', conversation_id='root')
    with pytest.raises(ValueError, match='identity'):
        await owner.save(invalid)


async def test_observer_failure_settles_and_delivers(caplog: pytest.LogCaptureFixture) -> None:
    async def fail(update: DelegationTaskEvent) -> None:
        raise ValueError('broken observer')

    owner = DelegationTasks(observer=fail)
    async with owner.opened():
        result = await delegate(owner)
        (record,) = owner.records.values()
        assert 'failed' in result and 'broken observer' in result
        assert record.delivered
        assert 'Task completion observer failed' in caplog.text


async def test_queued_reports_ack_generation_and_receiver_exit(tmp_path: Path) -> None:
    owner = DelegationTasks(directory=tmp_path)
    async with owner.opened():
        await delegate(owner)
        (record,) = owner.records.values()
        record.background, record.delivered = True, False
        calls: list[SystemPromptPart] = []

        def enqueue(part: SystemPromptPart) -> str:
            calls.append(part)
            return 'queued'

        with owner.receiving(conversation_id='root', parent_id=None, enqueue=enqueue):
            owner.queue_reports(conversation_id='root', parent_id=None)
            assert len(calls) == 1
            record.generation += 1
            await owner.acknowledge('queued')
            assert not record.delivered
            await owner.acknowledge('missing')
        with owner.receiving(conversation_id='root', parent_id=None, enqueue=lambda part: None):
            assert not record.delivered
        with owner.receiving(conversation_id='root', parent_id=None, enqueue=enqueue):
            await owner.acknowledge('queued')
            assert record.delivered
        await owner.wait_children(record.id)


async def test_interrupted_restore_and_copy_history(tmp_path: Path) -> None:
    owner = DelegationTasks(directory=tmp_path)
    async with owner.opened():
        await delegate(owner)
        (record,) = owner.records.values()
        record.messages = [ModelResponse(parts=[TextPart('saved')])]
        copied = owner.history(record.id)
        copied.clear()
        assert record.messages
    path = tmp_path / f'{record.id}.json'
    data = json.loads(path.read_text())
    data['status'] = 'running'
    path.write_text(json.dumps(data))
    async with DelegationTasks(directory=tmp_path).opened() as restored:
        saved = restored.records[record.id]
        assert saved.outcome == 'cancelled'
        assert 'process exit' in saved.output
        assert saved.messages == record.messages


async def test_cancellation_before_worker_starts_and_manual_resume(tmp_path: Path) -> None:
    owner = DelegationTasks(directory=tmp_path)
    async with owner.opened():
        await delegate(owner, background=True)
        (record,) = owner.records.values()
        await owner.cancel(record.id)
        assert record.outcome == 'cancelled'
        assert record.user_stopped
        await owner.allow_resume(record.id)
        assert 'done' in await delegate(owner, resume=record.id)


async def test_foreground_parent_cancellation_drains_worker() -> None:
    owner = DelegationTasks()
    started = asyncio.Event()

    async def wait(record: DelegationTask) -> str:
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    async with owner.opened():
        pending = asyncio.create_task(
            owner.delegate(
                agent_name='worker',
                prompt='',
                conversation_id='root',
                model=None,
                background=False,
                resume=None,
                run=wait,
            )
        )
        await started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        (record,) = owner.records.values()
        assert record.outcome == 'cancelled'


@pytest.mark.parametrize('snapshot', [False, True])
async def test_restore_interrupted_step_history(tmp_path: Path, snapshot: bool) -> None:
    store = FileStepStore(tmp_path / 'steps')
    directory = tmp_path / 'tasks'
    owner = DelegationTasks(directory=directory, step_store=store)
    async with owner.opened():
        assert len(owner.persistence_capabilities()) == 1
        await delegate(owner)
        (record,) = owner.records.values()
        if snapshot:
            agent = Agent(TestModel(custom_output_text='step evidence'), capabilities=[StepPersistence(store=store)])
            await agent.run('saved request', run_id=record.run_id)
    path = directory / f'{record.id}.json'
    data = json.loads(path.read_text())
    data['status'] = 'running'
    path.write_text(json.dumps(data))
    async with DelegationTasks(directory=directory, step_store=store).opened() as restored:
        saved = restored.records[record.id]
        assert saved.outcome == 'cancelled'
        assert bool(saved.messages) == snapshot
        if snapshot:
            assert 'step evidence' in str(saved.messages)


async def test_resume_closes_out_tool_calls_interrupted_by_process_exit(tmp_path: Path) -> None:
    owner = DelegationTasks(directory=tmp_path)
    async with owner.opened():
        await delegate(owner, conversation_id='parent')
        (record,) = owner.records.values()
        record.messages = [
            ModelRequest(parts=[UserPromptPart('start')]),
            ModelResponse(parts=[ToolCallPart('slow_tool', {}, tool_call_id='slow')]),
        ]
    path = tmp_path / f'{record.id}.json'
    data = json.loads(path.read_text())
    data['status'] = 'running'
    path.write_text(json.dumps(data))

    seen: list[ModelMessage] = []

    async def child_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.extend(messages)
        yield 'resumed result'

    child = Agent(FunctionModel(stream_function=child_stream), deps_type=object, name='worker')
    restored = DelegationTasks(directory=tmp_path)
    with anyio.fail_after(WAIT):
        async with restored.opened():
            with restored.bind():
                parent = Agent(
                    parent_model(resume=record.id),
                    capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)],
                )
                await parent.run('continue', conversation_id='parent')
    resumed = restored.records[record.id]
    assert (resumed.outcome, resumed.output) == ('ok', 'resumed result')
    returns = [
        part
        for message in seen
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert [(part.tool_call_id, part.outcome, part.content) for part in returns] == [
        ('slow', 'interrupted', INTERRUPTED_TOOL_RETURN_CONTENT)
    ]
    last = seen[-1]
    assert isinstance(last, ModelRequest)
    assert [part.content for part in last.parts if isinstance(part, UserPromptPart)] == ['inspect']


async def test_failed_parent_drains_descendants() -> None:
    owner = DelegationTasks()
    started = asyncio.Event()

    async def leaf(record: DelegationTask) -> str:
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    async def parent(record: DelegationTask) -> str:
        await owner.delegate(
            agent_name='leaf', prompt='', conversation_id='root', model=None, background=True, resume=None, run=leaf
        )
        await started.wait()
        raise ValueError('parent failed')

    with anyio.fail_after(WAIT):
        async with owner.opened():
            result = await owner.delegate(
                agent_name='parent',
                prompt='',
                conversation_id='root',
                model=None,
                background=False,
                resume=None,
                run=parent,
            )
            assert 'parent failed' in result
            records = list(owner.records.values())
            assert [r.outcome for r in records] == ['failed', 'cancelled']
            assert records[1].parent_id == records[0].id
            assert not records[1].user_stopped


async def test_targeted_stop_drains_nested_children() -> None:
    owner = DelegationTasks()
    started = asyncio.Event()

    async def leaf(record: DelegationTask) -> str:
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    async def parent(record: DelegationTask) -> str:
        await owner.delegate(
            agent_name='leaf', prompt='', conversation_id='root', model=None, background=True, resume=None, run=leaf
        )
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    with anyio.fail_after(WAIT):
        async with owner.opened():
            await owner.delegate(
                agent_name='parent',
                prompt='',
                conversation_id='root',
                model=None,
                background=True,
                resume=None,
                run=parent,
            )
            await started.wait()
            parent_record, child_record = owner.records.values()
            await owner.cancel(parent_record.id)
            assert parent_record.user_stopped and child_record.user_stopped
            assert parent_record.outcome == child_record.outcome == 'cancelled'


async def test_stop_during_acceptance_and_shutdown_before_start() -> None:
    saved, release = asyncio.Event(), asyncio.Event()

    class GatedOwner(DelegationTasks):
        async def save(self, record: DelegationTask) -> None:
            if not saved.is_set():
                saved.set()
                await release.wait()
            await super().save(record)

    owner = GatedOwner()
    async with owner.opened():
        pending = asyncio.create_task(delegate(owner))
        await saved.wait()
        (record,) = owner.records.values()
        await owner.cancel(record.id)
        release.set()
        assert 'stopped before execution' in await pending
    owner = DelegationTasks()
    async with owner.opened():
        await delegate(owner, background=True)
    (record,) = owner.records.values()
    assert record.outcome == 'cancelled'
    assert DelegationTasks.child_id() is None


async def test_promoted_foreground_wait_cancellation_keeps_child() -> None:
    owner = DelegationTasks()
    started, release = asyncio.Event(), asyncio.Event()

    async def child(record: DelegationTask) -> str:
        started.set()
        await release.wait()
        return 'child'  # pragma: lax no cover

    async with owner.opened():
        pending = asyncio.create_task(
            owner.delegate(
                agent_name='worker',
                prompt='',
                conversation_id='root',
                model=None,
                background=False,
                resume=None,
                run=child,
            )
        )
        await started.wait()
        (record,) = owner.records.values()
        owner.background(record.id)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert record.status == 'running'
        await owner.cancel(record.id)


@pytest.mark.parametrize('operation', ['close', 'cancel'])
async def test_bounded_save_cleanup(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    original = anyio.move_on_after

    def short(delay: float | None, *, shield: bool = False) -> anyio.CancelScope:
        return original(0.01, shield=shield)

    monkeypatch.setattr(anyio, 'move_on_after', short)

    class SlowOwner(DelegationTasks):
        stall: bool = False

        async def save(self, record: DelegationTask) -> None:
            if self.stall:
                await anyio.sleep_forever()
            await super().save(record)

    owner = SlowOwner()
    if operation == 'close':
        with pytest.raises(RuntimeError, match='cleanup timed out'):
            async with owner.opened():
                await delegate(owner)
                owner.stall = True
    else:
        async with owner.opened():
            await delegate(owner)
            (record,) = owner.records.values()
            owner.stall = True
            try:
                with pytest.raises(RuntimeError, match='cancellation cleanup'):
                    await owner.cancel(record.id)
            finally:
                owner.stall = False


async def test_child_budget_cannot_hide_parent_usage() -> None:
    from pydantic_ai.exceptions import UsageLimitExceeded

    owner = DelegationTasks()
    child = Agent(TestModel(custom_output_text='evidence'), deps_type=object, name='worker')
    usage = RunUsage()
    async with owner.opened():
        with owner.bind():
            parent = Agent(
                parent_model(),
                capabilities=[
                    SubAgents(agents=[SubAgent(child, usage_limits=UsageLimits(request_limit=10))], agent_folders=None)
                ],
            )
            with pytest.raises(UsageLimitExceeded, match='request_limit'):
                await parent.run('go', conversation_id='root', usage=usage, usage_limits=UsageLimits(request_limit=2))
            assert usage.requests == 2
            (record,) = owner.records.values()
            assert record.outcome == 'ok'


async def test_child_cost_budget_counts_from_the_spend_at_launch() -> None:
    # The parent already spent more than the child's budget; only what the child adds counts against it.
    owner = DelegationTasks()
    child = Agent(TestModel(custom_output_text='evidence'), deps_type=object, name='worker')
    usage = RunUsage(cost=Decimal('0.5'))
    async with owner.opened():
        with owner.bind():
            parent = Agent(
                parent_model(),
                capabilities=[
                    SubAgents(
                        agents=[SubAgent(child, usage_limits=UsageLimits(cost_limit=Decimal('0.25')))],
                        agent_folders=None,
                    )
                ],
            )
            await parent.run('go', conversation_id='root', usage=usage)
            (record,) = owner.records.values()
            assert record.outcome == 'ok'


async def test_rejects_cyclic_persisted_ancestry(tmp_path: Path) -> None:
    for identity, parent in [('a' * 32, 'b' * 32), ('b' * 32, 'a' * 32)]:
        (tmp_path / f'{identity}.json').write_text(
            json.dumps(
                {
                    'id': identity,
                    'parent_id': parent,
                    'agent_name': 'worker',
                    'prompt': '',
                    'conversation_id': 'root',
                    'status': 'finished',
                    'outcome': 'ok',
                }
            )
        )
    with pytest.raises(ValueError, match='Cyclic'):
        async with DelegationTasks(directory=tmp_path).opened():
            pytest.fail('Invalid ancestry must not reach task controls')  # pragma: no cover
