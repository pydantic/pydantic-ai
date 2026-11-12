"""Tests for `RunContext.in_flight`: work that outlives the step that started it, and the rule that ends a run with it.

They drive the run loop with `FunctionModel`, since what they check is when the run waits, cancels and records,
which no provider takes part in.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import anyio
import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunCancelled, RunContext
from pydantic_ai._enqueue import EnqueueGuard
from pydantic_ai.exceptions import ApprovalRequired, UserError
from pydantic_ai.in_flight import (
    InFlightRegistry,
    InFlightWork,
    InFlightWorkEvent,
    InFlightWorkPhase,
    InFlightWorkSnapshot,
    InFlightWorkState,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import RunUsage

from .conftest import no_leaked_tasks

READINESS_WAIT_TIMEOUT = 10


@dataclass
class _Emitted:
    """The `InFlightWorkEvent`s a run emitted, in order."""

    events: list[InFlightWorkEvent] = field(default_factory=list[InFlightWorkEvent])
    _changed: asyncio.Event = field(default_factory=asyncio.Event)

    def add(self, event: InFlightWorkEvent) -> None:
        self.events.append(event)
        self._changed.set()
        self._changed = asyncio.Event()

    @property
    def phases(self) -> list[InFlightWorkPhase]:
        return [event.phase for event in self.events]

    async def until(self, phase: InFlightWorkPhase, count: int = 1) -> None:
        """Wait until the run has emitted `count` events of `phase`, like the `'draining'` one it emits as it starts to wait."""
        while self.phases.count(phase) < count:
            await asyncio.wait_for(self._changed.wait(), READINESS_WAIT_TIMEOUT)


class _Recorder(list[AgentStreamEvent]):
    """Passes the registry's events on to the run's event buffer, and records them as they're emitted.

    The run's event stream drops the events emitted while no step streams, like those the run emits while it
    waits to end, and a step that does stream takes the events out of the buffer.
    """

    def __init__(self, buffer: list[AgentStreamEvent], emitted: _Emitted) -> None:
        super().__init__()
        self._buffer = buffer
        self._emitted = emitted

    def append(self, event: AgentStreamEvent) -> None:
        self._buffer.append(event)
        assert isinstance(event, InFlightWorkEvent)
        self._emitted.add(event)


@pytest.fixture
def emitted(monkeypatch: pytest.MonkeyPatch) -> _Emitted:
    emitted = _Emitted()
    init = InFlightRegistry.__init__

    def record(self: InFlightRegistry, event_stream_buffer: list[AgentStreamEvent]) -> None:
        init(self, _Recorder(event_stream_buffer, emitted))

    monkeypatch.setattr(InFlightRegistry, '__init__', record)
    return emitted


def _user_texts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def _snapshot(name: str, state: InFlightWorkState) -> InFlightWorkSnapshot:
    return InFlightWorkSnapshot(id=name, kind='test', name=name, on_end='drain', durable=False, state=state)


async def test_ending_with_output_waits_for_the_first_result_and_gives_the_model_a_turn(emitted: _Emitted):
    """The model gets a turn on each result as it lands: the run waits for the first one, then waits again."""
    seen: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(_user_texts(messages))
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='start', args={'name': 'a'}),
                    ToolCallPart(tool_name='start', args={'name': 'b'}),
                ]
            )
        return ModelResponse(parts=[TextPart(f'done {len(seen)}')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    async def start(ctx: RunContext, name: str) -> str:
        async def work() -> None:
            # `a` finishes once the run first waits to end, `b` once it waits again.
            await emitted.until('draining', 1 if name == 'a' else 2)
            ctx.enqueue(f'result {name}')

        ctx.in_flight.start(work, kind='test', name=name, id=name)
        return 'started'

    result = await agent.run('go')

    assert result.output == 'done 4'
    assert seen == [['go'], ['go'], ['go', 'result a'], ['go', 'result a', 'result b']]
    # The run stops waiting as soon as `a` has queued its result, before `a` itself has finished.
    assert emitted.phases == [
        'started',
        'started',
        'draining',
        'drained',
        'settled',
        'draining',
        'drained',
        'settled',
    ]
    assert emitted.events[2] == InFlightWorkEvent(
        phase='draining', work=(_snapshot('a', 'running'), _snapshot('b', 'running')), live=2
    )
    assert emitted.events[4] == InFlightWorkEvent(phase='settled', work=(_snapshot('a', 'succeeded'),), live=1)


async def test_ending_with_queued_content_gives_the_model_a_turn_before_waiting(emitted: _Emitted):
    """Content already queued when the model ends the run gets its turn without waiting for work."""
    seen: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(_user_texts(messages))
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])
        return ModelResponse(parts=[TextPart(f'done {len(seen)}')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    async def start(ctx: RunContext) -> str:
        async def work() -> None:
            await emitted.until('draining')
            ctx.enqueue('result')

        ctx.in_flight.start(work, kind='test', name='work', id='work')
        ctx.enqueue('note', priority='when_idle')
        return 'started'

    result = await agent.run('go')

    assert result.output == 'done 4'
    assert seen == [['go'], ['go'], ['go', 'note'], ['go', 'note', 'result']]
    assert emitted.phases == ['started', 'draining', 'drained', 'settled']


async def test_ending_with_output_gives_the_model_a_turn_once_work_queues_a_result(emitted: _Emitted):
    """Work that queues a result and keeps running doesn't hold back the model's turn on the result."""
    seen: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(_user_texts(messages))
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])
        return ModelResponse(parts=[TextPart(f'done {len(seen)}')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    async def start(ctx: RunContext) -> str:
        async def work() -> None:
            await emitted.until('draining')
            # From another thread, as a callback reporting progress might.
            await anyio.to_thread.run_sync(ctx.enqueue, 'partial')
            # Still running when the model gets its turn on `partial`, until the run waits again.
            await emitted.until('draining', 2)

        ctx.in_flight.start(work, kind='test', name='work', id='work')
        return 'started'

    result = await agent.run('go')

    assert result.output == 'done 3'
    assert seen == [['go'], ['go'], ['go', 'partial']]
    assert emitted.phases == ['started', 'draining', 'drained', 'draining', 'settled', 'drained']


async def test_work_settling_without_queuing_doesnt_end_the_wait(emitted: _Emitted):
    """Ending with output, the run keeps waiting when work settles without queuing anything while other draining work runs."""
    seen: list[list[str]] = []
    handles: dict[str, InFlightWork[None]] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(_user_texts(messages))
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])
        return ModelResponse(parts=[TextPart(f'done {len(seen)}')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    async def start(ctx: RunContext) -> str:
        silent = ctx.in_flight.register(kind='test', name='silent', id='silent', on_end='drain')

        async def work() -> None:
            await emitted.until('draining')
            ctx.in_flight.settle(silent)
            # Let the run check again after `silent` settled, before this work queues its result.
            await asyncio.sleep(0)
            ctx.enqueue('result')

        handles['work'] = ctx.in_flight.start(work, kind='test', name='work', id='work')
        return 'started'

    result = await agent.run('go')

    assert result.output == 'done 3'
    assert seen == [['go'], ['go'], ['go', 'result']]
    assert handles['work'].state == 'succeeded'
    assert emitted.phases == ['started', 'started', 'draining', 'settled', 'drained', 'settled']


async def test_run_stream_waits_for_work_but_doesnt_deliver_its_result(emitted: _Emitted):
    """`run_stream()` waits once its output has been consumed, but the output was already streamed, so the result goes nowhere."""
    handles: dict[str, InFlightWork[None]] = {}
    # `TestModel` calls the tool, then streams its answer.
    agent = Agent(TestModel(custom_output_text='final answer'))

    @agent.tool
    async def start(ctx: RunContext) -> str:
        async def work() -> None:
            await emitted.until('draining')
            ctx.enqueue('late result')

        handles['work'] = ctx.in_flight.start(work, kind='test', name='work', id='work')
        return 'started'

    async with agent.run_stream('go') as result:
        assert await result.get_output() == 'final answer'

    assert handles['work'].state == 'succeeded'
    assert 'late result' not in _user_texts(result.all_messages())
    assert emitted.phases == ['started', 'draining', 'drained', 'settled']


async def test_deferred_pause_waits_for_all_work_and_records_its_results(emitted: _Emitted):
    """A pause waits for every piece of `'drain'` work, records what it queued after the paused step, and stands."""
    model_calls = 0
    handles: dict[str, InFlightWork[None]] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name='start', args={'name': 'a'}),
                ToolCallPart(tool_name='start', args={'name': 'b'}),
                ToolCallPart(tool_name='needs_approval', args={}),
            ]
        )

    agent = Agent(FunctionModel(model_fn), output_type=[str, DeferredToolRequests])

    @agent.tool
    async def start(ctx: RunContext, name: str) -> str:
        async def work() -> None:
            await emitted.until('draining')
            if name == 'b':
                # Still running after `a` has settled, so the pause has to keep waiting.
                await handles['a'].wait()
            ctx.enqueue(f'result {name}')

        handles[name] = ctx.in_flight.start(work, kind='test', name=name, id=name)
        return 'started'

    @agent.tool_plain
    def needs_approval() -> str:
        raise ApprovalRequired

    result = await agent.run('go')

    assert isinstance(result.output, DeferredToolRequests)
    assert model_calls == 1
    messages = result.all_messages()
    assert len(messages) == 5
    assert _user_texts(messages) == ['go', 'result a', 'result b']
    assert emitted.phases == ['started', 'started', 'draining', 'settled', 'settled', 'drained']


async def test_cancelling_the_run_cancels_all_but_detached_work(emitted: _Emitted):
    """Cancelling the run, here while it waits to end, cancels its work, except work built to outlive it."""
    handles: dict[str, InFlightWork[None]] = {}
    cancelled: list[str] = []
    contexts: list[RunContext] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    async def start(ctx: RunContext) -> str:
        async def forever() -> None:
            await asyncio.Event().wait()

        async def cancel_the_run() -> None:
            await emitted.until('draining')
            ctx.cancel()
            await asyncio.Event().wait()

        contexts.append(ctx)
        handles['drain'] = ctx.in_flight.start(forever, kind='test', name='drain', id='drain')
        handles['cancel'] = ctx.in_flight.start(
            cancel_the_run, kind='test', name='cancel', id='cancel', on_end='cancel'
        )
        handles['job'] = ctx.in_flight.register(
            kind='test', name='job', id='job', on_end='drain', cancel=lambda: cancelled.append('job')
        )
        # Registered work lives elsewhere, so it's detached unless it says otherwise.
        handles['server'] = ctx.in_flight.register(
            kind='test', name='server', id='server', cancel=lambda: cancelled.append('server')
        )
        return 'started'

    with pytest.raises(RunCancelled):
        await agent.run('go')

    assert {name: work.state for name, work in handles.items()} == {
        'drain': 'cancelled',
        'cancel': 'cancelled',
        'job': 'cancelled',
        'server': 'running',
    }
    assert cancelled == ['job']

    # The registry has closed, but detached work can still be settled.
    registry = contexts[0].in_flight
    with pytest.raises(
        UserError, match=r'`ctx.in_flight.register\(\)` is not available because the agent run has ended'
    ):
        registry.register(kind='test', name='late', id='late')
    registry.settle(handles['server'])
    assert handles['server'].state == 'succeeded'
    # Nothing streams events once the run has ended, so none were emitted after the wait started.
    assert emitted.phases == ['started', 'started', 'started', 'started', 'draining']


async def test_a_failing_cancel_callback_doesnt_keep_other_work_running():
    """The run still cancels and waits for all its work when a registered work's `cancel` callback raises, then raises its error."""
    handles: dict[str, InFlightWork[Any]] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn))

    def fail_to_cancel() -> None:
        raise RuntimeError('process already gone')

    @agent.tool
    async def start(ctx: RunContext) -> str:
        # Registered first, so the run cancels it before the started work.
        handles['job'] = ctx.in_flight.register(
            kind='test', name='job', id='job', on_end='cancel', cancel=fail_to_cancel
        )
        handles['task'] = ctx.in_flight.start(
            asyncio.Event().wait, kind='test', name='task', id='task', on_end='cancel'
        )
        return 'started'

    with no_leaked_tasks(), pytest.raises(RuntimeError, match='process already gone'):
        await agent.run('go')

    assert handles['job'].state == 'cancelled'
    assert handles['task'].state == 'cancelled'


async def test_a_failing_cancel_callback_doesnt_replace_the_runs_own_error():
    """When the run fails anyway, the caller gets its error, not the one a `cancel` callback raised while it ended."""
    handles: dict[str, InFlightWork[None]] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])

    agent = Agent(FunctionModel(model_fn))

    def fail_to_cancel() -> None:
        raise RuntimeError('process already gone')

    @agent.tool
    async def start(ctx: RunContext) -> str:
        handles['job'] = ctx.in_flight.register(
            kind='test', name='job', id='job', on_end='cancel', cancel=fail_to_cancel
        )
        raise ValueError('the run fails')

    with pytest.raises(ValueError, match='the run fails'):
        await agent.run('go')

    assert handles['job'].state == 'cancelled'


async def test_work_failing_at_a_step_boundary_ends_the_run():
    """An error raised by started work ends the run at the next step boundary, as a failing tool would."""
    model_called = asyncio.Event()
    work: InFlightWork[None] | None = None

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])
        model_called.set()
        assert work is not None
        with pytest.raises(ValueError, match='boom'):
            await work.result()
        return ModelResponse(parts=[TextPart('not handled')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    async def start(ctx: RunContext) -> str:
        nonlocal work

        async def fail() -> None:
            await model_called.wait()
            raise ValueError('boom')

        work = ctx.in_flight.start(fail, kind='test', name='fail', id='fail')
        return 'started'

    with pytest.raises(ValueError, match='boom'):
        await agent.run('go')
    assert work is not None
    assert work.state == 'failed'


async def test_work_failing_while_the_run_waits_ends_the_run(emitted: _Emitted):
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    async def start(ctx: RunContext) -> str:
        async def fail() -> None:
            await emitted.until('draining')
            raise ValueError('boom')

        ctx.in_flight.start(fail, kind='test', name='fail', id='fail')
        return 'started'

    with pytest.raises(ValueError, match='boom'):
        await agent.run('go')
    assert emitted.phases == ['started', 'draining', 'settled']


async def test_registered_work_handles_and_waiting_for_idle():
    """Registered work settles when its producer says so, an error it settles with doesn't end the run, and detached work isn't waited for."""
    handles: dict[str, InFlightWork[None]] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='use_registry', args={})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    async def use_registry(ctx: RunContext) -> str:
        registry = ctx.in_flight
        job = registry.register(kind='job', name='job', id='job', on_end='drain')
        handles['server'] = registry.register(kind='job', name='server', id='server')
        assert handles['server'].on_end == 'detach'

        async def finish_job() -> int:
            registry.settle(job)
            return 42

        answer = registry.start(finish_job, kind='other', name='answer', id='answer')
        assert [work.id for work in registry.live(kind='job')] == ['job', 'server']
        await registry.wait_idle(kind='job')
        assert job.state == 'succeeded'
        await job.result()
        assert await answer.result() == 42
        answer.cancel()
        assert answer.state == 'succeeded'
        assert [work.id for work in registry.live()] == ['server']

        failing = registry.register(kind='other', name='failing', id='failing')
        registry.settle(failing, error=RuntimeError('job failed'))
        assert failing.state == 'failed'
        with pytest.raises(RuntimeError, match='job failed'):
            await failing.result()

        untracked = registry.register(kind='other', name='untracked', id='untracked')
        with pytest.raises(UserError, match=r'`ctx\.in_flight\.settle\(\)` must be called from the event loop'):
            await anyio.to_thread.run_sync(registry.settle, untracked)
        with pytest.raises(UserError, match=r'`ctx\.in_flight\.live\(\)` must be called from the event loop'):
            await anyio.to_thread.run_sync(registry.live)
        untracked.cancel()
        # Settling work that has already settled does nothing.
        registry.settle(untracked)
        assert untracked.state == 'cancelled'
        with pytest.raises(asyncio.CancelledError):
            await untracked.result()

        running = asyncio.Event()

        async def forever() -> None:
            running.set()
            await asyncio.Event().wait()

        slow = registry.start(forever, kind='other', name='slow', id='slow', on_end='cancel')
        with pytest.raises(UserError, match=r"In-flight work 'slow' was started with `start\(\)`"):
            registry.settle(slow)
        await asyncio.wait_for(running.wait(), READINESS_WAIT_TIMEOUT)
        slow.cancel()
        await slow.wait()
        assert slow.state == 'cancelled'

        with pytest.raises(UserError, match="In-flight work 'server' is already running in this run"):
            registry.register(kind='job', name='server', id='server')
        with pytest.raises(UserError, match='takes `on_end` of `drain` or `cancel`'):
            registry.start(
                forever,
                kind='other',
                name='detached',
                id='detached',
                on_end='detach',  # pyright: ignore[reportArgumentType]
            )
        return 'checked'

    result = await agent.run('go')

    assert result.output == 'done'
    assert handles['server'].state == 'running'


@pytest.mark.parametrize('method', ['start', 'register'])
async def test_adding_work_off_the_event_loop_raises(method: str):
    """A sync tool runs in a worker thread, where the registry's bookkeeping isn't safe to touch."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name='add_in_thread', args={})])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def add_in_thread(ctx: RunContext) -> None:
        if method == 'start':
            ctx.in_flight.start(lambda: asyncio.sleep(0), kind='test', name='work', id='work')
        else:
            ctx.in_flight.register(kind='test', name='work', id='work')

    with pytest.raises(UserError, match=rf'`ctx\.in_flight\.{method}\(\)` must be called from the event loop'):
        await agent.run('go')


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_adding_work_requires_asyncio():
    """The registry's tasks and events are asyncio's, so another backend gets an error that says so."""
    registry = InFlightRegistry([])
    with pytest.raises(UserError, match=r'`ctx\.in_flight\.register\(\)` requires the asyncio event loop'):
        registry.register(kind='test', name='work', id='work')


async def test_concurrent_runs_track_their_own_work():
    barrier = asyncio.Barrier(2)
    live: dict[str, list[str]] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='start', args={})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), deps_type=str)

    @agent.tool
    async def start(ctx: RunContext[str]) -> str:
        ctx.in_flight.start(asyncio.Event().wait, kind='test', name=ctx.deps, id=ctx.deps, on_end='cancel')
        # Both runs have started their work before either lists its own.
        await asyncio.wait_for(barrier.wait(), READINESS_WAIT_TIMEOUT)
        live[ctx.deps] = [work.id for work in ctx.in_flight.live()]
        return 'started'

    # Each run cancels and waits for the work it started before it returns.
    with no_leaked_tasks():
        await asyncio.gather(agent.run('go', deps='one'), agent.run('go', deps='two'))

    assert live == {'one': ['one'], 'two': ['two']}


async def test_events_reach_the_event_stream():
    """Work starting during a step is reported in that step's event stream."""
    # `TestModel` calls the tool, then answers; unlike a plain `FunctionModel`, it streams.
    agent = Agent(TestModel())

    @agent.tool
    async def start(ctx: RunContext) -> str:
        ctx.in_flight.start(asyncio.Event().wait, kind='test', name='work', id='work', on_end='cancel')
        return 'started'

    async with agent.run_stream_events('go') as stream:
        events = [event async for event in stream if isinstance(event, InFlightWorkEvent)]

    assert events == [
        InFlightWorkEvent(
            phase='started',
            work=(
                InFlightWorkSnapshot(
                    id='work', kind='test', name='work', on_end='cancel', durable=False, state='running'
                ),
            ),
            live=1,
        )
    ]


def test_in_flight_needs_a_run_outside_durable_units():
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    with pytest.raises(UserError, match=r'`ctx\.in_flight` is only available during an agent run'):
        _ = ctx.in_flight

    in_activity = RunContext(
        deps=None, model=TestModel(), usage=RunUsage(), pending_messages=EnqueueGuard('activity', 'workflow')
    )
    with pytest.raises(
        UserError,
        match=r'`ctx.in_flight` is not supported inside a durable activity: .* so work it started or registered '
        r'would be lost\.$',
    ):
        _ = in_activity.in_flight


def test_in_flight_work_event_round_trips():
    event = InFlightWorkEvent(phase='started', work=(_snapshot('a', 'running'),), live=1)
    adapter = TypeAdapter[AgentStreamEvent](AgentStreamEvent)

    assert event.kind == 'in_flight.work'
    assert event.capability_id is None
    assert adapter.validate_json(adapter.dump_json(event)) == event
