"""Persistent connection boundaries with deterministic duplex traffic, not provider timing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar

import anyio
import pytest

from pydantic_ai import Agent, AgentRunResult, RunCancelled, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import AgentStreamEvent, BinaryImage, ModelMessage, ModelResponse, SpeechPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import RealtimeModelSettings, RealtimeRun
from pydantic_ai.realtime.codec import (
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolResult,
)
from pydantic_ai.usage import RequestUsage

from .test_session import FakeRealtimeModel

READINESS_WAIT_TIMEOUT = 10


class DuplexConnection(RealtimeConnection):
    def __init__(self) -> None:
        self.events: asyncio.Queue[RealtimeCodecEvent] = asyncio.Queue()
        self.sent: list[RealtimeInput] = []
        self.iterations = 0

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)
        if isinstance(content, str):
            self.events.put_nowait(OutputTranscript(f'reply to {content}', output_text=True, is_final=True))
            self.events.put_nowait(ResponseDone())

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        self.iterations += 1
        while True:
            yield await self.events.get()


class CountedModel(FakeRealtimeModel):
    def __init__(self, connection: RealtimeConnection) -> None:
        super().__init__(connection)
        self.opens = 0
        self.closes = 0
        self.entry_task: asyncio.Task[object] | None = None

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[RealtimeConnection]:
        self.opens += 1
        self.entry_task = asyncio.current_task()
        # A real cancel scope makes entering in run A and exiting in run B fail, as providers can.
        with anyio.CancelScope():
            async with super().connect(
                messages=messages, model_settings=model_settings, model_request_parameters=model_request_parameters
            ) as connection:
                try:
                    yield connection
                finally:
                    assert asyncio.current_task() is self.entry_task
                    self.closes += 1


async def test_two_realtime_runs_share_one_connection_and_keep_independent_results():
    before = asyncio.all_tasks()
    connection = DuplexConnection()
    model = CountedModel(connection)
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:
            assert model.opens == 0
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with live.run() as first:
                    await first.send('first')
                assert first.result is not None
                assert first.result.output == 'reply to first'
                saved = first.all_messages()
                with pytest.raises(UserError, match='ordinary run'):
                    await owner.run('not while attached')
                async with live.run() as second:
                    with pytest.raises(UserError, match='run has ended'):
                        await first.send('stale')
                    await second.send('second')
            assert second.result is not None
            assert second.result.output == 'reply to second'
            assert first.result.run_id != second.result.run_id
            assert first.all_messages() == saved
            assert len(second.new_messages()) == 2
            assert {message.run_id for message in second.new_messages()} == {second.result.run_id}
            assert model.opens == connection.iterations == 1
            assert model.closes == 0
            assert owner.state.active_run_id is None
            first.result.all_messages()[0].parts = []
            assert all(message.parts for message in owner.conversation.messages)
        assert model.closes == 1
        result = await owner.run('after live')
        assert len([m for m in result.all_messages() if isinstance(m, ModelResponse)]) >= 3
    assert asyncio.all_tasks() == before


async def test_lazy_event_iterator_cannot_read_the_next_run():
    connection = DuplexConnection()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            async with live.run() as first:
                stale = aiter(first)
            async with live.run() as second:
                await second.send('second')
                with pytest.raises(UserError, match='run has ended'):
                    await anext(stale)


async def test_audio_producer_is_drained_before_next_run():
    before = asyncio.all_tasks()
    started, finalized = asyncio.Event(), asyncio.Event()
    connection = DuplexConnection()

    async def microphone() -> AsyncIterator[bytes]:
        try:
            started.set()
            await asyncio.Event().wait()
            yield b'never sent'
        finally:
            finalized.set()

    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            async with live.run() as first:
                producer = asyncio.create_task(first.send_audio(microphone()))
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await started.wait()
            assert finalized.is_set()
            await producer
            async with live.run() as second:
                await second.send('second')
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('waiting', [False, True])
async def test_run_owns_event_wrapper_cleanup_and_hook_handle(waiting: bool):
    before = asyncio.all_tasks()
    marker: ContextVar[str] = ContextVar('run-wrapper-marker', default='outside')
    calls: list[tuple[str, str]] = []
    handles: list[RealtimeRun] = []
    wrapper_waiting = asyncio.Event()

    class Lifecycle(AbstractCapability[str]):
        async def before_run(self, ctx: RunContext[str]) -> None:
            calls.append(('before', ctx.deps))
            assert ctx.metadata == {'label': ctx.deps}

        async def wrap_run_event_stream(
            self, ctx: RunContext[str], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterator[AgentStreamEvent]:
            assert isinstance(ctx.realtime_session, RealtimeRun)
            handles.append(ctx.realtime_session)
            task = asyncio.current_task()
            token = marker.set(ctx.deps)
            with anyio.CancelScope():
                try:
                    calls.append(('stream', ctx.deps))
                    async for event in stream:
                        yield event
                        if waiting:
                            wrapper_waiting.set()
                            await asyncio.Event().wait()
                finally:
                    assert asyncio.current_task() is task
                    assert marker.get() == ctx.deps
                    marker.reset(token)
                    calls.append(('stream-close', ctx.deps))

        async def after_run(self, ctx: RunContext[str], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            assert ('stream-close', ctx.deps) in calls
            assert isinstance(ctx.realtime_session, RealtimeRun)
            assert ctx.realtime_session is handles[-1]
            assert ctx.realtime_session.closed
            calls.append(('after', ctx.deps))
            return result

    agent = Agent(TestModel(), deps_type=str, capabilities=[Lifecycle()])
    async with agent.session(deps='default') as owner:
        async with owner.realtime(CountedModel(DuplexConnection())).connect() as live:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                for deps in ('first', 'second'):
                    reader = None
                    async with live.run(deps=deps, metadata={'label': deps}) as run:
                        for stale in handles:
                            with pytest.raises(UserError, match='run has ended'):
                                await stale.send('stale hook handle')
                        await run.send(deps)
                        iterator = aiter(run)
                        async for _ in iterator:
                            break
                        assert handles[-1] is run
                        if waiting:
                            wrapper_waiting.clear()
                            reader = asyncio.ensure_future(anext(iterator))
                            await wrapper_waiting.wait()
                    if reader is not None:
                        with pytest.raises(StopAsyncIteration):
                            await reader
            assert marker.get() == 'outside'
    assert calls == [
        (stage, deps) for deps in ('first', 'second') for stage in ('before', 'stream', 'stream-close', 'after')
    ]
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('during_after_run', [False, True])
async def test_attachment_exit_drains_active_run_before_transport(during_after_run: bool):
    before = asyncio.all_tasks()
    ready, cleaned = asyncio.Event(), asyncio.Event()
    model = CountedModel(DuplexConnection())

    class Lifecycle(AbstractCapability[None]):
        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            if during_after_run and ctx.realtime:
                try:
                    ready.set()
                    await asyncio.Event().wait()
                finally:
                    assert model.closes == 0
                    cleaned.set()
            return result

    async with Agent(TestModel(), deps_type=type(None), capabilities=[Lifecycle()]).session() as owner:
        async with owner.realtime(model).connect() as live:

            async def execute() -> None:
                try:
                    async with live.run() as run:
                        await run.send('first')
                        if not during_after_run:
                            try:
                                ready.set()
                                await asyncio.Event().wait()
                            finally:
                                assert model.closes == 0
                                cleaned.set()
                except asyncio.CancelledError:
                    pass

            task = asyncio.create_task(execute())
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await ready.wait()
        assert cleaned.is_set()
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await task
        assert model.closes == 1
        await owner.run('ordinary afterward')
    assert asyncio.all_tasks() == before


async def test_cleared_audio_does_not_leave_a_run_waiting_for_a_transcript():
    connection = DuplexConnection()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with live.run() as first:
                    await first.send_audio(b'\x00\x00' * 100)
                    await first.clear_audio()
                assert first.new_messages() == []
                async with live.run() as second:
                    await second.send('second')
            assert second.result is not None
            assert second.result.output == 'reply to second'


@pytest.mark.parametrize('clear_following_audio', [False, True])
async def test_run_waits_for_late_input_transcript(clear_following_audio: bool):
    connection = DuplexConnection()
    exiting = asyncio.Event()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:

            async def first_run() -> RealtimeRun:
                async with live.run() as run:
                    await run.send_audio(b'\x00\x00' * 100)
                    await run.commit_audio()
                    await run.create_response()
                    connection.events.put_nowait(OutputTranscript('answer', output_text=True, is_final=True))
                    connection.events.put_nowait(ResponseDone())
                    await run.wait_for_reply()
                    if clear_following_audio:
                        await run.send_audio(b'\x01\x01' * 100)
                        await run.clear_audio()
                    exiting.set()
                return run

            task = asyncio.create_task(first_run())
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await exiting.wait()
                assert not task.done()
                assert owner.state.active_run_id is not None
                connection.events.put_nowait(InputTranscript('question', is_final=True))
                first = await task
                async with live.run() as second:
                    await second.send('second')
            assert first.result is not None
            assert first.result.output == 'answer'
            assert isinstance(part := first.new_messages()[0].parts[0], SpeechPart)
            assert part.transcript == 'question'
            assert len(second.new_messages()) == 2
            assert all(message.run_id == first.result.run_id for message in first.new_messages())


async def test_tools_use_fresh_run_dependencies_metadata_and_context():
    before = asyncio.all_tasks()
    marker: ContextVar[str] = ContextVar('tool-run-marker')
    contexts: list[RunContext[str]] = []
    instances: list[AbstractCapability[str]] = []
    tool_started, release_tool = asyncio.Event(), asyncio.Event()

    class Lifecycle(AbstractCapability[str]):
        async def for_run(self, ctx: RunContext[str]) -> AbstractCapability[str]:
            instance = Lifecycle()
            instances.append(instance)
            return instance

        async def before_run(self, ctx: RunContext[str]) -> None:
            marker.set(ctx.deps)

    class ToolConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            if isinstance(content, str):
                self.events.put_nowait(ToolCall(tool_name='identify', tool_call_id=content, args='{}'))
                self.events.put_nowait(ResponseDone())
            elif isinstance(content, ToolResult):
                self.events.put_nowait(OutputTranscript(content.output, output_text=True, is_final=True))
                self.events.put_nowait(ResponseDone())

    agent = Agent(TestModel(), deps_type=str, capabilities=[Lifecycle()])

    @agent.tool
    async def identify(ctx: RunContext[str]) -> str:
        assert marker.get() == ctx.deps
        assert ctx.metadata == {'label': ctx.deps}
        contexts.append(ctx)
        tool_started.set()
        await release_tool.wait()
        assert marker.get() == ctx.deps
        return ctx.deps

    model = CountedModel(ToolConnection())
    async with agent.session(deps='default') as owner:
        async with owner.realtime(model).connect() as live:

            async def execute(label: str) -> RealtimeRun:
                async with live.run(deps=label, metadata={'label': label}) as run:
                    await run.send(label)
                return run

            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                for label in ('first', 'second'):
                    tool_started.clear()
                    release_tool.clear()
                    task = asyncio.create_task(execute(label))
                    await tool_started.wait()
                    assert not task.done()
                    release_tool.set()
                    run = await task
                    assert run.result is not None
                    assert run.result.output == label
                    assert contexts[-1].realtime_session is run
            assert len(instances) == 2
            assert instances[0] is not instances[1]
            assert contexts[0].run_id != contexts[1].run_id
            assert model.opens == 1
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('first_party', [False, True])
async def test_cancellation_drains_streams_and_closes_connection(first_party: bool):
    before = asyncio.all_tasks()
    started, cleaned = asyncio.Event(), asyncio.Event()
    model = CountedModel(DuplexConnection())

    async def microphone() -> AsyncIterator[bytes]:
        with anyio.CancelScope():
            try:
                started.set()
                await asyncio.Event().wait()
                yield b'never sent'
            finally:
                cleaned.set()

    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:
            producers: list[asyncio.Task[None]] = []

            async def execute() -> None:
                async with live.run() as run:
                    producers.append(asyncio.create_task(run.send_audio(microphone())))
                    await asyncio.Event().wait()

            task = asyncio.create_task(execute())
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await started.wait()
                if first_party:
                    owner.cancel()
                else:
                    task.cancel()
                with pytest.raises(RunCancelled if first_party else asyncio.CancelledError):
                    await task
            assert cleaned.is_set()
            await producers[0]
            assert model.closes == 1
            with pytest.raises(UserError, match='must not be closed'):
                async with live.run():
                    pytest.fail('A cancelled connection cannot be reused')
        await owner.run('ordinary afterward')
    assert asyncio.all_tasks() == before


async def test_cancelled_pull_keeps_its_unconsumed_event():
    entered, release = asyncio.Event(), asyncio.Event()
    produced: list[AgentStreamEvent] = []

    class PausedWrapper(AbstractCapability[None]):
        async def wrap_run_event_stream(
            self, ctx: RunContext[None], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterator[AgentStreamEvent]:
            entered.set()
            await release.wait()
            async for event in stream:
                produced.append(event)
                yield event

    agent = Agent(TestModel(), deps_type=type(None), capabilities=[PausedWrapper()])
    async with agent.session() as owner:
        async with owner.realtime(CountedModel(DuplexConnection())).connect() as live:
            async with live.run() as run:
                iterator = aiter(run)
                first = asyncio.ensure_future(anext(iterator))
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await entered.wait()
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                second = asyncio.ensure_future(anext(iterator))
                await run.send('hello')
                await run.wait_for_reply()
                release.set()
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    event = await second
                assert event is produced[0]


async def test_cancelled_run_aborts_blocked_audio_send_before_draining_producer():
    before = asyncio.all_tasks()
    sending, send_cancelled, release_send = asyncio.Event(), asyncio.Event(), asyncio.Event()
    source_closed = asyncio.Event()

    class BlockedAudioConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            sending.set()
            try:
                await release_send.wait()
            except asyncio.CancelledError:
                send_cancelled.set()
                raise

    async def microphone() -> AsyncIterator[bytes]:
        with anyio.CancelScope():
            try:
                yield b'\x00\x00' * 100
                await asyncio.Event().wait()
            finally:
                source_closed.set()

    model = CountedModel(BlockedAudioConnection())
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:
            producers: list[asyncio.Task[None]] = []

            async def execute() -> None:
                async with live.run() as run:
                    producers.append(asyncio.create_task(run.send_audio(microphone())))
                    await asyncio.Event().wait()

            task = asyncio.create_task(execute())
            try:
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await sending.wait()
                    task.cancel()
                    await send_cancelled.wait()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert source_closed.is_set()
                await producers[0]
                assert model.closes == 1
            finally:
                # The regression must report a failure rather than itself hanging during teardown.
                release_send.set()
                await asyncio.gather(task, *producers, return_exceptions=True)
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('operation', ['context', 'audio', 'clear', 'interrupt'])
async def test_abort_drains_in_flight_run_calls(operation: str):
    before = asyncio.all_tasks()
    sending, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls: list[asyncio.Task[None]] = []

    class BlockedConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            sending.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    model = CountedModel(BlockedConnection())
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:

            async def execute() -> None:
                async with live.run() as run:
                    if operation == 'context':
                        call = run.send('context only', respond=False)
                    elif operation == 'audio':
                        call = run.send_audio(b'\x00\x00' * 100)
                    elif operation == 'clear':
                        call = run.clear_audio()
                    else:
                        call = run.interrupt()
                    calls.append(asyncio.create_task(call))
                    await asyncio.Event().wait()

            task = asyncio.create_task(execute())
            try:
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await sending.wait()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert cancelled.is_set()
                assert calls[0].done()
                with pytest.raises(asyncio.CancelledError):
                    await calls[0]
                assert model.closes == 1
            finally:
                release.set()
                await asyncio.gather(task, *calls, return_exceptions=True)
    assert asyncio.all_tasks() == before


async def test_normal_exit_waits_for_in_flight_control_call():
    sending, release, exiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls: list[asyncio.Task[None]] = []

    class BlockedConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            sending.set()
            await release.wait()
            await super().send(content)

    model = CountedModel(BlockedConnection())
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:

            async def execute() -> None:
                async with live.run() as run:
                    calls.append(asyncio.create_task(run.clear_audio()))
                    await sending.wait()
                    exiting.set()

            task = asyncio.create_task(execute())
            try:
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await exiting.wait()
                    # A checkpoint after the driver's exit starts must still belong to this run.
                    await anyio.lowlevel.checkpoint()
                    assert owner.state.active_run_id is not None
                    release.set()
                    await task
                assert calls[0].done()
                await calls[0]
                async with live.run() as second:
                    await second.send('second')
            finally:
                release.set()
                await asyncio.gather(task, *calls, return_exceptions=True)


async def test_explicit_run_close_settles_without_spurious_boundary_error():
    model = CountedModel(DuplexConnection())
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:
            async with live.run() as run:
                await run.send('hello')
                await run.wait_for_reply()
                await run.close()
            assert run.result is not None
            assert run.result.output == 'reply to hello'
            with pytest.raises(UserError, match='closed'):
                async with live.run():
                    pytest.fail('Explicit close must revoke the attachment')
        assert model.closes == 1


async def test_image_retention_preserves_new_message_boundary_across_runs():
    image = BinaryImage(data=b'image', media_type='image/png')
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(DuplexConnection())).connect(retain_images_max=1) as live:
            async with live.run() as first:
                await first.send(image)
                await first.send('first')
            saved = first.all_messages()
            async with live.run() as second:
                await second.send(image)
                await second.send('second')
            assert second.result is not None
            assert len(second.new_messages()) == 3
            assert second.new_messages() == second.result.new_messages()
            assert all(message.run_id == second.result.run_id for message in second.new_messages())
            assert first.all_messages() == saved
            assert len(owner.conversation.messages) == 5


@pytest.mark.parametrize('explicit_close', [False, True])
async def test_connection_final_usage_updates_owner_without_mutating_results(explicit_close: bool):
    class FinalUsageConnection(DuplexConnection):
        async def _end_session(self) -> AsyncIterator[SessionUsage]:
            yield SessionUsage(RequestUsage(input_tokens=11, audio_seconds=1.5))

    class AccountExtraUsage(AbstractCapability[None]):
        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            result.usage.input_tokens += 100
            return result

    async with Agent(TestModel(), deps_type=type(None), capabilities=[AccountExtraUsage()]).session() as owner:
        async with owner.realtime(CountedModel(FinalUsageConnection())).connect() as live:
            async with live.run() as run:
                await run.send('hello')
                await run.wait_for_reply()
                if explicit_close:
                    await run.close()
            assert run.result is not None
            saved = run.result.usage.input_tokens
        assert owner.conversation.usage.input_tokens == 111
        assert owner.conversation.usage.audio_seconds == 1.5
        assert run.result.usage.input_tokens == saved
        assert run.usage.input_tokens == saved


@pytest.mark.parametrize('from_tool', [False, True])
async def test_explicit_close_while_run_exit_is_waiting(from_tool: bool):
    exiting, release_tool = asyncio.Event(), asyncio.Event()
    handles: list[RealtimeRun] = []

    class ToolConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            if isinstance(content, str):
                self.events.put_nowait(ToolCall(tool_name='end_call', tool_call_id='end-call', args='{}'))
                self.events.put_nowait(ResponseDone())

    agent = Agent(TestModel(), deps_type=type(None))

    @agent.tool
    async def end_call(ctx: RunContext[None]) -> None:
        await release_tool.wait()
        assert isinstance(ctx.realtime_session, RealtimeRun)
        await ctx.realtime_session.close()

    model = CountedModel(ToolConnection() if from_tool else DuplexConnection())
    async with agent.session() as owner:
        async with owner.realtime(model).connect() as live:

            async def execute() -> None:
                async with live.run() as run:
                    handles.append(run)
                    if from_tool:
                        await run.send('close via tool')
                    else:
                        # An input transcript that won't arrive keeps normal exit waiting.
                        await run.send_audio(b'\x00\x00' * 100)
                        await run.commit_audio()
                    exiting.set()

            task = asyncio.create_task(execute())
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await exiting.wait()
                assert not task.done()
                if from_tool:
                    release_tool.set()
                else:
                    await handles[0].close()
                await task
            assert handles[0].result is not None
            assert model.closes == 1
