"""Public persistent-run boundaries with controlled transport and cleanup ordering."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator

import anyio
import pytest

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, SpeechPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import RealtimeError, RealtimeRun, TranscriptUpdate
from pydantic_ai.realtime._lifecycle import TaggedEvent
from pydantic_ai.realtime.codec import (
    AudioDelta,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    TextContext,
    ToolCall,
)
from pydantic_ai.usage import RequestUsage

from .test_persistent_session import READINESS_WAIT_TIMEOUT, CountedModel, DuplexConnection, IdentifiedConnection


@pytest.mark.parametrize('delta', [False, True])
async def test_persistent_media_views_and_playback(delta: bool):
    before = asyncio.all_tasks()

    class MediaConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            assert content == 'hello'
            self.sent.append(content)
            self.events.put_nowait(AudioDelta(b'\x00\x00' * 16))
            self.events.put_nowait(OutputTranscript('hello back', is_final=True))
            self.events.put_nowait(ResponseDone())

    model = CountedModel(MediaConnection())
    model.profile['audio_input_sample_rate'] = 16000
    model.profile['audio_output_sample_rate'] = 24000
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with live.run() as run:
                    assert run.profile == model.profile
                    assert run.audio_input_sample_rate == 16000
                    assert run.audio_output_sample_rate == 24000
                    audio = run.stream_audio()
                    transcripts = run.stream_transcripts(delta=delta)
                    await run.send('hello')
                    assert await anext(audio) == b'\x00\x00' * 16
                    transcript = await anext(transcripts)
                    if delta:
                        assert isinstance(transcript, TranscriptUpdate)
                        assert transcript.delta == transcript.transcript == 'hello back'
                    else:
                        assert transcript == SpeechPart(speaker='assistant', transcript='hello back')
                    next_audio = asyncio.ensure_future(anext(audio))
                    await run.wait_for_reply()
                    await run.wait_for_playback()
                    assert await run.interrupt(played_bytes=32) is False
                with pytest.raises(StopAsyncIteration):
                    await next_audio
                # A started iterator stays exhausted, rather than reading a later run.
                with pytest.raises(StopAsyncIteration):
                    await anext(audio)
                async with live.run():
                    with pytest.raises(StopAsyncIteration):
                        await anext(audio)
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('hang_up', [False, True])
async def test_closing_unstarted_run_does_not_restart_receive_pump(hang_up: bool):
    before = asyncio.all_tasks()
    model = CountedModel(DuplexConnection())
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:
            async with live.run() as run:
                if hang_up:
                    await run.hang_up()
                else:
                    await run.close()
                assert run.closed
            assert model.closes == 1
            with pytest.raises(UserError, match='must not be closed'):
                async with live.run():
                    pass
    assert model.closes == 1
    assert asyncio.all_tasks() == before


async def test_persistent_rejects_changed_instructions_without_reconnecting():
    agent = Agent(TestModel(), deps_type=str)

    @agent.instructions
    def instructions(ctx: RunContext[str]) -> str:
        return ctx.deps

    model = CountedModel(DuplexConnection())
    async with agent.session(deps='first') as owner:
        async with owner.realtime(model).connect() as live:
            async with live.run() as first:
                await first.send('hello')
            with pytest.raises(UserError, match='cannot change its instructions'):
                async with live.run(deps='second'):
                    pass
            assert model.opens == 1
            assert first.result is not None
            assert first.result.output == 'reply to hello'


async def test_event_wrapper_cleanup_error_reaches_run_exit():
    before = asyncio.all_tasks()
    closed: list[str] = []

    class CleanupFailure(AbstractCapability[None]):
        async def wrap_run_event_stream(
            self, ctx: RunContext[None], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterator[AgentStreamEvent]:
            try:
                yield await anext(aiter(stream))
            finally:
                closed.append('wrapper')
                raise ValueError('wrapper cleanup failed')

    model = CountedModel(DuplexConnection())
    agent = Agent(TestModel(), deps_type=type(None), capabilities=[CleanupFailure()])
    async with agent.session() as owner:
        async with owner.realtime(model).connect() as live:
            with pytest.raises(ValueError, match='wrapper cleanup failed'):
                async with live.run() as run:
                    await run.send('hello')
                    await anext(aiter(run))
            assert closed == ['wrapper']
            assert model.closes == 1
    assert asyncio.all_tasks() == before


async def test_close_rejects_new_calls_while_prior_send_drains():
    before = asyncio.all_tasks()
    sending, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SlowCleanupConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            sending.set()
            try:
                await asyncio.Event().wait()
            finally:
                with anyio.CancelScope(shield=True):
                    cleaning.set()
                    await release.wait()

    connection = SlowCleanupConnection()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            async with live.run() as run:
                sender = asyncio.create_task(run.send('first', respond=False))
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await sending.wait()
                    closer = asyncio.create_task(run.close())
                    try:
                        await cleaning.wait()
                        assert not run.closed
                        with pytest.raises(UserError, match='run has ended'):
                            await run.send('too late')
                    finally:
                        release.set()
                        await closer
                    with pytest.raises(asyncio.CancelledError):
                        await sender
                assert connection.sent == [TextContext('first')]
    assert asyncio.all_tasks() == before


async def test_normal_exit_rejects_new_audio_while_source_cleanup_drains():
    before = asyncio.all_tasks()
    started, cleaning, release, leave = (asyncio.Event() for _ in range(4))

    async def microphone() -> AsyncIterator[bytes]:
        try:
            started.set()
            yield await asyncio.Future[bytes]()
        finally:
            with anyio.CancelScope(shield=True):
                cleaning.set()
                await release.wait()

    connection = DuplexConnection()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            handles: list[RealtimeRun] = []
            producers: list[asyncio.Task[None]] = []

            async def execute() -> None:
                async with live.run() as run:
                    handles.append(run)
                    producers.append(asyncio.create_task(run.send_audio(microphone())))
                    await leave.wait()

            task = asyncio.create_task(execute())
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await started.wait()
                leave.set()
                try:
                    await cleaning.wait()
                    with pytest.raises(UserError, match='no longer accepting audio streams'):
                        await handles[0].send_audio(microphone())
                finally:
                    release.set()
                    await task
                    await producers[0]
            assert connection.sent == []
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('fatal', [False, True])
async def test_receive_end_reaches_normal_run_exit(fatal: bool):
    before = asyncio.all_tasks()
    received = asyncio.Event()

    class EndsMidReply(DuplexConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            await self.events.get()
            yield OutputTranscript('partial', output_text=True)
            received.set()
            if fatal:
                raise RealtimeError(model_name='fake-realtime', message='receive failed')

    model = CountedModel(EndsMidReply())
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:
            with pytest.raises(
                RealtimeError if fatal else UserError,
                match='receive failed' if fatal else 'ended before the run could settle',
            ):
                async with live.run() as run:
                    await run.send('hello')
                    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                        await received.wait()
            assert model.closes == 1
        response = owner.conversation.messages[-1]
        assert isinstance(response, ModelResponse)
        assert response.state == 'interrupted'
    assert asyncio.all_tasks() == before


async def test_idle_eof_rejects_resume_without_reconnecting():
    ended = asyncio.Event()

    class EndsWhileIdle(DuplexConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            while True:
                event = await self.events.get()
                if isinstance(event, SessionUsage):
                    ended.set()
                    return
                yield event

    connection = EndsWhileIdle()
    model = CountedModel(connection)
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(model).connect() as live:
            async with live.run() as first:
                await first.send('first')
            assert first.result is not None
            saved = first.all_messages()
            connection.events.put_nowait(SessionUsage(RequestUsage()))
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await ended.wait()
            with pytest.raises(UserError, match='connection has ended'):
                async with live.run():
                    pass
            assert first.all_messages() == saved
            assert first.result.output == 'reply to first'
            assert model.opens == 1


async def test_stale_tagged_frames_do_not_change_run_history_or_usage():
    class StaleConnection(IdentifiedConnection):
        async def send(self, content: RealtimeInput) -> None:
            stale: list[TaggedEvent] = [
                (OutputTranscript('stale reply', is_final=True, output_text=True), True),
                (ToolCall(tool_name='forbidden', tool_call_id='stale', args='{}'), True),
                (SessionUsage(RequestUsage(input_tokens=100)), True),
            ]
            self.frames.put_nowait(stale)
            await super().send(content)

    agent = Agent(TestModel())

    @agent.tool_plain
    def forbidden() -> None:
        assert False, 'Stale tool calls must not execute'

    model = CountedModel(StaleConnection())
    async with agent.session() as owner:
        async with owner.realtime(model).connect() as live:
            for run_id in ('RUN-A', 'RUN-B'):
                async with live.run(run_id=run_id) as run:
                    await run.send(run_id)
                assert run.result is not None
                assert run.result.output == f'reply to {run_id}'
                assert run.result.usage.input_tokens == 0
                assert run.result.usage.tool_calls == 0
            assert [m.run_id for m in owner.conversation.messages if isinstance(m, ModelResponse)] == ['RUN-A', 'RUN-B']


async def test_merged_requests_settle_before_next_persistent_run():
    answer = asyncio.Event()
    first_response = asyncio.Event()

    class MergedConnection(DuplexConnection):
        merged = 0

        def _take_merged_response_requests(self) -> int:
            merged, self.merged = self.merged, 0
            return merged

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield await self.events.get()
            first_response.set()
            await answer.wait()
            yield ResponseDone()
            self.merged = 1
            yield OutputTranscript('Spain and Italy', output_text=True, is_final=True)
            yield ResponseDone()
            # Discard the scripted per-prompt replies superseded by the merged response.
            self.events = asyncio.Queue()
            async for event in super().__aiter__():
                yield event

    connection = MergedConnection()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with live.run() as first:
                    await first.send('France')
                    await first_response.wait()
                    await first.send('Spain')
                    await first.send('Italy')
                    answer.set()
                    await first.wait_for_reply()
                assert first.result is not None
                assert first.result.output == 'Spain and Italy'
                async with live.run() as second:
                    await second.send('next')
                assert second.result is not None
                assert second.result.output == 'reply to next'


async def test_normal_exit_discards_audio_buffered_before_source_cleanup():
    before = asyncio.all_tasks()
    buffered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def microphone() -> AsyncIterator[bytes]:
        try:
            # Wake the run owner before waking the source's consumer with this chunk.
            buffered.set()
            yield b'not sent after the boundary'
        finally:
            with anyio.CancelScope(shield=True):
                cleaning.set()
                await release.wait()

    connection = DuplexConnection()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            producers: list[asyncio.Task[None]] = []

            async def execute() -> None:
                async with live.run() as run:
                    producers.append(asyncio.create_task(run.send_audio(microphone())))
                    await buffered.wait()

            task = asyncio.create_task(execute())
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                try:
                    await cleaning.wait()
                    assert connection.sent == []
                finally:
                    release.set()
                    await task
                    await producers[0]
            assert connection.sent == []
    assert asyncio.all_tasks() == before
