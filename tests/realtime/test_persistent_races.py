"""Run-boundary races ordered by barriers rather than provider timing."""

from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack

import anyio
import pytest

import pydantic_ai.realtime._session as realtime_session_module
from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai._enqueue import PendingMessage
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime.codec import RealtimeInput, ResponseDone, ToolCall, ToolResult

from .test_persistent_session import READINESS_WAIT_TIMEOUT, CountedModel, DuplexConnection


async def test_enqueue_rechecks_closed_lease_after_worker_validation(monkeypatch: pytest.MonkeyPatch):
    # Validation and the queue's atomic append straddle a worker-thread boundary. Pause only the
    # validator, keeping public enqueue and normal run exit responsible for their real lease checks.
    before = asyncio.all_tasks()
    validating = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    validate = realtime_session_module._pending_message_text  # pyright: ignore[reportPrivateUsage]

    def paused_validation(pending: PendingMessage) -> str:
        text = validate(pending)
        loop.call_soon_threadsafe(validating.set)
        assert release.wait(READINESS_WAIT_TIMEOUT)
        return text

    connection = DuplexConnection()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                stack = AsyncExitStack()
                run = await stack.enter_async_context(live.run())
                run.stream_transcripts()
                monkeypatch.setattr(realtime_session_module, '_pending_message_text', paused_validation)
                worker = asyncio.create_task(asyncio.to_thread(run.enqueue, 'too late'))
                try:
                    async with stack:
                        await validating.wait()
                finally:
                    release.set()
                    await asyncio.gather(worker, return_exceptions=True)
                with pytest.raises(UserError, match='run has ended'):
                    await worker
                assert not owner.state.pending
                assert connection.sent == []
    assert asyncio.all_tasks() == before


async def test_late_tool_delivery_does_not_duplicate_completed_result():
    before = asyncio.all_tasks()
    sending = asyncio.Event()
    cancelled = asyncio.Event()

    class LateDeliveryConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            if isinstance(content, str):
                self.events.put_nowait(ToolCall(tool_name='effect', tool_call_id='call', args='{}'))
                self.events.put_nowait(ResponseDone())
            else:
                assert isinstance(content, ToolResult)
                sending.set()
                try:
                    await asyncio.Future[None]()
                except asyncio.CancelledError:
                    # The transport completes a send despite shutdown cancellation. The logical tool
                    # result was already recorded; neither close nor this return may record it again.
                    cancelled.set()

    calls: list[str] = []
    agent = Agent(TestModel())

    @agent.tool_plain
    def effect() -> str:
        calls.append('effect')
        return 'completed effect'

    connection = LateDeliveryConnection()
    model = CountedModel(connection)
    async with agent.session() as owner:
        async with owner.realtime(model).connect() as live:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with live.run() as run:
                    await run.send('start')
                    await sending.wait()
                    await run.close()
            assert cancelled.is_set()
            assert model.closes == 1
        returns = [
            part
            for message in owner.conversation.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert [(part.content, part.outcome) for part in returns] == [('completed effect', 'success')]
        assert calls == ['effect']
        assert len(owner.state.operations) == 1
        assert owner.state.operations[0].execution == 'completed'
        assert owner.state.operations[0].delivery == 'uncertain'
    assert asyncio.all_tasks() == before


async def test_attachment_closing_during_preparation_never_opens_transport():
    before = asyncio.all_tasks()
    preparing = asyncio.Event()
    model = CountedModel(DuplexConnection())
    agent = Agent(TestModel(), deps_type=type(None))

    @agent.instructions
    async def paused_instructions(ctx: RunContext[None]) -> str:
        preparing.set()
        try:
            return await asyncio.Future[str]()
        except asyncio.CancelledError:
            # Instructions are resolved after lifecycle admission. A callback swallowing cancellation
            # must still not enter the transport belonging to an attachment that is closing.
            return 'late instructions'

    async with agent.session() as owner:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with owner.realtime(model).connect() as live:

                async def start_run() -> None:
                    async with live.run():
                        pass

                task = asyncio.create_task(start_run())
                await preparing.wait()
            (error,) = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(error, UserError)
        assert str(error) == 'This realtime connection has closed.'
        assert model.opens == model.closes == 0
        assert owner.state.active_run_id is None
    assert asyncio.all_tasks() == before
