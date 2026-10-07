"""Persistent connection boundaries with deterministic duplex traffic, not provider timing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager

import anyio
import pytest

from pydantic_ai import Agent, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import RealtimeModelSettings
from pydantic_ai.realtime.codec import (
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
)

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
