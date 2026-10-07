"""Shared session ownership, using controlled duplex events rather than provider timing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, AgentRunResult, RunContext, UserError
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import BinaryImage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import RealtimeSession
from pydantic_ai.realtime.codec import ToolCall, ToolResult

from .test_session import BlockingRealtimeConnection, FakeRealtimeModel

READINESS_WAIT_TIMEOUT = 10


async def test_realtime_run_shares_owner_history_operations_and_dependencies():
    connection = BlockingRealtimeConnection([ToolCall(tool_name='value', tool_call_id='call', args='{}')])
    model = FakeRealtimeModel(connection, profile={'supports_session_seeding': True})
    contexts: list[RunContext[str]] = []
    agent = Agent(TestModel(call_tools=[]), deps_type=str)

    @agent.tool
    def value(ctx: RunContext[str]) -> str:
        contexts.append(ctx)
        return ctx.deps

    async with agent.session(deps='shared') as owner:
        first = await owner.run('before live')
        async with owner.realtime(model).session() as live:
            with pytest.raises(UserError, match='one run at a time'):
                await owner.run('overlap')
            with pytest.raises(UserError, match='one run at a time'):
                async with owner.realtime(model).session():
                    pass
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async for _ in live:
                    if any(isinstance(sent, ToolResult) for sent in connection.sent):
                        break
            checkpoint = owner.state
            assert checkpoint.active_run_id != first.run_id
            assert checkpoint.operations[0].execution == 'completed'
            assert checkpoint.conversation.conversation_id == first.conversation_id
        assert contexts[0].deps == 'shared'
        assert owner.state.active_run_id is None
        assert any(
            isinstance(part, ToolReturnPart) and part.content == 'shared'
            for message in owner.conversation.messages
            for part in message.parts
        )
        with pytest.raises(UserError, match='run has ended'):
            contexts[0].enqueue('stale')
        last = await owner.run('after live')
        assert last.conversation_id == first.conversation_id
        assert last.run_id not in (first.run_id, contexts[0].run_id)


@pytest.mark.parametrize('initial', [False, True])
async def test_realtime_owner_enqueue_wakes_driver(initial: bool):
    connection = BlockingRealtimeConnection([])
    model = FakeRealtimeModel(connection)
    agent = Agent()
    async with agent.session() as owner:
        if initial:
            owner.enqueue('queued from owner')
        async with owner.realtime(model).session() as live:
            if not initial:
                owner.enqueue('queued from owner')
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async for _ in live:
                    if connection.sent:
                        break
            assert connection.sent == ['queued from owner']
        assert any(
            isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) and part.content == 'queued from owner' for part in message.parts)
            for message in owner.conversation.messages
        )


async def test_realtime_invalid_inbox_transfer_preserves_all_pending_input():
    connection = BlockingRealtimeConnection([])
    image = BinaryImage(data=b'image', media_type='image/png')
    async with Agent(TestModel()).session() as owner:
        owner.enqueue('text first')
        owner.enqueue(image)
        pending = owner.state.pending
        with pytest.raises(UserError, match='text'):
            async with owner.realtime(FakeRealtimeModel(connection)).session():
                pytest.fail('Invalid input must fail before yielding the session')
        assert owner.state.pending == pending
        assert owner.state.active_run_id is None
        assert connection.sent == []
        result = await owner.run('continue with retained input')
        assert not owner.state.pending
        assert any(
            isinstance(part, UserPromptPart) and not isinstance(part.content, str) and image in part.content
            for message in result.all_messages()
            for part in message.parts
        )


@pytest.mark.parametrize('external', [False, True])
async def test_realtime_cancellation_releases_owner_and_drains_tools(external: bool):
    before = asyncio.all_tasks()
    started = asyncio.Event()
    stopped = asyncio.Event()
    agent = Agent(TestModel(call_tools=[]))

    @agent.tool_plain
    async def hold() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return 'unreachable'  # pragma: no cover

    connection = BlockingRealtimeConnection([ToolCall(tool_name='hold', tool_call_id='hold', args='{}')])
    async with agent.session() as owner:

        async def talk() -> None:
            async with owner.realtime(FakeRealtimeModel(connection)).session() as live:
                async for _ in live:
                    pass

        task = asyncio.create_task(talk())
        try:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await started.wait()
            if external:
                task.cancel()
            else:
                owner.cancel()
            with pytest.raises(asyncio.CancelledError if external else RunCancelled):
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await task
            assert stopped.is_set()
            assert owner.state.active_run_id is None
            assert owner.state.operations[0].execution == 'interrupted'
            assert any(
                isinstance(part, ToolReturnPart) and part.tool_call_id == 'hold'
                for message in owner.conversation.messages
                for part in message.parts
            )
            await owner.run('continue')
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert not (asyncio.all_tasks() - before)


@pytest.mark.parametrize('recover', [False, True])
async def test_realtime_hook_result_is_returned_to_owner(recover: bool):
    class FinishRun(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            if recover:
                return await handler()
            return AgentRunResult(output='short-circuit')

        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
            assert isinstance(error, ValueError)
            return AgentRunResult(output='recovered')

    agent = Agent(TestModel(), deps_type=type(None))
    async with agent.session() as owner:
        await owner.run('initial')
        model = FakeRealtimeModel(BlockingRealtimeConnection([]), profile={'supports_session_seeding': True})
        async with owner.realtime(model, capabilities=[FinishRun()]).session() as live:
            assert owner.state.active_run_id is not None
            if recover:
                await live.send('before error', respond=False)
                raise ValueError('caller failed')
            assert live.closed
        assert live.result is not None
        assert live.result.output == ('recovered' if recover else 'short-circuit')
        assert owner.conversation == live.result.conversation
        assert owner.state.active_run_id is None
        await owner.run('continue')


async def test_realtime_owner_cancel_reaches_before_run_hook():
    agent = Agent(TestModel(), deps_type=type(None))
    async with agent.session() as owner:

        class CancelBeforeRun(AbstractCapability[None]):
            async def before_run(self, ctx: RunContext[None]) -> None:
                owner.cancel()
                await asyncio.Event().wait()

        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            with pytest.raises(RunCancelled):
                async with owner.realtime(
                    FakeRealtimeModel(BlockingRealtimeConnection([])), capabilities=[CancelBeforeRun()]
                ).session():
                    pytest.fail('Cancelled preparation must not enter the caller body')
        assert owner.state.active_run_id is None
        await owner.run('continue')


@pytest.mark.parametrize('discard', [None, 'message_history', 'conversation_id', 'usage'])
async def test_realtime_wrapper_must_preserve_owner_conversation(discard: str | None):
    class Wrapped(WrapperAgent[None, str]):
        @asynccontextmanager
        async def _open_realtime_session(self, *args: Any, **kwargs: Any) -> AsyncGenerator[RealtimeSession]:
            if discard is not None:
                kwargs[discard] = None
            async with self.wrapped._open_realtime_session(*args, **kwargs) as live:
                yield live

    agent = Wrapped(Agent(TestModel()))
    async with agent.session() as owner:
        await owner.run('must remain')
        original = owner.conversation
        model = FakeRealtimeModel(BlockingRealtimeConnection([]), profile={'supports_session_seeding': True})
        if discard is None:
            async with owner.realtime(model).session():
                pass
        else:
            with pytest.raises(UserError, match='did not delegate'):
                async with owner.realtime(model).session():
                    pass
        assert owner.conversation == original
        assert owner.state.active_run_id is None


async def test_realtime_failed_attachment_can_recover_without_consuming_inbox():
    class Recover(AbstractCapability[None]):
        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
            assert isinstance(error, UserError)
            return AgentRunResult(output='recovered')

    async with Agent(TestModel(), deps_type=type(None)).session() as owner:
        owner.enqueue(BinaryImage(data=b'image', media_type='image/png'))
        pending = owner.state.pending
        async with owner.realtime(
            FakeRealtimeModel(BlockingRealtimeConnection([])), capabilities=[Recover()]
        ).session() as live:
            assert live.closed
            assert live.result is not None
            assert live.result.output == 'recovered'
        assert owner.state.pending == pending
        assert owner.state.active_run_id is None
        await owner.run('continue')


@pytest.mark.parametrize('second', ['second', BinaryImage(data=b'image', media_type='image/png')])
async def test_realtime_short_circuit_keeps_owner_inbox_order(second: str | BinaryImage):
    class Skip(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            return AgentRunResult(output='skipped')

    async with Agent(TestModel(), deps_type=type(None)).session() as owner:
        first_id = owner.enqueue('first')
        async with owner.realtime(FakeRealtimeModel(BlockingRealtimeConnection([])), capabilities=[Skip()]).session():
            second_id = owner.enqueue(second)
            assert [pending.enqueue_id for pending in owner.state.pending] == [first_id, second_id]
        assert [pending.enqueue_id for pending in owner.state.pending] == [first_id, second_id]
        result = await owner.run('continue')
        prompts = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        assert prompts == ['continue', 'first', second if isinstance(second, str) else [second]]
