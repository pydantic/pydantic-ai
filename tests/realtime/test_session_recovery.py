"""Checkpoint reconciliation against deterministic duplex tool traffic."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack

import anyio
import pytest

from pydantic_ai import Agent, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime.codec import OutputTranscript, RealtimeInput, ResponseDone, ToolCall, ToolResult
from pydantic_ai.session import SessionStateTypeAdapter

from .test_persistent_session import CountedModel, DuplexConnection

READINESS_WAIT_TIMEOUT = 10


@pytest.mark.parametrize('persistent', [False, True])
@pytest.mark.parametrize('after_cancel', [False, True])
@pytest.mark.parametrize('delayed_boundary', [False, True])
async def test_recover_realtime_tool_after_an_earlier_response(
    persistent: bool, after_cancel: bool, delayed_boundary: bool
):
    before = asyncio.all_tasks()
    entered = asyncio.Event()

    class ToolConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            if isinstance(content, str):
                self.events.put_nowait(
                    ToolCall(
                        tool_name='work',
                        tool_call_id=content,
                        args=json.dumps({'label': content}),
                        response_usage_follows=delayed_boundary,
                    )
                )
                if not delayed_boundary or content == 'first':
                    self.events.put_nowait(ResponseDone())
            else:
                assert isinstance(content, ToolResult)
                self.events.put_nowait(OutputTranscript('done', output_text=True, is_final=True))
                self.events.put_nowait(ResponseDone())

    agent = Agent(TestModel(call_tools=[]))

    @agent.tool_plain
    async def work(label: str) -> str:
        if label == 'second':
            entered.set()
            await asyncio.Event().wait()
        return label

    async with agent.session() as owner:
        async with AsyncExitStack() as stack:
            model = CountedModel(ToolConnection())
            if persistent:
                live = await stack.enter_async_context(owner.realtime(model).connect())
                async with live.run() as first:
                    await first.send('first')
                run = await stack.enter_async_context(live.run())
            else:
                run = await stack.enter_async_context(owner.realtime(model).session())
                await run.send('first')
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await run.wait_for_reply()
            await run.send('second')
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await entered.wait()
            if after_cancel:
                await run.close()
            checkpoint = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(owner.state))
            await run.close()
        original = SessionStateTypeAdapter.dump_json(checkpoint)
        second = next(op for op in checkpoint.operations if op.call.tool_call_id == 'second')
        assert second.call_index == 0
        verified = ModelRequest(
            parts=[
                ToolReturnPart('work', 'verified external result', tool_call_id='second'),
                UserPromptPart('external evidence'),
            ]
        )
        recovered = checkpoint.recover(
            tool_results={second.operation_id: verified},
            deliveries={
                op.operation_id: 'committed' for op in checkpoint.operations if op.call.tool_call_id == 'first'
            },
            abandon_run=True,
        )
        second_response = next(
            i
            for i, message in enumerate(recovered.conversation.messages)
            if isinstance(message, ModelResponse) and any(call.tool_call_id == 'second' for call in message.tool_calls)
        )
        request = recovered.conversation.messages[second_response + 1]
        assert isinstance(request, ModelRequest)
        assert request.parts == verified.parts
        assert checkpoint.operations[-1].execution in ('running', 'interrupted')
        assert SessionStateTypeAdapter.dump_json(checkpoint) == original
        if after_cancel:
            for message in checkpoint.conversation.messages:
                if isinstance(message, ModelRequest):
                    message.parts = [
                        ToolReturnPart('work', 'a real conflicting result', tool_call_id='second')
                        if isinstance(part, ToolReturnPart) and part.tool_call_id == 'second'
                        else part
                        for part in message.parts
                    ]
            conflicting = SessionStateTypeAdapter.dump_json(checkpoint)
            with pytest.raises(UserError, match='different result'):
                checkpoint.recover(
                    tool_results={second.operation_id: verified},
                    deliveries={
                        op.operation_id: 'committed' for op in checkpoint.operations if op.call.tool_call_id == 'first'
                    },
                    abandon_run=True,
                )
            assert SessionStateTypeAdapter.dump_json(checkpoint) == conflicting
        response = recovered.conversation.messages[second_response]
        assert isinstance(response, ModelResponse)
        if second.response_timestamp is not None:
            assert response.timestamp == second.response_timestamp
        async with agent.session(state=recovered) as resumed:
            await resumed.run('continue without repeating the effect')
    assert asyncio.all_tasks() == before


async def test_persistent_run_enqueue_is_delivered_and_revoked():
    sent = asyncio.Event()

    class AcknowledgedConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            await super().send(content)
            sent.set()

    connection = AcknowledgedConnection()
    async with Agent(TestModel()).session() as owner:
        async with owner.realtime(CountedModel(connection)).connect() as live:
            async with live.run() as first:
                enqueue_id = first.enqueue('queued input', priority='asap')
                assert enqueue_id is not None
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await sent.wait()
            async with live.run() as second:
                with pytest.raises(UserError, match='run has ended'):
                    first.enqueue('stale input')
                await second.send('next run')
            assert connection.sent == ['queued input', 'next run']


async def test_recover_parallel_realtime_tools_preserves_separate_results():
    before = asyncio.all_tasks()
    entered = {name: asyncio.Event() for name in ('a', 'b')}

    class ParallelConnection(DuplexConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            assert isinstance(content, str)
            for name in entered:
                self.events.put_nowait(
                    ToolCall(
                        tool_name='work',
                        tool_call_id=name,
                        args=json.dumps({'label': name}),
                        response_usage_follows=True,
                    )
                )
            self.events.put_nowait(ResponseDone())

    agent = Agent(TestModel(call_tools=[]))

    @agent.tool_plain
    async def work(label: str) -> str:
        entered[label].set()
        return await asyncio.Future[str]()

    async with agent.session() as owner:
        async with owner.realtime(CountedModel(ParallelConnection())).session() as run:
            await run.send('start')
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                for event in entered.values():
                    await event.wait()
            await run.close()
        checkpoint = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(owner.state))
    verified = {
        operation.operation_id: ModelRequest(
            parts=[
                ToolReturnPart(
                    'work', f'verified {operation.call.tool_call_id}', tool_call_id=operation.call.tool_call_id
                ),
                UserPromptPart(f'evidence {operation.call.tool_call_id}'),
            ]
        )
        for operation in checkpoint.operations
    }
    recovered = checkpoint.recover(tool_results=verified)
    returns = [
        part
        for message in recovered.conversation.messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert [(part.tool_call_id, part.content, part.outcome) for part in returns] == [
        ('a', 'verified a', 'success'),
        ('b', 'verified b', 'success'),
    ]
    requests = [
        message
        for message in recovered.conversation.messages
        if isinstance(message, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in message.parts)
    ]
    assert [request.parts for request in requests] == [request.parts for request in verified.values()]
    assert asyncio.all_tasks() == before
