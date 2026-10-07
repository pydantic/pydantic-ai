"""Effect/delivery boundaries need deterministic failures, not a provider cassette's happy path."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
from copy import deepcopy
from typing import Literal

import anyio
import pytest

from pydantic_ai import Agent, ModelRetry, RunContext, SessionStateTypeAdapter, UserError
from pydantic_ai._operations import (
    CompleteTool,
    InterruptTool,
    LoseDelivery,
    ObserveDelivery,
    ReconcileDelivery,
    StartDelivery,
    StartTool,
    ToolOperation,
    transition,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    BinaryImage,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import RealtimeEvent
from pydantic_ai.realtime.codec import (
    RealtimeCodecEvent,
    RealtimeInput,
    ResponseDone,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
)
from pydantic_ai.session import SessionState
from pydantic_ai.tool_manager import ToolManager

from .realtime.test_session import BlockingRealtimeConnection, FakeRealtimeConnection, FakeRealtimeModel

READINESS_WAIT_TIMEOUT = 10


@pytest.mark.parametrize('stream', [False, True])
async def test_tool_operation_is_completed_before_delivery_and_keeps_run_identity(stream: bool):
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1].parts[0], UserPromptPart):
            return ModelResponse(parts=[ToolCallPart('value', {}, tool_call_id='same')])
        return ModelResponse(parts=[TextPart('done')])

    async def streamed(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if isinstance(messages[-1].parts[0], UserPromptPart):
            yield {0: DeltaToolCall(name='value', json_args='{}', tool_call_id='same')}
        else:
            yield 'done'

    agent = Agent(FunctionModel(respond, stream_function=streamed))

    @agent.tool_plain
    def value() -> str:
        return 'stored result'

    async with agent.session() as session:
        for _ in range(2):
            if stream:
                async with session.run_stream('call') as result:
                    await result.get_output()
            else:
                await session.run('call')
        state = session.state
    assert len(state.operations) == 2
    assert len({op.operation_id for op in state.operations}) == 2
    assert len({op.run_id for op in state.operations}) == 2
    assert [(op.execution, op.delivery) for op in state.operations] == [('completed', 'committed')] * 2
    restored = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(state))
    assert restored == state


@pytest.mark.parametrize('stream', [False, True])
async def test_failed_model_delivery_does_not_erase_tool_completion(stream: bool):
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('value', {}, tool_call_id='same')])
        raise RuntimeError('delivery lost')

    async def streamed(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='value', json_args='{}', tool_call_id='same')}
        else:
            raise RuntimeError('delivery lost')

    agent = Agent(FunctionModel(respond, stream_function=streamed))

    @agent.tool_plain
    def value() -> str:
        nonlocal calls
        calls += 1
        return 'external effect completed'

    async with agent.session() as session:
        with pytest.raises(RuntimeError, match='delivery lost'):
            if stream:
                async with session.run_stream('call') as result:
                    await result.get_output()
            else:
                await session.run('call')
        (operation,) = session.state.operations
        assert (operation.execution, operation.delivery) == ('completed', 'uncertain')
        returned = operation.result[0].parts[0]
        assert isinstance(returned, ToolReturnPart)
        assert returned.content == 'external effect completed'
    assert calls == 1


@pytest.mark.parametrize('failure', ['error', 'cancel'])
async def test_realtime_preserves_completed_result_while_delivery_fails(failure: Literal['error', 'cancel']):
    sending = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class Connection(BlockingRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            if isinstance(content, ToolResult):
                sending.set()
                await release.wait()
                raise RuntimeError('socket write failed')
            await super().send(content)

    connection = Connection([ToolCall(tool_name='value', tool_call_id='call', args='{}'), ResponseDone()])
    agent = Agent()

    @agent.tool_plain
    def value() -> ToolReturn[str]:
        nonlocal calls
        calls += 1
        return ToolReturn('completed externally', content=[BinaryImage(data=b'\xff\x00', media_type='image/png')])

    events: list[RealtimeEvent] = []
    with pytest.raises(RuntimeError, match='socket write failed') if failure == 'error' else nullcontext():
        async with agent.realtime(FakeRealtimeModel(connection)).session() as session:

            async def collect() -> None:
                events.extend([event async for event in session])

            task = asyncio.create_task(collect())
            try:
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await sending.wait()
                (operation,) = session.tool_operations
                assert (operation.execution, operation.delivery) == ('completed', 'sending')
                returned = operation.result[0].parts[0]
                assert isinstance(returned, ToolReturnPart)
                assert returned.content == 'completed externally'
                snapshot = SessionState(conversation=session.conversation, operations=session.tool_operations)
                restored = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(snapshot))
                assert restored == snapshot
                if failure == 'error':
                    release.set()
                    await task
                else:
                    await session.close()
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert calls == 1
    (operation,) = session.tool_operations
    assert (operation.execution, operation.delivery) == ('completed', 'uncertain')
    returns = [p for m in session.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [(p.content, p.outcome) for p in returns] == [('completed externally', 'success')]


def test_operation_transition_does_not_mutate_or_reexecute_completed_tools():
    operation = ToolOperation(operation_id='stable', run_id='run', run_step=1, call=ToolCallPart('value', {}))
    original = deepcopy(operation)
    running, actions = transition(operation, StartTool())
    assert operation == original
    assert actions == ('execute_tool',)
    completed, actions = transition(running, CompleteTool([ModelRequest(parts=[ToolReturnPart('value', 'result')])]))
    assert actions == ('record_result',)
    sending, actions = transition(completed, StartDelivery())
    assert actions == ('deliver_result',)
    accepted, _ = transition(sending, ObserveDelivery('accepted'))
    uncertain, _ = transition(accepted, LoseDelivery())
    assert uncertain.delivery == 'uncertain'
    assert transition(uncertain, StartDelivery()) == (uncertain, ())
    assert transition(uncertain, ObserveDelivery('sent')) == (uncertain, ())
    assert transition(completed, InterruptTool()) == (completed, ())
    with pytest.raises(UserError, match='already started'):
        transition(uncertain, StartTool())
    reconciled, _ = transition(uncertain, ReconcileDelivery('ready'))
    assert reconciled.operation_id == operation.operation_id
    assert reconciled.result == completed.result
    assert transition(reconciled, StartDelivery())[1] == ('deliver_result',)
    committed, _ = transition(uncertain, ObserveDelivery('committed'))
    assert transition(committed, LoseDelivery()) == (committed, ())
    with pytest.raises(UserError, match='committed result'):
        transition(committed, ReconcileDelivery('ready'))


async def test_unconsumed_stream_does_not_confirm_result_delivery():
    async def streamed(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='value', json_args='{}', tool_call_id='call')}
        else:
            yield 'first'
            yield 'second'

    agent = Agent(FunctionModel(stream_function=streamed))

    @agent.tool_plain
    def value() -> str:
        return 'result'

    async with agent.session() as session:
        async with session.run_stream('call'):
            pass
        (operation,) = session.state.operations
        assert operation.execution == 'completed'
        assert operation.delivery == 'uncertain'


async def test_running_tool_snapshot_and_interruption_are_not_completion():
    agent = Agent(TestModel())

    @agent.tool_plain
    def fail_after_effect() -> str:
        (operation,) = session.state.operations
        assert operation.execution == 'running'
        assert operation.delivery == 'pending'
        assert operation.result == []
        raise RuntimeError('effect outcome unknown')

    async with agent.session() as session:
        with pytest.raises(RuntimeError, match='effect outcome unknown'):
            await session.run('call')
        (operation,) = session.state.operations
        assert operation.execution == 'interrupted'
        assert operation.result == []


async def test_successful_request_retry_confirms_previously_uncertain_delivery():
    requests = 0
    tool_calls = 0

    class Retry(AbstractCapability[None]):
        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise ModelRetry('try again') from error

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart('value', {}, tool_call_id='call')])
        if requests == 2:
            raise RuntimeError('delivery lost')
        assert any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[Retry()])

    @agent.tool_plain
    def value() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return 'result'

    async with agent.session() as session:
        assert (await session.run('call')).output == 'done'
        (operation,) = session.state.operations
        assert (operation.execution, operation.delivery) == ('completed', 'committed')
    assert tool_calls == 1


@pytest.mark.parametrize('ordered', [False, True])
async def test_provider_cancellation_during_delivery_emits_actual_completion(ordered: bool):
    sending = asyncio.Event()

    class Connection(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            if isinstance(content, ToolResult):
                sending.set()
                await asyncio.Event().wait()
            await super().send(content)

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_name='value', tool_call_id='call', args='{}')
            await sending.wait()
            yield ToolCallCancelled(tool_call_ids=['call'])

    agent = Agent()

    @agent.tool_plain
    def value() -> str:
        return 'done'

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        with ToolManager.parallel_execution_mode('parallel_ordered_events' if ordered else 'parallel'):
            async with agent.realtime(FakeRealtimeModel(Connection([]))).session() as session:
                events = [event async for event in session]
    returns = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert len(returns) == 1
    assert isinstance(returns[0], ToolReturnPart)
    assert (returns[0].content, returns[0].outcome) == ('done', 'success')
    (operation,) = session.tool_operations
    assert (operation.execution, operation.delivery) == ('completed', 'uncertain')
