"""Effect/delivery boundaries need deterministic failures, not a provider cassette's happy path."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
from copy import deepcopy
from typing import Literal

import anyio
import pytest

from pydantic_ai import (
    Agent,
    AgentRunResult,
    ApprovalRequired,
    DeferredToolRequests,
    ModelRetry,
    RunContext,
    SessionStateTypeAdapter,
    UserError,
)
from pydantic_ai._operations import (
    CompleteTool,
    InterruptTool,
    LoseDelivery,
    ObserveDelivery,
    ReconcileDelivery,
    SetOutputResult,
    StartDelivery,
    StartTool,
    ToolOperation,
    UpdateOutputCall,
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
from pydantic_ai.tools import ToolApproved

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


@pytest.mark.parametrize('dynamic', [False, True])
async def test_deferred_resume_preserves_operation_identity(dynamic: bool):
    calls: list[int] = []
    agent = Agent(TestModel(), deps_type=type(None), output_type=[str, DeferredToolRequests])

    @agent.tool(requires_approval=not dynamic)
    def value(ctx: RunContext[None], amount: int) -> int:
        if dynamic and not ctx.tool_call_approved:
            raise ApprovalRequired()
        calls.append(amount)
        return amount

    async with agent.session() as session:
        first = await session.run('call')
        assert isinstance(first.output, DeferredToolRequests)
        state = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(session.state))
        (original,) = state.operations
        assert (original.execution, original.delivery) == ('deferred', 'pending')
        results = first.output.build_results(approve_all=True)
        results.approvals[original.call.tool_call_id] = ToolApproved(override_args={'amount': 42})
    assert calls == []
    async with agent.session(state=state) as restored:
        resumed = await restored.run(deferred_tool_results=results)
        (completed,) = restored.state.operations
        assert completed.operation_id == original.operation_id
        assert completed.run_id == first.run_id != resumed.run_id
        assert (completed.execution, completed.delivery) == ('completed', 'committed')
        assert completed.result[0].run_id == resumed.run_id
    assert calls == [42]


@pytest.mark.parametrize('stream', [False, True])
async def test_output_function_failure_is_an_unresolved_operation(stream: bool):
    """Output functions can perform effects too, including on the run_stream fast path."""
    calls = 0

    async def finish(value: int) -> int:
        nonlocal calls
        calls += 1
        assert len(session.state.operations) == 1
        assert session.state.operations[0].execution == 'running'
        raise RuntimeError('output effect outcome unknown')

    agent = Agent(TestModel(), output_type=finish)
    async with agent.session() as session:
        with pytest.raises(RuntimeError, match='output effect outcome unknown'):
            if stream:
                async with session.run_stream('finish') as result:
                    await result.get_output()
            else:
                await session.run('finish')
        assert calls == 1
        (operation,) = session.state.operations
        assert operation.execution == 'interrupted'
        with pytest.raises(UserError, match='unresolved'):
            session.state.recover()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('output', [None, 7])
async def test_output_tool_completion_waits_for_provider_delivery(stream: bool, output: int | None):
    calls = 0

    async def finish(value: int) -> int | None:
        nonlocal calls
        calls += 1
        return output

    agent = Agent(TestModel(), output_type=finish)
    async with agent.session() as session:
        if stream:
            async with session.run_stream('finish') as result:
                assert await result.get_output() == output
                assert await result.get_output() == output
        else:
            assert (await session.run('finish')).output == output
        (operation,) = session.state.operations
        assert (operation.execution, operation.delivery) == ('completed', 'ready')
        part = operation.result[0].parts[0]
        assert isinstance(part, ToolReturnPart)
        assert calls == 1
        history_part = session.conversation.messages[-1].parts[0]
        assert part == history_part
        assert session.state.recover().operations == session.state.operations


@pytest.mark.parametrize('abandon', [False, True])
async def test_partial_output_processing_is_unresolved_until_final_validation(abandon: bool):
    seen: list[bool] = []

    async def finish(ctx: RunContext[None], value: str) -> str:
        seen.append(ctx.partial_output)
        assert session.state.operations[0].execution == 'running'
        return value

    async def streamed(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
        yield {0: DeltaToolCall(name='final_result', json_args='{"value":"one', tool_call_id='output')}
        yield {0: DeltaToolCall(json_args=' two"}')}

    agent = Agent(FunctionModel(stream_function=streamed), output_type=finish, deps_type=type(None))
    async with agent.session() as session:
        async with session.run_stream('finish') as result:
            async for _ in result.stream_output(debounce_by=None):
                if abandon:
                    break
        (operation,) = session.state.operations
        assert seen[0] is True
        if abandon:
            assert operation.execution == 'interrupted'
            assert operation.result == []
            with pytest.raises(UserError, match='unresolved'):
                session.state.recover()
        else:
            assert seen[-1] is False
            assert operation.call.args_as_dict() == {'value': 'one two'}
            assert operation.execution == 'completed'
            assert len(session.state.operations) == 1


@pytest.mark.parametrize('strategy', ['early', 'graceful', 'exhaustive'])
@pytest.mark.parametrize('retry', [False, True])
async def test_output_operation_status_matches_selected_history(
    strategy: Literal['early', 'graceful', 'exhaustive'], retry: bool
):
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 1}, tool_call_id='first'),
                    ToolCallPart('final_result', {'value': 2}, tool_call_id='second'),
                    ToolCallPart('work', {}, tool_call_id='work'),
                ]
            )
        return ModelResponse(parts=[TextPart('retried')])

    calls: list[int] = []

    async def finish(value: int) -> int:
        calls.append(value)
        return value

    agent = Agent(FunctionModel(respond), output_type=[str, finish], end_strategy=strategy)

    @agent.tool_plain
    async def work() -> str:
        if retry:
            raise ModelRetry('retry work')
        return 'done'

    async with agent.session() as session:
        await session.run('finish')
        state = session.state
    assert sorted(calls) == ([1, 2] if strategy == 'exhaustive' else [1])
    returns = [p for m in state.conversation.messages for p in m.parts if isinstance(p, ToolReturnPart)]
    output_operations = [op for op in state.operations if op.call.tool_name == 'final_result']
    assert len(output_operations) == len(calls)
    for op in output_operations:
        assert op.execution == 'completed'
        part = op.result[0].parts[0]
        assert part in returns
        assert op.delivery == ('committed' if retry and strategy != 'early' else 'ready')
    assert state.recover().operations == state.operations


@pytest.mark.parametrize('stream', [False, True])
async def test_output_cancellation_requires_explicit_reconciliation(stream: bool):
    started = anyio.Event()
    calls = 0
    checkpoint: SessionState | None = None

    async def finish(value: int) -> int:
        nonlocal calls, checkpoint
        calls += 1
        checkpoint = session.state
        started.set()
        await anyio.sleep_forever()
        return value

    agent = Agent(TestModel(), output_type=finish)
    async with agent.session() as session:

        async def execute() -> None:
            if stream:
                async with session.run_stream('finish') as result:
                    await result.get_output()
            else:
                await session.run('finish')

        async with anyio.create_task_group() as group:
            group.start_soon(execute)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await started.wait()
            group.cancel_scope.cancel()
        (operation,) = session.state.operations
        assert operation.execution == 'interrupted'
        assert checkpoint is not None
        checkpoint = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(checkpoint))
        with pytest.raises(UserError, match='unresolved'):
            checkpoint.recover(abandon_run=True)
        recovered = checkpoint.recover(
            abandon_run=True,
            tool_results={
                operation.operation_id: ModelRequest(
                    parts=[
                        ToolReturnPart(
                            operation.call.tool_name, 'Externally verified.', tool_call_id=operation.call.tool_call_id
                        )
                    ]
                )
            },
        )
    async with agent.session(state=recovered) as restored:
        await restored.run('continue', output_type=str)
    assert calls == 1


@pytest.mark.parametrize('owned', [False, True])
async def test_streamed_output_history_is_visible_to_after_run(owned: bool):
    seen: list[ModelMessage] = []

    class Observe(AbstractCapability[None]):
        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[int]) -> AgentRunResult[int]:
            seen.extend(result.all_messages())
            assert isinstance(seen[-1].parts[0], ToolReturnPart)
            assert result.usage.requests == 1
            return result

    agent = Agent(TestModel(), output_type=int, deps_type=type(None), capabilities=[Observe()])
    if owned:
        async with agent.session() as session:
            async with session.run_stream('finish') as result:
                await result.get_output()
            assert seen == result.all_messages() == session.conversation.messages
            assert session.conversation.usage.requests == 1
    else:
        async with agent.run_stream('finish') as result:
            await result.get_output()
        assert seen == result.all_messages()


def test_output_status_transitions_cannot_rewrite_a_delivered_result():
    """Pin reducer permission separately from runtime winner-selection behavior."""
    operation = ToolOperation(operation_id='output', run_id='run', run_step=1, call=ToolCallPart('finish', {}))
    request = ModelRequest(parts=[ToolReturnPart('finish', 'done', tool_call_id=operation.call.tool_call_id)])
    with pytest.raises(UserError, match='not running'):
        transition(operation, UpdateOutputCall(operation.call))
    with pytest.raises(UserError, match='cannot change'):
        transition(operation, SetOutputResult([request]))
    running, _ = transition(operation, StartTool())
    updated, actions = transition(running, UpdateOutputCall(ToolCallPart('finish', {'value': 1})))
    assert updated.call.args == {'value': 1}
    assert running.call.args == {}
    assert actions == ()
    completed, _ = transition(updated, CompleteTool([request]))
    replacement = ModelRequest(parts=[ToolReturnPart('finish', 'not selected')])
    settled, actions = transition(completed, SetOutputResult([replacement]))
    assert settled.result == [replacement]
    assert completed.result == [request]
    assert actions == ('record_result',)
    sending, _ = transition(settled, StartDelivery())
    with pytest.raises(UserError, match='cannot change'):
        transition(sending, SetOutputResult([request]))


async def test_iter_final_output_execution_records_full_arguments_after_partial_validation():
    seen: list[bool] = []

    async def finish(ctx: RunContext[None], value: str) -> str:
        seen.append(ctx.partial_output)
        if not ctx.partial_output:
            assert session.state.operations[0].call.args_as_dict() == {'value': value}
            raise RuntimeError('unknown final effect')
        return value

    async def streamed(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
        yield {0: DeltaToolCall(name='final_result', json_args='{"value":"one', tool_call_id='output')}
        yield {0: DeltaToolCall(json_args=' two"}')}

    agent = Agent(FunctionModel(stream_function=streamed), output_type=finish, deps_type=type(None))
    async with agent.session() as session:
        with pytest.raises(RuntimeError, match='unknown final effect'):
            async with session.iter('finish') as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as stream:
                            async for _ in stream.stream_output(debounce_by=None):
                                break
                            await stream.drain()
        (operation,) = session.state.operations
        assert operation.execution == 'interrupted'
        assert operation.call.args_as_dict() == {'value': 'one two'}
    assert seen == [True, False]
