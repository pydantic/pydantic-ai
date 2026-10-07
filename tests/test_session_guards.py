"""Admission and recovery guards use deterministic local state, not provider recordings."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import replace
from typing import Literal

import pytest

from pydantic_ai import Agent, Conversation, RunContext, UserError
from pydantic_ai._cancel import RunCancellation
from pydantic_ai._enqueue import PendingMessage, PendingMessageQueue
from pydantic_ai._operations import (
    CompleteTool,
    DeferTool,
    ObserveDelivery,
    OperationEvent,
    ReconcileDelivery,
    ReconcileTool,
    ToolOperation,
    begin_request_delivery,
    transition,
)
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai._session import ActiveRun, ModelResources, SessionRuntime
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.session import SessionState
from pydantic_ai.usage import RunUsage


@pytest.mark.parametrize(
    ('execution', 'event', 'message'),
    [
        ('pending', CompleteTool([]), 'not running'),
        ('completed', ReconcileTool([]), 'no unresolved outcome'),
        ('completed', DeferTool(), 'cannot be deferred'),
        ('pending', ObserveDelivery('sent'), 'no delivery to acknowledge'),
        ('pending', ReconcileDelivery('ready'), 'no completed result'),
    ],
)
def test_operation_rejects_invalid_effect_order(
    execution: Literal['pending', 'completed'], event: OperationEvent, message: str
):
    operation = ToolOperation(
        operation_id='op', run_id='run', run_step=1, call=ToolCallPart('tool'), execution=execution
    )
    with pytest.raises(UserError, match=message):
        transition(operation, event)
    assert operation.execution == execution
    assert operation.result == []


def test_request_delivery_does_not_send_an_incomplete_tool():
    """A malformed projection must not grant delivery merely because its history matches."""
    result = ModelRequest([ToolReturnPart('tool', 'result', tool_call_id='call')])
    operation = ToolOperation(
        operation_id='op',
        run_id='run',
        run_step=1,
        call=ToolCallPart('tool', tool_call_id='call'),
        execution='pending',
        delivery='ready',
        result=[result],
    )
    operations = {'op': operation}
    assert begin_request_delivery(operations, [result]) == []
    assert operations == {'op': operation}


async def test_model_resource_can_enter_a_previously_bound_definition():
    """Drivers may resolve a handle before acquiring its client; each scope must enter once."""
    entries: list[str] = []

    class Model(TestModel):
        async def __aenter__(self):
            entries.append('enter')
            return self

        async def __aexit__(self, *args: object):
            entries.append('exit')

    model = Model()
    resources = ModelResources()
    async with AsyncExitStack() as stack:
        resources.bind_stack(stack)
        assert await resources.get_model(model, enter_model=False) is model
        assert entries == []
        assert await resources.get_model(model) is model
        assert await resources.get_model(model) is model
        assert entries == ['enter']
    assert entries == ['enter', 'exit']


async def test_preentered_transparent_wrapper_does_not_reenter_its_client():
    entries: list[str] = []

    class Model(TestModel):
        async def __aenter__(self):
            entries.append('enter')
            return self

        async def __aexit__(self, *args: object):
            entries.append('exit')

    model = Model()
    wrapper = WrapperModel(model)
    async with wrapper:
        async with AsyncExitStack() as stack:
            resources = ModelResources(entered_model_ids={id(wrapper)})
            resources.bind_stack(stack)
            assert await resources.get_model(wrapper) is wrapper
            assert entries == ['enter']
    assert entries == ['enter', 'exit']


def test_runtime_closed_and_preparing_checkpoint_guards():
    runtime = SessionRuntime(Conversation())
    runtime.claim()
    runtime.cancel()  # cancellation is remembered before a driver has attached
    with pytest.raises(UserError, match='preparing a run'):
        runtime.snapshot()
    runtime.release()
    runtime.close()
    with pytest.raises(UserError, match='session has closed'):
        runtime.claim()


async def test_runtime_rejects_idle_steering():
    runtime = SessionRuntime(Conversation())
    with pytest.raises(UserError, match='active ordinary run'):
        await runtime.steer(['input'])


async def test_cancel_before_driver_binding_is_delivered():
    runtime = SessionRuntime(Conversation())
    runtime.claim()
    runtime.cancel()
    cancellation = RunCancellation()
    runtime.bind_cancellation(cancellation)
    assert cancellation.cancel_requested
    runtime.release()


def test_runtime_closed_queue_hands_input_back_to_idle_inbox():
    runtime = SessionRuntime(Conversation())
    runtime.claim()
    queue = PendingMessageQueue()
    runtime.attach(ActiveRun(run_id='run', pending_messages=queue, snapshot=Conversation))
    queue.close_and_take()
    pending = PendingMessage(messages=[ModelRequest([UserPromptPart('next')])])
    runtime.enqueue(pending)
    runtime.release()
    assert runtime.snapshot()[1] == [pending]


@pytest.mark.parametrize('conflict', ['conversation', 'message_history', 'conversation_id', 'usage'])
async def test_iter_rejects_session_state_override(conflict: str):
    agent = Agent(TestModel())
    async with agent.session() as session:
        with pytest.raises(UserError, match='session owns'):
            async with session.iter(
                conversation=Conversation() if conflict == 'conversation' else None,
                message_history=[] if conflict == 'message_history' else None,
                conversation_id='other' if conflict == 'conversation_id' else None,
                usage=RunUsage() if conflict == 'usage' else None,
            ):
                assert False, 'state conflict was accepted'


async def test_stream_events_can_be_abandoned_without_a_result():
    async with Agent(TestModel()).session() as session:
        async with session.run_stream_events('unused'):
            pass
        assert (await session.run('next')).output == 'success (no tool calls)'


def test_recovery_rejects_duplicate_operation_identity():
    operation = ToolOperation(operation_id='op', run_id='run', run_step=1, call=ToolCallPart('tool'))
    with pytest.raises(UserError, match='duplicate tool operation IDs'):
        SessionState(operations=[operation, operation]).recover()


def test_recovery_inserts_result_before_a_later_response():
    call = ToolCallPart('tool', tool_call_id='call')
    response = ModelResponse([call], run_id='run')
    later = ModelResponse([TextPart('later')], run_id='later')
    operation = ToolOperation(
        operation_id='op',
        run_id='run',
        run_step=1,
        call=call,
        execution='interrupted',
    )
    state = SessionState(conversation=Conversation(messages=[response, later]), operations=[operation])
    recovered = state.recover(tool_results={'op': ModelRequest([ToolReturnPart('tool', 'done', tool_call_id='call')])})
    assert recovered.conversation.messages[0] == response
    result = recovered.conversation.messages[1]
    assert isinstance(result, ModelRequest)
    assert result.parts[0] == recovered.operations[0].result[0].parts[0]
    assert recovered.conversation.messages[2] == later
    with pytest.raises(UserError, match='unambiguously locate'):
        replace(state, conversation=Conversation()).recover(
            tool_results={'op': ModelRequest([ToolReturnPart('tool', 'done', tool_call_id='call')])}
        )


async def test_session_preserves_multiple_resource_teardown_errors():
    class FailingClose(TestModel):
        async def __aexit__(self, *args: object):
            raise ValueError('resource failure')

    with pytest.raises(ExceptionGroup) as raised:
        async with Agent(FailingClose()).session() as session:
            await session.run('start')
            raise RuntimeError('body failure')
    assert raised.value.subgroup(ValueError) is not None
    assert raised.value.subgroup(RuntimeError) is not None


async def test_session_steering_uses_the_active_callback_context():
    agent = Agent(TestModel(), deps_type=type(None))

    @agent.tool
    async def check_steering(ctx: RunContext[None]) -> str:
        # Durable adapters can bind a context projection; session steering must use its guard.
        with set_current_run_context(ctx):
            with pytest.raises(UserError, match='active, steering-enabled'):
                await session.steer('not enabled')
        return 'checked'

    async with agent.session() as session:
        await session.run('call')
        async with session.iter('next') as run:
            with pytest.raises(UserError, match='active, steering-enabled'):
                await run.steer('not enabled')


async def test_session_enqueue_cannot_bypass_a_context_queue_guard():
    """A durable projection's queue takes precedence over a closure's live session queue."""
    runtime = SessionRuntime(Conversation())
    runtime.claim()
    queue = PendingMessageQueue()
    guarded = PendingMessageQueue()
    runtime.attach(ActiveRun(run_id='run', pending_messages=queue, snapshot=Conversation))
    context = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run', pending_messages=guarded)
    pending = PendingMessage(messages=[ModelRequest([UserPromptPart('next')])])
    with set_current_run_context(context):
        runtime.enqueue(pending)
    assert guarded.snapshot() == [pending]
    assert queue.snapshot() == []
    with pytest.raises(UserError, match='unavailable'):
        await context.steer('not attached')
    runtime.release()


def test_recovery_preserves_unrelated_parts_before_missing_tool_result():
    call = ToolCallPart('tool', tool_call_id='call')
    prompt = UserPromptPart('keep this')
    state = SessionState(
        conversation=Conversation(messages=[ModelResponse([call], run_id='run'), ModelRequest([prompt])]),
        operations=[ToolOperation(operation_id='op', run_id='run', run_step=1, call=call, execution='interrupted')],
    )
    recovered = state.recover(tool_results={'op': ModelRequest([ToolReturnPart('tool', 'done', tool_call_id='call')])})
    assert recovered.conversation.messages[1].parts[0] == prompt
    assert len(recovered.conversation.messages[1].parts) == 2


@pytest.mark.parametrize('stream', [False, True])
async def test_async_session_rejects_sync_execution_without_creating_a_coroutine(stream: bool):
    async with Agent(TestModel()).session() as session:
        with pytest.raises(RuntimeError, match=r'event loop|async'):
            if stream:
                with session.run_stream_sync('invalid'):
                    assert False, 'sync execution must not start inside an event loop'
            else:
                session.run_sync('invalid')
