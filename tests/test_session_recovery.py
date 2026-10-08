"""Checkpoint recovery uses controlled failures to prove effects are not replayed."""

from __future__ import annotations

from collections.abc import AsyncIterable
from copy import deepcopy
from dataclasses import replace
from typing import Literal

import pytest

from pydantic_ai import Agent, ApprovalRequired, CallDeferred, RunContext, SessionStateTypeAdapter, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryImage,
    DeferredToolRequestsEvent,
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
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.session import SessionState, ToolOperation
from pydantic_ai.tools import DeferredToolRequests, ToolDenied


@pytest.mark.parametrize('active', [False, True])
async def test_recover_uncertain_delivery_requires_explicit_decision(active: bool):
    checkpoints: list[SessionState] = []
    calls = 0
    fail = True

    def respond(history: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(history) == 1:
            return ModelResponse(parts=[ToolCallPart('value', {}, tool_call_id='call')])
        if fail:
            checkpoints.append(session.state)
            raise RuntimeError('connection lost')
        assert any(isinstance(p, ToolReturnPart) and p.content == 'done' for m in history for p in m.parts)
        return ModelResponse(parts=[TextPart('recovered')])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def value() -> str:
        nonlocal calls
        calls += 1
        return 'done'

    async with agent.session() as session:
        with pytest.raises(RuntimeError, match='connection lost'):
            await session.run('call')
        checkpoint = checkpoints[0] if active else session.state
    checkpoint = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(checkpoint))
    (operation,) = checkpoint.operations
    with pytest.raises(UserError, match=r'unfinished run|unresolved'):
        agent.session(state=checkpoint)
    with pytest.raises(UserError, match='unresolved'):
        checkpoint.recover(abandon_run=active)
    recovered = checkpoint.recover(deliveries={operation.operation_id: 'ready'}, abandon_run=active)
    assert checkpoint.operations[0].delivery == ('sending' if active else 'uncertain')
    fail = False
    async with agent.session(state=recovered) as session:
        assert (await session.run()).output == 'recovered'
        assert session.state.operations[0].delivery == 'committed'
    assert calls == 1


@pytest.mark.parametrize('active', [False, True])
async def test_recover_external_outcome_without_executing_again(active: bool):
    checkpoints: list[SessionState] = []
    effects: list[str] = []

    def respond(history: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(history) == 1:
            return ModelResponse(parts=[ToolCallPart('value', {}, tool_call_id='call')])
        returns = [p for m in history for p in m.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 1
        assert returns[0].content == 'verified in external store'
        return ModelResponse(parts=[TextPart('recovered')])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def value() -> str:
        effects.append('effect')
        checkpoints.append(session.state)
        raise RuntimeError('failed after effect')

    async with agent.session() as session:
        with pytest.raises(RuntimeError, match='failed after effect'):
            await session.run('call')
        checkpoint = checkpoints[0] if active else session.state
    checkpoint = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(checkpoint))
    original = deepcopy(checkpoint)
    (operation,) = checkpoint.operations
    verified_result = ModelRequest(
        parts=[
            ToolReturnPart('value', 'verified in external store', tool_call_id='call'),
            UserPromptPart([BinaryImage(data=b'\xff\x00', media_type='image/png')]),
        ]
    )
    if active:
        with pytest.raises(UserError, match='unfinished run'):
            checkpoint.recover(tool_results={operation.operation_id: verified_result})
    with pytest.raises(UserError, match='unresolved'):
        checkpoint.recover(abandon_run=active)
    recovered = checkpoint.recover(tool_results={operation.operation_id: verified_result}, abandon_run=active)
    assert checkpoint == original
    assert recovered.active_run_id is None
    recovered = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(recovered))
    async with agent.session(state=recovered) as resumed:
        assert (await resumed.run()).output == 'recovered'
        assert resumed.state.operations[0].execution == 'completed'
        assert resumed.state.operations[0].delivery == 'committed'
    assert effects == ['effect']


async def test_recover_completed_approval_from_resumed_run():
    checkpoints: list[SessionState] = []
    effects: list[str] = []
    image = BinaryImage(data=b'\xff\x00', media_type='image/png')
    agent = Agent(TestModel(), deps_type=type(None), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def value() -> ToolReturn:
        effects.append('effect')
        return ToolReturn(return_value='actual success', content=[image])

    async def capture(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            if isinstance(event, FunctionToolResultEvent):
                checkpoints.append(session.state)

    async with agent.session() as session:
        first = await session.run('call')
        assert isinstance(first.output, DeferredToolRequests)
        await session.run(
            deferred_tool_results=first.output.build_results(approve_all=True), event_stream_handler=capture
        )
    checkpoint = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(checkpoints[0]))
    (operation,) = checkpoint.operations
    assert operation.run_id == first.run_id != checkpoint.active_run_id
    recovered = checkpoint.recover(abandon_run=True)
    returns = [p for m in recovered.conversation.messages for p in m.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1
    assert (returns[0].content, returns[0].outcome) == ('actual success', 'success')
    assert any(
        isinstance(p, UserPromptPart) and not isinstance(p.content, str) and image in p.content
        for m in recovered.conversation.messages
        for p in m.parts
    )
    async with agent.session(state=recovered) as resumed:
        await resumed.run()
    assert effects == ['effect']


@pytest.mark.parametrize('kind', ['approval', 'dynamic-approval', 'external'])
@pytest.mark.parametrize('approve', [False, True])
async def test_recover_active_approval_checkpoint(
    kind: Literal['approval', 'dynamic-approval', 'external'], approve: bool
):
    checkpoints: list[SessionState] = []
    effects: list[str] = []
    agent = Agent(TestModel(), deps_type=type(None), output_type=[str, DeferredToolRequests])

    @agent.tool(requires_approval=kind == 'approval')
    def value(ctx: RunContext[None]) -> str:
        if kind == 'dynamic-approval' and not ctx.tool_call_approved:
            raise ApprovalRequired(metadata={'ticket': 'approval-42'})
        if kind == 'external':
            raise CallDeferred(metadata={'ticket': 'external-42'})
        effects.append('effect')
        return 'done'

    async def capture(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            if isinstance(event, DeferredToolRequestsEvent):
                checkpoints.append(session.state)

    async with agent.session() as session:
        await session.run('call', event_stream_handler=capture)
    checkpoint = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(checkpoints[0]))
    recovered = checkpoint.recover(abandon_run=True)
    requests = recovered.conversation.deferred_tool_requests
    assert requests is not None
    assert effects == []
    call_id = checkpoint.operations[0].call.tool_call_id
    if kind == 'external':
        assert requests.metadata == {call_id: {'ticket': 'external-42'}}
        results = requests.build_results()
        results.calls[call_id] = 'externally completed'
    else:
        if kind == 'dynamic-approval':
            assert requests.metadata == {call_id: {'ticket': 'approval-42'}}
        results = requests.build_results(approve_all=approve)
        if not approve:
            results.approvals[call_id] = ToolDenied('not authorized')
    async with agent.session(state=recovered) as resumed:
        await resumed.run(deferred_tool_results=results)
        assert resumed.state.operations[0].operation_id == checkpoint.operations[0].operation_id
        assert resumed.state.operations[0].execution == 'completed'
    assert effects == (['effect'] if approve and kind != 'external' else [])


def test_recovery_rejects_unknown_operations_and_mismatched_results():
    call = ToolCallPart('value', {}, tool_call_id='call')
    checkpoint = SessionState(
        operations=[ToolOperation(operation_id='op', run_id='run', run_step=1, call=call, execution='interrupted')]
    )
    with pytest.raises(UserError, match='Unknown tool operation'):
        checkpoint.recover(deliveries={'other': 'ready'})
    with pytest.raises(UserError, match='must answer only'):
        checkpoint.recover(
            tool_results={'op': ModelRequest(parts=[ToolReturnPart('other', 'bad', tool_call_id='call')])}
        )
    committed = replace(checkpoint.operations[0], execution='completed', delivery='committed')
    with pytest.raises(UserError, match='cannot be made pending'):
        SessionState(operations=[committed]).recover(deliveries={'op': 'ready'})
