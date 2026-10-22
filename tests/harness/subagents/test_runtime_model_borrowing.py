"""Runtime model selection for delegated agents."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager

import anyio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.subagents import DelegationTaskEvent, DelegationTasks, SubAgent, SubAgents

READINESS_WAIT_TIMEOUT = 10


def _user_prompt(messages: list[ModelMessage]) -> str:
    prompt = next(part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart))
    assert isinstance(prompt, str)
    return prompt


def _delegate_results(messages: Sequence[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task'
    ]


async def test_child_without_a_model_borrows_the_parent_run_model() -> None:
    """A run-level connected-model stand-in is used by the parent and its child.

    This verifies public runtime model forwarding with a provider-free model. It does
    not exercise OpenAI's concrete WebSocket handle, whose lifecycle needs a socket-backed test.
    """
    calls: list[str] = []

    async def borrowed_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = _user_prompt(messages)
        calls.append(prompt)
        if prompt == 'inspect':
            return ModelResponse(parts=[TextPart('borrowed result')])
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('parent finished')])
        return ModelResponse(
            parts=[ToolCallPart('delegate_task', {'agent_name': 'worker', 'task': 'inspect'}, tool_call_id='call-1')]
        )

    async def unused_parent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the run-level model should override the parent agent model')  # pragma: no cover

    child = Agent(name='worker')
    parent = Agent(
        FunctionModel(function=unused_parent_model),
        capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)],
    )

    result = await parent.run('go', model=FunctionModel(function=borrowed_model))

    assert calls == ['go', 'inspect', 'go']
    assert _delegate_results(result.all_messages()) == ['borrowed result']


async def test_background_child_reports_expired_inherited_model_after_parent_scope() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    connected = False

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal connected
        prompt = _user_prompt(messages)
        if prompt == 'inspect':
            started.set()
            await release.wait()
            assert not connected
            raise RuntimeError('connected model scope has closed')
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('parent finished')])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    'delegate_task',
                    {'agent_name': 'worker', 'task': 'inspect', 'background': True},
                    tool_call_id='call-1',
                )
            ]
        )

    async def observe(update: DelegationTaskEvent) -> None:
        if update.task.status == 'finished':
            finished.set()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        response = await respond(messages, info)
        # The inherited background call raises after its connection scope closes.
        for part in response.parts:  # pragma: no cover
            assert isinstance(part, TextPart)
            yield part.content

    @asynccontextmanager
    async def connected_scope() -> AsyncGenerator[FunctionModel, None]:
        nonlocal connected
        connected = True
        try:
            yield FunctionModel(function=respond, stream_function=stream)
        finally:
            connected = False

    async def unused_parent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the run-level model should override the parent agent model')  # pragma: no cover

    owner = DelegationTasks(observer=observe)
    child = Agent(name='worker')
    parent = Agent(
        FunctionModel(function=unused_parent_model),
        capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)],
    )

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with owner.opened():
            async with connected_scope() as connected_model:
                with owner.bind():
                    await parent.run('go', model=connected_model)
                    await started.wait()
            release.set()
            await finished.wait()

    (record,) = owner.records.values()
    assert record.outcome == 'failed'
    assert 'connected model scope has closed' in record.output


async def test_background_child_uses_its_own_model_after_parent_scope() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    parent_calls: list[str] = []
    connected = False

    async def connected_parent(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal connected
        prompt = _user_prompt(messages)
        if not connected:
            raise RuntimeError('connected model scope has closed')  # pragma: no cover
        parent_calls.append(prompt)
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('parent finished')])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    'delegate_task',
                    {'agent_name': 'worker', 'task': 'inspect', 'background': True},
                    tool_call_id='call-1',
                )
            ]
        )

    async def own_child(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('own model result')])

    async def own_child_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        response = await own_child(messages, info)
        for part in response.parts:
            assert isinstance(part, TextPart)
            yield part.content

    async def observe(update: DelegationTaskEvent) -> None:
        if update.task.status == 'finished':
            finished.set()

    @asynccontextmanager
    async def connected_scope() -> AsyncGenerator[FunctionModel, None]:
        nonlocal connected
        connected = True
        try:
            yield FunctionModel(function=connected_parent)
        finally:
            connected = False

    async def unused_parent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the run-level model should override the parent agent model')  # pragma: no cover

    owner = DelegationTasks(observer=observe)
    child = Agent(FunctionModel(function=own_child, stream_function=own_child_stream), name='worker')
    parent = Agent(
        FunctionModel(function=unused_parent_model),
        capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)],
    )

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with owner.opened():
            async with connected_scope() as connected_model:
                with owner.bind():
                    await parent.run('go', model=connected_model)
                    await started.wait()
            release.set()
            await finished.wait()

    (record,) = owner.records.values()
    assert parent_calls == ['go', 'go']
    assert record.outcome == 'ok'
    assert record.output == 'own model result'
