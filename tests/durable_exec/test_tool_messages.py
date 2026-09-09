from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import pytest
from anyio import Event, create_task_group, fail_after
from pydantic import TypeAdapter

from pydantic_ai import Agent, FunctionToolset, ModelMessage, ModelRequest, ModelResponse, RunContext
from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.durable_exec import BaseDurabilityCapability, DurabilityEngineSpec, RoleBasedOperationConfig
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._operation import DurableOperationId, ToolsetCallToolId
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend
from pydantic_ai.durable_exec._operation_names import JournalOperationNamer
from pydantic_ai.durable_exec._tool_messages import (
    RecordedToolCallResult,
    ToolCallWithMessages,
    record_tool_call_result,
    replay_tool_messages,
    tool_message_replay_scope,
)
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    guard_run_context,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.usage import RunUsage

READINESS_WAIT_TIMEOUT = 30


@dataclass
class ToolJournal:
    result: object | None = None
    executions: int = 0
    replays: int = 0
    retry_failure: bool = False
    replay_barrier: Event | None = None


class MessageBackend(CallableOperationBackend[None]):
    def __init__(self, journal: ToolJournal) -> None:
        super().__init__(
            namer=JournalOperationNamer('messages'),
            config=RoleBasedOperationConfig(model=None, event=None, capability=None, tool=None),
        )
        self.journal = journal

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        if not isinstance(operation_id, ToolsetCallToolId):
            return await body()
        if self.journal.result is None:
            self.journal.executions += 1
            try:
                self.journal.result = await body()
            except RuntimeError:
                if not self.journal.retry_failure:
                    raise
                self.journal.executions += 1
                self.journal.result = await body()
        else:
            self.journal.replays += 1
            if self.journal.replay_barrier is not None:
                if self.journal.replays == 2:
                    self.journal.replay_barrier.set()
                await self.journal.replay_barrier.wait()
        return self.journal.result


class MessageDurability(BaseDurabilityCapability[object]):
    engine_spec = DurabilityEngineSpec(
        engine_name='Message journal', durable_unit_noun='unit', durable_container_noun='run', codec=JSON_CODEC
    )

    def __init__(self, journal: ToolJournal) -> None:
        super().__init__(name='messages')
        self.journal = journal

    @property
    def in_durable_context(self) -> bool:
        return True

    def get_durable_operation_backend(self) -> MessageBackend:
        return MessageBackend(self.journal)


def user_prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


@pytest.mark.parametrize('priority', ['asap', 'when_idle'])
@pytest.mark.parametrize('dynamic', [False, True])
async def test_recorded_messages_replay_once_per_run(priority: Literal['asap', 'when_idle'], dynamic: bool) -> None:
    """Replay one JSON record twice, including after the first delivery drains the queue."""
    journal = ToolJournal()
    toolset = FunctionToolset[object](id='messages')

    @toolset.tool
    async def enqueue(ctx: RunContext[object]) -> str:
        assert get_current_run_context() is ctx
        ctx.enqueue('recorded', priority=priority)
        return 'done'

    def get_toolset(ctx: RunContext[object]) -> FunctionToolset[object]:
        return toolset

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls = sum(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if calls < 2:
            return ModelResponse(parts=[ToolCallPart('enqueue', '{}', tool_call_id=f'call-{calls}')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(respond),
        toolsets=[DynamicToolset(get_toolset, id='dynamic-messages') if dynamic else toolset],
        capabilities=[MessageDurability(journal)],
    )
    for _ in range(2):
        result = await agent.run('start')
        assert result.output == 'done'
        assert user_prompts(result.all_messages()) == ['start', 'recorded']
    assert journal.executions == 1
    assert journal.replays == 3


async def test_concurrent_runs_receive_the_same_recorded_message() -> None:
    journal = ToolJournal()

    def enqueue(ctx: RunContext[object]) -> str:
        assert get_current_run_context() is ctx
        ctx.enqueue('recorded')
        return 'done'

    agent = Agent(TestModel(), tools=[enqueue], capabilities=[MessageDurability(journal)])
    await agent.run('warm up')
    journal.replay_barrier = Event()
    prompts: list[list[str]] = []

    async def run(prompt: str) -> None:
        result = await agent.run(prompt)
        prompts.append(user_prompts(result.all_messages()))

    with fail_after(READINESS_WAIT_TIMEOUT):
        async with create_task_group() as group:
            group.start_soon(run, 'first')
            group.start_soon(run, 'second')
    assert sorted(prompts) == [['first', 'recorded'], ['second', 'recorded']]
    assert journal.executions == 1
    assert journal.replays == 2


async def test_failed_durable_attempt_does_not_deliver_its_messages() -> None:
    journal = ToolJournal(retry_failure=True)

    async def enqueue(ctx: RunContext[object]) -> str:
        if journal.executions == 1:
            ctx.enqueue('abandoned')
            raise RuntimeError('retry this durable unit')
        ctx.enqueue('committed')
        return 'done'

    agent = Agent(TestModel(), tools=[enqueue], capabilities=[MessageDurability(journal)])
    result = await agent.run('start')
    assert user_prompts(result.all_messages()) == ['start', 'committed']
    assert journal.executions == 2


async def test_model_retry_replays_its_enqueued_message() -> None:
    journal = ToolJournal()

    async def enqueue(ctx: RunContext[object]) -> str:
        ctx.enqueue('correction')
        raise ModelRetry('use the correction')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(
            isinstance(part, RetryPromptPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('enqueue', '{}')])

    agent = Agent(FunctionModel(respond), tools=[enqueue], capabilities=[MessageDurability(journal)])
    for _ in range(2):
        result = await agent.run('start')
        assert user_prompts(result.all_messages()) == ['start', 'correction']
    assert journal.executions == 1
    assert journal.replays == 1


async def test_recorded_message_identity_and_cached_objects_survive_replay() -> None:
    """The wire identity and reusable object graph are contracts of the recorded output."""
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), pending_messages=[])
    durable_ctx = guard_run_context(ctx, unit_noun='unit', container_noun='run')

    async def enqueue() -> str | None:
        return durable_ctx.enqueue('recorded', priority='when_idle')

    adapter: TypeAdapter[RecordedToolCallResult] = TypeAdapter(RecordedToolCallResult)
    record = await record_tool_call_result(durable_ctx, enqueue())
    restored = adapter.validate_json(adapter.dump_json(record))
    assert isinstance(restored, ToolCallWithMessages)
    assert ctx.pending_messages == []

    with tool_message_replay_scope():
        enqueue_id = unwrap_tool_call_result(replay_tool_messages(restored, ctx))
        assert ctx.pending_messages is not None
        assert len(ctx.pending_messages) == 1
        pending = ctx.pending_messages.pop()
        assert pending.enqueue_id == enqueue_id
        assert pending.priority == 'when_idle'
        pending.messages[0].run_id = 'mutated by history processing'
        replay_tool_messages(restored, ctx)
        assert ctx.pending_messages == []

        with tool_message_replay_scope():
            replay_tool_messages(restored, ctx)
            assert len(ctx.pending_messages) == 1
            assert ctx.pending_messages.pop().messages[0].run_id is None

        replay_tool_messages(restored, ctx)
        assert ctx.pending_messages == []


async def test_tool_results_without_messages_keep_the_old_wire_shape() -> None:
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), pending_messages=[])
    durable_ctx = guard_run_context(ctx, unit_noun='unit', container_noun='run')

    async def tool() -> str:
        return 'legacy result'

    old_adapter: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)
    adapter: TypeAdapter[RecordedToolCallResult] = TypeAdapter(RecordedToolCallResult)
    old_wire = old_adapter.dump_json(await wrap_tool_call_result(tool()))
    new_wire = adapter.dump_json(await record_tool_call_result(durable_ctx, tool()))
    assert new_wire == old_wire
    assert unwrap_tool_call_result(replay_tool_messages(adapter.validate_json(old_wire), ctx)) == 'legacy result'
