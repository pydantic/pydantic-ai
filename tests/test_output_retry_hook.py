from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, assert_type

import pytest
from pydantic import ValidationError

from pydantic_ai import (
    Agent,
    EndStrategy,
    ModelRetry,
    PromptedOutput,
    RunContext,
    Tool,
    ToolOutput,
    capture_run_messages,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    Hooks,
    OutputContext,
    OutputRetryContext,
    OutputRetrySource,
    RawOutput,
    WrapperCapability,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryFeedbackCause,
    RetryFeedbackPart,
    RetryInfo,
    ToolCallPart,
    ToolRetryCause,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize('registration', ['constructor', 'decorator', 'timeout_decorator', 'wrapped'])
async def test_hook_registration_and_context(
    registration: Literal['constructor', 'decorator', 'timeout_decorator', 'wrapped'],
) -> None:
    received: list[OutputRetryContext] = []

    async def record(ctx: RunContext[str], *, retry_context: OutputRetryContext) -> None:
        assert_type(retry_context.retry, RetryInfo[RetryFeedbackCause | ToolRetryCause])
        assert_type(retry_context.source, OutputRetrySource)
        assert ctx.deps == 'dependency'
        assert (ctx.retry, ctx.max_retries) == (1, 1)
        received.append(retry_context)

    hooks = Hooks[str]()
    if registration == 'constructor':
        hooks = Hooks(output_retry=record)
    elif registration == 'timeout_decorator':
        hooks.on.output_retry(timeout=10)(record)
    else:
        hooks.on.output_retry(record)
    capability = WrapperCapability(wrapped=hooks) if registration == 'wrapped' else hooks
    agent = Agent(TestModel(custom_output_text='draft'), deps_type=str, capabilities=[capability])

    @agent.output_validator
    def validate(ctx: RunContext[str], output: str) -> str:
        if ctx.retry == 0:
            raise ModelRetry('Finish the draft.')
        return output

    result = await agent.run('Answer.', deps='dependency')
    [retry] = received
    assert retry.response is result.all_messages()[1]
    assert retry.retry == RetryInfo(cause='model_retry', attempt=1, max_retries=1)
    assert (retry.source, retry.tool_name, retry.tool_call_id) == ('response_output', None, None)


async def test_notifications_follow_reverse_capability_order() -> None:
    order: list[str] = []

    @dataclass
    class Record(AbstractCapability[object]):
        label: str

        async def on_output_retry(self, ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
            order.append(self.label)

    agent = Agent(TestModel(custom_output_text='draft'), capabilities=[Record('outer'), Record('inner')])

    @agent.output_validator
    def validate(ctx: RunContext[object], output: str) -> str:
        if ctx.retry == 0:
            raise ModelRetry('Finish the draft.')
        return output

    await agent.run('Answer.')
    assert order == ['inner', 'outer']


async def test_terminal_exhaustion_does_not_notify() -> None:
    received: list[OutputRetryContext] = []
    hooks = Hooks[object]()

    @hooks.on.output_retry
    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:  # pragma: no cover
        received.append(retry_context)

    agent = Agent(TestModel(custom_output_text='draft'), capabilities=[hooks], retries={'output': 0})

    @agent.output_validator
    def validate(output: str) -> str:
        raise ModelRetry('Finish the draft.')

    with pytest.raises(UnexpectedModelBehavior, match='output retries'):
        await agent.run('Answer.')
    assert received == []


@pytest.mark.parametrize('end_strategy', ['early', 'graceful', 'exhaustive'])
async def test_accepted_sibling_notifies_before_later_exhaustion(end_strategy: EndStrategy) -> None:
    received: list[tuple[str | None, int | None, int]] = []

    def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('retryable', {'response': 'invalid'}, 'one'),
                ToolCallPart('also_retryable', {'response': 'invalid'}, 'two'),
                ToolCallPart('exhausted', {'response': 'invalid'}, 'three'),
            ]
        )

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        received.append((retry_context.tool_name, retry_context.retry.attempt, ctx.retry))

    agent = Agent(
        FunctionModel(invalid),
        output_type=[
            ToolOutput(int, name='retryable', max_retries=1),
            ToolOutput(int, name='also_retryable', max_retries=1),
            ToolOutput(int, name='exhausted', max_retries=0),
        ],
        capabilities=[Hooks(output_retry=record)],
        end_strategy=end_strategy,
    )
    with pytest.raises(UnexpectedModelBehavior, match='output retries'):
        await agent.run('Answer.')
    assert received == [('retryable', 1, 2), ('also_retryable', 1, 2)]


async def test_none_output_validator_retry_notifies() -> None:
    received: list[OutputRetryContext] = []

    def empty(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        received.append(retry_context)

    agent = Agent(FunctionModel(empty), output_type=str | None, capabilities=[Hooks(output_retry=record)])

    @agent.output_validator
    def validate(ctx: RunContext[object], output: str | None) -> str | None:
        if ctx.retry == 0:
            raise ModelRetry('Check once more.')
        return output

    assert (await agent.run('Answer.')).output is None
    [retry] = received
    assert retry.source == 'response_output'
    assert retry.retry.attempt == 1


async def test_recovered_schema_error_does_not_notify() -> None:
    received: list[OutputRetryContext] = []

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:  # pragma: no cover
        received.append(retry_context)

    def recover(
        ctx: RunContext[object],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> object:
        return 42

    agent = Agent(
        TestModel(custom_output_text='invalid'),
        output_type=PromptedOutput(int),
        capabilities=[Hooks(output_retry=record, output_validate_error=recover)],
    )
    assert (await agent.run('Answer.')).output == 42
    assert received == []


@pytest.mark.parametrize('error', [ValueError, ModelRetry, UnexpectedModelBehavior])
async def test_notification_exception_stops_run_and_preserves_tool_returns(error: type[Exception]) -> None:
    def fail(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        assert retry_context.source == 'output_tool'
        raise error('notification failed')

    agent = Agent(
        TestModel(custom_output_args={'response': 'invalid'}),
        output_type=int,
        capabilities=[Hooks(output_retry=fail)],
    )
    with capture_run_messages() as messages, pytest.raises(error, match='notification failed'):
        await agent.run('Answer.')
    final = messages[-1]
    assert isinstance(final, ModelRequest)
    assert final.state == 'interrupted'


@dataclass(frozen=True)
class Case:
    end_strategy: EndStrategy
    expected_retries: tuple[tuple[str, int], ...] = ()


CASES: tuple[Case, ...] = (
    Case('early'),
    Case('graceful'),
    Case('exhaustive', (('bad', 1),)),
)


@pytest.mark.parametrize('case', [pytest.param(case, id=case.end_strategy) for case in CASES])
@pytest.mark.parametrize('method', ['run', 'run_stream'])
async def test_final_result_notifies_accepted_output_tool_retries(
    case: Case, method: Literal['run', 'run_stream']
) -> None:
    received: list[OutputRetryContext] = []
    calls: list[str] = []

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('request')
        return ModelResponse(
            parts=[ToolCallPart('good', {'response': 42}, 'good'), ToolCallPart('bad', {'response': 'invalid'}, 'bad')]
        )

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
        calls.append('request')
        yield {
            0: DeltaToolCall(name='good', json_args='{"response":42}', tool_call_id='good'),
            1: DeltaToolCall(name='bad', json_args='{"response":"invalid"}', tool_call_id='bad'),
        }

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        assert ctx.retry == 1
        assert retry_context.source == 'output_tool'
        assert retry_context.tool_call_id == 'bad'
        received.append(retry_context)

    agent = Agent(
        FunctionModel(response, stream_function=stream),
        deps_type=object,
        output_type=[ToolOutput(int, name='good'), ToolOutput(int, name='bad', max_retries=1)],
        end_strategy=case.end_strategy,
        capabilities=[Hooks(output_retry=record)],
    )
    if method == 'run':
        result = await agent.run('Answer.')
        assert result.output == 42
        messages = result.all_messages()
    else:
        async with agent.run_stream('Answer.') as streamed:
            assert await streamed.get_output() == 42
            messages = streamed.all_messages()

    assert tuple((retry.tool_name, retry.retry.attempt) for retry in received) == case.expected_retries
    assert (
        tuple(
            (part.tool_name, part.retry.attempt)
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.retry is not None
        )
        == case.expected_retries
    )
    assert calls == ['request']


async def test_function_tool_error_does_not_notify_uncharged_output_retries() -> None:
    received: list[OutputRetryContext] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
        yield {
            0: DeltaToolCall(name='good', json_args='{"response":42}', tool_call_id='good'),
            1: DeltaToolCall(name='bad', json_args='{"response":"invalid"}', tool_call_id='bad'),
            2: DeltaToolCall(name='explode', json_args='{}', tool_call_id='explode'),
        }

    def explode() -> int:
        raise RuntimeError('tool failed')

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        received.append(retry_context)

    agent = Agent(
        FunctionModel(stream_function=stream),
        deps_type=object,
        output_type=[ToolOutput(int, name='good'), ToolOutput(int, name='bad', max_retries=1)],
        tools=[Tool(explode, sequential=True)],
        end_strategy='exhaustive',
        capabilities=[Hooks(output_retry=record)],
    )
    with pytest.raises(RuntimeError, match='tool failed'):
        async with agent.run_stream('Answer.') as streamed:
            await streamed.get_output()
    assert received == []


@pytest.mark.parametrize('with_hook', [False, True])
async def test_absorbed_validator_cancellation_keeps_ordinary_retry_behavior(with_hook: bool) -> None:
    received: list[OutputRetryContext] = []

    async def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        received.append(retry_context)

    capabilities: list[AbstractCapability[object]] = [Hooks(output_retry=record)] if with_hook else []
    agent = Agent(TestModel(custom_output_text='ready'), capabilities=capabilities)

    @agent.output_validator
    async def validate(ctx: RunContext[object], output: str) -> str:
        if ctx.retry == 0:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass
            raise ModelRetry('Check once more.')
        return output

    result = await agent.run('Answer.')
    assert result.output == 'ready'
    assert result.usage.requests == 2
    assert len(received) == int(with_hook)
    request = result.all_messages()[2]
    assert isinstance(request, ModelRequest)
    assert any(isinstance(part, RetryFeedbackPart) for part in request.parts)


async def test_cancelled_validator_does_not_notify() -> None:
    received: list[OutputRetryContext] = []

    async def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:  # pragma: no cover
        received.append(retry_context)

    agent = Agent(TestModel(custom_output_text='draft'), capabilities=[Hooks(output_retry=record)])

    @agent.output_validator
    async def validate(output: str) -> str:
        raise asyncio.CancelledError('cancel validation')

    with pytest.raises(asyncio.CancelledError):
        await agent.run('Answer.')
    assert received == []
