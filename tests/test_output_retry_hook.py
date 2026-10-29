from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import ValidationError

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.capabilities import AbstractCapability, Hooks, OutputContext, OutputRetryContext, WrapperCapability
from pydantic_ai.capabilities.abstract import RawOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, RetryInfo
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize('registration', ['constructor', 'decorator', 'timeout_decorator', 'wrapped'])
async def test_hook_registration_and_context(
    registration: Literal['constructor', 'decorator', 'timeout_decorator', 'wrapped'],
) -> None:
    received: list[OutputRetryContext] = []

    async def record(ctx: RunContext[str], *, retry_context: OutputRetryContext) -> None:
        assert ctx.deps == 'dependency'
        assert (ctx.retry, ctx.max_retries) == (1, 1)
        received.append(retry_context)

    hooks = Hooks[str]()
    if registration == 'constructor':
        hooks = Hooks(after_output_retry=record)
    elif registration == 'timeout_decorator':
        hooks.on.after_output_retry(timeout=10)(record)
    else:
        hooks.on.after_output_retry(record)
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

        async def after_output_retry(self, ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
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

    @hooks.on.after_output_retry
    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        received.append(retry_context)

    agent = Agent(TestModel(custom_output_text='draft'), capabilities=[hooks], retries={'output': 0})

    @agent.output_validator
    def validate(output: str) -> str:
        raise ModelRetry('Finish the draft.')

    with pytest.raises(UnexpectedModelBehavior, match='output retries'):
        await agent.run('Answer.')
    assert received == []


async def test_none_output_validator_retry_notifies() -> None:
    received: list[OutputRetryContext] = []

    def empty(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        received.append(retry_context)

    agent = Agent(FunctionModel(empty), output_type=str | None, capabilities=[Hooks(after_output_retry=record)])

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

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
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
        capabilities=[Hooks(after_output_retry=record, output_validate_error=recover)],
    )
    assert (await agent.run('Answer.')).output == 42
    assert received == []


@pytest.mark.parametrize('error', [ValueError, ModelRetry])
async def test_notification_exception_stops_run_and_preserves_tool_returns(error: type[Exception]) -> None:
    def fail(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        assert retry_context.source == 'output_tool'
        raise error('notification failed')

    agent = Agent(
        TestModel(custom_output_args={'response': 'invalid'}),
        output_type=int,
        capabilities=[Hooks(after_output_retry=fail)],
    )
    with capture_run_messages() as messages, pytest.raises(error, match='notification failed'):
        await agent.run('Answer.')
    final = messages[-1]
    assert isinstance(final, ModelRequest)
    assert final.state == 'interrupted'
