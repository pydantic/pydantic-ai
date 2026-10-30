from __future__ import annotations

import pytest

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import Hooks, ModelFallback
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel


async def test_output_retirement_restricts_sticky_fallback_and_resets_next_run() -> None:
    calls: list[str] = []
    steps: list[tuple[int, str]] = []
    backup_calls = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('primary')
        raise ModelHTTPError(status_code=500, model_name='primary')

    def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal backup_calls
        calls.append('backup')
        backup_calls += 1
        return ModelResponse(parts=[TextPart('draft' if backup_calls == 2 else 'ready')])

    def final(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('final')
        raise ModelHTTPError(status_code=500, model_name='final')

    def observe(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        steps.append((ctx.run_step, ctx.model.model_name))
        return request_context

    agent = Agent(
        FunctionModel(primary, model_name='primary'),
        deps_type=object,
        capabilities=[
            ModelFallback(
                FunctionModel(backup, model_name='backup'),
                FunctionModel(final, model_name='final'),
                sticky=True,
                output_retry_threshold=1,
            ),
            Hooks(before_model_request=observe),
        ],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    first = await agent.run('Answer.')
    assert first.output == 'ready'
    assert calls == ['primary', 'backup']
    steps.clear()

    with pytest.raises(FallbackExceptionGroup):
        await agent.run('Continue.', message_history=first.all_messages())
    assert calls == ['primary', 'backup', 'backup', 'final']
    assert steps[-1] == (2, 'final')

    assert (await agent.run('Continue again.', message_history=first.all_messages())).output == 'ready'
    assert calls == ['primary', 'backup', 'backup', 'final', 'backup']


async def test_output_only_switch_does_not_start_sticky_history() -> None:
    calls: list[str] = []

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('primary')
        return ModelResponse(parts=[TextPart('draft')])

    def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('backup')
        return ModelResponse(parts=[TextPart('ready')])

    agent = Agent(
        FunctionModel(primary, model_name='primary'),
        capabilities=[ModelFallback(FunctionModel(backup, model_name='backup'), sticky=True, output_retry_threshold=1)],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    first = await agent.run('Answer.')
    assert first.output == 'ready'
    assert (await agent.run('Continue.', message_history=first.all_messages())).output == 'ready'
    assert calls == ['primary', 'backup', 'primary', 'backup']
