from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from pydantic_ai import Agent, ModelRetry, RunContext, ToolDefinition
from pydantic_ai.capabilities import (
    AbstractCapability,
    ModelFallback,
    SelectModel,
    WrapModelRequestHandler,
    WrapperCapability,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestContext, ModelSelectionContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings


@dataclass
class ObserveStepModel(AbstractCapability[object]):
    phases: list[tuple[str, int, str]]
    attempts: list[tuple[int, str, int]]

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.phases.append(('before', ctx.run_step, request_context.model.model_name))
        assert ctx.model is request_context.model
        return request_context

    async def wrap_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        self.phases.append(('wrap', ctx.run_step, request_context.model.model_name))
        assert ctx.model is request_context.model
        return await handler(request_context)

    async def prepare_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.attempts.append((ctx.run_step, request_context.model.model_name, request_context.attempt))
        return request_context

    async def prepare_tools(self, ctx: RunContext[object], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        self.phases.append(('tools', ctx.run_step, ctx.model.model_name))
        return tool_defs


@pytest.mark.parametrize('source', ['agent', 'run', 'override', 'selector_before', 'selector_after', 'wrapped'])
async def test_output_switch_selects_before_step_hooks(
    source: Literal['agent', 'run', 'override', 'selector_before', 'selector_after', 'wrapped'],
) -> None:
    phases: list[tuple[str, int, str]] = []
    attempts: list[tuple[int, str, int]] = []
    selected_steps: list[int] = []
    backup_calls = 0

    def primary_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('draft')])

    def backup_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal backup_calls
        backup_calls += 1
        if backup_calls == 1:
            return ModelResponse(parts=[ToolCallPart('lookup', {})])
        return ModelResponse(parts=[TextPart('ready')])

    primary = FunctionModel(primary_response, model_name='primary')
    backup = FunctionModel(backup_response, model_name='backup')
    fallback = ModelFallback[object](backup, output_retry_threshold=1)
    policy = WrapperCapability(fallback) if source == 'wrapped' else fallback
    capabilities: list[AbstractCapability[object]] = [ObserveStepModel(phases, attempts), policy]

    def select(ctx: ModelSelectionContext[object]) -> Model:
        selected_steps.append(ctx.run_step)
        assert ctx.run_step == 1, 'The persistent candidate supersedes the ordinary selector.'
        return primary

    if source == 'selector_before':
        capabilities.insert(0, SelectModel(select))
    elif source == 'selector_after':
        capabilities.append(SelectModel(select))

    def settings(ctx: RunContext[object]) -> ModelSettings:
        phases.append(('settings', ctx.run_step, ctx.model.model_name))
        return {'temperature': 0.2}

    def lookup(ctx: RunContext[object]) -> str:
        phases.append(('tool', ctx.run_step, ctx.model.model_name))
        return 'evidence'

    other = TestModel(custom_output_text='wrong')
    agent = Agent(
        primary if source in {'agent', 'wrapped'} else other,
        deps_type=object,
        capabilities=capabilities,
        tools=[lookup],
        model_settings=settings,
        retries={'output': 2},
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    if source == 'override':
        with agent.override(model=primary):
            result = await agent.run('Answer.', model=other)
    elif source == 'run':
        result = await agent.run('Answer.', model=primary)
    else:
        result = await agent.run('Answer.')

    assert result.output == 'ready'
    assert attempts == [(1, 'primary', 1), (2, 'backup', 1), (3, 'backup', 1)]
    assert all(model == ('primary' if step == 1 else 'backup') for _, step, model in phases)
    for step in (1, 2, 3):
        assert {'before', 'wrap', 'settings', 'tools'} <= {phase for phase, number, _ in phases if number == step}
    assert ('tool', 2, 'backup') in phases
    assert bool(selected_steps) == source.startswith('selector_')


@pytest.mark.parametrize('pin', ['known', 'outside', 'missing'])
async def test_output_policy_preserves_suspended_selection(pin: Literal['known', 'outside', 'missing']) -> None:
    phases: list[tuple[str, int, str]] = []
    attempts: list[tuple[int, str, int]] = []

    def draft(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('draft')])

    primary = FunctionModel(draft, model_name='primary')
    pinned = FunctionModel(draft, model_name='pinned')
    backup = TestModel(custom_output_text='ready')
    fallback = ModelFallback[object](pinned, backup, output_retry_threshold=1)
    agent = Agent(
        primary,
        deps_type=object,
        capabilities=[ObserveStepModel(phases, attempts), fallback],
        retries={'output': 3},
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value.endswith('draft'):
            raise ModelRetry('Finish the draft.')
        return value

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Answer.')]),
        ModelResponse(
            parts=[TextPart('partial')],
            state='suspended',
            provider_response_id='job',
            model_id=pinned.model_id if pin == 'known' else 'outside:model' if pin == 'outside' else None,
        ),
    ]
    result = await agent.run(message_history=history)

    assert result.output == 'ready'
    expected: list[tuple[int, str, int]] = (
        [(1, 'pinned', 1), (2, backup.model_name, 1)]
        if pin == 'known'
        else [
            (1, 'primary', 1),
            (2, 'pinned', 1),
            (3, backup.model_name, 1),
        ]
    )
    assert attempts == expected
    for step, model, _ in expected:
        assert ('before', step, model) in phases
        assert ('wrap', step, model) in phases


@pytest.mark.parametrize('policy', ['disabled', 'empty', 'unlisted'])
async def test_request_only_suspended_hook_behavior_is_unchanged(
    policy: Literal['disabled', 'empty', 'unlisted'],
) -> None:
    phases: list[tuple[str, int, str]] = []
    attempts: list[tuple[int, str, int]] = []
    served: list[str] = []

    def pinned_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        served.append('pinned')
        return ModelResponse(parts=[TextPart('ready')])

    pinned = FunctionModel(pinned_response, model_name='pinned')
    threshold: dict[str, int] | None = (
        None if policy == 'disabled' else {} if policy == 'empty' else {'unused:model': 1}
    )
    agent = Agent(
        TestModel(model_name='primary'),
        deps_type=object,
        capabilities=[
            ObserveStepModel(phases, attempts),
            ModelFallback(pinned, output_retry_threshold=threshold),
        ],
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Answer.')]),
        ModelResponse(
            parts=[TextPart('partial')], state='suspended', provider_response_id='job', model_id=pinned.model_id
        ),
    ]
    assert (await agent.run(message_history=history)).output.endswith('ready')
    assert served == ['pinned']
    assert ('before', 1, 'primary') in phases
    assert ('wrap', 1, 'primary') in phases
    assert attempts == [(1, 'primary', 1), (1, 'pinned', 2)]
