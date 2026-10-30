from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Literal

import pytest

from pydantic_ai import Agent, ModelRetry, RunContext, ToolDefinition
from pydantic_ai.capabilities import (
    AbstractCapability,
    CombinedCapability,
    Hooks,
    ModelFallback,
    ResolveModelId,
    SelectModel,
    WrapModelRequestHandler,
    WrapperCapability,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior
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


@dataclass(frozen=True)
class SelectionCase:
    source: Literal['agent', 'fallback_default', 'run', 'override', 'selector_before', 'selector_after', 'wrapped']
    expected_models: tuple[str, ...] = ('primary', 'backup', 'backup')
    expected_output: str | None = 'ready'
    selected_steps: tuple[int, ...] = ()
    warning: str | None = None


SELECTION_CASES: tuple[SelectionCase, ...] = (
    SelectionCase('agent'),
    SelectionCase('fallback_default'),
    SelectionCase('run', ('primary', 'primary', 'primary'), None, warning='output_retry_threshold=1'),
    SelectionCase('override', ('primary', 'primary', 'primary'), None),
    SelectionCase('selector_before', ('primary', 'primary', 'primary'), None, (1, 1, 2, 3)),
    SelectionCase('selector_after', ('primary', 'primary', 'primary'), None, (1, 1, 2, 3)),
    SelectionCase('wrapped'),
)


@pytest.mark.parametrize('case', [pytest.param(case, id=case.source) for case in SELECTION_CASES])
async def test_output_switch_selects_before_step_hooks(case: SelectionCase) -> None:
    source = case.source
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
    fallback = ModelFallback[object](
        *([primary, backup] if source == 'fallback_default' else [backup]), output_retry_threshold=1
    )
    policy = WrapperCapability(fallback) if source == 'wrapped' else fallback
    capabilities: list[AbstractCapability[object]] = [ObserveStepModel(phases, attempts), policy]

    def select(ctx: ModelSelectionContext[object]) -> Model:
        selected_steps.append(ctx.run_step)
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
        primary if source in {'agent', 'wrapped'} else None if source == 'fallback_default' else other,
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

    output: str | None = None
    with (
        pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries')
        if case.expected_output is None
        else nullcontext()
    ):
        with pytest.warns(UserWarning, match=case.warning) if case.warning is not None else nullcontext():
            if source == 'override':
                with agent.override(model=primary):
                    output = (await agent.run('Answer.', model=other)).output
            elif source == 'run':
                output = (await agent.run('Answer.', model=primary)).output
            else:
                output = (await agent.run('Answer.')).output
    assert output == case.expected_output
    assert attempts == [(step, model, 1) for step, model in enumerate(case.expected_models, 1)]
    assert all(model == case.expected_models[step - 1] for _, step, model in phases)
    assert backup_calls == case.expected_models.count('backup')
    if case.expected_output is not None:
        assert ('tool', 2, 'backup') in phases
    for step in (1, 2, 3):
        assert {'before', 'wrap', 'settings', 'tools'} <= {phase for phase, number, _ in phases if number == step}
    # Resolving the run-bound fallback copy re-evaluates the bootstrap selector on step one.
    assert tuple(selected_steps) == case.selected_steps


@pytest.mark.parametrize('threshold', [None, 1])
async def test_wrapped_container_keeps_output_selection_a_default(threshold: int | None) -> None:
    primary = TestModel(custom_output_text='primary')
    selected = TestModel(custom_output_text='selected')
    agent = Agent(
        capabilities=[
            SelectModel(lambda ctx: selected),
            WrapperCapability(CombinedCapability([ModelFallback(primary, output_retry_threshold=threshold)])),
        ]
    )

    # Static container contributions retain their existing precedence. The opt-in run-bound
    # output selector stays a default even through a wrapped container.
    assert (await agent.run('Answer.')).output == ('primary' if threshold is None else 'selected')


async def test_output_counts_survive_serving_model_reconstruction() -> None:
    calls: list[str] = []
    reconstructed: list[FunctionModel] = []
    hooks = Hooks[object]()

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('primary')
        return ModelResponse(parts=[TextPart('draft')])

    def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('backup')
        return ModelResponse(parts=[TextPart('ready')])

    @hooks.on.before_model_request
    def reconstruct(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        if request_context.model.model_name == 'primary':
            model = FunctionModel(primary, model_name='primary')
            reconstructed.append(model)
            return replace(request_context, model=model)
        return request_context

    agent = Agent(
        FunctionModel(primary, model_name='primary'),
        capabilities=[hooks, ModelFallback(FunctionModel(backup), output_retry_threshold=2)],
        retries={'output': 3},
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    assert (await agent.run('Answer.')).output == 'ready'
    assert calls == ['primary', 'primary', 'backup']
    assert reconstructed[0] is not reconstructed[1]


async def test_output_counts_use_configured_aliases() -> None:
    served: list[str | None] = []
    hooks = Hooks[object]()
    primary = TestModel(custom_output_text='draft')
    models: dict[str, Model] = {'first': primary, 'second': primary, 'backup': TestModel(custom_output_text='ready')}

    @hooks.on.before_model_request
    def record(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        served.append(request_context.model_id)
        return request_context

    agent = Agent(
        'first',
        capabilities=[
            hooks,
            ModelFallback('second', 'backup', output_retry_threshold={'first': 2, 'second': 2}),
            ResolveModelId(lambda ctx, model_id: models.get(model_id)),
        ],
        retries={'output': 4},
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    assert (await agent.run('Answer.')).output == 'ready'
    assert served == ['first', 'first', 'second', 'second', 'backup']


@dataclass(frozen=True)
class SuspendedCase:
    id: str
    pin: Literal['known', 'outside', 'missing']
    mapped: bool
    expected_models: tuple[str, ...]


SUSPENDED_CASES: tuple[SuspendedCase, ...] = (
    SuspendedCase('known-uniform', 'known', False, ('pinned', 'backup')),
    SuspendedCase('known-mapped', 'known', True, ('pinned', 'backup')),
    SuspendedCase('outside-uniform', 'outside', False, ('primary', 'pinned', 'backup')),
    SuspendedCase('outside-mapped', 'outside', True, ('primary', 'pinned', 'backup')),
    SuspendedCase('missing-uniform', 'missing', False, ('primary', 'pinned', 'backup')),
    SuspendedCase('missing-mapped', 'missing', True, ('primary', 'pinned', 'backup')),
)


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in SUSPENDED_CASES])
async def test_output_policy_preserves_suspended_selection(case: SuspendedCase) -> None:
    phases: list[tuple[str, int, str]] = []
    attempts: list[tuple[int, str, int]] = []

    def draft(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('draft')])

    primary = FunctionModel(draft, model_name='primary')
    pinned = FunctionModel(draft, model_name='pinned')
    backup = TestModel(custom_output_text='ready', model_name='backup')
    threshold: int | dict[str, int] = {primary.model_id: 1, pinned.model_id: 1} if case.mapped else 1
    fallback = ModelFallback[object](pinned, backup, output_retry_threshold=threshold)
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

    pin_ids: dict[str, str | None] = {'known': pinned.model_id, 'outside': 'outside:model', 'missing': None}
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Answer.')]),
        ModelResponse(
            parts=[TextPart('partial')],
            state='suspended',
            provider_response_id='job',
            model_id=pin_ids[case.pin],
        ),
    ]
    result = await agent.run(message_history=history)

    assert result.output == 'ready'
    expected: list[tuple[int, str, int]] = [(step, model, 1) for step, model in enumerate(case.expected_models, 1)]
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
