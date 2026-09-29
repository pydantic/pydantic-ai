from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date

import pytest
from pydantic import BaseModel
from render.workflows import TaskContext, Workflows
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext


@dataclass
class Deps:
    tenant: str


class TypedPayload(BaseModel):
    due: date
    labels: list[str]


class TypedDeps(TypedDict):
    tenant: str


class CountDeps(TypedDict):
    retries: int


def typed_argument_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('typed arguments arrived')])
    return ModelResponse(
        parts=[
            ToolCallPart(
                'typed_lookup',
                {
                    'payload': {'due': '2026-09-15', 'labels': ['render', 'pydantic']},
                    'dates': ['2026-09-16', '2026-09-17'],
                },
                tool_call_id='typed-lookup',
            )
        ]
    )


def invalid_then_valid_typed_argument_model(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    del info
    parts = [part for message in messages for part in message.parts]
    if any(isinstance(part, ToolReturnPart) for part in parts):
        return ModelResponse(parts=[TextPart('typed arguments recovered')])
    due = '2026-09-15' if any(isinstance(part, RetryPromptPart) for part in parts) else 'not-a-date'
    return ModelResponse(
        parts=[
            ToolCallPart(
                'typed_lookup',
                {'payload': {'due': due, 'labels': ['render']}, 'dates': ['2026-09-16']},
                tool_call_id='typed-lookup',
            )
        ]
    )


async def test_model_and_function_transports_round_trip_public_run_context() -> None:
    seen: list[tuple[Deps, str | None, RunUsage, UsageLimits | None, object]] = []
    model = TestModel(call_tools=['lookup'])
    runtime = RenderWorkflows[Deps](Workflows(), deps_type=Deps)
    agent = Agent[Deps, str](
        model,
        name='transport-agent',
        deps_type=Deps,
        capabilities=[runtime],
    )

    @agent.tool
    async def lookup(ctx: RunContext[Deps], value: str) -> str:
        seen.append((ctx.deps, ctx.run_id, ctx.usage, ctx.usage_limits, ctx.model))
        return f'{ctx.deps.tenant}:{value}'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        result = await agent.run(
            'look up status',
            deps=Deps('acme'),
            usage=RunUsage(requests=2, input_tokens=10),
            usage_limits=UsageLimits(request_limit=8),
        )
        return result.output

    run_agent = runtime.task(run_agent_impl)
    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)

    assert seen
    deps, run_id, usage, usage_limits, child_model = seen[0]
    assert deps == Deps('acme')
    assert run_id is not None
    assert usage.requests >= 3
    assert usage_limits == UsageLimits(request_limit=8)
    assert child_model is model
    assert 'transport-agent__model.request' in context.task_names
    assert 'transport-agent__function_toolset__<agent>.call_tool' in context.task_names


async def test_function_arguments_are_revalidated_after_the_json_boundary() -> None:
    seen: list[tuple[TypedPayload, list[date]]] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(typed_argument_model),
        name='typed-arguments',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool_plain
    async def typed_lookup(payload: TypedPayload, dates: list[date]) -> str:
        seen.append((payload, dates))
        return 'ok'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('use typed arguments')).output

    run_agent = runtime.task(run_agent_impl)
    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)

    assert await pending == 'typed arguments arrived'
    assert seen == [
        (
            TypedPayload(due=date(2026, 9, 15), labels=['render', 'pydantic']),
            [date(2026, 9, 16), date(2026, 9, 17)],
        )
    ]
    assert 'typed-arguments__function_toolset__<agent>.call_tool' in context.task_names


async def test_invalid_typed_arguments_follow_normal_model_retry_control_flow() -> None:
    seen: list[TypedPayload] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(invalid_then_valid_typed_argument_model),
        name='invalid-typed-arguments',
        deps_type=type(None),
        retries=1,
        capabilities=[runtime],
    )

    @agent.tool_plain
    async def typed_lookup(payload: TypedPayload, dates: list[date]) -> str:
        assert dates == [date(2026, 9, 16)]
        seen.append(payload)
        return 'ok'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('retry invalid typed arguments')).output

    run_agent = runtime.task(run_agent_impl)
    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)

    assert await pending == 'typed arguments recovered'
    assert seen == [TypedPayload(due=date(2026, 9, 15), labels=['render'])]
    call_task = 'invalid-typed-arguments__function_toolset__<agent>.call_tool'
    # Pydantic rejects the malformed date before dispatching the tool call. The
    # corrected retry is the only invocation that crosses the task boundary.
    assert context.task_names.count(call_task) == 1


async def test_parameterized_mapping_dependencies_round_trip() -> None:
    seen: list[dict[str, str]] = []
    deps_type = dict[str, str]
    runtime = RenderWorkflows[dict[str, str]](Workflows(), deps_type=deps_type)
    agent = Agent[dict[str, str], str](
        TestModel(call_tools=['lookup']),
        name='mapping-deps',
        deps_type=deps_type,
        capabilities=[runtime],
    )

    @agent.tool
    async def lookup(ctx: RunContext[dict[str, str]], value: str) -> str:
        seen.append(ctx.deps)
        return f'{ctx.deps["tenant"]}:{value}'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('lookup', deps={'tenant': 'acme'})).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)
    assert seen == [{'tenant': 'acme'}]


async def test_typed_dict_dependencies_round_trip() -> None:
    seen: list[TypedDeps] = []
    runtime = RenderWorkflows[TypedDeps](Workflows(), deps_type=TypedDeps)
    agent = Agent[TypedDeps, str](
        TestModel(call_tools=['lookup']),
        name='typed-dict-deps',
        deps_type=TypedDeps,
        capabilities=[runtime],
    )

    @agent.tool
    async def lookup(ctx: RunContext[TypedDeps], value: str) -> str:
        seen.append(ctx.deps)
        return f'{ctx.deps["tenant"]}:{value}'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('lookup', deps=TypedDeps(tenant='acme'))).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)
    assert seen == [TypedDeps(tenant='acme')]


async def test_invalid_typed_dict_dependencies_fail_codec_validation() -> None:
    runtime = RenderWorkflows[CountDeps](Workflows(), deps_type=CountDeps)
    agent = Agent[CountDeps, str](
        TestModel(),
        name='invalid-typed-dict-deps',
        deps_type=CountDeps,
        capabilities=[runtime],
    )

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        invalid_deps = {'retries': 'not-an-integer'}
        return (await agent.run('inspect', deps=invalid_deps)).output  # type: ignore[arg-type]

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)
    with pytest.raises(UserWarning, match='Expected `int`'):
        await pending


async def _run_in_workflow(
    agent: Agent[None, str], runtime: RenderWorkflows[None], context: TaskContext, *, model: str | None = None
) -> str:
    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('inspect', model=model) if model else await agent.run('inspect')).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    return await pending


async def test_child_task_model_is_guarded_when_the_worker_registers_no_instance() -> None:
    # A model-name string is deliberately not resolved when the agent is bound, so nothing is
    # registered under `default` and a child task has no instance to attach to its run context.
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str]('test', name='string-model-agent', deps_type=type(None), capabilities=[runtime])

    @agent.tool
    async def inspect_model(ctx: RunContext[None]) -> str:
        return str(ctx.model)

    with pytest.raises(UserError, match="'model' is not available on 'RenderRunContext'"):
        await _run_in_workflow(agent, runtime, RecordingTaskContext())


async def test_registered_model_id_resolves_to_its_instance_in_the_child_task() -> None:
    alternate = TestModel(custom_output_text='from the alternate model')
    runtime = RenderWorkflows[None](Workflows(), models={'alternate': alternate})
    agent = Agent[None, str](
        TestModel(call_tools=['inspect_model']),
        name='registry-agent',
        deps_type=type(None),
        capabilities=[runtime],
    )
    seen: list[object] = []

    @agent.tool
    async def inspect_model(ctx: RunContext[None]) -> str:
        seen.append(ctx.model)
        return 'ok'

    output = await _run_in_workflow(agent, runtime, RecordingTaskContext(), model='alternate')

    assert output == 'from the alternate model'
    assert seen == [alternate]


async def test_unregistered_model_instance_is_rejected_before_dispatch() -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](TestModel(), name='instance-agent', deps_type=type(None), capabilities=[runtime])
    context = RecordingTaskContext()

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('inspect', model=TestModel(custom_output_text='unregistered'))).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)

    # An instance cannot cross the task boundary, so the run fails before any task is started
    # rather than rebuilding a different model from its name on the worker.
    with pytest.raises(UserError, match='was not registered with `RenderWorkflows`'):
        await pending
    assert context.task_names == []
