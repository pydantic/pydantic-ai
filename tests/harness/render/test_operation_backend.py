from __future__ import annotations

from collections.abc import AsyncIterable

import pytest
from render.workflows import Options, Retry, Workflows

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolset, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext, RecordingWorkflows, run_agent_in_task


def build_agent(*, options: Options | None = None) -> tuple[Agent[None, str], RenderWorkflows[None]]:
    app = Workflows()
    runtime = RenderWorkflows[None](app, deps_type=type(None), model_options=options)
    agent = Agent[None, str](
        TestModel(),
        name='support',
        deps_type=type(None),
        capabilities=[runtime],
    )
    return agent, runtime


async def test_runs_inline_outside_render_context() -> None:
    agent, runtime = build_agent()
    assert isinstance((await agent.run('inline')).output, str)
    assert runtime.in_durable_context is False


async def test_dispatches_registered_task_and_activates_child_context() -> None:
    agent, runtime = build_agent()
    context = RecordingTaskContext()

    assert isinstance(await run_agent_in_task(agent, runtime, context), str)
    assert context.task_names == ['support__model.request']


async def test_invocation_options_that_differ_from_registration_are_rejected() -> None:
    runtime = RenderWorkflows[None](
        Workflows(),
        deps_type=type(None),
        tool_options=Options(timeout_seconds=60),
        resolve_tool_options=lambda operation_id, tool, tool_name: Options(timeout_seconds=60 if tool is None else 120),
    )
    agent = Agent[None, str](
        TestModel(call_tools=['ping']),
        name='per-call-options',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def ping(ctx: RunContext[None]) -> str:
        del ctx
        return 'pong'

    # Render fixes a task's retry, timeout, and plan when the task is registered, so options
    # resolved later for one tool cannot take effect and are rejected instead of ignored.
    with pytest.raises(UserError, match='fixed when an agent is bound'):
        await run_agent_in_task(agent, runtime, RecordingTaskContext())


async def test_named_toolsets_register_and_use_distinct_options() -> None:
    app = RecordingWorkflows()
    fast_options = Options(timeout_seconds=60, plan='starter')
    slow_options = Options(timeout_seconds=300, plan='standard')
    registration_calls: list[tuple[str, object | None, str]] = []

    def resolve_options(operation_id: object, tool: object | None, tool_name: str) -> Options | None:
        toolset_id = getattr(operation_id, 'toolset_id', '')
        if tool is None:
            registration_calls.append((toolset_id, tool, tool_name))
        if toolset_id == 'fast-tools':
            return fast_options
        if toolset_id == 'slow-tools':
            return slow_options
        return None

    async def fast_lookup() -> str:
        return 'fast'

    async def slow_lookup() -> str:
        return 'slow'

    runtime = RenderWorkflows[None](
        app,
        deps_type=type(None),
        resolve_tool_options=resolve_options,
    )
    agent = Agent[None, str](
        TestModel(call_tools=['fast_lookup']),
        name='toolset-options',
        deps_type=type(None),
        toolsets=[
            FunctionToolset([fast_lookup], id='fast-tools'),
            FunctionToolset([slow_lookup], id='slow-tools'),
        ],
        capabilities=[runtime],
    )

    assert app.options['toolset-options__function_toolset__fast-tools.call_tool'] == fast_options
    assert app.options['toolset-options__function_toolset__slow-tools.call_tool'] == slow_options
    assert ('fast-tools', None, '') in registration_calls
    assert ('slow-tools', None, '') in registration_calls
    assert isinstance(await run_agent_in_task(agent, runtime, RecordingTaskContext()), str)


async def test_registered_task_options_are_snapshotted_when_the_agent_is_bound() -> None:
    """Registration copies the caller's options, so later mutation cannot change a live task.

    `TaskDefinition` publishes only `name` and `func`, so the registered retry, timeout, and plan
    cannot be read back through the public Render SDK. The snapshot is observable instead through
    the invocation check: the resolver hands back the very object that was registered, and the
    call is rejected only because registration kept a copy of its earlier values.
    """
    options = Options(retry=Retry(max_retries=2, wait_duration_ms=100), timeout_seconds=90, plan='flex')
    runtime = RenderWorkflows[None](
        Workflows(),
        deps_type=type(None),
        tool_options=options,
        resolve_tool_options=lambda operation_id, tool, tool_name: options,
    )
    agent = Agent[None, str](
        TestModel(call_tools=['ping']),
        name='snapshot-options',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def ping(ctx: RunContext[None]) -> str:
        del ctx
        return 'pong'

    assert options.retry is not None
    options.retry.max_retries = 99
    options.timeout_seconds = 1

    with pytest.raises(UserError, match='fixed when an agent is bound'):
        await run_agent_in_task(agent, runtime, RecordingTaskContext())


@pytest.mark.parametrize('with_events', [False, True])
def test_options_resolver_failure_registers_no_tasks_and_app_can_be_reused(with_events: bool) -> None:
    """Even late per-tool configuration failures leave the app usable for a corrected agent."""
    app = RecordingWorkflows()

    async def events(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:
            pass

    def lookup() -> str:
        return 'found'

    def invalid_options(operation: object, tool: object | None, name: str) -> None:
        if name == 'lookup':
            raise ValueError('invalid lookup options')

    with pytest.raises(ValueError, match='invalid lookup options'):
        Agent(
            TestModel(),
            name='resolver-failure',
            deps_type=type(None),
            toolsets=[FunctionToolset([lookup], id='lookup')],
            capabilities=[
                RenderWorkflows[None](
                    app,
                    event_stream_handler=events if with_events else None,
                    resolve_tool_options=invalid_options,
                )
            ],
        )
    assert app.registered_task_names == []

    Agent(
        TestModel(),
        name='resolver-failure',
        deps_type=type(None),
        toolsets=[FunctionToolset([lookup], id='lookup')],
        capabilities=[RenderWorkflows[None](app)],
    )
    assert 'resolver-failure__function_toolset__lookup.call_tool' in app.registered_task_names
    assert len(app.registered_task_names) == len(set(app.registered_task_names))


async def test_reused_capability_registers_each_agents_tasks_once() -> None:
    app = RecordingWorkflows()
    runtime = RenderWorkflows[None](app)
    agents = [
        Agent(TestModel(), name=name, deps_type=type(None), capabilities=[runtime]) for name in ('first', 'second')
    ]
    registered_names = app.registered_task_names.copy()
    assert len(registered_names) == len(set(registered_names))
    for agent in agents:
        context = RecordingTaskContext()
        with runtime.activate(context):
            await agent.run('go')
        assert context.task_names == [f'{agent.name}__model.request']
    # Running the bound copies must not register additional operations.
    assert [name for name in app.registered_task_names if '__' in name] == registered_names


def test_different_static_tool_options_fail_before_registration() -> None:
    app = RecordingWorkflows()

    def slow() -> str:
        return 'slow'

    with pytest.raises(UserError, match='separate named toolsets'):
        Agent(
            TestModel(),
            name='unsupported-per-tool',
            toolsets=[FunctionToolset([slow], id='lookup')],
            capabilities=[
                RenderWorkflows(
                    app,
                    resolve_tool_options=lambda op, tool, name: (
                        Options(timeout_seconds=300) if name == 'slow' else None
                    ),
                )
            ],
        )
    assert app.registered_task_names == []
