"""Black-box completion contracts for Render-backed agent operations."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Literal, ParamSpec, TypeVar

import anyio
from render.workflows import Options, TaskContext, TaskDefinition

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_ai_harness import RenderWorkflows
from pydantic_ai_harness.subagents import SubAgent, SubAgents

from .conftest import RecordingTaskContext, RecordingWorkflows, run_agent_in_task

P = ParamSpec('P')
R = TypeVar('R')


class NestedTaskContext(RecordingTaskContext):
    """Execute public task definitions while recording nested run depth."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, int]] = []
        self.stack: list[str] = []

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self.calls.append((task.name, len(self.stack) + 1))
        self.stack.append(task.name)
        try:
            return await super().run(task, *args, **kwargs)
        finally:
            self.stack.pop()


def three_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call three statically known tools once, then finish."""
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(
        parts=[
            ToolCallPart('fast_lookup', {}, tool_call_id='fast'),
            ToolCallPart('slow_lookup', {}, tool_call_id='slow'),
            ToolCallPart('inline_lookup', {}, tool_call_id='inline'),
        ]
    )


def resolve_lookup_options(
    operation_id: object, tool: object | None, tool_name: str
) -> Options | Literal[False] | None:
    """Assign Options to named toolsets and opt one function tool out of a Render task."""
    del tool
    if getattr(operation_id, 'toolset_id', '') == 'fast':
        return Options(timeout_seconds=30, plan='starter')
    if getattr(operation_id, 'toolset_id', '') == 'slow':
        return Options(timeout_seconds=300, plan='standard')
    return False if tool_name == 'inline_lookup' else None


def build_toolset_options_agent() -> tuple[Agent[None, str], RenderWorkflows[None], RecordingWorkflows]:
    """Two named toolsets with different policies and one inline opt-out."""

    async def fast_lookup() -> str:
        return 'fast'

    async def slow_lookup() -> str:
        return 'slow'

    async def inline_lookup() -> str:
        return 'inline'

    app = RecordingWorkflows()
    runtime = RenderWorkflows[None](app, deps_type=type(None), resolve_tool_options=resolve_lookup_options)
    agent = Agent[None, str](
        FunctionModel(three_tools),
        name='per-function',
        deps_type=type(None),
        toolsets=[
            FunctionToolset([fast_lookup], id='fast'),
            FunctionToolset([slow_lookup], id='slow'),
            FunctionToolset([inline_lookup], id='inline'),
        ],
        capabilities=[runtime],
    )
    return agent, runtime, app


async def test_named_toolsets_apply_policies_and_function_opt_out() -> None:
    agent, runtime, app = build_toolset_options_agent()
    _, _, second_app = build_toolset_options_agent()
    registered = {
        name: value
        for name, value in app.options.items()
        if any(f'__function_toolset__{toolset}.' in name for toolset in ('fast', 'slow'))
        and name.endswith('.call_tool')
    }
    second = {
        name: value
        for name, value in second_app.options.items()
        if any(f'__function_toolset__{toolset}.' in name for toolset in ('fast', 'slow'))
        and name.endswith('.call_tool')
    }

    assert registered == second
    assert len(registered) == 2
    assert 'per-function__function_toolset__fast.call_tool' in registered
    assert 'per-function__function_toolset__slow.call_tool' in registered
    assert {value.timeout_seconds for value in registered.values()} == {30, 300}
    assert {value.plan for value in registered.values()} == {'starter', 'standard'}

    context = NestedTaskContext()
    assert await run_agent_in_task(agent, runtime, context) == 'done'
    invoked = [name for name, _depth in context.calls if name in registered]
    assert sorted(invoked) == sorted(registered)
    assert not any('__function_toolset__inline.' in name for name, _depth in context.calls)


def delegate_once(agent_name: str) -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('parent done')])
        return ModelResponse(
            parts=[ToolCallPart('delegate_task', {'agent_name': agent_name, 'task': 'work'}, tool_call_id='delegate')]
        )

    return FunctionModel(model)


def tool_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('child done')])
    return ModelResponse(parts=[ToolCallPart('child_tool', {}, tool_call_id='child-tool')])


def build_nested_agents() -> tuple[Agent[None, str], RenderWorkflows[None], RecordingWorkflows]:
    """Build explicit parent and child agents against one Workflows app."""
    app = RecordingWorkflows()
    child_runtime = RenderWorkflows[None](app, deps_type=type(None))
    child = Agent[None, str](
        FunctionModel(tool_then_finish),
        name='nested-child',
        deps_type=type(None),
        capabilities=[child_runtime],
    )

    @child.tool_plain
    async def child_tool() -> str:
        return 'tool done'

    parent_runtime = RenderWorkflows[None](
        app,
        deps_type=type(None),
        resolve_tool_options=lambda _operation, _tool, name: False if name == 'delegate_task' else None,
    )
    parent = Agent[None, str](
        delegate_once('nested-child'),
        name='nested-parent',
        deps_type=type(None),
        capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None), parent_runtime],
    )
    return parent, parent_runtime, app


async def test_explicit_subagent_uses_shared_app_for_direct_child_task_runs() -> None:
    parent, parent_runtime, _ = build_nested_agents()
    context = NestedTaskContext()
    assert await run_agent_in_task(parent, parent_runtime, context) == 'parent done'
    assert ('nested-child__model.request', 1) in context.calls
    assert ('nested-child__function_toolset__<agent>.call_tool', 1) in context.calls
    assert not [name for name, _depth in context.calls if '__function_toolset__sub_agents' in name]


def returned_tool_names(messages: list[ModelMessage]) -> set[str]:
    """Return tool names with results in serialized model history."""
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolReturnPart)}


def grandchild_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call the grandchild's function tool once, then finish."""
    del info
    if 'grandchild_tool' in returned_tool_names(messages):
        return ModelResponse(parts=[TextPart('grandchild done')])
    return ModelResponse(parts=[ToolCallPart('grandchild_tool', {}, tool_call_id='grandchild-tool')])


def child_delegating_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call a child tool, delegate to the grandchild, then finish."""
    del info
    returned = returned_tool_names(messages)
    if 'delegate_task' in returned:
        return ModelResponse(parts=[TextPart('child done')])
    if 'child_tool' in returned:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    'delegate_task',
                    {'agent_name': 'nested-grandchild', 'task': 'finish the work'},
                    tool_call_id='child-to-grandchild',
                )
            ]
        )
    return ModelResponse(parts=[ToolCallPart('child_tool', {}, tool_call_id='child-tool')])


def build_two_level_agents() -> tuple[Agent[None, str], RenderWorkflows[None], RecordingWorkflows]:
    """Build parent, child, and grandchild agents against one Workflows app."""
    app = RecordingWorkflows()
    grandchild_runtime = RenderWorkflows[None](app, deps_type=type(None))
    grandchild = Agent[None, str](
        FunctionModel(grandchild_model),
        name='nested-grandchild',
        deps_type=type(None),
        capabilities=[grandchild_runtime],
    )

    @grandchild.tool_plain
    async def grandchild_tool() -> str:
        return 'grandchild tool done'

    child_runtime = RenderWorkflows[None](
        app,
        deps_type=type(None),
        resolve_tool_options=lambda _operation, _tool, name: False if name == 'delegate_task' else None,
    )
    child = Agent[None, str](
        FunctionModel(child_delegating_model),
        name='nested-child-two-level',
        deps_type=type(None),
        capabilities=[
            SubAgents(agents=[SubAgent(grandchild)], agent_folders=None),
            child_runtime,
        ],
    )

    @child.tool_plain
    async def child_tool() -> str:
        return 'child tool done'

    parent_runtime = RenderWorkflows[None](
        app,
        deps_type=type(None),
        resolve_tool_options=lambda _operation, _tool, name: False if name == 'delegate_task' else None,
    )
    parent = Agent[None, str](
        delegate_once('nested-child-two-level'),
        name='nested-parent-two-level',
        deps_type=type(None),
        capabilities=[
            SubAgents(agents=[SubAgent(child)], agent_folders=None),
            parent_runtime,
        ],
    )
    return parent, parent_runtime, app


async def test_two_level_explicit_delegation_runs_supported_operations_and_terminates() -> None:
    parent, runtime, _ = build_two_level_agents()
    context = NestedTaskContext()

    with anyio.fail_after(5):
        assert await run_agent_in_task(parent, runtime, context) == 'parent done'

    names = [name for name, _depth in context.calls]
    assert names.count('nested-child-two-level__model.request') == 3
    assert names.count('nested-child-two-level__function_toolset__<agent>.call_tool') == 1
    assert names.count('nested-grandchild__model.request') == 2
    assert names.count('nested-grandchild__function_toolset__<agent>.call_tool') == 1
    assert all(depth == 1 for name, depth in context.calls if name.startswith(('nested-child', 'nested-grandchild')))
    assert not [name for name in names if '__function_toolset__sub_agents' in name]


def two_sibling_delegations(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Issue two sibling delegations once, based only on message history."""
    del info
    if returned_tool_names(messages):
        return ModelResponse(parts=[TextPart('parent done')])
    return ModelResponse(
        parts=[
            ToolCallPart(
                'delegate_task',
                {'agent_name': 'budget-worker', 'task': 'first'},
                tool_call_id='first-delegation',
            ),
            ToolCallPart(
                'delegate_task',
                {'agent_name': 'budget-worker', 'task': 'second'},
                tool_call_id='second-delegation',
            ),
        ]
    )


async def test_inline_subagents_charges_concurrent_max_calls_before_awaiting() -> None:
    """The budget is scoped narrowly to one active parent task run in this process."""
    executions: list[str] = []

    def worker_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        executions.append('ran')
        return ModelResponse(parts=[TextPart('worker done')])

    app = RecordingWorkflows()
    runtime = RenderWorkflows[None](
        app,
        deps_type=type(None),
        resolve_tool_options=lambda _operation, _tool, name: False if name == 'delegate_task' else None,
    )
    worker = Agent[None, str](
        FunctionModel(worker_model),
        name='budget-worker',
        deps_type=type(None),
    )
    parent = Agent[None, str](
        FunctionModel(two_sibling_delegations),
        name='budget-parent',
        deps_type=type(None),
        capabilities=[
            SubAgents(agents=[SubAgent(worker, max_calls=1)], agent_folders=None),
            runtime,
        ],
    )
    context = NestedTaskContext()

    @runtime.task
    async def run_budget_parent(ctx: TaskContext) -> tuple[str, list[str]]:
        del ctx
        result = await parent.run('go')
        returns = [
            str(part.content)
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task'
        ]
        return result.output, returns

    pending = run_budget_parent.func(context)
    assert inspect.isawaitable(pending)
    with anyio.fail_after(5):
        output, returns = await pending

    assert output == 'parent done'
    assert executions == ['ran']
    assert len(returns) == 2
    assert 'worker done' in returns
    assert any("Delegate budget for 'budget-worker' is exhausted" in value for value in returns)
    assert not [name for name, _depth in context.calls if '__function_toolset__sub_agents' in name]


def renamed_for_model(func: Callable[[], Awaitable[str]], model_facing_name: str) -> Tool[None]:
    """Expose a statically known function tool under a different model-facing name."""

    async def prepare(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition:
        del ctx
        return replace(tool_def, name=model_facing_name)

    return Tool[None](func, prepare=prepare)


def tool_returns(messages: list[ModelMessage]) -> dict[str, str]:
    """Return model-facing tool names mapped to their returned content."""
    return {
        part.tool_name: str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }


def call_renamed_tool(model_facing_name: str) -> FunctionModel:
    """Call one model-facing tool name once, then report what came back."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        returned = tool_returns(messages)
        if returned:
            return ModelResponse(
                parts=[TextPart('|'.join(f'{name}={value}' for name, value in sorted(returned.items())))]
            )
        return ModelResponse(parts=[ToolCallPart(model_facing_name, {}, tool_call_id=model_facing_name)])

    return FunctionModel(model)


def resolve_prepared_options(
    operation_id: object, tool: object | None, tool_name: str
) -> Options | Literal[False] | None:
    """Opt out a prepared tool by the original name held by its toolset."""
    del operation_id, tool
    if tool_name == 'inline_report':
        return False
    return None


def build_prepared_agent(
    tools: list[Tool[None]], *, agent_name: str, toolset_id: str, called_name: str
) -> tuple[Agent[None, str], RenderWorkflows[None], RecordingWorkflows]:
    """One named FunctionToolset whose tools are renamed for the model by `prepare`."""
    app = RecordingWorkflows()
    runtime = RenderWorkflows[None](app, deps_type=type(None), resolve_tool_options=resolve_prepared_options)
    agent = Agent[None, str](
        call_renamed_tool(called_name),
        name=agent_name,
        deps_type=type(None),
        toolsets=[FunctionToolset[None](tools, id=toolset_id)],
        capabilities=[runtime],
    )
    return agent, runtime, app


async def slow_report() -> str:
    return 'slow'


async def quick_report() -> str:
    return 'quick'


async def inline_report() -> str:
    return 'inline'


async def test_prepared_rename_routes_registration_and_invocation_to_one_task() -> None:
    """Renaming a tool for the model does not change its toolset task identity."""
    agent, runtime, app = build_prepared_agent(
        [renamed_for_model(slow_report, 'report'), Tool[None](quick_report)],
        agent_name='prepared-options',
        toolset_id='prepared',
        called_name='report',
    )
    registered = [
        name
        for name in app.registered_task_names
        if '__function_toolset__prepared' in name and name.endswith('.call_tool')
    ]
    slow_task = 'prepared-options__function_toolset__prepared.call_tool'

    assert registered == [slow_task]

    context = NestedTaskContext()
    with anyio.fail_after(5):
        output = await run_agent_in_task(agent, runtime, context)

    assert output == 'report=slow'
    assert [name for name, _depth in context.calls if name.endswith('.call_tool')] == [slow_task]


async def test_prepared_rename_with_resolver_false_stays_inline() -> None:
    """A renamed opted-out tool uses no child run, even though its toolset has shared tasks."""
    agent, runtime, app = build_prepared_agent(
        [renamed_for_model(inline_report, 'summary'), Tool[None](quick_report)],
        agent_name='prepared-inline',
        toolset_id='opted-out',
        called_name='summary',
    )
    registered = sorted(name for name in app.registered_task_names if '__function_toolset__opted-out' in name)

    assert registered == [
        'prepared-inline__function_toolset__opted-out.call_tool',
        'prepared-inline__function_toolset__opted-out.validate_args',
    ]
    assert not [name for name in app.registered_task_names if 'inline_report' in name or 'summary' in name]

    context = NestedTaskContext()
    with anyio.fail_after(5):
        output = await run_agent_in_task(agent, runtime, context)

    assert output == 'summary=inline'
    assert not [name for name, _depth in context.calls if '__function_toolset__opted-out' in name]


def test_toolset_task_names_survive_reordering_and_added_tools() -> None:
    """A tool name matching another toolset ID cannot rename that toolset's task."""

    async def part() -> str:
        return 'part'

    async def other() -> str:
        return 'other'

    async def added() -> str:
        return 'added'

    async def shared_tool() -> str:
        return 'shared'

    expected: set[str] | None = None
    for tools in ([part, other], [other, part], [added, other, part]):
        app = RecordingWorkflows()
        Agent(
            FunctionModel(three_tools),
            name='stable',
            toolsets=[FunctionToolset(tools, id='x'), FunctionToolset([shared_tool], id='x.part')],
            capabilities=[RenderWorkflows(app)],
        )
        names = set(app.registered_task_names)
        assert len(names) == len(app.registered_task_names)
        assert {'stable__function_toolset__x.call_tool', 'stable__function_toolset__x.part.call_tool'} <= names
        if expected is None:
            expected = names
        else:
            assert names == expected
