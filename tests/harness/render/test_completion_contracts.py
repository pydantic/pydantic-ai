"""Black-box completion contracts for Render-backed agent operations."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Concatenate, Literal, ParamSpec, Protocol, TypeVar, overload

import anyio
from render.workflows import Options, Retry, TaskContext, TaskDefinition, Workflows

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness import RenderWorkflows
from pydantic_ai_harness.subagents import SubAgent, SubAgents

from .conftest import RecordingTaskContext

P = ParamSpec('P')
R = TypeVar('R')


class RecordingTaskDecorator(Protocol):
    """Typed decorator returned by `RegistrationLog.task(...)`."""

    @overload
    def __call__(self, func: Callable[Concatenate[TaskContext, P], Awaitable[R]], /) -> TaskDefinition[P, R]: ...

    @overload
    def __call__(self, func: Callable[Concatenate[TaskContext, P], R], /) -> TaskDefinition[P, R]: ...


class RegistrationLog(Workflows):
    """Record task names and registration-time Options through the public decorator."""

    def __init__(self) -> None:
        super().__init__()
        self.options: dict[str, Options] = {}
        self.names: list[str] = []

    @overload
    def task(
        self,
        func: Callable[Concatenate[TaskContext, P], Awaitable[R]],
        /,
    ) -> TaskDefinition[P, R]: ...

    @overload
    def task(
        self,
        func: Callable[Concatenate[TaskContext, P], R],
        /,
    ) -> TaskDefinition[P, R]: ...

    @overload
    def task(
        self,
        *,
        name: str | None = None,
        retry: Retry | None = None,
        timeout_seconds: int | None = None,
        plan: str | None = None,
    ) -> RecordingTaskDecorator: ...

    def task(
        self,
        func: Callable[..., object] | None = None,
        *,
        name: str | None = None,
        retry: Retry | None = None,
        timeout_seconds: int | None = None,
        plan: str | None = None,
    ) -> object:
        decorate = super().task(name=name, retry=retry, timeout_seconds=timeout_seconds, plan=plan)

        def record(target: Callable[..., object]) -> TaskDefinition[..., object]:
            definition = decorate(target)
            self.names.append(definition.name)
            self.options[definition.name] = Options(retry=retry, timeout_seconds=timeout_seconds, plan=plan)
            return definition

        return record if func is None else record(func)


class NestedTaskContext(RecordingTaskContext):
    """Execute public task definitions while recording nested run depth."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.stack: list[str] = []

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self.calls.append((task.name, len(self.stack) + 1))
        self.stack.append(task.name)
        try:
            result = task.func(self, *args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        finally:
            self.stack.pop()


async def run_in_task(agent: Agent[None, str], runtime: RenderWorkflows[None], context: TaskContext) -> str:
    """Run `agent` from a public Render task entry point."""

    @runtime.task
    async def root(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('go')).output

    pending = root.func(context)
    assert inspect.isawaitable(pending)
    return await pending


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
    """Assign distinct Options to two tools and opt the third out of a Render task."""
    del operation_id, tool
    if tool_name == 'fast_lookup':
        return Options(timeout_seconds=30, plan='starter')
    if tool_name == 'slow_lookup':
        return Options(timeout_seconds=300, plan='standard')
    return False if tool_name == 'inline_lookup' else None


def build_per_tool_agent() -> tuple[Agent[None, str], RenderWorkflows[None], RegistrationLog]:
    """One named FunctionToolset with two remote tools and one inline opt-out."""

    async def fast_lookup() -> str:
        return 'fast'

    async def slow_lookup() -> str:
        return 'slow'

    async def inline_lookup() -> str:
        return 'inline'

    app = RegistrationLog()
    runtime = RenderWorkflows[None](app, deps_type=type(None), resolve_tool_options=resolve_lookup_options)
    agent = Agent[None, str](
        FunctionModel(three_tools),
        name='per-function',
        deps_type=type(None),
        toolsets=[FunctionToolset([fast_lookup, slow_lookup, inline_lookup], id='lookups')],
        capabilities=[runtime],
    )
    return agent, runtime, app


async def test_named_toolset_registers_stable_per_function_tasks_and_inline_opt_out() -> None:
    agent, runtime, app = build_per_tool_agent()
    _, _, second_app = build_per_tool_agent()
    registered = {
        name: value
        for name, value in app.options.items()
        if '__function_toolset__lookups' in name and name.endswith('.call_tool')
    }
    second = {
        name: value
        for name, value in second_app.options.items()
        if '__function_toolset__lookups' in name and name.endswith('.call_tool')
    }

    assert registered == second
    assert len(registered) == 2
    assert any('fast_lookup' in name for name in registered)
    assert any('slow_lookup' in name for name in registered)
    assert {value.timeout_seconds for value in registered.values()} == {30, 300}
    assert {value.plan for value in registered.values()} == {'starter', 'standard'}

    context = NestedTaskContext()
    assert await run_in_task(agent, runtime, context) == 'done'
    invoked = [name for name, _depth in context.calls if name in registered]
    assert sorted(invoked) == sorted(registered)
    assert not any('inline_lookup' in name for name, _depth in context.calls)


class OwnedUnnamedTools(AbstractCapability[None]):
    """A capability-owned FunctionToolset the caller never names."""

    id = 'owned_tools'

    def __init__(self) -> None:
        async def owned_lookup() -> str:
            return 'owned'

        self.toolset = FunctionToolset[None]([owned_lookup])

    def get_toolset(self) -> AbstractToolset[None]:
        return self.toolset


async def test_unnamed_capability_toolset_is_never_privately_renamed() -> None:
    capability = OwnedUnnamedTools()
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['owned_lookup']),
        name='owned',
        deps_type=type(None),
        capabilities=[capability, runtime],
    )

    assert capability.toolset.id is None
    context = NestedTaskContext()
    assert isinstance(await run_in_task(agent, runtime, context), str)
    assert not [name for name, _depth in context.calls if '__function_toolset__owned_tools' in name]


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


def build_nested_agents() -> tuple[Agent[None, str], RenderWorkflows[None], RegistrationLog]:
    """Build explicit parent and child agents against one Workflows app."""
    app = RegistrationLog()
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

    parent_runtime = RenderWorkflows[None](app, deps_type=type(None))
    parent = Agent[None, str](
        delegate_once('nested-child'),
        name='nested-parent',
        deps_type=type(None),
        capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None), parent_runtime],
    )
    return parent, parent_runtime, app


def test_explicit_subagent_delegate_tool_is_not_registered() -> None:
    _, _, app = build_nested_agents()

    assert not [name for name in app.options if '__function_toolset__sub_agents' in name]


async def test_explicit_subagent_uses_shared_app_for_direct_child_task_runs() -> None:
    parent, parent_runtime, _ = build_nested_agents()
    context = NestedTaskContext()
    assert await run_in_task(parent, parent_runtime, context) == 'parent done'
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


def build_two_level_agents() -> tuple[Agent[None, str], RenderWorkflows[None], RegistrationLog]:
    """Build parent, child, and grandchild agents against one Workflows app."""
    app = RegistrationLog()
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

    child_runtime = RenderWorkflows[None](app, deps_type=type(None))
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

    parent_runtime = RenderWorkflows[None](app, deps_type=type(None))
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
    parent, runtime, app = build_two_level_agents()
    assert not [name for name in app.options if '__function_toolset__sub_agents' in name]
    context = NestedTaskContext()

    with anyio.fail_after(5):
        assert await run_in_task(parent, runtime, context) == 'parent done'

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

    app = RegistrationLog()
    runtime = RenderWorkflows[None](app, deps_type=type(None))
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
    assert not [name for name in app.options if '__function_toolset__sub_agents' in name]
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
    """Select per-tool Options by the original static name a toolset holds a tool under."""
    del operation_id, tool
    if tool_name == 'slow_report':
        return Options(timeout_seconds=300, plan='standard')
    if tool_name == 'inline_report':
        return False
    return None


def build_prepared_agent(
    tools: list[Tool[None]], *, agent_name: str, toolset_id: str, called_name: str
) -> tuple[Agent[None, str], RenderWorkflows[None], RegistrationLog]:
    """One named FunctionToolset whose tools are renamed for the model by `prepare`."""
    app = RegistrationLog()
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
    """A renamed tool keeps its original static name as task identity and options."""
    agent, runtime, app = build_prepared_agent(
        [renamed_for_model(slow_report, 'report'), Tool[None](quick_report)],
        agent_name='prepared-options',
        toolset_id='prepared',
        called_name='report',
    )
    registered = [name for name in app.names if '__function_toolset__prepared' in name and name.endswith('.call_tool')]
    slow_task = 'prepared-options__function_toolset__prepared.slow_report.call_tool'

    assert sorted(registered) == sorted(
        [slow_task, 'prepared-options__function_toolset__prepared.quick_report.call_tool']
    )
    assert not [name for name in registered if '.report.' in name]
    assert app.options[slow_task].timeout_seconds == 300
    assert app.options[slow_task].plan == 'standard'

    context = NestedTaskContext()
    with anyio.fail_after(5):
        output = await run_in_task(agent, runtime, context)

    assert output == 'report=slow'
    assert [name for name, _depth in context.calls if name.endswith('.call_tool')] == [slow_task]


async def test_prepared_rename_with_resolver_false_stays_inline() -> None:
    """A renamed tool the resolver opts out of has no task definition and runs inline."""
    agent, runtime, app = build_prepared_agent(
        [renamed_for_model(inline_report, 'summary'), Tool[None](quick_report)],
        agent_name='prepared-inline',
        toolset_id='opted-out',
        called_name='summary',
    )
    registered = sorted(name for name in app.names if '__function_toolset__opted-out' in name)

    assert registered == [
        'prepared-inline__function_toolset__opted-out.quick_report.call_tool',
        'prepared-inline__function_toolset__opted-out.quick_report.validate_args',
    ]
    assert not [name for name in app.names if 'inline_report' in name or 'summary' in name]

    context = NestedTaskContext()
    with anyio.fail_after(5):
        output = await run_in_task(agent, runtime, context)

    assert output == 'summary=inline'
    assert not [name for name, _depth in context.calls if '__function_toolset__opted-out' in name]


def resolve_collision_options(
    operation_id: object, tool: object | None, tool_name: str
) -> Options | Literal[False] | None:
    """Give one tool of toolset `x` its own Options so it needs its own task."""
    del operation_id, tool
    return Options(timeout_seconds=45) if tool_name == 'part' else None


def call_part_then_shared(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call the per-tool task's tool, then the shared task's tool, one step each."""
    del info
    returned = returned_tool_names(messages)
    if 'shared_tool' in returned:
        return ModelResponse(parts=[TextPart('collide done')])
    if 'part' in returned:
        return ModelResponse(parts=[ToolCallPart('shared_tool', {}, tool_call_id='shared')])
    return ModelResponse(parts=[ToolCallPart('part', {}, tool_call_id='part')])


def build_colliding_agent() -> tuple[Agent[None, str], RenderWorkflows[None], RegistrationLog]:
    """Toolset `x` with tool `part` alongside a toolset whose own ID is `x.part`."""

    async def part() -> str:
        return 'per-tool'

    async def other() -> str:
        return 'other'

    async def shared_tool() -> str:
        return 'shared'

    app = RegistrationLog()
    runtime = RenderWorkflows[None](app, deps_type=type(None), resolve_tool_options=resolve_collision_options)
    agent = Agent[None, str](
        FunctionModel(call_part_then_shared),
        name='collide',
        deps_type=type(None),
        toolsets=[
            FunctionToolset[None]([part, other], id='x'),
            FunctionToolset[None]([shared_tool], id='x.part'),
        ],
        capabilities=[runtime],
    )
    return agent, runtime, app


def test_per_tool_and_shared_task_names_do_not_collide() -> None:
    """Binding succeeds with unique names and the shared task keeps the name it had."""
    _, _, app = build_colliding_agent()
    shared_task = 'collide__function_toolset__x.part.call_tool'

    assert len(app.names) == len(set(app.names))
    assert app.names.count(shared_task) == 1
    per_tool = [
        name
        for name in app.names
        if name.endswith('.call_tool') and name != shared_task and '__function_toolset__x' in name
    ]
    assert len(per_tool) == 2
    assert any('other' in name for name in per_tool)
    assert any('part' in name for name in per_tool)


async def test_colliding_identities_route_each_tool_to_its_own_task() -> None:
    """Both tools run, through two different task definitions, with no partial registration."""
    agent, runtime, app = build_colliding_agent()
    shared_task = 'collide__function_toolset__x.part.call_tool'
    context = NestedTaskContext()

    with anyio.fail_after(5):
        output = await run_in_task(agent, runtime, context)

    assert output == 'collide done'
    call_tool_runs = [name for name, _depth in context.calls if name.endswith('.call_tool')]
    assert len(call_tool_runs) == 2
    part_task, shared_run = call_tool_runs
    assert shared_run == shared_task
    assert part_task != shared_task
    assert 'part' in part_task
    assert app.options[part_task].timeout_seconds == 45
    assert app.options[shared_task].timeout_seconds is None
