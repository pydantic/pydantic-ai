from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import timedelta
from typing import Protocol

import anyio
import pytest

from pydantic_ai import Agent, DeferredToolRequests, FunctionToolset
from pydantic_ai.durable_exec._operation import ToolsetCallToolId
from pydantic_ai.exceptions import CallDeferred, ModelRetry, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from ..._inline_snapshot import snapshot

try:
    from temporalio import activity, workflow
    from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.exceptions import ApplicationError, CancelledError as TemporalCancelledError, TerminatedError
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.workflow import (
        ChildWorkflowCancellationType,
        ChildWorkflowConfig,
        _Definition as WorkflowDefinition,
    )

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
    from pydantic_ai.durable_exec.temporal._durability import worker_registrations
    from pydantic_ai.durable_exec.temporal._operation_backend import registered_child_workflow_names

    from ._shared import TASK_QUEUE, _workflow_failure_cause  # pyright: ignore[reportPrivateUsage]
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = [pytest.mark.xdist_group(name='temporal-durability')]

CHILD_WORKFLOW_TYPE = 'agent__child_workflow_agent__toolset__child_workflow_tools__call_tool__child_workflow'


_CHILD_WORKFLOW = ChildWorkflowConfig()


def _metadata_error_agent(name: str, toolset: FunctionToolset[None]) -> Agent[None, str]:
    return Agent(
        TestModel(call_tools='all'),
        name=name,
        deps_type=type(None),
        toolsets=[toolset],
        capabilities=[TemporalDurability()],
    )


extra_key_toolset = FunctionToolset[None](id='extra_key_tools')


@extra_key_toolset.tool_plain(metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW, 'summary': 'invalid'}})
async def extra_key_tool() -> str:
    return 'unreachable'  # pragma: no cover


sync_child_toolset = FunctionToolset[None](id='sync_child_tools')


@sync_child_toolset.tool_plain(metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW}})
def sync_child_tool() -> str:
    return 'unreachable'  # pragma: no cover


invalid_config_toolset = FunctionToolset[None](id='invalid_config_tools')


@invalid_config_toolset.tool_plain(
    metadata={'temporal': {'child_workflow': ChildWorkflowConfig(unknown_option=True)}}  # pyright: ignore[reportCallIssue]
)
async def invalid_config_tool() -> str:
    return 'unreachable'  # pragma: no cover


late_child_toolset = FunctionToolset[None](id='late_child_tools')
_metadata_error_agents = {
    agent.name: agent
    for agent in [
        _metadata_error_agent('child_extra_key', extra_key_toolset),
        _metadata_error_agent('child_sync_tool', sync_child_toolset),
        _metadata_error_agent('child_invalid_config', invalid_config_toolset),
        _metadata_error_agent('child_added_late', late_child_toolset),
    ]
}


async def late_child_tool() -> str:
    return 'unreachable'  # pragma: no cover


@workflow.defn
class MetadataErrorWorkflow:
    @workflow.run
    async def run(self, agent_name: str) -> str:
        return (await _metadata_error_agents[agent_name].run('call the tool')).output


@pytest.mark.parametrize(
    ('agent_name', 'message'),
    [
        (
            'child_extra_key',
            "Tool 'extra_key_tool' has invalid Temporal metadata: `child_workflow` must be the only key.",
        ),
        (
            'child_sync_tool',
            "Temporal metadata for tool 'sync_child_tool' selects a child workflow, but non-async tools cannot run "
            'in workflow code. Make the tool function async instead.',
        ),
        ('child_invalid_config', "Invalid Temporal `ChildWorkflowConfig` in tool 'invalid_config_tool' metadata:"),
    ],
)
async def test_child_workflow_metadata_errors_fail_the_workflow(client: Client, agent_name: str, message: str) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MetadataErrorWorkflow],
        plugins=[AgentPlugin(_metadata_error_agents[agent_name])],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                MetadataErrorWorkflow.run,
                agent_name,
                id=f'test_child_metadata_error_{agent_name}',
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == UserError.__name__
    assert cause.message.startswith(message)


async def test_child_workflow_metadata_added_after_worker_configured(client: Client) -> None:
    late_agent = _metadata_error_agents['child_added_late']
    # The plugin reads the toolset's tools now, before the child-workflow tool exists, so it registers no child workflow.
    plugin = AgentPlugin(late_agent)
    late_child_toolset.add_function(late_child_tool, metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW}})
    try:
        # Unsandboxed, since the sandbox would re-import this module without the tool added here.
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[MetadataErrorWorkflow],
            plugins=[plugin],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            with pytest.raises(WorkflowFailureError) as exc_info:
                await client.execute_workflow(
                    MetadataErrorWorkflow.run,
                    'child_added_late',
                    id='test_child_workflow_metadata_added_after_worker_configured',
                    task_queue=TASK_QUEUE,
                )
    finally:
        late_child_toolset.tools.pop('late_child_tool')

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == UserError.__name__
    assert cause.message == snapshot(
        "Toolset 'late_child_tools' has a tool configured to run as a Temporal child workflow, but its child "
        'workflow was not registered with the worker, because no tool in the toolset had `child_workflow` metadata '
        'when the worker was configured. Set the metadata on the tool, or with a `SetToolMetadata` capability, '
        'before creating the worker, and register the worker through `PydanticAIPlugin` or `AgentPlugin`.'
    )


decorated_agent = Agent(
    TestModel(call_tools='all'),
    name='decorated_child_workflow_agent',
    deps_type=type(None),
    capabilities=[TemporalDurability()],
)


# Decorated after the agent is built: the worker, not the agent constructor, decides which child workflows to register.
@decorated_agent.tool_plain(metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW}})
async def decorated_child_tool() -> str:
    return 'ran in a child workflow'


@workflow.defn
class DecoratedChildWorkflowAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await decorated_agent.run('call the tool')).output


async def test_agent_decorated_tool_runs_as_child_workflow(client: Client) -> None:
    workflow_id = 'test_agent_decorated_tool_runs_as_child_workflow'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DecoratedChildWorkflowAgentWorkflow],
        plugins=[AgentPlugin(decorated_agent)],
    ):
        result = await client.execute_workflow(
            DecoratedChildWorkflowAgentWorkflow.run, id=workflow_id, task_queue=TASK_QUEUE
        )

    assert 'ran in a child workflow' in result
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    assert _child_start(history).workflow_type.name == (
        'agent__decorated_child_workflow_agent__toolset__<agent>__call_tool__child_workflow'
    )


failing_toolset = FunctionToolset[None](id='failing_child_tools')


@failing_toolset.tool_plain(metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW}})
async def failing_child_tool() -> str:
    raise ValueError('Something broke in the child.')


failing_agent = _metadata_error_agent('failing_child_workflow_agent', failing_toolset)


@workflow.defn
class FailingChildWorkflowAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await failing_agent.run('call the tool')).output


async def test_child_workflow_tool_exception_fails_the_child(client: Client) -> None:
    """An ordinary exception fails the child workflow, like an activity, instead of retrying its workflow task."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FailingChildWorkflowAgentWorkflow],
        plugins=[AgentPlugin(failing_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                FailingChildWorkflowAgentWorkflow.run,
                id='test_child_workflow_tool_exception_fails_the_child',
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert (cause.type, cause.message) == ('ValueError', 'Something broke in the child.')


user_error_toolset = FunctionToolset[None](id='user_error_child_tools')


@user_error_toolset.tool_plain(
    metadata={
        'temporal': {
            'child_workflow': ChildWorkflowConfig(
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(milliseconds=1))
            )
        }
    }
)
async def user_error_child_tool() -> str:
    raise UserError('Misconfigured.')


user_error_agent = _metadata_error_agent('user_error_child_workflow_agent', user_error_toolset)


@workflow.defn
class UserErrorChildWorkflowAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await user_error_agent.run('call the tool')).output


async def test_child_workflow_retry_policy_skips_non_retryable_errors(client: Client) -> None:
    """Like an activity, a child with a multi-attempt `retry_policy` doesn't retry the framework's deterministic errors."""
    workflow_id = 'test_child_workflow_retry_policy_skips_non_retryable_errors'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UserErrorChildWorkflowAgentWorkflow],
        plugins=[AgentPlugin(user_error_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                UserErrorChildWorkflowAgentWorkflow.run, id=workflow_id, task_queue=TASK_QUEUE
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert (cause.type, cause.message) == ('UserError', 'Misconfigured.')
    child_id = _child_start(await client.get_workflow_handle(workflow_id).fetch_history()).workflow_id
    # The child's latest run is its first attempt: the `UserError` wasn't retried.
    child_history = await client.get_workflow_handle(child_id).fetch_history()
    assert child_history.events[0].workflow_execution_started_event_attributes.attempt == 1


@activity.defn
async def failing_nested_activity() -> str:
    raise ValueError('The nested activity failed.')


nested_failure_toolset = FunctionToolset[None](id='nested_failure_tools')


@nested_failure_toolset.tool_plain(metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW}})
async def nested_failure_tool() -> str:
    return await workflow.execute_activity(
        failing_nested_activity,
        start_to_close_timeout=timedelta(seconds=10),
        retry_policy=RetryPolicy(maximum_attempts=1),
    )


nested_failure_agent = _metadata_error_agent('nested_failure_child_workflow_agent', nested_failure_toolset)


@workflow.defn
class NestedFailureChildWorkflowAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await nested_failure_agent.run('call the tool')).output


async def test_child_workflow_tool_temporal_error_passes_through(client: Client) -> None:
    """A Temporal error from the tool body, like a failed nested activity, fails the child unchanged."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[NestedFailureChildWorkflowAgentWorkflow],
        activities=[failing_nested_activity],
        plugins=[AgentPlugin(nested_failure_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                NestedFailureChildWorkflowAgentWorkflow.run,
                id='test_child_workflow_tool_temporal_error_passes_through',
                task_queue=TASK_QUEUE,
            )

    # Not re-wrapped as an `ActivityError`-typed `ApplicationError`: the innermost failure is the activity's own.
    cause = _workflow_failure_cause(exc_info.value)
    assert (cause.type, cause.message) == ('ValueError', 'The nested activity failed.')


async def test_child_workflow_metadata_rejects_non_function_tool_operation() -> None:
    """Pin the guard for a non-function tool operation that resolves child-workflow metadata.

    Through the public API, child-workflow metadata only reaches this resolver for function tool calls, so the
    guard can't be triggered by running an agent; this calls the resolver directly with an MCP operation id.
    """
    mcp_metadata_toolset = FunctionToolset[None](id='mcp_metadata_tools')

    @mcp_metadata_toolset.tool_plain(metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW}})
    async def child_tool() -> str:
        return 'unreachable'  # pragma: no cover

    durability = TemporalDurability.from_agent(_metadata_error_agent('child_mcp_operation', mcp_metadata_toolset))
    assert durability is not None
    tools = await mcp_metadata_toolset.get_tools(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))

    with pytest.raises(
        UserError, match=re.escape('Temporal child workflows are only supported for function tool calls.')
    ):
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('mcp', toolset_id='mcp_metadata_tools'), tools['child_tool'], 'child_tool'
        )


retry_toolset = FunctionToolset[None](id='retry_child_tools')


@retry_toolset.tool(metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW}})
async def flaky_child_tool(ctx: RunContext[None]) -> str:
    if ctx.retry == 0:
        raise ModelRetry('Try again.')
    return 'succeeded on retry'


retry_agent = _metadata_error_agent('retry_child_workflow_agent', retry_toolset)


@workflow.defn
class RetryChildWorkflowAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await retry_agent.run('call the tool')).output


async def test_child_workflow_tool_model_retry(client: Client) -> None:
    workflow_id = 'test_child_workflow_tool_model_retry'
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[RetryChildWorkflowAgentWorkflow], plugins=[AgentPlugin(retry_agent)]
    ):
        result = await client.execute_workflow(
            RetryChildWorkflowAgentWorkflow.run, id=workflow_id, task_queue=TASK_QUEUE
        )

    assert 'succeeded on retry' in result
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    # The `ModelRetry` crossed back from the first child as a retry, and the model's second call started another child.
    assert len(_child_starts(history)) == 2
    await Replayer(
        workflows=[RetryChildWorkflowAgentWorkflow],
        data_converter=pydantic_data_converter,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ).replay_workflow(history)


deferred_toolset = FunctionToolset[None](id='deferred_child_tools')


@deferred_toolset.tool_plain(metadata={'temporal': {'child_workflow': _CHILD_WORKFLOW}})
async def deferred_child_tool() -> str:
    raise CallDeferred


deferred_agent = Agent(
    TestModel(call_tools='all'),
    name='deferred_child_workflow_agent',
    deps_type=type(None),
    output_type=[str, DeferredToolRequests],
    toolsets=[deferred_toolset],
    capabilities=[TemporalDurability()],
)


@workflow.defn
class DeferredChildWorkflowAgentWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        output = (await deferred_agent.run('call the tool')).output
        assert isinstance(output, DeferredToolRequests)
        return [call.tool_name for call in output.calls]


async def test_child_workflow_tool_call_deferred(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DeferredChildWorkflowAgentWorkflow],
        plugins=[AgentPlugin(deferred_agent)],
    ):
        deferred_tool_names = await client.execute_workflow(
            DeferredChildWorkflowAgentWorkflow.run, id='test_child_workflow_tool_call_deferred', task_queue=TASK_QUEUE
        )

    assert deferred_tool_names == ['deferred_child_tool']


@activity.defn
async def nested_child_activity(value: str) -> str:
    return f'delegated: {value}'


toolset = FunctionToolset[None](id='child_workflow_tools')
validated_values: list[str] = []


def validate_durable_delegate(ctx: RunContext[None], value: str) -> None:
    validated_values.append(value)


@toolset.tool_plain(
    args_validator=validate_durable_delegate,
    metadata={
        'temporal': {
            'child_workflow': ChildWorkflowConfig(execution_timeout=timedelta(seconds=30)),
        }
    },
)
async def durable_delegate(value: str) -> str:
    return await workflow.execute_activity(
        nested_child_activity,
        value,
        start_to_close_timeout=timedelta(seconds=10),
    )


agent = Agent(
    TestModel(call_tools='all'),
    name='child_workflow_agent',
    deps_type=type(None),
    toolsets=[toolset],
    capabilities=[TemporalDurability()],
)


@workflow.defn
class ChildWorkflowAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await agent.run('delegate this')).output


def _child_starts(history: WorkflowHistory):
    return [
        event.start_child_workflow_execution_initiated_event_attributes
        for event in history.events
        if event.HasField('start_child_workflow_execution_initiated_event_attributes')
    ]


def _child_start(history: WorkflowHistory):
    starts = _child_starts(history)
    assert len(starts) == 1
    return starts[0]


class _HistoryHandle(Protocol):
    async def fetch_history(self) -> WorkflowHistory: ...


async def _wait_for_child_start(handle: _HistoryHandle) -> tuple[str, str]:
    with anyio.fail_after(10):
        while True:
            history = await handle.fetch_history()
            starts = [
                event.start_child_workflow_execution_initiated_event_attributes
                for event in history.events
                if event.HasField('start_child_workflow_execution_initiated_event_attributes')
            ]
            if starts:
                start = starts[0]
                return start.workflow_type.name, start.workflow_id
            await asyncio.sleep(0.01)


async def test_tool_call_runs_as_child_workflow(client: Client) -> None:
    workflow_id = 'test_tool_call_runs_as_child_workflow'
    validated_values.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ChildWorkflowAgentWorkflow],
        activities=[nested_child_activity],
        plugins=[AgentPlugin(agent)],
    ):
        result = await client.execute_workflow(
            ChildWorkflowAgentWorkflow.run,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

    assert 'delegated:' in result
    assert validated_values == ['a']
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    child_start = _child_start(history)
    assert child_start.workflow_type.name == CHILD_WORKFLOW_TYPE
    assert re.fullmatch(
        rf'{workflow_id}--[0-9a-f]{{8}}-[0-9a-f]{{4}}-4[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}',
        child_start.workflow_id,
    )

    child_history = await client.get_workflow_handle(child_start.workflow_id).fetch_history()
    nested_activity_types = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in child_history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]
    assert nested_activity_types == ['nested_child_activity']

    # Replaying pins the workflow-generated UUID: a non-deterministic child ID would mismatch history. It also
    # replays in a process where no worker was configured, as a standalone determinism check would.
    registered = set(registered_child_workflow_names)
    registered_child_workflow_names.clear()
    try:
        await Replayer(workflows=[ChildWorkflowAgentWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(history)
    finally:
        registered_child_workflow_names.update(registered)


sub_agent = Agent(TestModel(), name='nested_sub_agent', capabilities=[TemporalDurability()])
delegation_toolset = FunctionToolset[None](id='delegation_tools')


@delegation_toolset.tool_plain(metadata={'temporal': {'child_workflow': ChildWorkflowConfig()}})
async def delegate_to_sub_agent(task: str) -> str:
    return (await sub_agent.run(task)).output


delegating_agent = Agent(
    TestModel(call_tools='all'),
    name='delegating_agent',
    deps_type=type(None),
    toolsets=[delegation_toolset],
    capabilities=[TemporalDurability()],
)


@workflow.defn
class DelegatingAgentWorkflow:
    __pydantic_ai_agents__ = [delegating_agent, sub_agent]

    @workflow.run
    async def run(self) -> str:
        return (await delegating_agent.run('delegate this')).output


def _scheduled_activity_types(history: WorkflowHistory) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]


async def test_child_workflow_runs_nested_durable_agent(client: Client) -> None:
    workflow_id = 'test_child_workflow_runs_nested_durable_agent'
    # The agent is also listed in `__pydantic_ai_agents__`; its child workflow is registered once.
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[DelegatingAgentWorkflow], plugins=[AgentPlugin(delegating_agent)]
    ):
        result = await client.execute_workflow(DelegatingAgentWorkflow.run, id=workflow_id, task_queue=TASK_QUEUE)

    assert 'success (no tool calls)' in result
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    child_history = await client.get_workflow_handle(_child_start(history).workflow_id).fetch_history()
    # The nested agent's model request is a durable activity in the child's history, not the parent's.
    assert _scheduled_activity_types(child_history) == ['agent__nested_sub_agent__model_request']
    assert 'agent__nested_sub_agent__model_request' not in _scheduled_activity_types(history)


class _ChildWorkflow(Protocol):
    async def run(self, params: None) -> str: ...


def _child_workflow_registration(agent: Agent[None, str], toolset_id: str) -> type[_ChildWorkflow]:
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    registrations = [
        registration
        for registration in worker_registrations(durability)[1]
        if (workflow_name := WorkflowDefinition.must_from_class(registration).name) is not None
        and f'__toolset__{toolset_id}__call_tool__child_workflow' in workflow_name
    ]
    assert len(registrations) == 1
    return registrations[0]


async def test_child_workflow_rejects_top_level_start(client: Client) -> None:
    child_workflow = _child_workflow_registration(agent, 'child_workflow_tools')
    workflow_id = 'test_child_workflow_rejects_top_level_start'
    async with Worker(client, task_queue=TASK_QUEUE, workflows=[child_workflow]):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(child_workflow.run, None, id=workflow_id, task_queue=TASK_QUEUE)

    assert isinstance(exc_info.value.__cause__, ApplicationError)
    assert exc_info.value.__cause__.non_retryable
    assert str(exc_info.value.__cause__) == 'Pydantic AI tool-call workflows must be started as child workflows.'


def _collision_agent(name: str, result: str) -> Agent[None, str]:
    collision_toolset = FunctionToolset(id='collision_tools')

    @collision_toolset.tool_plain(metadata={'temporal': {'child_workflow': ChildWorkflowConfig()}})
    async def collision_tool() -> str:
        return result

    return Agent(
        TestModel(call_tools='all'),
        name=name,
        toolsets=[collision_toolset],
        capabilities=[TemporalDurability()],
    )


hyphen_agent = _collision_agent('foo-bar', 'hyphen')
underscore_agent = _collision_agent('foo_bar', 'underscore')


@workflow.defn
class CollidingAgentNamesWorkflow:
    @workflow.run
    async def run(self, use_hyphen: bool) -> str:
        selected_agent = hyphen_agent if use_hyphen else underscore_agent
        return (await selected_agent.run('call the tool')).output


async def test_child_workflow_classes_do_not_collide(client: Client) -> None:
    hyphen_child = _child_workflow_registration(hyphen_agent, 'collision_tools')
    underscore_child = _child_workflow_registration(underscore_agent, 'collision_tools')

    assert hyphen_child is not underscore_child
    assert hyphen_child.__qualname__ != underscore_child.__qualname__
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CollidingAgentNamesWorkflow],
        plugins=[AgentPlugin(hyphen_agent), AgentPlugin(underscore_agent)],
    ):
        hyphen_result = await client.execute_workflow(
            CollidingAgentNamesWorkflow.run,
            True,
            id='test_child_workflow_class_hyphen',
            task_queue=TASK_QUEUE,
        )
        underscore_result = await client.execute_workflow(
            CollidingAgentNamesWorkflow.run,
            False,
            id='test_child_workflow_class_underscore',
            task_queue=TASK_QUEUE,
        )

    assert hyphen_result is not None
    assert 'hyphen' in hyphen_result
    assert underscore_result is not None
    assert 'underscore' in underscore_result


def _cancellation_agent(name: str, cancellation_type: ChildWorkflowCancellationType) -> Agent[None, str]:
    cancellation_toolset = FunctionToolset(id='cancellation_tools')

    @cancellation_toolset.tool_plain(
        name='wait_forever',
        metadata={
            'temporal': {
                'child_workflow': ChildWorkflowConfig(
                    execution_timeout=timedelta(seconds=30),
                    cancellation_type=cancellation_type,
                ),
            }
        },
    )
    async def wait_forever() -> str:
        await workflow.wait_condition(lambda: False)
        return 'unreachable'  # pragma: no cover

    return Agent(
        TestModel(call_tools='all'),
        name=name,
        toolsets=[cancellation_toolset],
        capabilities=[TemporalDurability()],
    )


wait_cancellation_agent = _cancellation_agent(
    'wait_cancellation_agent', ChildWorkflowCancellationType.WAIT_CANCELLATION_COMPLETED
)
try_cancel_agent = _cancellation_agent('try_cancel_agent', ChildWorkflowCancellationType.TRY_CANCEL)


@workflow.defn
class WaitCancellationWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await wait_cancellation_agent.run('wait')).output


@workflow.defn
class TryCancelWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await try_cancel_agent.run('wait')).output


@pytest.mark.parametrize(
    ('parent_workflow', 'cancel_agent'),
    [
        (WaitCancellationWorkflow, wait_cancellation_agent),
        (TryCancelWorkflow, try_cancel_agent),
    ],
    ids=['wait-cancellation-completed', 'try-cancel'],
)
async def test_parent_cancellation_reaches_child_without_livelock(
    client: Client,
    parent_workflow: type[WaitCancellationWorkflow] | type[TryCancelWorkflow],
    cancel_agent: Agent[None, str],
) -> None:
    workflow_id = f'test_child_cancellation_{parent_workflow.__name__}'
    run: Callable[..., object] = parent_workflow.run
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[parent_workflow],
        plugins=[AgentPlugin(cancel_agent)],
    ):
        handle = await client.start_workflow(run, id=workflow_id, task_queue=TASK_QUEUE)
        child_type, child_id = await _wait_for_child_start(handle)
        assert child_type.endswith('__call_tool__child_workflow')

        await handle.cancel()
        with anyio.fail_after(10):
            with pytest.raises(WorkflowFailureError) as parent_error:
                await handle.result()
        assert isinstance(parent_error.value.__cause__, TemporalCancelledError)

        child_handle = client.get_workflow_handle(child_id)
        with anyio.fail_after(10):
            with pytest.raises(WorkflowFailureError) as child_error:
                await child_handle.result()

        parent_history = await handle.fetch_history()

    assert not [event for event in parent_history.events if 'WORKFLOW_TASK_FAILED' in str(event.event_type)]
    assert any(
        event.HasField('request_cancel_external_workflow_execution_initiated_event_attributes')
        for event in parent_history.events
    )
    if parent_workflow is WaitCancellationWorkflow:
        assert isinstance(child_error.value.__cause__, TemporalCancelledError)
    else:
        assert isinstance(child_error.value.__cause__, TemporalCancelledError | TerminatedError)
