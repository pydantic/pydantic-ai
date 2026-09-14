from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import timedelta
from typing import Protocol

import anyio
import pytest
from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError, CancelledError as TemporalCancelledError, TerminatedError
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
from temporalio.workflow import (
    ChildWorkflowCancellationType,
    ChildWorkflowConfig,
    _Definition as WorkflowDefinition,
)

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec._operation import ToolsetCallToolId
from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from ._shared import TASK_QUEUE

pytestmark = pytest.mark.anyio

CHILD_WORKFLOW_TYPE = 'agent__child_workflow_agent__toolset__child_workflow_tools__call_tool__child_workflow'


async def test_child_workflow_metadata_validation() -> None:
    async def async_tool() -> str:
        return 'async'  # pragma: no cover

    def sync_tool() -> str:
        return 'sync'  # pragma: no cover

    validation_toolset = FunctionToolset[None](id='validation_tools')
    validation_toolset.add_function(
        async_tool,
        metadata={'temporal': {'child_workflow': ChildWorkflowConfig(), 'summary': 'invalid'}},
    )
    validation_toolset.add_function(
        sync_tool,
        metadata={'temporal': {'child_workflow': ChildWorkflowConfig()}},
    )
    validation_agent = Agent(
        TestModel(),
        name='child_workflow_validation',
        deps_type=type(None),
        toolsets=[validation_toolset],
        capabilities=[TemporalDurability()],
    )
    durability = TemporalDurability.from_agent(validation_agent)
    assert durability is not None
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    tools = await validation_toolset.get_tools(ctx)

    with pytest.raises(
        UserError,
        match=re.escape("Tool 'async_tool' has invalid Temporal metadata: `child_workflow` must be the only key."),
    ):
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('function', toolset_id='validation_tools'), tools['async_tool'], 'async_tool'
        )

    tools['async_tool'].tool_def.metadata = {'temporal': {'child_workflow': ChildWorkflowConfig()}}
    with pytest.raises(
        UserError, match=re.escape('Temporal child workflows are only supported for function tool calls.')
    ):
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('mcp', toolset_id='validation_tools'), tools['async_tool'], 'async_tool'
        )

    with pytest.raises(
        UserError,
        match=re.escape(
            "Temporal metadata for tool 'sync_tool' selects a child workflow, but non-async tools cannot run in "
            'workflow code. Make the tool function async instead.'
        ),
    ):
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('function', toolset_id='validation_tools'), tools['sync_tool'], 'sync_tool'
        )


async def test_child_workflow_metadata_rejects_invalid_config() -> None:
    async def child_tool() -> str:
        return 'done'  # pragma: no cover

    validation_toolset = FunctionToolset[None](id='invalid_config_tools')
    validation_toolset.add_function(
        child_tool,
        metadata={
            'temporal': {
                'child_workflow': ChildWorkflowConfig(unknown_option=True),  # pyright: ignore[reportCallIssue]
            }
        },
    )
    validation_agent = Agent(
        TestModel(),
        name='invalid_child_workflow_config',
        deps_type=type(None),
        toolsets=[validation_toolset],
        capabilities=[TemporalDurability()],
    )
    durability = TemporalDurability.from_agent(validation_agent)
    assert durability is not None
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    tools = await validation_toolset.get_tools(ctx)

    with pytest.raises(
        UserError,
        match=re.escape("Invalid Temporal `ChildWorkflowConfig` in tool 'child_tool' metadata:"),
    ):
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('function', toolset_id='invalid_config_tools'), tools['child_tool'], 'child_tool'
        )


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


def _child_start(history: WorkflowHistory):
    starts = [
        event.start_child_workflow_execution_initiated_event_attributes
        for event in history.events
        if event.HasField('start_child_workflow_execution_initiated_event_attributes')
    ]
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

    # Replaying pins the workflow-generated UUID: a non-deterministic child ID would mismatch history.
    await Replayer(
        workflows=[ChildWorkflowAgentWorkflow],
        data_converter=pydantic_data_converter,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ).replay_workflow(history)


class _ChildWorkflow(Protocol):
    async def run(self, params: None) -> str: ...


def _child_workflow_registration(agent: Agent[None, str], toolset_id: str) -> type[_ChildWorkflow]:
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    registrations = [
        registration
        for registration in durability.temporal_registrations
        if isinstance(registration, type)
        and (workflow_name := WorkflowDefinition.must_from_class(registration).name) is not None
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
