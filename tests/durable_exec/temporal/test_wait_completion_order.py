"""Tool calls that complete in one workflow task replay in the order they were recorded.

Both tool tasks come back from one `asyncio.wait` call, whose `done` set iterates in an order that follows task
identity, so a worker replaying the history can iterate it differently than the worker that recorded it. Each
`FunctionToolResultEvent` is one event-handler activity, and Temporal matches activities on replay by type and
sequence, not by input: a replay that reorders the result events hands each handler activity the other tool's result
without a nondeterminism error. https://github.com/pydantic/pydantic-ai/issues/11182
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field

import anyio
import pytest

from pydantic_ai import Agent, ModelMessage, RunContext
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

try:
    from temporalio import workflow
    from temporalio.api.enums.v1 import EventType
    from temporalio.client import Client
    from temporalio.worker import (
        Interceptor,
        Replayer,
        StartActivityInput,
        UnsandboxedWorkflowRunner,
        Worker,
        WorkflowInboundInterceptor,
        WorkflowInterceptorClassInput,
        WorkflowOutboundInterceptor,
    )
    from temporalio.workflow import ActivityHandle

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability

    from ._shared import BASE_ACTIVITY_CONFIG
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = [pytest.mark.xdist_group(name='temporal-durability')]

TOOLS = ('tool_a', 'tool_b')


@dataclass
class _Gate:
    """Holds both tool activities until the test releases them, so they complete together."""

    arrived: set[str] = field(default_factory=set[str])
    both_arrived: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


_gate = _Gate()


async def _gated(name: str) -> str:
    _gate.arrived.add(name)
    if _gate.arrived == set(TOOLS):
        _gate.both_arrived.set()
    await _gate.release.wait()
    return f'{name} done'


async def tool_a() -> str:
    return await _gated('tool_a')


async def tool_b() -> str:
    return await _gated('tool_b')


async def _call_both_tools(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    if len(messages) == 1:
        yield {
            0: DeltaToolCall(name='tool_a', json_args='{}', tool_call_id='call_a'),
            1: DeltaToolCall(name='tool_b', json_args='{}', tool_call_id='call_b'),
        }
    else:
        yield 'done'


async def _handle_events(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in stream:
        pass


_agent = Agent(
    FunctionModel(stream_function=_call_both_tools),
    name='completion_order',
    tools=[tool_a, tool_b],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_handler=_handle_events)],
)


@workflow.defn
class CompletionOrderWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _agent.run(prompt)).output


@dataclass
class _DoneOrder:
    """How an `asyncio.wait` that returns both tool tasks iterates its `done` set: by task name, either way round."""

    reverse: bool = False
    tools_done_together: bool = False


def _task_name(future: asyncio.Future[object]) -> str:
    # Temporal's workflow loop appends ` (workflow: ..., id: ..., run: ...)` to every task name.
    return future.get_name().partition(' (')[0] if isinstance(future, asyncio.Task) else ''


class _NameOrderedDone(set[asyncio.Future[object]]):
    """A `done` set that iterates by task name, one of the orders a plain set can produce."""

    def __init__(self, done: Iterable[asyncio.Future[object]], *, reverse: bool):
        super().__init__(done)
        self.reverse = reverse

    def __iter__(self) -> Iterator[asyncio.Future[object]]:
        return iter(sorted(super().__iter__(), key=_task_name, reverse=self.reverse))


@pytest.fixture
def done_order(monkeypatch: pytest.MonkeyPatch) -> _DoneOrder:
    order = _DoneOrder()
    real_wait = asyncio.wait

    async def wait(
        fs: Iterable[asyncio.Future[object]], *, timeout: float | None = None, return_when: str = asyncio.ALL_COMPLETED
    ) -> tuple[set[asyncio.Future[object]], set[asyncio.Future[object]]]:
        done, pending = await real_wait(fs, timeout=timeout, return_when=return_when)
        if not set(TOOLS) <= {_task_name(future) for future in done}:
            return done, pending
        order.tools_done_together = True
        return _NameOrderedDone(done, reverse=order.reverse), pending

    monkeypatch.setattr(asyncio, 'wait', wait)
    return order


_replayed_event_inputs: list[tuple[str, str]] = []


class _RecordEventHandlerInputs(Interceptor):
    def workflow_interceptor_class(self, input: WorkflowInterceptorClassInput) -> type[WorkflowInboundInterceptor]:
        return _RecordEventHandlerInputsInbound


class _RecordEventHandlerInputsInbound(WorkflowInboundInterceptor):
    def init(self, outbound: WorkflowOutboundInterceptor) -> None:
        super().init(_RecordEventHandlerInputsOutbound(outbound))


class _RecordEventHandlerInputsOutbound(WorkflowOutboundInterceptor):
    def start_activity(self, input: StartActivityInput) -> ActivityHandle[object]:
        if input.activity.endswith('__event_stream_handler'):
            payload = workflow.payload_converter().to_payloads([input.args[0]])[0]
            _replayed_event_inputs.append(_event_key(payload.data))
        return super().start_activity(input)


def _event_key(data: bytes) -> tuple[str, str]:
    event = json.loads(data)['event']
    return event['event_kind'], event['part']['tool_call_id']


def _workflow_worker(client: Client, task_queue: str, *, max_cached_workflows: int) -> Worker:
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[CompletionOrderWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        max_cached_workflows=max_cached_workflows,
    )


async def test_tool_results_completed_together_replay_in_recorded_order(client: Client, done_order: _DoneOrder):
    """A replay that iterates the `done` set the other way round schedules the event-handler inputs it recorded.

    Both tool completions land in one workflow task by construction: the workflow worker stops while the tools are
    held, the tools are released, and a fresh workflow worker starts once history holds both completions. The
    recording iterates `done` in call order and the replay in reverse; the replay must still send each handler
    activity the result the history holds for it. The workers and the replayer run unsandboxed so the patched
    `asyncio.wait` and the gate are the module state the workflow and the tools see.
    """
    global _gate
    _gate = _Gate()
    task_queue = f'completion-order-{uuid.uuid4()}'
    with anyio.fail_after(30):
        # Activities only: this worker has no workflows, so it never polls workflow tasks.
        async with Worker(client, task_queue=task_queue, plugins=[AgentPlugin(_agent)]):
            # No cache, so no sticky queue holds the next workflow task for the stopped worker.
            async with _workflow_worker(client, task_queue, max_cached_workflows=0):
                handle = await client.start_workflow(
                    CompletionOrderWorkflow.run, args=['go'], id=task_queue, task_queue=task_queue
                )
                await _gate.both_arrived.wait()
            _gate.release.set()
            # The next workflow task only starts once history holds both tool completions.
            events = handle.fetch_history_events(wait_new_event=True)
            tool_activities: set[int] = set()
            completed = 0
            while completed < len(TOOLS):
                event = await anext(events)
                if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
                    if event.activity_task_scheduled_event_attributes.activity_type.name.endswith('__call_tool'):
                        tool_activities.add(event.event_id)
                elif event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
                    completed += event.activity_task_completed_event_attributes.scheduled_event_id in tool_activities
            async with _workflow_worker(client, task_queue, max_cached_workflows=100):
                assert await handle.result() == 'done'
        history = await handle.fetch_history()

    recorded = [
        _event_key(event.activity_task_scheduled_event_attributes.input.payloads[0].data)
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
        and event.activity_task_scheduled_event_attributes.activity_type.name.endswith('__event_stream_handler')
    ]
    assert recorded == [
        ('function_tool_call', 'call_a'),
        ('function_tool_call', 'call_b'),
        ('function_tool_result', 'call_a'),
        ('function_tool_result', 'call_b'),
    ]

    done_order.reverse = True
    done_order.tools_done_together = False
    _replayed_event_inputs.clear()
    await Replayer(
        workflows=[CompletionOrderWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=client.data_converter,
        interceptors=[_RecordEventHandlerInputs()],
    ).replay_workflow(history)
    assert done_order.tools_done_together
    assert _replayed_event_inputs == recorded
