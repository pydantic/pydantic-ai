"""Conformance tests against the Durable Task Scheduler emulator, without Azure or LLM credentials.

Run `make integration-azure-durable-task` with an emulator on localhost:8080, hub default.
Override AZURE_DURABLE_TASK_TEST_ENDPOINT / AZURE_DURABLE_TASK_TEST_TASKHUB as needed.
AZURE_DURABLE_TASK_REQUIRE_LIVE=1 fails instead of skipping an unavailable service.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from threading import Barrier, Lock
from typing import Literal, NoReturn, TypeVar
from uuid import uuid4

import grpc
import pytest
from durabletask import task
from durabletask.client import TaskHubGrpcClient
from durabletask.worker import ConcurrencyOptions, TaskHubGrpcWorker
from pydantic import BaseModel, JsonValue, TypeAdapter

from pydantic_ai import (
    Agent,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
    ToolApproved,
    ToolDenied,
    UserError,
)
from pydantic_ai._utils import is_str_dict
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

_WAIT_SECONDS = 30.0
DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
InputT = TypeVar('InputT')
Workflow = Callable[[task.OrchestrationContext, InputT], Generator[task.Task[object], object, JsonValue]]
_JSON = TypeAdapter[JsonValue](JsonValue)


class _Decision(BaseModel):
    action: Literal['approve', 'deny', 'retry', 'return']
    override_args: dict[str, JsonValue] | None = None
    message: str = 'Denied by reviewer'
    value: JsonValue = None


class _Review(BaseModel):
    pause: int
    decisions: dict[str, _Decision]


@dataclass
class _ReviewExample:
    """An approval and external-call review loop, with every decision kind a reviewer can send."""

    agent: Agent[None, str | DeferredToolRequests]
    orchestrator: Workflow[str]

    @staticmethod
    def send_review(
        client: TaskHubGrpcClient, instance_id: str, *, pause: int, decisions: dict[str, JsonValue]
    ) -> None:
        review = _Review.model_validate({'pause': pause, 'decisions': decisions})
        state = client.get_orchestration_state(instance_id)
        pending = state.get_custom_status() if state is not None else None
        if not is_str_dict(pending) or pending.get('pause') != pause:
            raise UserError('The reviewed pause is no longer pending.')
        client.raise_orchestration_event(instance_id, f'review:{pause}', data=review.model_dump(mode='json'))


def _review_example(model: Model, *, agent_name: str) -> _ReviewExample:
    """Build the review agent and its orchestrator, named for this test run."""
    agent = Agent[None, str | DeferredToolRequests](
        model,
        name=agent_name,
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        capabilities=[AzureDurableTaskDurability()],
    )

    @agent.tool_plain(requires_approval=True)
    def approve_payment(amount: int) -> str:
        return f'Paid {amount}'

    @agent.tool_plain
    def external_lookup() -> str:
        raise CallDeferred()

    def review_workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, JsonValue]:
        result = yield from run_durable(ctx, lambda: agent.run(prompt))
        pause = 0
        while isinstance(result.output, DeferredToolRequests):
            requests = result.output
            ctx.set_custom_status(
                {
                    'pause': pause,
                    'event': f'review:{pause}',
                    'requests': TypeAdapter(DeferredToolRequests).dump_python(requests, mode='json'),
                }
            )
            payload = yield ctx.wait_for_external_event(f'review:{pause}')
            review = _Review.model_validate(payload)
            if review.pause != pause:
                raise UserError('The decision does not match the reviewed pause.')
            if set(review.decisions) != {call.tool_call_id for call in [*requests.approvals, *requests.calls]}:
                raise UserError('Decisions must match the reviewed tool-call IDs exactly.')
            deferred = DeferredToolResults()
            for call in requests.approvals:
                decision = review.decisions[call.tool_call_id]
                if decision.action == 'approve':
                    deferred.approvals[call.tool_call_id] = ToolApproved(override_args=decision.override_args)
                else:
                    deferred.approvals[call.tool_call_id] = ToolDenied(message=decision.message)
            for call in requests.calls:
                decision = review.decisions[call.tool_call_id]
                if decision.action == 'retry':
                    deferred.calls[call.tool_call_id] = ModelRetry(decision.message)
                else:
                    deferred.calls[call.tool_call_id] = decision.value
            previous = result
            result = yield from run_durable(
                ctx,
                lambda: agent.run(
                    message_history=previous.all_messages(),
                    deferred_tool_results=deferred,
                    usage=previous.usage,
                    conversation_id=previous.conversation_id,
                ),
            )
            pause += 1
        ctx.set_custom_status(None)
        return {'output': result.output, 'requests': result.usage.requests}

    review_workflow.__name__ = f'{agent_name}_workflow'
    return _ReviewExample(agent, review_workflow)


def _unavailable(message: str) -> NoReturn:
    if os.environ.get('AZURE_DURABLE_TASK_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
        pytest.fail(message)
    pytest.skip(message)


@dataclass
class _Hub:
    endpoint: str
    taskhub: str
    client: TaskHubGrpcClient

    def metadata(self) -> list[tuple[str, str]]:
        return [('taskhub', self.taskhub)]


@dataclass
class _Dependencies:
    tenant: str


@dataclass
class _Pending:
    pause: int
    event: str
    requests: DeferredToolRequests


@pytest.fixture(scope='module')
def hub() -> Generator[_Hub]:
    """Own the SDK client after verifying gRPC readiness and task-hub access."""
    endpoint = os.environ.get('AZURE_DURABLE_TASK_TEST_ENDPOINT', 'localhost:8080')
    taskhub = os.environ.get('AZURE_DURABLE_TASK_TEST_TASKHUB', 'default')
    with grpc.insecure_channel(endpoint) as channel:
        try:
            grpc.channel_ready_future(channel).result(timeout=_WAIT_SECONDS)
        except grpc.FutureTimeoutError:
            _unavailable(f'No Durable Task Scheduler at {endpoint} (AZURE_DURABLE_TASK_TEST_ENDPOINT).')
    with TaskHubGrpcClient(host_address=endpoint, metadata=[('taskhub', taskhub)], secure_channel=False) as client:
        try:
            client.get_orchestration_state(f'harness-readiness-{uuid4().hex}')
        except grpc.RpcError as exc:
            _unavailable(f'Durable Task Scheduler at {endpoint}, hub {taskhub!r}, is unavailable: {exc}')
        yield _Hub(endpoint, taskhub, client)


@contextmanager
def _running_worker(hub: _Hub, agent: AbstractAgent[DepsT, OutputT], orchestrator: Workflow[InputT]) -> Generator[None]:
    worker = TaskHubGrpcWorker(
        host_address=hub.endpoint,
        metadata=hub.metadata(),
        secure_channel=False,
        concurrency_options=ConcurrencyOptions(maximum_concurrent_activity_work_items=1),
    )
    # Restrict dispatch to these registrations when using an existing shared hub.
    worker.use_work_item_filters()
    worker.add_orchestrator(orchestrator)
    durability = AzureDurableTaskDurability.from_agent(agent)
    assert durability is not None
    with durability.register(worker):
        try:
            worker.start()
            yield
        finally:
            worker.stop()


def _output(hub: _Hub, instance_id: str) -> JsonValue:
    state = hub.client.wait_for_orchestration_completion(instance_id, timeout=_WAIT_SECONDS)
    assert state is not None
    state.raise_if_failed()
    return _JSON.validate_python(state.get_output())


def _pending(hub: _Hub, instance_id: str, *, pause: int = 0) -> _Pending:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        state = hub.client.get_orchestration_state(instance_id)
        assert state is not None
        state.raise_if_failed()
        status = state.get_custom_status()
        if status is not None:
            pending = TypeAdapter(_Pending).validate_python(status)
            if pending.pause == pause:
                return pending
        time.sleep(0.05)
    pytest.fail(f'Orchestration {instance_id!r} did not reach review pause {pause}.')


def _tool_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    if returns:
        return ModelResponse(parts=[TextPart(str(returns[-1].content))])
    return ModelResponse(parts=[ToolCallPart(info.function_tools[0].name, {}, tool_call_id='call')])


class TestLiveAzureDurableTask:
    """Verify native dispatch, recorded replay and user-owned review patterns against the service."""

    def test_typed_output(self, hub: _Hub) -> None:
        class Output(BaseModel):
            value: int

        agent = Agent(
            TestModel(custom_output_args={'value': 42}),
            name=f'typed-{uuid4().hex}',
            output_type=Output,
            capabilities=[AzureDurableTaskDurability()],
        )

        def workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, JsonValue]:
            result = yield from run_durable(ctx, lambda: agent.run(prompt))
            return {'output': result.output.model_dump(mode='json'), 'requests': result.usage.requests}

        workflow.__name__ = f'{agent.name}_workflow'
        with _running_worker(hub, agent, workflow):
            instance_id = hub.client.schedule_new_orchestration(workflow, input='Hi')
            assert _output(hub, instance_id) == {'output': {'value': 42}, 'requests': 1}

    def test_tool_activity(self, hub: _Hub) -> None:
        calls: list[str] = []
        agent = Agent(
            FunctionModel(_tool_model),
            name=f'tool/<{uuid4().hex}>/\u00e9`+:',
            deps_type=str,
            capabilities=[
                AzureDurableTaskDurability(
                    tool_options=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
                )
            ],
        )

        @agent.tool
        async def identity(ctx: RunContext[str]) -> str:
            calls.append('activity')
            return f'activity {ctx.deps}'

        def workflow(ctx: task.OrchestrationContext, value: str) -> Generator[task.Task[object], object, JsonValue]:
            result = yield from run_durable(ctx, lambda: agent.run('Hi', deps=value))
            return {'output': result.output, 'requests': result.usage.requests}

        workflow.__name__ = f'tool_{uuid4().hex}_workflow'
        with _running_worker(hub, agent, workflow):
            instance_id = hub.client.schedule_new_orchestration(workflow, input='result')
            assert _output(hub, instance_id) == {'output': 'activity result', 'requests': 2}
        assert calls == ['activity']

    def test_inline_tool_metadata(self, hub: _Hub) -> None:
        calls: list[bool] = []
        agent = Agent(
            TestModel(custom_output_text='done'),
            name=f'inline-{uuid4().hex}',
            capabilities=[AzureDurableTaskDurability()],
        )
        durability = AzureDurableTaskDurability.from_agent(agent)
        assert durability is not None

        @agent.tool_plain(metadata={'azure_durable_task': False})
        def deterministic() -> int:
            calls.append(durability.in_durable_context)
            return 42

        def workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, JsonValue]:
            result = yield from run_durable(ctx, lambda: agent.run(prompt))
            return result.output

        workflow.__name__ = f'inline_{uuid4().hex}_workflow'
        with _running_worker(hub, agent, workflow):
            instance_id = hub.client.schedule_new_orchestration(workflow, input='Hi')
            assert _output(hub, instance_id) == 'done'
        assert calls
        assert all(calls)

    def test_parallel_tool_activities(self, hub: _Hub) -> None:
        barrier = Barrier(2, timeout=_WAIT_SECONDS)
        calls: list[int] = []
        name = f'parallel-{uuid4().hex}'

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[ToolCallPart('identity', {'value': value}, tool_call_id=str(value)) for value in (1, 2)]
                )
            return ModelResponse(parts=[TextPart('parallel')])

        def build() -> tuple[Agent[None, str], Workflow[str]]:
            agent = Agent(FunctionModel(model), name=name, capabilities=[AzureDurableTaskDurability()])

            @agent.tool_plain
            def identity(value: int) -> int:
                barrier.wait()
                calls.append(value)
                return value

            def workflow(
                ctx: task.OrchestrationContext, prompt: str
            ) -> Generator[task.Task[object], object, JsonValue]:
                result = yield from run_durable(ctx, lambda: agent.run(prompt))
                return result.output

            workflow.__name__ = f'{name}_workflow'
            return agent, workflow

        first, first_workflow = build()
        second, second_workflow = build()
        with _running_worker(hub, first, first_workflow), _running_worker(hub, second, second_workflow):
            instance_id = hub.client.schedule_new_orchestration(first_workflow, input='Hi')
            assert _output(hub, instance_id) == 'parallel'
        assert sorted(calls) == [1, 2]

    @pytest.mark.parametrize('decision', ['approve', 'deny', 'retry'])
    def test_documented_review_round_trip(self, hub: _Hub, decision: str) -> None:
        requests = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal requests
            requests += 1
            if requests == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('approve_payment', {'amount': 1}, tool_call_id='call_X'),
                        ToolCallPart('external_lookup', {}, tool_call_id='call_x'),
                    ]
                )
            if decision == 'retry' and requests == 2:
                assert any(
                    isinstance(part, ToolReturnPart) and part.outcome == 'retried' and 'Try again' in str(part.content)
                    for message in messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                )
                return ModelResponse(parts=[ToolCallPart('external_lookup', {}, tool_call_id='call_x')])
            content = [
                str(part.content)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
            return ModelResponse(parts=[TextPart('|'.join(content))])

        example = _review_example(FunctionModel(model), agent_name=f'review-{uuid4().hex}')
        with _running_worker(hub, example.agent, example.orchestrator):
            instance_id = hub.client.schedule_new_orchestration(example.orchestrator, input='Review')
            pending = _pending(hub, instance_id)
            assert pending.requests.approvals[0].tool_call_id == 'call_X'
            assert pending.requests.calls[0].tool_call_id == 'call_x'
            decisions: dict[str, JsonValue] = {
                'call_X': {'action': 'approve', 'override_args': {'amount': 7}},
                'call_x': {'action': 'return', 'value': {'status': 'complete'}},
            }
            if decision == 'deny':
                decisions['call_X'] = {'action': 'deny', 'message': 'Payment blocked'}
            elif decision == 'retry':
                decisions['call_x'] = {'action': 'retry', 'message': 'Try again'}
            example.send_review(hub.client, instance_id, pause=pending.pause, decisions=decisions)
            if decision == 'retry':
                next_pause = _pending(hub, instance_id, pause=1)
                assert next_pause.requests.calls[0].tool_call_id == 'call_x'
                with pytest.raises(UserError, match='no longer pending'):
                    example.send_review(hub.client, instance_id, pause=pending.pause, decisions=decisions)
                example.send_review(
                    hub.client,
                    instance_id,
                    pause=next_pause.pause,
                    decisions={'call_x': {'action': 'return', 'value': {'status': 'complete'}}},
                )
            output = _output(hub, instance_id)
        assert isinstance(output, dict)
        assert output['requests'] == (3 if decision == 'retry' else 2)
        text = output['output']
        assert isinstance(text, str) and 'complete' in text
        assert ('Paid 7' in text) is (decision != 'deny')
        if decision == 'deny':
            assert 'Payment blocked' in text

    def test_replacement_worker_replays_completed_activity(self, hub: _Hub) -> None:
        name = f'recovery-{uuid4().hex}'
        model_calls: list[str] = []

        def build(worker_name: str) -> _ReviewExample:
            def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                model_calls.append(worker_name)
                if len(messages) == 1:
                    return ModelResponse(parts=[ToolCallPart('approve_payment', {'amount': 7}, tool_call_id='call')])
                return _tool_model(messages, info)

            return _review_example(FunctionModel(model), agent_name=name)

        first, second = build('A'), build('B')
        with _running_worker(hub, first.agent, first.orchestrator):
            instance_id = hub.client.schedule_new_orchestration(first.orchestrator, input='Review')
            pending = _pending(hub, instance_id)
            assert model_calls == ['A']
        with _running_worker(hub, second.agent, second.orchestrator):
            second.send_review(hub.client, instance_id, pause=pending.pause, decisions={'call': {'action': 'approve'}})
            assert _output(hub, instance_id) == {'output': 'Paid 7', 'requests': 2}
        assert model_calls == ['A', 'B']

    def test_dependencies_are_reconstructed_on_each_worker(self, hub: _Hub) -> None:
        barrier = Barrier(2, timeout=_WAIT_SECONDS)
        lock = Lock()
        used: list[tuple[str, _Dependencies]] = []
        name = f'deps-{uuid4().hex}'

        def build(worker_name: str) -> tuple[Agent[_Dependencies, str], Workflow[str]]:
            agent = Agent(
                FunctionModel(_tool_model),
                name=name,
                deps_type=_Dependencies,
                capabilities=[AzureDurableTaskDurability()],
            )

            @agent.tool
            def identity(ctx: RunContext[_Dependencies]) -> str:
                barrier.wait()
                with lock:
                    used.append((worker_name, ctx.deps))
                return ctx.deps.tenant

            def workflow(
                ctx: task.OrchestrationContext, tenant: str
            ) -> Generator[task.Task[object], object, JsonValue]:
                result = yield from run_durable(ctx, lambda: agent.run('Hi', deps=_Dependencies(tenant)))
                return result.output

            workflow.__name__ = f'{name}_workflow'
            return agent, workflow

        first, first_workflow = build('A')
        second, second_workflow = build('B')
        with _running_worker(hub, first, first_workflow), _running_worker(hub, second, second_workflow):
            instances = [
                hub.client.schedule_new_orchestration(first_workflow, input=tenant)
                for tenant in ('tenant-a', 'tenant-b')
            ]
            assert [_output(hub, instance_id) for instance_id in instances] == ['tenant-a', 'tenant-b']
        assert {worker for worker, _ in used} == {'A', 'B'}
        assert {deps.tenant for _, deps in used} == {'tenant-a', 'tenant-b'}
        assert len({id(deps) for _, deps in used}) == 2
