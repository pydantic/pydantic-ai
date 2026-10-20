"""Live Durable Task Scheduler conformance tests.

The in-process suite replays native SDK histories but cannot prove gRPC dispatch,
external-event delivery, worker replacement or worker-local dependency lifetimes.
These tests use the emulator, the core SDK's insecure transport and local models;
no Azure credentials or LLM API are needed.

Run `make integration-durabletask` after starting
`mcr.microsoft.com/dts/dts-emulator:latest` on ports 8080 and 8082. Override
`DURABLETASK_TEST_ENDPOINT` and `DURABLETASK_TEST_TASKHUB` for an existing server.
An unreachable endpoint skips locally; `DURABLETASK_REQUIRE_LIVE=1` makes it fail.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from threading import Barrier, Lock
from typing import NoReturn, TypeVar
from uuid import uuid4

import grpc
import pytest
from durabletask.client import TaskHubGrpcClient
from durabletask.worker import ConcurrencyOptions, TaskHubGrpcWorker
from pydantic import BaseModel, JsonValue

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability, DurableTaskRun

_WAIT_SECONDS = 30.0
DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


def _requires_live() -> bool:
    return os.environ.get('DURABLETASK_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}


def _unavailable(message: str) -> NoReturn:
    if _requires_live():
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
    worker: str
    token: str


@pytest.fixture(scope='module')
def hub() -> Generator[_Hub]:
    """Wait for gRPC readiness, verify the selected hub and own the test client."""
    endpoint = os.environ.get('DURABLETASK_TEST_ENDPOINT', 'localhost:8080')
    taskhub = os.environ.get('DURABLETASK_TEST_TASKHUB', 'default')
    # The dashboard serves HTML even for unknown paths; it is not a gRPC readiness check.
    with grpc.insecure_channel(endpoint) as channel:
        try:
            grpc.channel_ready_future(channel).result(timeout=_WAIT_SECONDS if _requires_live() else 1)
        except grpc.FutureTimeoutError:
            _unavailable(f'No Durable Task Scheduler at {endpoint} (DURABLETASK_TEST_ENDPOINT).')
    with TaskHubGrpcClient(host_address=endpoint, metadata=[('taskhub', taskhub)], secure_channel=False) as client:
        try:
            client.get_orchestration_state(f'harness-readiness-{uuid4().hex}')
        except grpc.RpcError as exc:
            _unavailable(f'Durable Task Scheduler at {endpoint}, hub {taskhub!r}, is unavailable: {exc}')
        yield _Hub(endpoint, taskhub, client)


@contextmanager
def _running_worker(hub: _Hub, durable: DurableTaskAgent[DepsT, OutputT]) -> Generator[None]:
    worker = TaskHubGrpcWorker(
        host_address=hub.endpoint,
        metadata=hub.metadata(),
        secure_channel=False,
        concurrency_options=ConcurrencyOptions(maximum_concurrent_activity_work_items=1),
    )
    # Do not consume other tests' or applications' work from a shared hub.
    worker.use_work_item_filters()
    with durable.register(worker):
        try:
            worker.start()
            yield
        finally:
            worker.stop()


def _start(
    hub: _Hub, durable: DurableTaskAgent[DepsT, OutputT], *, deps_input: JsonValue = None
) -> DurableTaskRun[OutputT]:
    return durable.start(hub.client, 'Hi', deps_input=deps_input, instance_id=f'harness-{uuid4().hex}')


def _pending_approval(hub: _Hub, run: DurableTaskRun[OutputT]) -> str:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        state = hub.client.get_orchestration_state(run.instance_id)
        assert state is not None
        assert state.failure_details is None, state.failure_details
        if state.get_custom_status() is not None:
            requests = run.pending_tools()
            assert len(requests.approvals) == 1
            return requests.approvals[0].tool_call_id
        time.sleep(0.05)
    pytest.fail(f'Orchestration {run.instance_id!r} did not request approval.')


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


class TestLiveDurableTask:
    """Exercise service boundaries that replaying in-process histories cannot verify."""

    def test_typed_output(self, hub: _Hub) -> None:
        class Output(BaseModel):
            value: int

        agent = Agent(
            TestModel(custom_output_args={'value': 42}),
            name=f'typed-{uuid4().hex}',
            output_type=Output,
            capabilities=[DurableTaskDurability()],
        )
        durable = DurableTaskAgent(agent)
        with _running_worker(hub, durable):
            result = _start(hub, durable).result(timeout=_WAIT_SECONDS)
        assert result.output == Output(value=42)
        assert result.usage.requests == 1

    def test_tool_activity(self, hub: _Hub) -> None:
        calls: list[str] = []

        @asynccontextmanager
        async def dependencies(value: JsonValue) -> AsyncGenerator[str]:
            assert isinstance(value, str)
            yield value

        agent = Agent(
            FunctionModel(_tool_model),
            name=f'tool-{uuid4().hex}',
            deps_type=str,
            capabilities=[DurableTaskDurability(deps_factory=dependencies)],
        )

        @agent.tool
        async def identity(ctx: RunContext[str]) -> str:
            calls.append('activity')
            return f'activity {ctx.deps}'

        durable = DurableTaskAgent(agent)
        with _running_worker(hub, durable):
            result = _start(hub, durable, deps_input='result').result(timeout=_WAIT_SECONDS)
        assert result.output == 'activity result'
        assert calls == ['activity']
        assert result.usage.requests == 2

    def test_parallel_tool_activities(self, hub: _Hub) -> None:
        # A single activity slot on each worker forces the rendezvous across workers.
        barrier = Barrier(2, timeout=_WAIT_SECONDS)
        calls: list[int] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[ToolCallPart('identity', {'value': value}, tool_call_id=str(value)) for value in (1, 2)]
                )
            return ModelResponse(parts=[TextPart('parallel')])

        name = f'parallel-{uuid4().hex}'

        def build() -> DurableTaskAgent[None, str]:
            agent = Agent(FunctionModel(model), name=name, capabilities=[DurableTaskDurability()])

            @agent.tool_plain
            def identity(value: int) -> int:
                barrier.wait()
                calls.append(value)
                return value

            return DurableTaskAgent(agent)

        first, second = build(), build()
        with _running_worker(hub, first), _running_worker(hub, second):
            result = _start(hub, first).result(timeout=_WAIT_SECONDS)
        assert result.output == 'parallel'
        assert sorted(calls) == [1, 2]

    def test_approval_round_trip(self, hub: _Hub) -> None:
        calls: list[str] = []
        agent = Agent(
            FunctionModel(_tool_model),
            name=f'approval-{uuid4().hex}',
            output_type=[str, DeferredToolRequests],
            capabilities=[DurableTaskDurability()],
        )

        @agent.tool_plain(requires_approval=True)
        async def approve_payment() -> str:
            calls.append('approved')
            return 'paid'

        durable = DurableTaskAgent(agent)
        with _running_worker(hub, durable):
            run = _start(hub, durable)
            tool_call_id = _pending_approval(hub, run)
            assert calls == []
            run.approve(tool_call_id)
            assert run.result(timeout=_WAIT_SECONDS).output == 'paid'
        assert calls == ['approved']

    def test_replacement_worker_replays_the_completed_activity(self, hub: _Hub) -> None:
        """Stop at a persisted approval boundary, not a race with the next activity."""
        name = f'recovery-{uuid4().hex}'
        model_calls: list[str] = []
        tool_calls: list[str] = []

        def build(worker_name: str) -> DurableTaskAgent[None, str | DeferredToolRequests]:
            def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                model_calls.append(worker_name)
                return _tool_model(messages, info)

            agent = Agent(
                FunctionModel(model),
                name=name,
                output_type=[str, DeferredToolRequests],
                capabilities=[DurableTaskDurability()],
            )

            @agent.tool_plain(requires_approval=True)
            async def identity() -> str:
                tool_calls.append(worker_name)
                return 'recovered'

            return DurableTaskAgent(agent)

        first, second = build('A'), build('B')
        with _running_worker(hub, first):
            run = _start(hub, first)
            tool_call_id = _pending_approval(hub, run)
            assert model_calls == ['A']
            assert tool_calls == []
        with _running_worker(hub, second):
            restored = second.get_run(hub.client, run.instance_id)
            restored.approve(tool_call_id)
            assert restored.result(timeout=_WAIT_SECONDS).output == 'recovered'
        assert model_calls == ['A', 'B']
        assert tool_calls == ['B']

    def test_dependencies_are_reconstructed_on_each_worker(self, hub: _Hub) -> None:
        barrier = Barrier(2, timeout=_WAIT_SECONDS)
        lock = Lock()
        entered: list[_Dependencies] = []
        exited: list[_Dependencies] = []
        used: list[_Dependencies] = []
        name = f'deps-{uuid4().hex}'

        def build(worker_name: str) -> DurableTaskAgent[_Dependencies, str]:
            @asynccontextmanager
            async def dependencies(value: JsonValue) -> AsyncGenerator[_Dependencies]:
                assert isinstance(value, str)
                deps = _Dependencies(value, worker_name, uuid4().hex)
                with lock:
                    entered.append(deps)
                try:
                    yield deps
                finally:
                    with lock:
                        exited.append(deps)

            agent = Agent(
                FunctionModel(_tool_model),
                name=name,
                deps_type=_Dependencies,
                capabilities=[DurableTaskDurability(deps_factory=dependencies)],
            )

            @agent.tool
            def identity(ctx: RunContext[_Dependencies]) -> str:
                assert ctx.deps.worker == worker_name
                barrier.wait()
                with lock:
                    used.append(ctx.deps)
                return ctx.deps.tenant

            return DurableTaskAgent(agent)

        first, second = build('A'), build('B')
        with _running_worker(hub, first), _running_worker(hub, second):
            runs = [_start(hub, first, deps_input=tenant) for tenant in ('tenant-a', 'tenant-b')]
            assert [run.result(timeout=_WAIT_SECONDS).output for run in runs] == ['tenant-a', 'tenant-b']
        assert {deps.worker for deps in used} == {'A', 'B'}
        assert len({deps.token for deps in entered}) == len(entered)
        assert sorted(deps.token for deps in entered) == sorted(deps.token for deps in exited)
