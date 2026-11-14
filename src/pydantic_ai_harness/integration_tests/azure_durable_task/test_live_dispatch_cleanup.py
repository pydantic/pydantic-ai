"""Verify dispatch teardown through the public capability and a real scheduler."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Generator
from threading import Lock
from uuid import uuid4

import grpc
import pytest
from durabletask import task
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable


@pytest.mark.parametrize('cancel_during_teardown', [False, True])
def test_dispatch_closes_suspended_agent_scope(cancel_during_teardown: bool) -> None:
    """A suspended dispatch drains its async root without losing its scheduled model activity."""
    endpoint = os.environ.get('AZURE_DURABLE_TASK_TEST_ENDPOINT', 'localhost:8080')
    taskhub = os.environ.get('AZURE_DURABLE_TASK_TEST_TASKHUB', 'default')
    with grpc.insecure_channel(endpoint) as channel:
        try:
            grpc.channel_ready_future(channel).result(timeout=30)
        except grpc.FutureTimeoutError:
            message = f'No Durable Task Scheduler at {endpoint} (AZURE_DURABLE_TASK_TEST_ENDPOINT).'
            if os.environ.get('AZURE_DURABLE_TASK_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
                pytest.fail(message)
            pytest.skip(message)

    roots: list[asyncio.Task[object]] = []
    closed: list[bool] = []
    lock = Lock()

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        with lock:
            assert closed == [True]
            assert roots[0].cancelled() and roots[0].get_loop().is_closed()
        return ModelResponse(parts=[TextPart('cleaned up')])

    agent = Agent(
        FunctionModel(model),
        name=f'cleanup-{uuid4().hex}',
        capabilities=[AzureDurableTaskDurability()],
    )

    def workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
        async def body() -> str:
            root = asyncio.current_task()
            assert root is not None
            with lock:
                roots.append(root)
            try:
                return (await agent.run(prompt)).output
            finally:
                await asyncio.sleep(0)
                with lock:
                    closed.append(True)

        try:
            return (yield from run_durable(ctx, body))
        except GeneratorExit:
            if cancel_during_teardown:
                raise asyncio.CancelledError('dispatch teardown') from None
            raise

    workflow.__name__ = f'cleanup_{uuid4().hex}'
    worker = DurableTaskSchedulerWorker(
        host_address=endpoint, taskhub=taskhub, token_credential=None, secure_channel=False
    )
    worker.use_work_item_filters()
    worker.add_orchestrator(workflow)
    durability = AzureDurableTaskDurability.from_agent(agent)
    assert durability is not None
    with durability.register(worker):
        try:
            worker.start()
            with DurableTaskSchedulerClient(
                host_address=endpoint, taskhub=taskhub, token_credential=None, secure_channel=False
            ) as client:
                instance_id = client.schedule_new_orchestration(workflow, input='Hi')
                state = client.wait_for_orchestration_completion(instance_id, timeout=30)
                assert state is not None
                state.raise_if_failed()
                assert state.get_output() == 'cleaned up'
        finally:
            worker.stop()
    with lock:
        assert closed == [True, True]
        assert all(root.done() and root.get_loop().is_closed() for root in roots)
