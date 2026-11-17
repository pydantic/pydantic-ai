from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import replace

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
    from durabletask.grpc_options import GrpcChannelOptions, GrpcRetryPolicyOptions
    from durabletask.worker import TaskHubGrpcWorker

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


class TestAzureDurableTaskDurability:
    @pytest.mark.parametrize('managed', [False, True])
    @pytest.mark.parametrize('configured', [False, True])
    @pytest.mark.parametrize(
        'user_agent,expected_user_agent',
        [
            (None, 'pydantic-ai'),
            ('', 'pydantic-ai'),
            ('my-app/1.0', 'my-app/1.0 pydantic-ai'),
            ('my-app/1.0 pydantic-ai', 'my-app/1.0 pydantic-ai'),
            ('pydantic-ai-extension', 'pydantic-ai-extension pydantic-ai'),
        ],
    )
    def test_worker_user_agent(
        self, managed: bool, configured: bool, user_agent: str | None, expected_user_agent: str
    ) -> None:
        """Inspect the SDK channel configuration without starting a network worker."""
        options = (
            GrpcChannelOptions(
                max_receive_message_length=1000,
                max_send_message_length=2000,
                keepalive_time_ms=3000,
                keepalive_timeout_ms=4000,
                keepalive_permit_without_calls=True,
                retry_policy=GrpcRetryPolicyOptions(max_attempts=5),
                raw_options=[('grpc.max_connection_idle_ms', 5000)],
            )
            if configured
            else None
        )
        if user_agent is not None:
            options = options or GrpcChannelOptions()
            options.raw_options.append(('grpc.primary_user_agent', user_agent))
        original = replace(options, raw_options=list(options.raw_options)) if options is not None else None
        worker = (
            DurableTaskSchedulerWorker(
                host_address='localhost:8080', taskhub='test', token_credential=None, channel_options=options
            )
            if managed
            else TaskHubGrpcWorker(channel_options=options)
        )
        expected_raw_options: list[tuple[str, object]] = [('grpc.max_connection_idle_ms', 5000)] if configured else []
        expected_raw_options.append(('grpc.primary_user_agent', expected_user_agent))

        with ExitStack() as stack:
            for name in ('first', 'second'):
                agent = Agent(TestModel(), name=name, capabilities=[AzureDurableTaskDurability()])
                durability = AzureDurableTaskDurability.from_agent(agent)
                assert durability is not None
                stack.enter_context(durability.register(worker))
                # Attribution is consumed by the SDK's lazy channel creation, not by an HTTP client.
                registered_options = worker._channel_options  # pyright: ignore[reportPrivateUsage]
                assert registered_options is not None
                assert registered_options.raw_options == expected_raw_options
                assert registered_options is not options
                if options is not None:
                    assert registered_options.raw_options is not options.raw_options
                assert replace(registered_options, raw_options=[]) == replace(
                    options or GrpcChannelOptions(), raw_options=[]
                )
                assert options == original

        assert worker._channel_options == registered_options  # pyright: ignore[reportPrivateUsage]

    def test_worker_without_channel_options(self, durable_worker: InMemoryWorker) -> None:
        """Keep SDK-compatible in-memory workers independent of network channel configuration."""
        agent = Agent(TestModel(), name='stub', capabilities=[AzureDurableTaskDurability()])
        durability = AzureDurableTaskDurability.from_agent(agent)
        assert durability is not None
        assert not hasattr(durable_worker, '_channel_options')
        with durability.register(durable_worker):
            assert not hasattr(durable_worker, '_channel_options')


def test_worker_completes_synchronous_orchestrators(durable_worker: InMemoryWorker) -> None:
    def orchestrator(context: task.OrchestrationContext, payload: object) -> str:
        return 'done'

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.schedule(orchestrator)
    state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() == 'done'


def test_worker_waits_for_an_external_event(durable_worker: InMemoryWorker) -> None:
    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, str]:
        response = yield context.wait_for_external_event('resume')
        assert isinstance(response, str)
        return response

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.schedule(orchestrator)
    with pytest.raises(TimeoutError, match='has not completed'):
        durable_worker.wait(instance_id)
    durable_worker.raise_event(instance_id, 'resume', data='resumed')
    state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() == 'resumed'
