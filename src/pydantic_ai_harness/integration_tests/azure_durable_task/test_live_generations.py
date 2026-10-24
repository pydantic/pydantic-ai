"""Native SDK generation rollover and retained-version dispatch against the release emulator."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Protocol
from uuid import uuid4

import pytest
from durabletask import task
from durabletask.client import TaskHubGrpcClient
from durabletask.worker import TaskHubGrpcWorker, VersionFailureStrategy, VersioningOptions, VersionMatchStrategy
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable
from tests.harness.azure_durable_task.examples import load_generation_example

from .test_live_azure_durable_task import hub as hub


class LiveHub(Protocol):
    """Shared live fixture contract without coupling to its implementation class."""

    endpoint: str
    client: TaskHubGrpcClient

    def metadata(self) -> list[tuple[str, str]]: ...


@contextmanager
def worker_scope(
    hub: LiveHub,
    durability: AzureDurableTaskDurability[None],
    *,
    previous_versions: dict[str, Agent[None, str | DeferredToolRequests]] | None = None,
) -> Generator[TaskHubGrpcWorker]:
    """Keep activity runtimes alive until this test's filtered SDK worker stops."""
    worker = TaskHubGrpcWorker(host_address=hub.endpoint, metadata=hub.metadata(), secure_channel=False)
    worker.use_work_item_filters()
    with durability.register(worker, previous_versions=previous_versions):
        try:
            yield worker
        finally:
            worker.stop()


class TestLiveGenerations:
    """Prove native continuation, version stamping and retained worker dispatch against the scheduler."""

    def test_published_rollover_spans_three_generations(self, hub: LiveHub) -> None:
        calls: list[int] = []
        run_ids: list[str | None] = []
        conversations: list[str | None] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            count = sum(
                isinstance(part, ToolReturnPart)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
            if count == 4:
                return ModelResponse(parts=[TextPart('four steps')])
            return ModelResponse(parts=[ToolCallPart('count', {'value': count}, tool_call_id=f'call-{count}')])

        example = load_generation_example(FunctionModel(model), agent_name=f'live-generation-{uuid4().hex}')

        @example.agent.tool
        async def count(ctx: RunContext[None], value: int) -> int:
            calls.append(value)
            run_ids.append(ctx.run_id)
            conversations.append(ctx.conversation_id)
            return value

        durability = AzureDurableTaskDurability.from_agent(example.agent)
        assert durability is not None
        with worker_scope(hub, durability) as worker:
            worker.add_orchestrator(example.orchestrator)
            worker.start()
            instance_id = hub.client.schedule_new_orchestration(example.orchestrator, input={'prompt': 'Count.'})
            state = hub.client.wait_for_orchestration_completion(instance_id, timeout=30)
            assert state is not None
            state.raise_if_failed()
            assert state.get_output() == {'output': 'four steps', 'requests': 5, 'generation': 2}
        assert calls == [0, 1, 2, 3]
        assert len(set(run_ids)) == 2
        assert len(set(conversations)) == 1

    def test_old_paused_run_drains_while_new_version_starts(self, hub: LiveHub) -> None:
        calls: list[str] = []
        suffix = uuid4().hex

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(message, ModelResponse) for message in messages):
                return ModelResponse(parts=[TextPart('review complete')])
            return ModelResponse(parts=[ToolCallPart('effect', {}, tool_call_id='Call-A')])

        def make_agent(version: str) -> Agent[None, str | DeferredToolRequests]:
            agent = Agent(
                FunctionModel(model),
                name=f'drain-{suffix}-{version}',
                output_type=[str, DeferredToolRequests],
                deps_type=type(None),
                capabilities=[AzureDurableTaskDurability()],
            )

            @agent.tool_plain(requires_approval=True)
            async def effect() -> str:
                calls.append(version)
                return version

            return agent

        old, new = make_agent('1.0'), make_agent('2.0')
        agents = {'1.0': old, '2.0': new}

        def workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, JsonValue]:
            version = ctx.version
            assert version is not None
            selected = agents[version]
            result = yield from run_durable(ctx, lambda: selected.run(prompt))
            assert isinstance(result.output, DeferredToolRequests)
            ctx.set_custom_status({'pause': 0, 'version': version})
            decision = yield ctx.wait_for_external_event('review:0')
            assert decision == 'deny'
            resumed = yield from run_durable(
                ctx,
                lambda: selected.run(
                    message_history=result.all_messages(),
                    deferred_tool_results=DeferredToolResults(approvals={'Call-A': ToolDenied('Denied')}),
                    usage=result.usage,
                    conversation_id=result.conversation_id,
                    run_id=f'{ctx.instance_id}:review:0:resume',
                ),
            )
            assert isinstance(resumed.output, str)
            return {
                'output': resumed.output,
                'requests': resumed.usage.requests,
                'version': version,
            }

        workflow.__name__ = f'drain_{suffix}_workflow'
        old_capability = AzureDurableTaskDurability.from_agent(old)
        capability = AzureDurableTaskDurability.from_agent(new)
        assert old_capability is not None and capability is not None
        with worker_scope(hub, old_capability) as worker:
            worker.use_versioning(
                VersioningOptions(
                    version='1.0',
                    match_strategy=VersionMatchStrategy.CURRENT_OR_OLDER,
                    failure_strategy=VersionFailureStrategy.REJECT,
                )
            )
            worker.add_orchestrator(workflow)
            worker.start()
            old_id = hub.client.schedule_new_orchestration(workflow, input='Old review.', version='1.0')
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = hub.client.get_orchestration_state(old_id)
                assert state is not None
                state.raise_if_failed()
                if state.get_custom_status() == {'pause': 0, 'version': '1.0'}:
                    break
                time.sleep(0.05)
            else:
                pytest.fail('Old instance did not reach its paused review')
        with worker_scope(hub, capability, previous_versions={'1.0': old}) as worker:
            worker.use_versioning(
                VersioningOptions(
                    version='2.0',
                    match_strategy=VersionMatchStrategy.CURRENT_OR_OLDER,
                    failure_strategy=VersionFailureStrategy.REJECT,
                )
            )
            worker.add_orchestrator(workflow)
            worker.start()
            new_id = hub.client.schedule_new_orchestration(workflow, input='New review.', version='2.0')
            for instance_id, version in [(old_id, '1.0'), (new_id, '2.0')]:
                hub.client.raise_orchestration_event(instance_id, 'review:0', data='deny')
                state = hub.client.wait_for_orchestration_completion(instance_id, timeout=30)
                assert state is not None
                state.raise_if_failed()
                assert TypeAdapter(JsonValue).validate_python(state.get_output()) == {
                    'output': 'review complete',
                    'requests': 2,
                    'version': version,
                }
        assert calls == []
