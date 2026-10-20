from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, CallDeferred, DeferredToolRequests, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError, OrchestrationStatus

    from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


@dataclass
class _DataclassDeps:
    tenant: str


class _ModelDeps(BaseModel):
    tenant: str


class TestDurableTaskRun:
    def test_approval_preserves_usage_and_new_messages(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'),
            output_type=[str, DeferredToolRequests],
            name='approval-result',
            capabilities=[DurableTaskDurability()],
        )

        @agent.tool_plain(requires_approval=True)
        def submit() -> str:
            return 'submitted'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            run = durable.start(durable_worker.client, 'Hi')
            durable_worker.run_until_idle()
            run.approve(run.pending_tools().approvals[0].tool_call_id)
            result = run.result()

        assert result.output == 'done'
        assert (result.usage.requests, len(result.new_messages()), len(result.all_messages())) == (2, 4, 4)
        responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
        assert result.usage.input_tokens == sum(message.usage.input_tokens for message in responses)
        assert result.usage.output_tokens == sum(message.usage.output_tokens for message in responses)

    def test_missing_result_is_not_a_timeout(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='missing-result', capabilities=[DurableTaskDurability()])
        run = DurableTaskAgent(agent).get_run(durable_worker.client, 'missing')
        with pytest.raises(UserError, match=r"'missing'.*not found"):
            run.result()
        with pytest.raises(UserError, match=r"'missing'.*not found"):
            run.wait_for_pause(timeout=0)

    def test_sdk_timeout_propagates(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='timeout-result', capabilities=[DurableTaskDurability()])
        run = DurableTaskAgent(agent).get_run(durable_worker.client, 'slow')
        timeout = TimeoutError('Timed-out waiting for the orchestration to complete')
        durable_worker.client_mock.wait_for_orchestration_completion.side_effect = timeout
        with pytest.raises(TimeoutError) as exc:
            run.result(timeout=0.25)
        assert exc.value is timeout
        durable_worker.client_mock.wait_for_orchestration_completion.assert_called_once_with('slow', timeout=0.25)

    def test_user_error_is_terminal_under_native_retries(self, durable_worker: InMemoryWorker) -> None:
        effects: list[str] = []
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='terminal-user-error',
            capabilities=[
                DurableTaskDurability(
                    retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
                )
            ],
        )

        @agent.tool_plain
        def submit() -> str:
            effects.append('written')
            raise UserError('The tool configuration is invalid.')

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            run = durable.start(durable_worker.client, 'Hi')
            durable_worker.run_until_idle()
            if durable_worker.timers:
                durable_worker.fire_timers()
            with pytest.raises(OrchestrationFailedError, match='The tool configuration is invalid'):
                run.result()
            with pytest.raises(OrchestrationFailedError, match='The tool configuration is invalid'):
                run.wait_for_pause(timeout=0)
        assert effects == ['written']
        assert durable_worker.timers == []

    @pytest.mark.parametrize('deps_type', [_DataclassDeps, _ModelDeps])
    def test_typed_dependencies_are_serialized(
        self, durable_worker: InMemoryWorker, deps_type: type[_DataclassDeps] | type[_ModelDeps]
    ) -> None:
        values: list[str] = []
        agent = Agent(
            TestModel(custom_output_text='done'),
            deps_type=deps_type,
            name='typed-deps',
            capabilities=[DurableTaskDurability()],
        )

        @agent.tool
        def read(ctx: RunContext[_DataclassDeps | _ModelDeps]) -> str:
            values.append(ctx.deps.tenant)
            return ctx.deps.tenant

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            run = durable.start(durable_worker.client, 'Hi', deps_input=deps_type(tenant='acme'))
            assert json.loads(durable_worker.instances[run.instance_id].input or 'null') == {
                'prompt': 'Hi',
                'deps_input': {'tenant': 'acme'},
            }
            assert run.result().output == 'done'
        assert values == ['acme']

    @pytest.mark.parametrize('deferred', [False, True])
    def test_wait_for_pause_and_completion(
        self, durable_worker: InMemoryWorker, monkeypatch: pytest.MonkeyPatch, deferred: bool
    ) -> None:
        now = 0.0
        sleeps: list[float] = []

        def sleep(delay: float) -> None:
            nonlocal now
            now += delay
            sleeps.append(delay)
            durable_worker.run_until_idle()

        monkeypatch.setattr('pydantic_ai_harness.durabletask._agent.monotonic', lambda: now)
        monkeypatch.setattr('pydantic_ai_harness.durabletask._agent.sleep', sleep)
        agent = Agent(
            TestModel(custom_output_text='done'),
            output_type=[str, DeferredToolRequests],
            name='wait-pause',
            capabilities=[DurableTaskDurability()],
        )

        @agent.tool_plain(requires_approval=not deferred)
        def submit() -> str:
            if deferred:
                raise CallDeferred()
            return 'submitted'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            run = durable.start(durable_worker.client, 'Hi')
            with pytest.raises(UserError, match='no pending tool requests'):
                run.pending_tools()
            pending = run.wait_for_pause(timeout=30)
            assert pending is not None
            assert sleeps == [0.1]
            assert run.wait_for_pause(timeout=0) == pending
            if deferred:
                run.resolve(pending.calls[0].tool_call_id, 'submitted')
            else:
                run.approve(pending.approvals[0].tool_call_id)
            assert run.result().output == 'done'
            assert run.wait_for_pause(timeout=0) is None

    def test_wait_for_pause_times_out_without_real_waits(
        self, durable_worker: InMemoryWorker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = 0.0
        sleeps: list[float] = []

        def sleep(delay: float) -> None:
            nonlocal now
            now += delay
            sleeps.append(delay)

        monkeypatch.setattr('pydantic_ai_harness.durabletask._agent.monotonic', lambda: now)
        monkeypatch.setattr('pydantic_ai_harness.durabletask._agent.sleep', sleep)
        agent = Agent(TestModel(), name='wait-timeout', capabilities=[DurableTaskDurability()])
        run = DurableTaskAgent(agent).start(durable_worker.client, 'Hi')
        with pytest.raises(TimeoutError, match='did not pause or finish'):
            run.wait_for_pause(timeout=0.25)
        assert sleeps == pytest.approx([0.1, 0.1, 0.05])
        assert now == 0.25

    @pytest.mark.parametrize('timeout', [-1, float('inf'), float('nan')])
    def test_wait_for_pause_rejects_invalid_timeout(self, durable_worker: InMemoryWorker, timeout: float) -> None:
        agent = Agent(TestModel(), name='invalid-timeout', capabilities=[DurableTaskDurability()])
        run = DurableTaskAgent(agent).get_run(durable_worker.client, 'missing')
        with pytest.raises(UserError, match='finite and non-negative'):
            run.wait_for_pause(timeout=timeout)
        durable_worker.client_mock.get_orchestration_state.assert_not_called()

    def test_wait_for_pause_rejects_termination(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='wait-terminated', capabilities=[DurableTaskDurability()])
        run = DurableTaskAgent(agent).start(durable_worker.client, 'Hi')
        durable_worker.instances[run.instance_id].status = OrchestrationStatus.TERMINATED
        with pytest.raises(UserError, match='status TERMINATED'):
            run.wait_for_pause(timeout=0)
