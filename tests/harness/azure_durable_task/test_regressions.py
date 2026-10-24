from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

    from .worker import (
        InMemoryWorker,
        completed,
        registered_agent,
        schedule_agent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


@dataclass
class _DataclassDeps:
    tenant: str


class _ModelDeps(BaseModel):
    tenant: str


class TestAzureDurableTaskDurability:
    def test_user_error_is_terminal_under_native_retries(self, durable_worker: InMemoryWorker) -> None:
        effects: list[str] = []
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='terminal-user-error',
            capabilities=[
                AzureDurableTaskDurability(
                    retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
                )
            ],
        )

        @agent.tool_plain
        def submit() -> str:
            effects.append('written')
            raise UserError('The tool configuration is invalid.')

        with registered_agent(durable_worker, agent):
            run = schedule_agent(durable_worker, agent, 'Hi', deps=None)
            durable_worker.run_until_idle()
            assert durable_worker.timers == []
            with pytest.raises(OrchestrationFailedError, match='The tool configuration is invalid'):
                completed(durable_worker, run)
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
            capabilities=[AzureDurableTaskDurability()],
        )

        @agent.tool
        def read(ctx: RunContext[_DataclassDeps | _ModelDeps]) -> str:
            values.append(ctx.deps.tenant)
            return ctx.deps.tenant

        with registered_agent(durable_worker, agent):
            run = schedule_agent(durable_worker, agent, 'Hi', deps=deps_type(tenant='acme'))
            assert durable_worker.step()
            payload = durable_worker.activity_queue[0][3]
            assert payload is not None
            assert json.loads(payload)['deps_input'] == {'tenant': 'acme'}
            assert completed(durable_worker, run) == 'done'
        assert values == ['acme']
