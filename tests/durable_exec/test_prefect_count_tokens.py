"""The token count `UsageLimits(count_tokens_before_request=True)` asks for, under `PrefectDurability`."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits

from ..conftest import try_import

with try_import() as imports_successful:
    from prefect import flow
    from prefect.context import TaskRunContext

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
    pytest.mark.usefixtures('prefect_test_server'),
    pytest.mark.xdist_group(name='prefect'),
]


@pytest.fixture(autouse=True)
def blockbuster_excluded_modules() -> tuple[str, ...]:
    """Prefect inspects flow source synchronously while constructing its tasks."""
    return ('pydantic_ai.durable_exec.prefect',)


async def test_prefect_count_tokens_runs_in_its_own_task() -> None:
    """The count runs in a task of its own ahead of the request's task."""
    tasks: list[tuple[str, str | None]] = []

    def task_name() -> str | None:
        context = TaskRunContext.get()
        return context.task.name if context is not None else None

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tasks.append(('request', task_name()))
        return ModelResponse(parts=[TextPart('ok')])

    class CountingModel(FunctionModel):
        async def count_tokens(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> RequestUsage:
            tasks.append(('count_tokens', task_name()))
            return RequestUsage(input_tokens=5)

    agent = Agent(CountingModel(reply), name='prefect_count_tokens', capabilities=[PrefectDurability()])

    @flow
    async def run_flow() -> str:
        limits = UsageLimits(input_tokens_limit=100, count_tokens_before_request=True)
        return (await agent.run('hi', usage_limits=limits)).output

    assert await run_flow() == 'ok'
    assert tasks == [
        ('count_tokens', 'Count Tokens: function:reply:'),
        ('request', 'Model Request: function:reply:'),
    ]
