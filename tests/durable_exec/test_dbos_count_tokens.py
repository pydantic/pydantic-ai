"""The token count `UsageLimits(count_tokens_before_request=True)` asks for, under `DBOSDurability` (sqlite).

Not VCR tests: the behavior under test is which steps a workflow records and what a forked re-execution replays,
which only the DBOS system database can show.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits

from ..conftest import detach_dbos_logging, without_admin_server

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('DBOS is not installed', allow_module_level=True)

pytestmark = pytest.mark.xdist_group(name='dbos')


@pytest.fixture(scope='module')
async def dbos(tmp_path_factory: pytest.TempPathFactory) -> AsyncGenerator[DBOS]:
    # An async module-scoped fixture keeps one event loop for the module: DBOS binds its async
    # machinery to the loop it is launched under, and a per-test loop would leave it on a closed one.
    dbos_sqlite_file = tmp_path_factory.mktemp('dbos') / 'dbostest.sqlite'
    dbos_config: DBOSConfig = {
        'name': 'pydantic_dbos_count_tokens',
        'system_database_url': f'sqlite:///{dbos_sqlite_file}',
        'enable_otlp': False,
    }
    dbos = DBOS(config=without_admin_server(dbos_config))
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()
        detach_dbos_logging()


async def test_dbos_count_tokens_is_a_recorded_step_that_replays(dbos: DBOS) -> None:
    """The count is a step ahead of the request, and a replay returns it, with the key it wrote, without counting again.

    Forking from past the last step re-executes the workflow function against every recorded step, the way recovery
    does. The key stands in for `AnthropicModel`'s record of a stale-thinking retry, which later counts read.
    """
    calls: list[tuple[str, int | None]] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(('request', DBOS.step_id))
        return ModelResponse(parts=[TextPart('ok')])

    class CountingModel(FunctionModel):
        async def count_tokens(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> RequestUsage:
            calls.append(('count_tokens', DBOS.step_id))
            request = next(message for message in reversed(messages) if isinstance(message, ModelRequest))
            request.metadata = {'__pydantic_ai__': {'counted': True}}
            return RequestUsage(input_tokens=5)

    agent = Agent(CountingModel(reply), name='dbos_count_tokens', capabilities=[DBOSDurability[object]()])

    @DBOS.workflow()
    async def run_workflow() -> list[dict[str, Any] | None]:
        limits = UsageLimits(input_tokens_limit=100, count_tokens_before_request=True)
        result = await agent.run('hi', usage_limits=limits)
        return [message.metadata for message in result.all_messages() if isinstance(message, ModelRequest)]

    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await run_workflow() == [{'__pydantic_ai__': {'counted': True}}]
    assert calls == [('count_tokens', 1), ('request', 2)]
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps] == [
        'dbos_count_tokens__model.count_tokens',
        'dbos_count_tokens__model.request',
    ]

    calls.clear()
    fork = await dbos.fork_workflow_async(workflow_id, max(step['function_id'] for step in steps) + 1)
    assert await fork.get_result() == [{'__pydantic_ai__': {'counted': True}}]
    assert calls == []
