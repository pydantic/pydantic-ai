"""Memory tools registered and executed by a real Prefect flow."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.conftest import try_import

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED, temporary_settings
    from prefect.testing.utilities import prefect_test_harness

    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai_harness.memory import InMemoryStore, Memory

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='Prefect or Harness is not installed'),
    pytest.mark.xdist_group(name='prefect'),
]


@pytest.fixture(scope='module')
def prefect_server() -> Iterator[None]:
    with temporary_settings({PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED: False}):
        with prefect_test_harness(server_startup_timeout=60):
            yield


async def test_memory_tools_run_in_prefect(prefect_server: None) -> None:
    store = InMemoryStore()

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returned:
            return ModelResponse(parts=[ToolCallPart('write_memory', {'content': 'Prefer short answers.'})])
        if len(returned) == 1:
            return ModelResponse(parts=[ToolCallPart('read_memory', {'file': 'MEMORY.md'})])
        return ModelResponse(parts=[TextPart(str(returned[-1].content))])

    agent = Agent(
        FunctionModel(model),
        name='memory-prefect',
        capabilities=[Memory(store=store, inject_memory=False), PrefectDurability()],
    )

    @flow
    async def read_and_write_memory() -> str:
        return (await agent.run('Remember my preference, then read it.')).output

    assert await read_and_write_memory() == 'Prefer short answers.\n'
    saved = await store.read('main/MEMORY.md', max_chars=100)
    assert saved is not None
    assert saved.content == 'Prefer short answers.\n'
