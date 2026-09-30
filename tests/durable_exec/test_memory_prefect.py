"""Memory tools registered and executed by a real Prefect flow."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.conftest import try_import

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED, temporary_settings
    from prefect.testing.utilities import prefect_test_harness

    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai_harness.memory import InMemoryStore, Memory, MemoryStore

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='Prefect or Harness is not installed'),
    pytest.mark.xdist_group(name='prefect'),
]


@pytest.fixture(scope='module')
def prefect_server() -> Iterator[None]:
    with temporary_settings({PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED: False}):
        with prefect_test_harness(server_startup_timeout=60):
            yield


@pytest.mark.parametrize('resolve_store', [False, True])
@pytest.mark.parametrize('prefix', ['', 'tenant_'])
async def test_memory_tools_run_in_prefect(prefect_server: None, resolve_store: bool, prefix: str) -> None:
    store = InMemoryStore()
    resolved_stores: list[InMemoryStore] = []

    def resolver(ctx: RunContext[object]) -> MemoryStore:
        selected = InMemoryStore()
        resolved_stores.append(selected)
        return selected

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returned:
            return ModelResponse(parts=[ToolCallPart(f'{prefix}write_memory', {'content': 'Prefer short answers.'})])
        if len(returned) == 1:
            return ModelResponse(parts=[ToolCallPart(f'{prefix}read_memory', {'file': 'MEMORY.md'})])
        return ModelResponse(parts=[TextPart(str(returned[-1].content))])

    memory = Memory(store=store, store_resolver=resolver if resolve_store else None, inject_memory=False)
    agent = Agent(
        FunctionModel(model),
        name='memory-prefect',
        capabilities=[memory.prefix_tools(prefix.removesuffix('_')) if prefix else memory, PrefectDurability()],
    )

    @flow
    async def read_and_write_memory() -> str:
        return (await agent.run('Remember my preference, then read it.')).output

    assert await read_and_write_memory() == 'Prefer short answers.\n'
    assert len(resolved_stores) == (1 if resolve_store else 0)
    saved = await (resolved_stores[0] if resolve_store else store).read('main/MEMORY.md', max_chars=100)
    assert saved is not None
    assert saved.content == 'Prefer short answers.\n'
