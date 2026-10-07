from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from itertools import count

import pytest

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolSearchReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

pytestmark = [pytest.mark.benchmark]


@pytest.fixture
def blockbuster_enabled() -> bool:
    return False


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(params=[256, 8192], ids=['256-tools', '8192-tools'])
async def searchable_agent(
    request: pytest.FixtureRequest, searches: int, changing_corpus: bool
) -> tuple[Agent[int, str], Iterator[int]]:
    async def lookup() -> str:
        return 'ok'

    async def prepare(ctx: RunContext[int], tool_def: ToolDefinition) -> ToolDefinition:
        description = 'Lookup unrelated records.' if ctx.deps % 2 else 'Lookup account records and profile fields.'
        return replace(tool_def, description=f'{description} revision {ctx.deps}')

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        completed = sum(
            isinstance(part, ToolSearchReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if completed < searches:
            return ModelResponse(
                parts=[
                    ToolCallPart('search_tools', {'queries': ['account profile']}, tool_call_id=f'search_{completed}')
                ]
            )
        return ModelResponse(parts=[TextPart('ok')])

    shared_schema = Tool(lookup).function_schema
    catalog = FunctionToolset[int](
        tools=[
            Tool(
                lookup,
                name=f'lookup_{index}',
                description=f'Lookup account records and profile fields for service {index % 8}.',
                defer_loading=True,
                function_schema=shared_schema,
                prepare=prepare if changing_corpus and index == 0 else None,
            )
            for index in range(request.param)
        ]
    )
    agent = Agent(
        FunctionModel(respond, profile=ModelProfile(supported_native_tools=frozenset())),
        deps_type=int,
        toolsets=[catalog],
        capabilities=[ToolSearch()],
    )
    revisions = count()
    assert await lookup() == 'ok'
    assert (await agent.run('Find account tools', deps=next(revisions))).output == 'ok'
    return agent, revisions


@pytest.mark.parametrize('searches', [1, 5], ids=['first-search', 'five-searches'])
@pytest.mark.parametrize('changing_corpus', [False, True], ids=['stable-corpus', 'changing-corpus'])
@pytest.mark.benchmark(max_time=15)
async def test_keyword_tool_search(
    searchable_agent: tuple[Agent[int, str], Iterator[int]], searches: int, changing_corpus: bool
) -> None:
    agent, revisions = searchable_agent
    revision = next(revisions)
    result = await agent.run('Find account tools', deps=revision)
    pages = [
        [match['name'] for match in part.discovered_tools]
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart)
    ]
    assert result.output == 'ok'
    assert result.usage.requests == searches + 1
    offset = revision % 2 if changing_corpus else 0
    assert pages == [
        [f'lookup_{index}' for index in range(offset + page * 10, offset + (page + 1) * 10)] for page in range(searches)
    ]
