from __future__ import annotations

from collections.abc import Sequence

import pytest

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolSearchReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.toolsets import FunctionToolset


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def make_agent(
    corpora: Sequence[Sequence[tuple[str, str | None]]], queries: Sequence[Sequence[str]], *, max_results: int = 10
) -> Agent[None, str]:
    async def lookup() -> str:
        raise NotImplementedError

    shared_schema = Tool(lookup).function_schema

    async def catalog(ctx: RunContext[None]) -> FunctionToolset[None]:
        completed = sum(
            isinstance(part, ToolSearchReturnPart)
            for message in ctx.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        return FunctionToolset(
            tools=[
                Tool(lookup, name=name, description=description, defer_loading=True, function_schema=shared_schema)
                for name, description in corpora[min(completed, len(corpora) - 1)]
            ]
        )

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        completed = sum(
            isinstance(part, ToolSearchReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if completed < len(queries):
            return ModelResponse(
                parts=[
                    ToolCallPart('search_tools', {'queries': queries[completed]}, tool_call_id=f'search-{completed}')
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    return Agent(
        FunctionModel(respond, profile=ModelProfile(supported_native_tools=frozenset())),
        deps_type=type(None),
        toolsets=[catalog],
        capabilities=[ToolSearch(max_results=max_results)],
    )


@pytest.mark.parametrize(
    'first,second,queries,max_results,expected',
    [
        pytest.param(
            [('one', 'alpha'), ('two', 'beta')],
            [('one', 'beta'), ('two', 'alpha')],
            [['alpha'], ['alpha']],
            10,
            [['one'], ['two']],
            id='description',
        ),
        pytest.param(
            [('alpha_first', None)],
            [('alpha_second', None)],
            [['alpha'], ['alpha']],
            10,
            [['alpha_first'], ['alpha_second']],
            id='name',
        ),
        pytest.param(
            [('one', 'alpha')],
            [('two', 'alpha')],
            [['alpha'], ['alpha']],
            10,
            [['one'], ['two']],
            id='membership',
        ),
        pytest.param(
            [('a', 'alpha gamma'), ('b', 'alpha'), ('c', 'alpha gamma delta')],
            [('b', 'alpha'), ('c', 'alpha gamma delta'), ('a', 'alpha gamma')],
            [['alpha gamma'], ['alpha gamma']],
            1,
            [['a'], ['c']],
            id='order',
        ),
        pytest.param(
            [('get_me', 'profile'), ('comment', 'other'), ('me', None)],
            [('get_me', 'profile'), ('comment', 'other'), ('me', None)],
            [['ME', 'me'], ['comment']],
            10,
            [['get_me', 'me'], ['comment']],
            id='word-boundaries',
        ),
        pytest.param(
            [('kelvin', 'Kİ café'), ('ascii', 'k i'), ('other', '猫')],
            [('kelvin', 'Kİ café'), ('ascii', 'k i'), ('other', '猫')],
            [['Kİ'], ['CAFÉ']],
            10,
            [['kelvin'], ['kelvin']],
            id='unicode',
        ),
    ],
)
async def test_keyword_index_tracks_current_corpus(
    first: list[tuple[str, str | None]],
    second: list[tuple[str, str | None]],
    queries: list[list[str]],
    max_results: int,
    expected: list[list[str]],
) -> None:
    result = await make_agent([first, second], queries, max_results=max_results).run('search')
    pages = [
        [match['name'] for match in part.discovered_tools]
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart)
    ]
    assert result.output == 'done'
    assert pages == expected


@pytest.mark.parametrize(
    'max_results,expected', [(-1, ['z', 'b']), (0, []), (1, ['z']), (2, ['z', 'b']), (10, ['z', 'b', 'a'])]
)
async def test_keyword_index_preserves_result_slicing(max_results: int, expected: list[str]) -> None:
    corpus = [('z', 'alpha beta'), ('a', 'alpha'), ('b', 'alpha beta'), ('k', 'unmatched')]
    result = await make_agent([corpus], [['alpha', 'beta', 'alpha']], max_results=max_results).run('search')
    returns = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart)
    ]
    assert len(returns) == 1
    assert returns[0].content == {'discovered_tools': [{'name': name} for name in expected]}


async def test_keyword_index_preserves_no_matches() -> None:
    result = await make_agent([[('lookup', 'alpha')]], [['absent']]).run('search')
    returns = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart)
    ]
    assert len(returns) == 1
    assert returns[0].content == {
        'discovered_tools': [],
        'message': 'No matching tools found. The tools you need may not be available.',
    }
