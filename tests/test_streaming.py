import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


async def test_streamed_text_response():
    m = TestModel()

    agent = Agent(m, deps=None)

    @agent.retriever_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result = await agent.run_stream('Hello')

    assert not result.is_structured()
    assert not result.is_complete
    response = await result.get_response()
    assert response == snapshot('{"ret_a":"a-apple"}')
    assert result.is_complete


async def test_streamed_structured_response():
    m = TestModel()

    agent = Agent(m, deps=None, result_type=tuple[str, str])

    result = await agent.run_stream('')

    assert result.is_structured()
    assert not result.is_complete
    response = await result.get_response()
    assert response == snapshot(('a', 'a'))
    assert result.is_complete


async def test_streamed_text_stream():
    m = TestModel(custom_result_text='The cat sat on the mat.')

    agent = Agent(m, deps=None)

    result = await agent.run_stream('Hello')
    assert not result.is_structured()
    # typehint to test (via static typing) that the stream type is correctly inferred
    chunks: list[str] = [c async for c in result.stream()]
    # one chunk due to group_by_temporal
    assert chunks == snapshot(['The cat sat on the mat.'])
    assert result.is_complete

    result = await agent.run_stream('Hello')
    assert [c async for c in result.stream(debounce_by=None)] == snapshot(
        [
            'The ',
            'The cat ',
            'The cat sat ',
            'The cat sat on ',
            'The cat sat on the ',
            'The cat sat on the mat.',
        ]
    )

    result = await agent.run_stream('Hello')
    assert [c async for c in result.stream(text_delta=True, debounce_by=None)] == snapshot(
        ['The ', 'cat ', 'sat ', 'on ', 'the ', 'mat.']
    )
