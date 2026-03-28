from datetime import timezone
from pathlib import Path

import pytest

from pydantic_ai import Agent, InMemoryStore, SQLiteMemoryStore, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInt, IsNow, IsStr

pytestmark = pytest.mark.anyio


async def test_in_memory_store_persists_across_runs() -> None:
    store = InMemoryStore()
    agent = Agent(TestModel(custom_output_text='ok'), memory=store)

    r1 = await agent.run('hello', session_id='s1')
    assert r1.output == 'ok'
    assert r1.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )

    r2 = await agent.run('hello again', session_id='s1')
    assert r2.output == 'ok'
    assert r2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok')],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_sqlite_memory_store_clear(tmp_path: Path) -> None:
    db_path = tmp_path / 'memory.db'

    store = SQLiteMemoryStore(db_path)
    agent = Agent(TestModel(custom_output_text='ok'), memory=store)

    r1 = await agent.run('hello', session_id='s1')
    assert r1.output == 'ok'

    await store.clear('s1')

    r2 = await agent.run('hello again', session_id='s1')
    assert r2.output == 'ok'
    assert r2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok')],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_in_memory_store_clear() -> None:
    store = InMemoryStore()
    agent = Agent(TestModel(custom_output_text='ok'), memory=store)

    r1 = await agent.run('hello', session_id='s1')
    assert r1.output == 'ok'

    await store.clear('s1')

    r2 = await agent.run('hello again', session_id='s1')
    assert r2.output == 'ok'
    assert r2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok')],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_sqlite_memory_store_persists_across_agents(tmp_path: Path) -> None:
    db_path = tmp_path / 'memory.db'

    store1 = SQLiteMemoryStore(db_path)
    agent1 = Agent(TestModel(custom_output_text='ok'), memory=store1)

    r1 = await agent1.run('first', session_id='s1')
    assert r1.output == 'ok'
    assert r1.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='first', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )

    store2 = SQLiteMemoryStore(db_path)
    agent2 = Agent(TestModel(custom_output_text='ok'), memory=store2)

    r2 = await agent2.run('second', session_id='s1')
    assert r2.output == 'ok'
    assert r2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='first', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='second', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok')],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_session_id_without_memory_raises() -> None:
    agent = Agent(TestModel(custom_output_text='ok'))

    with pytest.raises(UserError, match='session_id'):
        await agent.run('hello', session_id='s1')


async def test_message_history_and_session_id_raises() -> None:
    store = InMemoryStore()
    agent = Agent(TestModel(custom_output_text='ok'), memory=store)

    r1 = await agent.run('hello', session_id='s1')

    with pytest.raises(UserError, match='message_history'):
        await agent.run('hello again', message_history=r1.all_messages(), session_id='s1')


async def test_memory_save_failure_raises_user_error() -> None:
    class FailingMemoryStore:
        async def load(self, session_id: str) -> list[ModelMessage]:
            return []

        async def save(self, session_id: str, messages: list[ModelMessage]) -> None:
            raise RuntimeError('save failed')

        async def clear(self, session_id: str) -> None:
            pass

    agent = Agent(TestModel(custom_output_text='ok'), memory=FailingMemoryStore())

    with pytest.raises(UserError, match='saving message history'):
        await agent.run('hello', session_id='s1')


async def test_run_stream_session_id_not_consumed_warns() -> None:
    store = InMemoryStore()
    agent = Agent(TestModel(custom_output_text='ok'), memory=store)

    with pytest.warns(UserWarning, match='no final result was produced'):
        async with agent.run_stream('hello', session_id='s1'):
            pass
