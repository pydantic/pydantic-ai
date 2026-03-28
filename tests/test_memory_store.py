import pytest

from pydantic_ai import Agent, InMemoryStore, SQLiteMemoryStore
from pydantic_ai.models.test import TestModel


pytestmark = pytest.mark.anyio


async def test_in_memory_store_persists_across_runs() -> None:
    store = InMemoryStore()
    agent = Agent(TestModel(custom_output_text='ok'), memory=store)

    r1 = await agent.run('hello', session_id='s1')
    assert len(r1.all_messages()) > 0

    r2 = await agent.run('hello again', session_id='s1')
    assert len(r2.all_messages()) >= len(r1.all_messages())


async def test_sqlite_memory_store_persists_across_agents(tmp_path) -> None:
    db_path = tmp_path / 'memory.db'

    store1 = SQLiteMemoryStore(db_path)
    agent1 = Agent(TestModel(custom_output_text='ok'), memory=store1)

    r1 = await agent1.run('first', session_id='s1')
    assert len(r1.all_messages()) > 0

    store2 = SQLiteMemoryStore(db_path)
    agent2 = Agent(TestModel(custom_output_text='ok'), memory=store2)

    r2 = await agent2.run('second', session_id='s1')
    assert len(r2.all_messages()) >= len(r1.all_messages())
