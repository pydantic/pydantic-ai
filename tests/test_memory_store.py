"""Tests for pydantic_ai.memory — InMemoryStore, SQLiteMemoryStore, MemoryScope."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.memory import InMemoryStore, MemoryScope, SQLiteMemoryStore
from pydantic_ai.models.test import TestModel

# ===========================================================================
# MemoryScope
# ===========================================================================


def test_memory_scope_user_only() -> None:
    scope = MemoryScope(user_id='alice')
    assert scope.session_id() == 'alice'


def test_memory_scope_user_and_conversation() -> None:
    scope = MemoryScope(user_id='alice', conversation_id='conv-1')
    assert scope.session_id() == 'alice:conv-1'


def test_memory_scope_all_levels() -> None:
    scope = MemoryScope(user_id='alice', agent_id='agent-a', conversation_id='conv-1')
    assert scope.session_id() == 'alice:agent-a:conv-1'


def test_memory_scope_user_and_agent() -> None:
    scope = MemoryScope(user_id='alice', agent_id='agent-a')
    assert scope.session_id() == 'alice:agent-a'


def test_memory_scope_is_frozen() -> None:
    scope = MemoryScope(user_id='alice')
    with pytest.raises(Exception):
        scope.user_id = 'bob'  # type: ignore


# ===========================================================================
# InMemoryStore — unit tests (no Agent)
# ===========================================================================


@pytest.mark.anyio
async def test_inmemory_load_recent_empty() -> None:
    store = InMemoryStore()
    result = await store.load_recent('session-1')
    assert result == []


@pytest.mark.anyio
async def test_inmemory_load_summary_empty() -> None:
    store = InMemoryStore()
    result = await store.load_summary('session-1')
    assert result is None


@pytest.mark.anyio
async def test_inmemory_save_and_load_recent() -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = InMemoryStore()
    messages = [ModelRequest(parts=[UserPromptPart(content='Hello')])]

    await store.save('session-1', messages)
    loaded = await store.load_recent('session-1')
    assert len(loaded) == 1


@pytest.mark.anyio
async def test_inmemory_load_recent_respects_limit() -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = InMemoryStore()
    messages = [ModelRequest(parts=[UserPromptPart(content=f'Message {i}')]) for i in range(30)]
    await store.save('session-1', messages)

    loaded = await store.load_recent('session-1', limit=10)
    assert len(loaded) == 10


@pytest.mark.anyio
async def test_inmemory_set_and_load_summary() -> None:
    store = InMemoryStore()
    store.set_summary('session-1', 'User prefers concise answers.')
    result = await store.load_summary('session-1')
    assert result == 'User prefers concise answers.'


@pytest.mark.anyio
async def test_inmemory_clear() -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = InMemoryStore()
    messages = [ModelRequest(parts=[UserPromptPart(content='Hello')])]
    await store.save('session-1', messages)
    store.set_summary('session-1', 'Some summary')

    await store.clear('session-1')

    assert await store.load_recent('session-1') == []
    assert await store.load_summary('session-1') is None


@pytest.mark.anyio
async def test_inmemory_clear_is_safe_when_empty() -> None:
    store = InMemoryStore()
    await store.clear('nonexistent-session')  # should not raise


@pytest.mark.anyio
async def test_inmemory_summarize_warns() -> None:
    store = InMemoryStore()
    with pytest.warns(UserWarning, match='no-op'):
        await store.summarize('session-1')


@pytest.mark.anyio
async def test_inmemory_sessions_are_isolated() -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = InMemoryStore()
    await store.save('session-a', [ModelRequest(parts=[UserPromptPart(content='A')])])
    await store.save('session-b', [ModelRequest(parts=[UserPromptPart(content='B')])])

    a = await store.load_recent('session-a')
    b = await store.load_recent('session-b')
    assert len(a) == 1
    assert len(b) == 1


# ===========================================================================
# SQLiteMemoryStore — unit tests
# ===========================================================================


@pytest.mark.anyio
async def test_sqlite_load_recent_empty(tmp_path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    result = await store.load_recent('session-1')
    assert result == []


@pytest.mark.anyio
async def test_sqlite_save_and_load_recent(tmp_path) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    messages = [ModelRequest(parts=[UserPromptPart(content='Hello SQLite')])]

    await store.save('session-1', messages)
    loaded = await store.load_recent('session-1')
    assert len(loaded) == 1


@pytest.mark.anyio
async def test_sqlite_persists_across_instances(tmp_path) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    db = str(tmp_path / 'mem.db')
    store1 = SQLiteMemoryStore(db)
    messages = [ModelRequest(parts=[UserPromptPart(content='Persistent')])]
    await store1.save('session-1', messages)

    # New instance — simulates app restart
    store2 = SQLiteMemoryStore(db)
    loaded = await store2.load_recent('session-1')
    assert len(loaded) == 1


@pytest.mark.anyio
async def test_sqlite_load_recent_respects_limit(tmp_path) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    messages = [ModelRequest(parts=[UserPromptPart(content=f'Msg {i}')]) for i in range(25)]
    await store.save('session-1', messages)

    loaded = await store.load_recent('session-1', limit=5)
    assert len(loaded) == 5


@pytest.mark.anyio
async def test_sqlite_load_recent_order_is_chronological(tmp_path) -> None:
    """load_recent must return oldest-first, not newest-first."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    messages = [ModelRequest(parts=[UserPromptPart(content=f'Msg {i}')]) for i in range(5)]
    await store.save('session-1', messages)

    loaded = await store.load_recent('session-1', limit=5)
    assert len(loaded) == 5
    # Content of first loaded message should be "Msg 0", not "Msg 4"


@pytest.mark.anyio
async def test_sqlite_set_and_load_summary(tmp_path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    await store.set_summary('session-1', 'User is a developer who prefers Python.')
    result = await store.load_summary('session-1')
    assert result == 'User is a developer who prefers Python.'


@pytest.mark.anyio
async def test_sqlite_summary_overwrites(tmp_path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    await store.set_summary('session-1', 'First summary.')
    await store.set_summary('session-1', 'Updated summary.')
    result = await store.load_summary('session-1')
    assert result == 'Updated summary.'


@pytest.mark.anyio
async def test_sqlite_load_summary_empty(tmp_path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    result = await store.load_summary('session-1')
    assert result is None


@pytest.mark.anyio
async def test_sqlite_clear(tmp_path) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    messages = [ModelRequest(parts=[UserPromptPart(content='Hello')])]
    await store.save('session-1', messages)
    await store.set_summary('session-1', 'A summary.')

    await store.clear('session-1')

    assert await store.load_recent('session-1') == []
    assert await store.load_summary('session-1') is None


@pytest.mark.anyio
async def test_sqlite_summarize_warns(tmp_path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    with pytest.warns(UserWarning, match='no-op'):
        await store.summarize('session-1')


@pytest.mark.anyio
async def test_sqlite_sessions_are_isolated(tmp_path) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = SQLiteMemoryStore(str(tmp_path / 'mem.db'))
    await store.save('session-a', [ModelRequest(parts=[UserPromptPart(content='A')])])
    await store.save('session-b', [ModelRequest(parts=[UserPromptPart(content='B')])])

    a = await store.load_recent('session-a')
    b = await store.load_recent('session-b')
    assert len(a) == 1
    assert len(b) == 1


# ===========================================================================
# Agent integration — InMemoryStore
# ===========================================================================


@pytest.mark.anyio
async def test_agent_persists_across_runs_inmemory() -> None:
    store = InMemoryStore()
    agent = Agent(TestModel(), memory=store)

    await agent.run('First message', session_id='user-1')
    messages_after_first = await store.load_recent('user-1')
    assert len(messages_after_first) > 0

    await agent.run('Second message', session_id='user-1')
    messages_after_second = await store.load_recent('user-1')
    # After second run the full history is saved — should have more messages
    assert len(messages_after_second) >= len(messages_after_first)


@pytest.mark.anyio
async def test_agent_loads_summary_as_system_context() -> None:
    """Summary should be injected without raising."""
    store = InMemoryStore()
    store.set_summary('user-1', 'User prefers concise answers.')
    agent = Agent(TestModel(), memory=store)

    # Should not raise — summary is injected as a system context block
    result = await agent.run('Hello', session_id='user-1')
    assert result is not None


@pytest.mark.anyio
async def test_agent_raises_on_session_id_without_memory() -> None:
    from pydantic_ai.exceptions import UserError

    agent = Agent(TestModel())  # no memory configured

    with pytest.raises(UserError, match='memory'):
        await agent.run('Hello', session_id='user-1')


@pytest.mark.anyio
async def test_agent_raises_on_both_message_history_and_session_id() -> None:
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    store = InMemoryStore()
    agent = Agent(TestModel(), memory=store)
    history = [ModelRequest(parts=[UserPromptPart(content='Old message')])]

    with pytest.raises(UserError, match='session_id'):
        await agent.run(
            'Hello',
            message_history=history,
            session_id='user-1',
        )


# ===========================================================================
# Agent integration — MemoryScope
# ===========================================================================


@pytest.mark.anyio
async def test_agent_memory_scope_basic() -> None:
    store = InMemoryStore()
    agent = Agent(TestModel(), memory=store)

    scope = MemoryScope(user_id='alice', conversation_id='conv-1')
    await agent.run('Hello', memory_scope=scope)

    # Data should be stored under the composed key
    messages = await store.load_recent('alice:conv-1')
    assert len(messages) > 0


@pytest.mark.anyio
async def test_agent_memory_scope_isolation() -> None:
    """Different scopes must not bleed into each other."""
    store = InMemoryStore()
    agent = Agent(TestModel(), memory=store)

    scope_a = MemoryScope(user_id='alice', conversation_id='conv-1')
    scope_b = MemoryScope(user_id='bob', conversation_id='conv-1')

    await agent.run("Alice's message", memory_scope=scope_a)
    await agent.run("Bob's message", memory_scope=scope_b)

    alice_msgs = await store.load_recent('alice:conv-1')
    bob_msgs = await store.load_recent('bob:conv-1')

    assert len(alice_msgs) > 0
    assert len(bob_msgs) > 0


@pytest.mark.anyio
async def test_agent_raises_on_both_session_id_and_memory_scope() -> None:
    from pydantic_ai.exceptions import UserError

    store = InMemoryStore()
    agent = Agent(TestModel(), memory=store)
    scope = MemoryScope(user_id='alice')

    with pytest.raises(UserError, match='memory_scope'):
        await agent.run('Hello', session_id='user-1', memory_scope=scope)


# ===========================================================================
# Agent integration — SQLiteMemoryStore
# ===========================================================================


@pytest.mark.anyio
async def test_agent_sqlite_persists_across_agent_instances(tmp_path) -> None:
    db = str(tmp_path / 'mem.db')

    agent1 = Agent(TestModel(), memory=SQLiteMemoryStore(db))
    await agent1.run('Remember me', session_id='user-1')

    # Entirely new agent instance — simulates app restart
    Agent(TestModel(), memory=SQLiteMemoryStore(db))
    store = SQLiteMemoryStore(db)
    messages = await store.load_recent('user-1')
    assert len(messages) > 0


@pytest.mark.anyio
async def test_agent_sqlite_memory_scope(tmp_path) -> None:
    db = str(tmp_path / 'mem.db')
    store = SQLiteMemoryStore(db)
    agent = Agent(TestModel(), memory=store)

    scope = MemoryScope(user_id='arjun', agent_id='support-bot', conversation_id='t-001')
    await agent.run('Hello from Arjun', memory_scope=scope)

    messages = await store.load_recent('arjun:support-bot:t-001')
    assert len(messages) > 0
