from __future__ import annotations

import pytest

from pydantic_ai import Agent, Conversation, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.conversation_store import ConversationConflict, InMemoryConversationStore
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests

pytestmark = pytest.mark.anyio


def _entry_count(store: InMemoryConversationStore) -> int:
    return len(store._entries)  # pyright: ignore[reportPrivateUsage]


async def test_a_saved_conversation_loads_whole() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    result = await agent.run('Hello')

    await store.save(result.conversation)
    loaded = await store.load(result.conversation.conversation_id)

    assert loaded == result.conversation
    assert await store.load('unknown') is None


async def test_continuing_a_conversation_stores_only_its_new_messages() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    first = await agent.run('Hello')
    await store.save(first.conversation)
    conversation = await store.load(first.conversation.conversation_id)
    assert conversation is not None

    second = await agent.run('Again', conversation=conversation)
    await store.save(second.conversation)

    assert _entry_count(store) == len(second.conversation.messages)
    loaded = await store.load(first.conversation.conversation_id)
    assert loaded == second.conversation


async def test_a_rewritten_history_branches_off_the_last_shared_message() -> None:
    store = InMemoryConversationStore()
    messages = [
        ModelRequest(parts=[UserPromptPart('one')]),
        ModelResponse(parts=[TextPart('two')]),
        ModelRequest(parts=[UserPromptPart('three')]),
    ]
    conversation = Conversation(messages=messages)
    await store.save(conversation)
    compacted = Conversation(
        messages=[messages[0], ModelResponse(parts=[TextPart('summary')])],
        conversation_id=conversation.conversation_id,
    )

    await store.save(compacted)

    assert await store.load(conversation.conversation_id) == compacted
    # The first message is shared; the rewritten one is a new entry beside the two it replaced.
    assert _entry_count(store) == 4


async def test_a_paused_conversation_keeps_what_it_is_waiting_on_and_its_scope() -> None:
    store = InMemoryConversationStore()
    requests = DeferredToolRequests(metadata={'call': {'reason': 'approval'}})
    conversation = Conversation(deferred_tool_requests=requests)

    first = await store.save(conversation, scope='user-1')
    second = await store.save(conversation)

    assert (first.scope, second.scope) == ('user-1', 'user-1')
    assert (first.revision, second.revision) == (0, 1)
    loaded = await store.load(conversation.conversation_id)
    assert loaded is not None
    assert loaded.deferred_tool_requests == requests


async def test_a_save_from_a_stale_copy_conflicts() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)
    stale = await store.read_record(conversation.conversation_id)
    assert stale is not None
    await store.save(conversation)

    with pytest.raises(ConversationConflict, match='changed in the store since it was loaded'):
        await store.write_record(stale, expected_revision=stale.revision)


async def test_a_fork_shares_the_history_it_keeps() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    result = await agent.run('Hello')
    original = result.conversation
    await store.save(original, scope='user-1')
    stored_entries = _entry_count(store)

    whole = await store.fork(original.conversation_id, new_conversation_id='whole')
    first_only = await store.fork(original.conversation_id, at=1, scope='user-2')

    assert whole.messages == original.messages
    assert whole.usage == original.usage
    assert whole.conversation_id == 'whole'
    assert first_only.messages == original.messages[:1]
    assert first_only.conversation_id != original.conversation_id
    assert _entry_count(store) == stored_entries
    record = await store.read_record(first_only.conversation_id)
    assert record is not None
    assert record.scope == 'user-2'
    assert record.forked_from == (await store.read_history(record.head_id))[-1].id
    assert (await store.fork(original.conversation_id, at=0)).messages == []

    continued = await agent.run('Again', conversation=first_only)
    await store.save(continued.conversation)
    assert await store.load(first_only.conversation_id) == continued.conversation
    assert await store.load(original.conversation_id) == original


async def test_forking_a_missing_conversation_or_past_its_end_raises() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)

    with pytest.raises(UserError, match="Conversation 'unknown' is not in the store"):
        await store.fork('unknown')
    with pytest.raises(UserError, match='`at` must be between 0 and 1'):
        await store.fork(conversation.conversation_id, at=2)


async def test_conversations_list_newest_first_by_scope() -> None:
    store = InMemoryConversationStore()
    first = Conversation()
    second = Conversation()
    other = Conversation()
    await store.save(first, scope='user-1')
    await store.save(other, scope='user-2')
    await store.save(second, scope='user-1')

    listed = await store.list_conversations(scope='user-1')

    assert [record.conversation_id for record in listed] == [second.conversation_id, first.conversation_id]
    assert len(await store.list_conversations()) == 3


async def test_deleting_a_conversation_keeps_the_entries_a_fork_still_uses() -> None:
    store = InMemoryConversationStore()
    messages = [ModelRequest(parts=[UserPromptPart('one')]), ModelResponse(parts=[TextPart('two')])]
    original = Conversation(messages=messages)
    await store.save(original)
    fork = await store.fork(original.conversation_id, at=1)

    await store.delete(original.conversation_id)

    assert await store.load(original.conversation_id) is None
    assert await store.load(fork.conversation_id) == fork
    assert _entry_count(store) == 1
    await store.delete(fork.conversation_id)
    assert _entry_count(store) == 0


async def test_a_saved_request_a_run_sends_is_stored_again_with_its_run_metadata() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('queued')])])
    await store.save(conversation)
    loaded = await store.load(conversation.conversation_id)
    assert loaded is not None

    result = await Agent(TestModel()).run(conversation=loaded)
    await store.save(result.conversation)

    # The run fills in the request's `run_id`, so the save branches off before it.
    assert result.conversation.messages[0] != conversation.messages[0]
    assert _entry_count(store) == 3
    assert await store.load(conversation.conversation_id) == result.conversation
