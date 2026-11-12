from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from unittest.mock import Mock

import pytest
from pydantic import TypeAdapter

from pydantic_ai import (
    Agent,
    AgentRunResult,
    Conversation,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    messages as _messages,
)
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.conversation import ConversationTypeAdapter
from pydantic_ai.conversation_store import (
    ConversationConflict,
    ConversationEntry,
    ConversationRecord,
    InMemoryConversationStore,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelSettingsDeltaPart, ToolCallPart, ToolReturnPart, repair_messages
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def _entry_count(store: InMemoryConversationStore) -> int:
    return len(store._entries)  # pyright: ignore[reportPrivateUsage]


async def test_a_saved_conversation_loads_whole() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    conversation = (await agent.run('Hello')).conversation

    await store.save(conversation)
    loaded = await store.load(conversation.conversation_id)

    assert loaded == conversation
    assert conversation.revision is not None
    assert await store.load('unknown') is None


async def test_continuing_a_conversation_stores_only_its_new_messages() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    first = (await agent.run('Hello')).conversation
    await store.save(first)

    second = (await agent.run('Again', conversation=first)).conversation
    await store.save(second)

    assert second.revision not in (None, first.revision)
    assert _entry_count(store) == len(second.messages)
    assert await store.load(first.conversation_id) == second


async def test_the_store_keeps_its_own_copy_of_each_message() -> None:
    store = InMemoryConversationStore()
    request = ModelRequest(parts=[UserPromptPart('one')])
    conversation = Conversation(messages=[request])
    await store.save(conversation)

    request.run_id = 'changed after saving'
    loaded = await store.load(conversation.conversation_id)
    assert loaded is not None
    loaded.messages[0].run_id = 'changed after loading'

    reloaded = await store.load(conversation.conversation_id)
    assert reloaded is not None
    assert reloaded.messages[0].run_id is None


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
        revision=conversation.revision,
    )

    await store.save(compacted)

    assert await store.load(conversation.conversation_id) == compacted
    # The first message is shared; the rewritten one is a new entry beside the two it replaced.
    assert _entry_count(store) == 4


async def test_a_paused_conversation_keeps_what_it_is_waiting_on_and_its_scope() -> None:
    store = InMemoryConversationStore()
    requests = DeferredToolRequests(metadata={'call': {'reason': 'approval'}})
    conversation = Conversation(deferred_tool_requests=requests)

    await store.save(conversation, scope='user-1')
    first = await store.read_record(conversation.conversation_id)
    await store.save(conversation)
    second = await store.read_record(conversation.conversation_id)

    assert first is not None and second is not None
    assert (first.scope, second.scope) == ('user-1', 'user-1')
    assert first.revision != second.revision == conversation.revision
    loaded = await store.load(conversation.conversation_id)
    assert loaded is not None
    assert loaded.deferred_tool_requests == requests


async def test_a_save_from_a_stale_copy_conflicts() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)
    mine = await store.load(conversation.conversation_id)
    theirs = await store.load(conversation.conversation_id)
    assert mine is not None and theirs is not None
    mine.messages.append(ModelResponse(parts=[TextPart('mine')]))
    theirs.messages.append(ModelResponse(parts=[TextPart('theirs')]))
    await store.save(mine)

    with pytest.raises(ConversationConflict, match='was saved or deleted since it was loaded') as conflict:
        await store.save(theirs)

    assert conflict.value.conversation_id == conversation.conversation_id
    assert await store.load(conversation.conversation_id) == mine
    assert _entry_count(store) == 2


async def test_a_new_conversation_under_a_taken_id_or_one_deleted_since_loading_conflicts() -> None:
    store = InMemoryConversationStore()
    taken = Conversation(conversation_id='taken')
    await store.save(taken)

    with pytest.raises(ConversationConflict):
        await store.save(Conversation(conversation_id='taken'))
    await store.delete('taken')
    with pytest.raises(ConversationConflict):
        await store.save(taken)


async def test_a_record_written_over_a_stale_revision_conflicts() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation()
    await store.save(conversation)
    stale = await store.read_record(conversation.conversation_id)
    assert stale is not None
    await store.save(conversation)

    with pytest.raises(ConversationConflict):
        await store.write_record(stale, expected_revision=stale.revision)


async def test_a_fork_shares_the_history_it_keeps() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    original = (await agent.run('Hello')).conversation
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
    assert record.forked_from == original.conversation_id
    assert record.forked_at == (await store.read_history(record.head_id))[-1].id
    assert (await store.fork(original.conversation_id, at=0)).messages == []

    assert first_only.revision == record.revision
    continued = (await agent.run('Again', conversation=first_only)).conversation
    await store.save(continued)
    assert await store.load(first_only.conversation_id) == continued
    assert await store.load(original.conversation_id) == original
    saved = await store.read_record(first_only.conversation_id)
    assert saved is not None
    assert (saved.forked_from, saved.forked_at) == (record.forked_from, record.forked_at)


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
    assert [record.conversation_id for record in await store.list_conversations(limit=1)] == [second.conversation_id]
    with pytest.raises(UserError, match='`limit` must not be negative'):
        await store.list_conversations(limit=-1)


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


async def test_a_saved_request_a_run_sends_is_not_stored_again() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('queued')])])
    await store.save(conversation)
    loaded = await store.load(conversation.conversation_id)
    assert loaded is not None

    continued = (await Agent(TestModel()).run(conversation=loaded)).conversation
    await store.save(continued)

    # The run fills in the request's `run_id`, but the save appends only what the run added.
    assert continued.messages[0] != conversation.messages[0]
    assert _entry_count(store) == 2
    reloaded = await store.load(conversation.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == [conversation.messages[0], continued.messages[1]]


async def test_a_saved_conversation_carries_its_revision_into_the_next_run() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    conversation = (await agent.run('Hello')).conversation
    await store.save(conversation)

    result = await agent.run('Again', conversation=conversation)
    conversation = result.conversation
    await store.save(conversation)

    record = await store.read_record(conversation.conversation_id)
    assert record is not None
    assert conversation.revision == record.revision
    # Each access builds a new conversation, so changing one doesn't change the next.
    assert result.conversation is not conversation
    assert result.conversation.revision != conversation.revision


async def test_a_whole_fork_keeps_what_a_paused_conversation_is_waiting_on() -> None:
    store = InMemoryConversationStore()
    requests = DeferredToolRequests(metadata={'call': {'reason': 'approval'}})
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])], deferred_tool_requests=requests)
    await store.save(conversation)

    whole = await store.fork(conversation.conversation_id)
    partial = await store.fork(conversation.conversation_id, at=0)

    assert whole.deferred_tool_requests == requests
    loaded = await store.load(whole.conversation_id)
    assert loaded is not None
    assert loaded.deferred_tool_requests == requests
    assert partial.deferred_tool_requests is None


async def test_the_store_keeps_its_own_copy_of_each_record() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(deferred_tool_requests=DeferredToolRequests(metadata={'call': {}}))
    await store.save(conversation, scope='user-1')

    conversation.usage.requests = 99
    assert conversation.deferred_tool_requests is not None
    conversation.deferred_tool_requests.metadata['call']['changed'] = True
    (listed,) = await store.list_conversations()
    listed.usage.requests = 42

    loaded = await store.load(conversation.conversation_id)
    assert loaded is not None
    assert loaded.usage.requests == 0
    assert loaded.deferred_tool_requests == DeferredToolRequests(metadata={'call': {}})


class _RacingStore(InMemoryConversationStore):
    """Runs `race` the first time a history is read, as another writer acting between a store's two reads would."""

    def __init__(self) -> None:
        super().__init__()
        self.race: Callable[[], Awaitable[None]] | None = None

    async def read_history(self, head_id: str | None, *, limit: int | None = None) -> list[ConversationEntry]:
        race, self.race = self.race, None
        if race is not None:
            await race()
        return await super().read_history(head_id, limit=limit)


async def test_a_conversation_deleted_while_it_is_read_loads_as_missing_and_conflicts_on_save() -> None:
    store = _RacingStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)

    store.race = lambda: store.delete(conversation.conversation_id)
    assert await store.load(conversation.conversation_id) is None

    await store.save(other := Conversation(messages=[ModelRequest(parts=[UserPromptPart('two')])]))
    # Without a known `new_message_index`, the save reads the stored history to compare with it.
    other.new_message_index = None
    store.race = lambda: store.delete(other.conversation_id)
    with pytest.raises(ConversationConflict):
        await store.save(other)


async def test_a_conversation_replaced_while_it_is_read_loads_as_replaced() -> None:
    store = _RacingStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)
    rewritten = Conversation(
        messages=[ModelRequest(parts=[UserPromptPart('rewritten')])], conversation_id=conversation.conversation_id
    )

    async def replace_it() -> None:
        # Deleting collects the history being read; a rewrite would keep it as a branch.
        await store.delete(conversation.conversation_id)
        await store.save(rewritten)

    store.race = replace_it
    assert await store.load(conversation.conversation_id) == rewritten


async def test_forking_a_conversation_deleted_while_it_is_read_raises_that_it_is_gone() -> None:
    store = _RacingStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)

    store.race = lambda: store.delete(conversation.conversation_id)
    with pytest.raises(UserError, match='is not in the store'):
        await store.fork(conversation.conversation_id)


async def test_a_fork_of_a_history_released_before_it_is_written_conflicts() -> None:
    class _DeleteBeforeWrite(InMemoryConversationStore):
        source: str | None = None

        async def write_record(
            self,
            record: ConversationRecord,
            *,
            new_entries: Sequence[ConversationEntry] = (),
            expected_revision: str | None,
        ) -> None:
            if self.source is not None:
                source, self.source = self.source, None
                await self.delete(source)
            await super().write_record(record, new_entries=new_entries, expected_revision=expected_revision)

    store = _DeleteBeforeWrite()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)

    store.source = conversation.conversation_id
    with pytest.raises(ConversationConflict):
        await store.fork(conversation.conversation_id, new_conversation_id='fork')
    assert await store.load('fork') is None


async def test_saving_the_same_conversation_twice_at_once_conflicts() -> None:
    class _SaveWhileReading(InMemoryConversationStore):
        race: Callable[[], Awaitable[None]] | None = None

        async def read_record(self, conversation_id: str) -> ConversationRecord | None:
            race, self.race = self.race, None
            if race is not None:
                await race()
            return await super().read_record(conversation_id)

    store = _SaveWhileReading()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)
    conversation.messages.append(ModelResponse(parts=[TextPart('two')]))

    async def save_it_too() -> None:
        await store.save(conversation)

    store.race = save_it_too
    with pytest.raises(ConversationConflict):
        await store.save(conversation)


async def test_a_fork_keeps_an_empty_conversation_id_it_is_given() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation()
    await store.save(conversation)

    fork = await store.fork(conversation.conversation_id, new_conversation_id='')

    assert fork.conversation_id == ''


async def test_a_history_missing_an_entry_of_an_unchanged_record_fails_to_load() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    (entry,) = await store.save(conversation)
    del store._entries[entry.id]  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(KeyError):
        await store.load(conversation.conversation_id)


async def test_a_copy_loaded_before_a_delete_cant_be_saved_over_a_conversation_created_again() -> None:
    store = InMemoryConversationStore()
    original = Conversation(conversation_id='reused', messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(original)
    stale = await store.load('reused')
    assert stale is not None
    await store.delete('reused')
    recreated = Conversation(conversation_id='reused', messages=[ModelRequest(parts=[UserPromptPart('two')])])
    await store.save(recreated)

    stale.messages.append(ModelResponse(parts=[TextPart('stale')]))
    with pytest.raises(ConversationConflict):
        await store.save(stale)
    assert await store.load('reused') == recreated


async def _stored_conversation(store: InMemoryConversationStore, turns: int) -> Conversation:
    agent = Agent(TestModel())
    conversation = (await agent.run('turn 0')).conversation
    for turn in range(1, turns):
        conversation = (await agent.run(f'turn {turn}', conversation=conversation)).conversation
    await store.save(conversation)
    return conversation


async def test_a_run_that_trims_its_history_still_appends_to_the_stored_conversation() -> None:
    """A capability that bounds what the model sees doesn't shrink the stored conversation."""
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=4)
    loaded = await store.load(stored.conversation_id)
    assert loaded is not None
    trimming = Agent(TestModel(), capabilities=[ProcessHistory(lambda messages: messages[-2:])])

    result = await trimming.run('turn 4', conversation=loaded)
    assert len(result.all_messages()) < len(stored.messages)
    await store.save(result.conversation)

    reloaded = await store.load(stored.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == [*stored.messages, *result.new_messages()]


async def test_a_conversation_loaded_as_a_window_appends_to_the_whole_one() -> None:
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=3)

    window = await store.load(stored.conversation_id, limit=2)
    assert window is not None
    assert window.messages == stored.messages[-2:]
    result = await Agent(TestModel()).run('turn 3', conversation=window)
    await store.save(result.conversation)

    reloaded = await store.load(stored.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == [*stored.messages, *result.new_messages()]


async def test_messages_added_to_a_loaded_conversation_are_saved_with_the_next_run() -> None:
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=2)
    loaded = await store.load(stored.conversation_id)
    assert loaded is not None
    turn = ModelRequest(parts=[UserPromptPart('a new turn')])
    loaded.messages.append(turn)

    result = await Agent(TestModel()).run(conversation=loaded)
    conversation = result.conversation
    assert conversation.new_message_index == len(stored.messages)
    await store.save(conversation)

    reloaded = await store.load(stored.conversation_id)
    assert reloaded is not None
    assert reloaded.messages[: len(stored.messages)] == stored.messages
    assert reloaded.messages[len(stored.messages)].parts == turn.parts
    assert len(reloaded.messages) == len(stored.messages) + 2


async def test_messages_added_to_a_loaded_conversation_are_saved_with_the_parts_the_run_adds_to_them() -> None:
    """A part the run adds to an unsaved request, like a recorded settings change, doesn't make the save compare the whole history.

    Here the history is trimmed too, so comparing it would replace the stored conversation with the trimmed one.
    """
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=2)
    loaded = await store.load(stored.conversation_id)
    assert loaded is not None
    loaded.messages.append(ModelRequest(parts=[UserPromptPart('a new turn')]))
    trimming = Agent(TestModel(), capabilities=[ProcessHistory(lambda messages: messages[-1:])])

    result = await trimming.run(conversation=loaded, model_settings={'thinking': 'low'})
    turn = result.all_messages()[0]
    assert isinstance(turn, ModelRequest)
    assert [type(part) for part in turn.parts] == [UserPromptPart, ModelSettingsDeltaPart]
    await store.save(result.conversation)

    reloaded = await store.load(stored.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == [*stored.messages, turn, *result.new_messages()]
    record = await store.read_record(stored.conversation_id)
    assert record is not None and record.branches == []


async def test_unsaved_messages_a_capability_rewrote_fall_back_to_comparing_with_the_stored_history() -> None:
    """Without knowing which stored messages the rewritten history matches, the save branches instead of guessing."""
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=2)
    loaded = await store.load(stored.conversation_id)
    assert loaded is not None
    loaded.messages.append(ModelRequest(parts=[UserPromptPart('dropped by the capability')]))
    dropping = Agent(TestModel(), capabilities=[ProcessHistory(lambda messages: messages[:-1])])

    result = await dropping.run('and this', conversation=loaded)

    assert result.conversation.new_message_index is None


async def test_a_run_result_keeps_where_its_conversation_s_unsaved_messages_start_across_serialization() -> None:
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=2)
    loaded = await store.load(stored.conversation_id)
    assert loaded is not None
    loaded.messages.append(ModelRequest(parts=[UserPromptPart('a new turn')]))
    result = await Agent(TestModel()).run(conversation=loaded)

    adapter: TypeAdapter[AgentRunResult[str]] = TypeAdapter(AgentRunResult[str])
    restored = adapter.validate_json(adapter.dump_json(result))

    assert restored.conversation.new_message_index == result.conversation.new_message_index == len(stored.messages)
    plain = await Agent(TestModel()).run('hi')
    assert 'conversation_unsaved' not in adapter.dump_python(plain)


async def test_branching_a_conversation_continues_from_an_earlier_message_and_keeps_the_rest() -> None:
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=2)
    loaded = await store.load(stored.conversation_id)
    assert loaded is not None

    branched = await store.branch(stored.conversation_id, at=2, expected_revision=loaded.revision)
    assert branched.messages == stored.messages[:2]
    assert _entry_count(store) == 4
    result = await Agent(TestModel()).run('a different second turn', conversation=branched)
    await store.save(result.conversation)

    reloaded = await store.load(stored.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == [*stored.messages[:2], *result.new_messages()]
    assert _entry_count(store) == 6
    with pytest.raises(ConversationConflict):
        await store.branch(stored.conversation_id, at=1, expected_revision=loaded.revision)


async def test_branching_keeps_what_a_paused_conversation_waits_on_only_when_nothing_is_dropped() -> None:
    store = InMemoryConversationStore()
    requests = DeferredToolRequests(metadata={'call': {}})
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])], deferred_tool_requests=requests)
    await store.save(conversation)

    whole = await store.branch(conversation.conversation_id, at=1, expected_revision=conversation.revision)
    assert whole.deferred_tool_requests == requests
    partial = await store.branch(conversation.conversation_id, at=0, expected_revision=whole.revision)
    assert partial.deferred_tool_requests is None
    assert partial.messages == []


async def test_branching_a_missing_conversation_or_past_its_end_raises() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)

    with pytest.raises(UserError, match="Conversation 'unknown' is not in the store"):
        await store.branch('unknown', at=0, expected_revision=None)
    with pytest.raises(UserError, match='`at` must be between 0 and 1'):
        await store.branch(conversation.conversation_id, at=2, expected_revision=conversation.revision)


async def test_branching_a_conversation_deleted_while_it_is_read_raises_that_it_is_gone() -> None:
    store = _RacingStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation)

    store.race = lambda: store.delete(conversation.conversation_id)
    with pytest.raises(UserError, match='is not in the store'):
        await store.branch(conversation.conversation_id, at=0, expected_revision=conversation.revision)


async def test_loading_with_a_negative_limit_raises() -> None:
    with pytest.raises(UserError, match='`limit` must not be negative'):
        await InMemoryConversationStore().load('any', limit=-1)


async def test_a_save_returns_the_entries_it_stored_and_they_load_with_their_ids() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    result = await agent.run('one')
    conversation = result.conversation

    first = await store.save(conversation)
    assert [entry.message for entry in first] == conversation.messages
    result = await agent.run('two', conversation=conversation)
    conversation = result.conversation
    second = await store.save(conversation)
    assert [entry.message for entry in second] == result.new_messages()
    assert second[0].parent_id == first[-1].id
    assert await store.save(conversation) == []

    entries = await store.load_entries(conversation.conversation_id)
    assert entries is not None
    assert [entry.id for entry in entries] == [entry.id for entry in [*first, *second]]
    window = await store.load_entries(conversation.conversation_id, limit=1)
    assert window is not None
    assert [entry.id for entry in window] == [second[-1].id]
    assert await store.load_entries('unknown') is None


@pytest.mark.parametrize('edit', ['delete', 'insert', 'replace', 'replace_earlier', 'truncate'])
async def test_a_loaded_conversation_edited_before_its_stored_end_is_compared_instead(edit: str) -> None:
    """Once the messages before `new_message_index` change, it no longer marks the stored end, so nothing is dropped."""
    store = InMemoryConversationStore()
    first, second = ModelRequest(parts=[UserPromptPart('one')]), ModelResponse(parts=[TextPart('two')])
    await store.save(Conversation(messages=[first, second], conversation_id='c'))
    loaded = await store.load('c')
    assert loaded is not None
    if edit == 'delete':
        del loaded.messages[0]
    elif edit == 'insert':
        loaded.messages.insert(0, ModelRequest(parts=[UserPromptPart('zero')]))
    elif edit == 'replace':
        loaded.messages[1] = ModelResponse(parts=[TextPart('two, edited')])
    elif edit == 'replace_earlier':
        loaded.messages[0] = ModelRequest(parts=[UserPromptPart('one, edited')])
    else:
        del loaded.messages[1]
    if edit != 'truncate':
        loaded.messages.append(ModelRequest(parts=[UserPromptPart('three')]))

    await store.save(loaded)

    reloaded = await store.load('c')
    assert reloaded is not None
    assert reloaded.messages == loaded.messages


async def test_a_run_from_a_loaded_conversation_edited_before_its_stored_end_saves_the_edit() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    saved = (await agent.run('one')).conversation
    await store.save(saved)
    loaded = await store.load(saved.conversation_id)
    assert loaded is not None
    loaded.messages[0] = ModelRequest(parts=[UserPromptPart('one, edited')])

    result = await agent.run('two', conversation=loaded)
    adapter: TypeAdapter[AgentRunResult[str]] = TypeAdapter(AgentRunResult[str])
    await store.save(adapter.validate_json(adapter.dump_json(result)))

    reloaded = await store.load(saved.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == result.all_messages()


async def test_messages_added_to_a_conversation_while_it_is_saved_are_saved_next_time() -> None:
    store = _WriteRacingStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    added = ModelResponse(parts=[TextPart('two')])

    async def add() -> None:
        conversation.messages.append(added)

    store.race = add
    await store.save(conversation)
    assert [entry.message for entry in await store.save(conversation)] == [added]

    reloaded = await store.load(conversation.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == conversation.messages


class _WriteRacingStore(InMemoryConversationStore):
    """Runs `race` once, right after a record is written."""

    def __init__(self) -> None:
        super().__init__()
        self.race: Callable[[], Awaitable[None]] | None = None

    async def write_record(
        self,
        record: ConversationRecord,
        *,
        new_entries: Sequence[ConversationEntry] = (),
        expected_revision: str | None,
    ) -> None:
        await super().write_record(record, new_entries=new_entries, expected_revision=expected_revision)
        race, self.race = self.race, None
        if race is not None:
            await race()


async def test_a_window_carried_into_a_run_with_instructions_appends_to_the_whole_conversation() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel(), instructions='Be helpful')
    first = await agent.run('one')
    second = await agent.run('two', conversation=first.conversation)
    await store.save(second)
    window = await store.load(second.conversation.conversation_id, limit=2)
    assert window is not None
    window.messages.append(ModelRequest(parts=[UserPromptPart('three')]))

    third = await agent.run(conversation=window)
    await store.save(third)

    reloaded = await store.load(window.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == [*second.all_messages(), *third.all_messages()[-2:]]


async def _no_history_read() -> None:
    raise AssertionError('the history was read')  # pragma: no cover


async def test_a_serialized_window_appends_to_the_whole_conversation() -> None:
    store = _RacingStore()
    agent = Agent(TestModel())
    second = await agent.run('two', conversation=(await agent.run('one')).conversation)
    await store.save(second)
    window = await store.load(second.conversation.conversation_id, limit=2)
    assert window is not None
    revived = ConversationTypeAdapter.validate_json(ConversationTypeAdapter.dump_json(window))

    third = await agent.run('three', conversation=revived)
    adapter: TypeAdapter[AgentRunResult[str]] = TypeAdapter(AgentRunResult[str])
    await store.save(adapter_result := adapter.validate_json(adapter.dump_json(third)))
    # Saved, the result's conversation is known to match the store, so continuing it reads no history.
    continued = adapter_result.conversation
    continued.messages.append(ModelRequest(parts=[UserPromptPart('four')]))
    store.race = _no_history_read
    await store.save(continued)
    store.race = None

    reloaded = await store.load(window.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == [*second.all_messages(), *third.new_messages(), continued.messages[-1]]
    record = await store.read_record(window.conversation_id)
    assert record is not None
    assert record.branches == []


@pytest.mark.parametrize('edited', [0, 1])
async def test_an_edited_serialized_window_branches_inside_the_window(edited: int) -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    second = await agent.run('two', conversation=(await agent.run('one')).conversation)
    await store.save(second)
    window = await store.load(second.conversation.conversation_id, limit=2)
    assert window is not None
    revived = ConversationTypeAdapter.validate_json(ConversationTypeAdapter.dump_json(window))
    revived.messages[edited] = ModelResponse(parts=[TextPart('edited')])

    await store.save(revived)

    reloaded = await store.load(window.conversation_id)
    assert reloaded is not None
    assert reloaded.messages == [*second.all_messages()[:2], *revived.messages]


@dataclasses.dataclass
class _Answer:
    text: str


def _round_trip(conversation: Conversation) -> Conversation:
    return ConversationTypeAdapter.validate_json(ConversationTypeAdapter.dump_json(conversation))


async def test_a_conversation_serialized_between_output_tool_turns_appends_each_turn() -> None:
    """A run merges the stored output tool result with its new prompt, so the save compares the stored messages as the run cleaned them."""
    store = InMemoryConversationStore()
    agent = Agent(TestModel(), output_type=_Answer)
    added: list[ModelMessage] = []
    for turn in range(4):
        loaded = await store.load('c') or Conversation(conversation_id='c')
        result = await agent.run(f'turn {turn}', conversation=_round_trip(loaded))
        await store.save(_round_trip(result.conversation))
        added.extend(result.new_messages())

    reloaded = await store.load('c')
    assert reloaded is not None
    assert reloaded.messages == added
    record = await store.read_record('c')
    assert record is not None
    assert record.branches == []


class _HistoryReadsStore(InMemoryConversationStore):
    """Records the `limit` of each history it reads."""

    def __init__(self) -> None:
        super().__init__()
        self.limits: list[int | None] = []

    async def read_history(self, head_id: str | None, *, limit: int | None = None) -> list[ConversationEntry]:
        self.limits.append(limit)
        return await super().read_history(head_id, limit=limit)


async def _stored_output_tool_turns(store: InMemoryConversationStore) -> Conversation:
    """Three output tool turns saved in-process, so the store keeps each turn's messages as the run added them."""
    agent = Agent(TestModel(), output_type=_Answer)
    conversation = Conversation(conversation_id='c')
    for turn in range(3):
        result = await agent.run(f'turn {turn}', conversation=conversation)
        await store.save(result)
        conversation = result.conversation
    stored = await store.load('c')
    assert stored is not None
    return stored


async def test_a_serialized_window_of_output_tool_turns_appends_to_the_whole_conversation() -> None:
    store = _HistoryReadsStore()
    stored = await _stored_output_tool_turns(store)
    window = await store.load('c', limit=5)
    assert window is not None
    result = await Agent(TestModel(), output_type=_Answer).run('turn 3', conversation=_round_trip(window))
    revived = _round_trip(result.conversation)
    # The run merged two of the 5 messages it was given.
    assert revived.new_message_index == 4
    store.limits.clear()

    await store.save(revived)
    reads = list(store.limits)

    reloaded = await store.load('c')
    assert reloaded is not None
    assert reloaded.messages == [*stored.messages, *result.new_messages()]
    record = await store.read_record('c')
    assert record is not None
    assert record.branches == []
    # The save reads stored messages back only until, cleaned the same way, they're as many as the
    # run was sent: the one entry before the 4 it compared, not the whole history.
    assert reads == [4, 1]


def _look_up_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Calls the `look_up` tool, and once it returned, the output tool."""
    if any(isinstance(part, ToolReturnPart) and part.tool_name == 'look_up' for part in messages[-1].parts):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'text': 'found'})])
    return ModelResponse(parts=[ToolCallPart('look_up', {})])


async def test_a_serialized_window_starting_inside_a_tool_call_turn_appends_to_the_whole_conversation() -> None:
    """Reading back two entries can add two cleaned messages: a tool call restores the result the shorter history dropped as orphaned."""
    agent = Agent(FunctionModel(_look_up_then_answer), output_type=_Answer)

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    store = _HistoryReadsStore()
    conversation = Conversation(conversation_id='c')
    for turn in range(4):
        result = await agent.run(f'turn {turn}', conversation=conversation)
        await store.save(result)
        conversation = result.conversation
    stored = await store.load('c')
    assert stored is not None
    window = await store.load('c', limit=9)
    assert window is not None
    result = await agent.run('turn 4', conversation=_round_trip(window))
    revived = _round_trip(result.conversation)
    assert revived.new_message_index == 8
    store.limits.clear()

    await store.save(revived)
    reads = list(store.limits)

    reloaded = await store.load('c')
    assert reloaded is not None
    assert reloaded.messages == [*stored.messages, *result.new_messages()]
    record = await store.read_record('c')
    assert record is not None
    assert record.branches == []
    # The last 8 entries clean to 6 messages and the last 10 to 9, so the history the run was sent
    # starts in between, at the last 9.
    assert reads == [8, 2]


async def test_an_edited_serialized_window_of_requests_that_merge_reads_back_in_doubling_steps() -> None:
    """Requests in a row clean to one, so each entry read back adds no message: the save reads twice as many each time, then branches as before."""
    store = _HistoryReadsStore()
    requests: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(f'request {i}')]) for i in range(16)]
    await store.save(Conversation(messages=requests, conversation_id='c'))
    window = await store.load('c', limit=2)
    assert window is not None
    revived = _round_trip(window)
    revived.messages[1] = ModelRequest(parts=[UserPromptPart('edited')])
    store.limits.clear()

    await store.save(revived)
    reads = list(store.limits)

    reloaded = await store.load('c')
    assert reloaded is not None
    assert reloaded.messages == [*requests[:-1], revived.messages[1]]
    record = await store.read_record('c')
    assert record is not None
    assert len(record.branches) == 1
    assert reads == [2, 1, 2, 4, 8]


class _NoEarlierHistoryStore(InMemoryConversationStore):
    """Once `head_id` is set, returns no entries for a history that ends anywhere else, which `read_history`'s contract rules out."""

    head_id: str | None = None

    async def read_history(self, head_id: str | None, *, limit: int | None = None) -> list[ConversationEntry]:
        if self.head_id is not None and head_id != self.head_id:
            return []
        return await super().read_history(head_id, limit=limit)


async def test_a_store_that_reads_no_earlier_history_branches_as_without_reading_back() -> None:
    store = _NoEarlierHistoryStore()
    stored = await _stored_output_tool_turns(store)
    window = await store.load('c', limit=5)
    assert window is not None
    result = await Agent(TestModel(), output_type=_Answer).run('turn 3', conversation=_round_trip(window))
    record = await store.read_record('c')
    assert record is not None
    store.head_id = record.head_id

    await store.save(_round_trip(result.conversation))

    store.head_id = None
    reloaded = await store.load('c')
    assert reloaded is not None
    assert reloaded.messages == [*stored.messages[:-4], *result.all_messages()]
    record = await store.read_record('c')
    assert record is not None
    assert len(record.branches) == 1


async def test_a_serialized_window_whose_run_closed_out_a_tool_call_appends_to_the_whole_conversation() -> None:
    """A run sends a result for a stored tool call nothing answered, so it was sent more messages than the stored ones it came from."""
    store = InMemoryConversationStore()
    stored: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('turn 0')]),
        ModelResponse(parts=[TextPart('answer 0')]),
        ModelRequest(parts=[UserPromptPart('turn 1')]),
        ModelResponse(parts=[ToolCallPart('look_up', {}, tool_call_id='unanswered')]),
        ModelResponse(parts=[TextPart('answer 1')]),
    ]
    await store.save(Conversation(messages=stored, conversation_id='c'))
    window = await store.load('c', limit=3)
    assert window is not None
    result = await Agent(TestModel()).run('turn 2', conversation=_round_trip(window))
    revived = _round_trip(result.conversation)
    assert revived.new_message_index == 4

    await store.save(revived)

    reloaded = await store.load('c')
    assert reloaded is not None
    assert reloaded.messages == [*stored, *result.new_messages()]
    record = await store.read_record('c')
    assert record is not None
    assert record.branches == []


def _output_tool_turns(turns: int) -> list[ModelMessage]:
    """`turns` output tool turns as an in-process save stores them, each tool result in a request of its own."""
    messages: list[ModelMessage] = []
    for turn in range(turns):
        messages += [
            ModelRequest(parts=[UserPromptPart(f'turn {turn}')]),
            ModelResponse(parts=[ToolCallPart('final_result', {'text': 'a'}, tool_call_id=f'call {turn}')]),
            ModelRequest(
                parts=[ToolReturnPart('final_result', 'Final result processed.', tool_call_id=f'call {turn}')]
            ),
        ]
    return messages


@pytest.mark.parametrize(('edited', 'cleanings'), [(False, 3), (True, 12)])
async def test_a_long_serialized_conversation_is_compared_cleaning_few_of_the_histories_it_could_have_been_sent(
    edited: bool, cleanings: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleaning a history takes time in its length, so the save cleans only a logarithmic number of the stored histories that end at its last message."""
    store = InMemoryConversationStore()
    stored = _output_tool_turns(200)
    await store.save(Conversation(messages=stored, conversation_id='c'))
    loaded = await store.load('c')
    assert loaded is not None
    sent = repair_messages(loaded.messages, repair_last_response=False)
    if edited:
        sent[len(sent) // 2] = ModelResponse(parts=[TextPart('edited')])
    turn = ModelRequest(parts=[UserPromptPart('turn 200')])
    serialized = Conversation(
        messages=[*sent, turn], conversation_id='c', revision=loaded.revision, new_message_index=len(sent)
    )
    repair = Mock(wraps=_messages.repair_messages)
    monkeypatch.setattr(_messages, 'repair_messages', repair)

    await store.save(serialized)

    assert repair.call_count == cleanings
    record = await store.read_record('c')
    assert record is not None
    assert len(record.branches) == edited


async def test_a_serialized_window_whose_conversation_is_deleted_while_more_of_it_is_read_conflicts() -> None:
    store = _RacingStore()
    await _stored_output_tool_turns(store)
    window = await store.load('c', limit=5)
    assert window is not None
    result = await Agent(TestModel(), output_type=_Answer).run('turn 3', conversation=_round_trip(window))

    async def delete_on_the_next_read() -> None:
        store.race = lambda: store.delete('c')

    # The save's first history read is the window it compares; the delete lands on the next one,
    # which reads further back.
    store.race = delete_on_the_next_read
    with pytest.raises(ConversationConflict):
        await store.save(_round_trip(result.conversation))


async def test_a_conversation_copied_after_loading_is_compared_instead() -> None:
    store = InMemoryConversationStore()
    await store.save(Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])], conversation_id='c'))
    loaded = await store.load('c')
    assert loaded is not None
    copied = dataclasses.replace(loaded, messages=[*loaded.messages[1:], ModelRequest(parts=[UserPromptPart('two')])])

    await store.save(copied)

    reloaded = await store.load('c')
    assert reloaded is not None
    assert reloaded.messages == copied.messages


async def test_a_conversation_can_branch_back_to_a_branch_it_left() -> None:
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=2)
    original = await store.load_entries(stored.conversation_id)
    assert original is not None

    branched = await store.branch(stored.conversation_id, at=2, expected_revision=stored.revision)
    edited = (await Agent(TestModel()).run('an edited second turn', conversation=branched)).conversation
    edited_tip = (await store.save(edited))[-1].id
    record = await store.read_record(stored.conversation_id)
    assert record is not None
    assert record.branches == [original[-1].id]

    back = await store.branch(stored.conversation_id, from_entry=original[-1].id, expected_revision=edited.revision)

    assert back.messages == stored.messages
    record = await store.read_record(stored.conversation_id)
    assert record is not None
    assert (record.head_id, record.branches) == (original[-1].id, [edited_tip])


async def test_branching_needs_exactly_one_known_point() -> None:
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=1)

    with pytest.raises(UserError, match='exactly one of `at` and `from_entry`'):
        await store.branch(stored.conversation_id, expected_revision=stored.revision)
    with pytest.raises(UserError, match='exactly one of `at` and `from_entry`'):
        await store.branch(stored.conversation_id, at=0, from_entry='x', expected_revision=stored.revision)
    with pytest.raises(UserError, match="Entry 'unknown' is not in the history"):
        await store.branch(stored.conversation_id, from_entry='unknown', expected_revision=stored.revision)


async def test_a_fork_can_start_from_a_branch_the_conversation_left() -> None:
    store = InMemoryConversationStore()
    stored = await _stored_conversation(store, turns=2)
    original = await store.load_entries(stored.conversation_id)
    assert original is not None
    await store.branch(stored.conversation_id, at=0, expected_revision=stored.revision)

    fork = await store.fork(stored.conversation_id, from_entry=original[1].id)

    assert fork.messages == stored.messages[:2]
    assert fork.usage == RunUsage()


async def test_a_save_that_rewrites_the_history_records_the_branch_it_left() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    (entry,) = await store.save(conversation)
    conversation.messages = [ModelRequest(parts=[UserPromptPart('rewritten')])]
    conversation.new_message_index = None

    await store.save(conversation)

    record = await store.read_record(conversation.conversation_id)
    assert record is not None
    assert record.branches == [entry.id]
    await store.save(other := Conversation(messages=[ModelRequest(parts=[UserPromptPart('other')])]))
    await store.delete(other.conversation_id)
    # Deleting another conversation collects its entries, but keeps the branch this one left.
    assert _entry_count(store) == 2


async def test_a_scope_keeps_a_conversation_to_its_owner() -> None:
    store = InMemoryConversationStore()
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
    await store.save(conversation, scope='alice')
    conversation_id = conversation.conversation_id

    assert await store.load(conversation_id, scope='bob') is None
    assert await store.load_entries(conversation_id, scope='bob') is None
    assert await store.load(conversation_id, scope='alice') is not None
    with pytest.raises(UserError, match='is not in the store'):
        await store.branch(conversation_id, at=0, expected_revision=conversation.revision, scope='bob')
    with pytest.raises(UserError, match='is not in the store'):
        await store.fork(conversation_id, source_scope='bob')
    with pytest.raises(ConversationConflict):
        await store.save(conversation, scope='bob')
    await store.delete(conversation_id, scope='bob')
    assert await store.load(conversation_id) is not None
    await store.delete(conversation_id, scope='alice')
    assert await store.load(conversation_id) is None


async def test_saving_a_result_lets_its_conversation_continue_and_save_again() -> None:
    store = InMemoryConversationStore()
    agent = Agent(TestModel())
    result = await agent.run('one')
    await store.save(result)

    result = await agent.run('two', conversation=result.conversation)
    await store.save(result)
    assert await store.save(result) == []

    loaded = await store.load(result.conversation_id)
    assert loaded is not None
    assert loaded.messages == result.all_messages()
    adapter: TypeAdapter[AgentRunResult[str]] = TypeAdapter(AgentRunResult[str])
    restored = adapter.validate_json(adapter.dump_json(result))
    assert restored.conversation.revision == loaded.revision
    assert restored.conversation.new_message_index == len(loaded.messages)


async def test_a_conversation_replaced_under_another_scope_while_it_is_deleted_is_kept() -> None:
    store = _RecordRacingStore()
    await store.save(Conversation(conversation_id='c'), scope='alice')

    async def replace() -> None:
        await store.delete('c')
        await store.save(Conversation(conversation_id='c'), scope='bob')

    store.race = replace
    await store.delete('c', scope='alice')

    record = await store.read_record('c')
    assert record is not None
    assert record.scope == 'bob'


class _RecordRacingStore(InMemoryConversationStore):
    """Runs `race` once, right after a record is read."""

    def __init__(self) -> None:
        super().__init__()
        self.race: Callable[[], Awaitable[None]] | None = None

    async def read_record(self, conversation_id: str) -> ConversationRecord | None:
        record = await super().read_record(conversation_id)
        race, self.race = self.race, None
        if race is not None:
            await race()
        return record
