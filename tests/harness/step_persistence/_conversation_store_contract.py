"""What every Harness `ConversationStore` backend must do, run against each by a subclass.

A subclass provides the `backend` fixture: the store under test plus counts of the entries and
message bodies it keeps, and a way to delete one of either to simulate a damaged store.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import anyio
import pytest

from pydantic_ai import (
    Agent,
    BinaryImage,
    Conversation,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.conversation_store import (
    ConversationConflict,
    ConversationEntry,
    ConversationRecord,
    ConversationStore,
)
from pydantic_ai.models.test import TestModel
from tests._timeouts import READINESS_WAIT_TIMEOUT


@dataclass
class Backend:
    store: ConversationStore
    count_entries: Callable[[], Awaitable[int]]
    count_messages: Callable[[], Awaitable[int]]
    delete_entry: Callable[[str], Awaitable[None]]
    delete_messages: Callable[[], Awaitable[None]]
    stored_message_bytes: Callable[[], Awaitable[int]]
    collects_garbage: bool = True
    """Whether deleting a conversation deletes the entries and messages nothing else uses."""
    atomic: bool = True
    """Whether a record write that conflicts leaves none of its entries behind."""


def _prompt(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)])


async def _saved(
    store: ConversationStore, conversation: Conversation, *, scope: str | None = None
) -> ConversationRecord:
    await store.save(conversation, scope=scope)
    record = await store.read_record(conversation.conversation_id)
    assert record is not None
    return record


def _hold_writes_until_both_read(store: ConversationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `write_record` wait for a second writer, so two saves both read the store before either writes."""
    both_read = anyio.Event()
    waiting = 0
    write_record = store.write_record

    async def write_after_both_read(
        record: ConversationRecord,
        *,
        new_entries: Sequence[ConversationEntry] = (),
        expected_revision: str | None,
    ) -> None:
        nonlocal waiting
        waiting += 1
        if waiting == 2:
            both_read.set()
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await both_read.wait()
        await write_record(record, new_entries=new_entries, expected_revision=expected_revision)

    monkeypatch.setattr(store, 'write_record', write_after_both_read)


class ConversationStoreContract:
    @pytest.fixture
    def backend(self) -> Backend:
        raise NotImplementedError

    async def test_a_conversation_round_trips_and_continuing_stores_only_new_messages(self, backend: Backend) -> None:
        store = backend.store
        agent = Agent(TestModel())
        first = (await agent.run('Hello')).conversation
        await store.save(first)
        loaded = await store.load(first.conversation_id)
        assert loaded == first

        second = (await agent.run('Again', conversation=loaded)).conversation
        await store.save(second)

        assert await store.load(first.conversation_id) == second
        assert await backend.count_entries() == len(second.messages)
        assert await store.load('unknown') is None

    async def test_a_message_two_conversations_share_is_stored_once(self, backend: Backend) -> None:
        shared = _prompt('the same')
        await backend.store.save(Conversation(messages=[shared]))
        await backend.store.save(Conversation(messages=[shared]))

        assert (await backend.count_entries(), await backend.count_messages()) == (2, 1)

    async def test_a_record_written_over_a_stale_revision_conflicts(self, backend: Backend) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one')])
        record = await _saved(store, conversation)

        for expected_revision in (None, 'stale'):
            entry = ConversationEntry(parent_id=record.head_id, message=_prompt('two'))
            stale = dataclasses.replace(record, head_id=entry.id, revision='new')
            with pytest.raises(ConversationConflict):
                await store.write_record(stale, new_entries=[entry], expected_revision=expected_revision)

        assert await store.load(conversation.conversation_id) == conversation
        assert await backend.count_entries() == (1 if backend.atomic else 3)

    async def test_a_title_set_since_a_save_read_the_conversation_survives_the_save(self, backend: Backend) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one')])
        await store.save(conversation)

        await store.update(conversation.conversation_id, title='Trip', expected_title=None)
        updated = await store.update(conversation.conversation_id, metadata={'days': 3, 'from': 'Oslo'})
        assert updated == await store.read_record(conversation.conversation_id)
        conversation.messages.append(ModelResponse(parts=[TextPart('two')]))
        saved = await _saved(store, conversation)

        assert (saved.title, saved.metadata) == ('Trip', {'days': 3, 'from': 'Oslo'})
        with pytest.raises(ConversationConflict):
            await store.write_details(conversation.conversation_id, {'title': 'Oslo'}, expected_revision='stale')
        with pytest.raises(ConversationConflict):
            await store.update(conversation.conversation_id, metadata=None, expected_title=None)
        renamed = await store.write_details(
            conversation.conversation_id,
            {'title': 'Oslo'},
            expected_revision=saved.revision,
            expected={'metadata': {'from': 'Oslo', 'days': 3}},
        )
        assert renamed.title == 'Oslo'

    async def test_conversations_list_newest_first_by_scope_and_limit(self, backend: Backend) -> None:
        store = backend.store
        saved_at = datetime(2026, 1, 1, tzinfo=UTC)
        for minute, (conversation_id, scope) in enumerate(
            [('a', 'user-1'), ('b', 'user-2'), ('c', 'user-1'), ('d', 'user-1')]
        ):
            record = ConversationRecord(
                conversation_id=conversation_id,
                head_id=None,
                scope=scope,
                updated_at=saved_at + timedelta(minutes=minute),
            )
            await store.write_record(record, expected_revision=None)

        async def listed(*, scope: str | None = None, limit: int | None = None) -> list[str]:
            return [record.conversation_id for record in await store.list_conversations(scope=scope, limit=limit)]

        assert await listed() == ['d', 'c', 'b', 'a']
        assert await listed(scope='user-1') == ['d', 'c', 'a']
        assert await listed(scope='user-1', limit=2) == ['d', 'c']
        assert await listed(limit=1) == ['d']
        assert await listed(limit=0) == []

    async def test_a_fork_shares_entries_and_deleting_keeps_what_it_uses(self, backend: Backend) -> None:
        store = backend.store
        original = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')])])
        await store.save(original)
        fork = await store.fork(original.conversation_id, at=1)
        assert await backend.count_entries() == 2

        await store.delete(original.conversation_id)

        assert await store.load(original.conversation_id) is None
        assert await store.load(fork.conversation_id) == fork
        if backend.collects_garbage:
            assert (await backend.count_entries(), await backend.count_messages()) == (1, 1)
        await store.delete(fork.conversation_id)
        if backend.collects_garbage:
            assert (await backend.count_entries(), await backend.count_messages()) == (0, 0)
        else:
            assert (await backend.count_entries(), await backend.count_messages()) == (2, 2)

    async def test_deleting_keeps_a_message_another_entry_holds(self, backend: Backend) -> None:
        shared = _prompt('the same')
        first = Conversation(messages=[shared])
        await backend.store.save(first)
        await backend.store.save(Conversation(messages=[shared]))

        await backend.store.delete(first.conversation_id)

        expected = (1, 1) if backend.collects_garbage else (2, 1)
        assert (await backend.count_entries(), await backend.count_messages()) == expected

    async def test_an_entry_reads_back_alone(self, backend: Backend) -> None:
        conversation = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')])])
        record = await _saved(backend.store, conversation)
        assert record.head_id is not None

        entry = await backend.store.read_entry(record.head_id)

        assert entry.message == conversation.messages[1]
        assert entry.parent_id is not None
        assert (await backend.store.read_entry(entry.parent_id)).parent_id is None
        with pytest.raises(KeyError):
            await backend.store.read_entry('unknown')

    async def test_large_parts_are_stored_as_media(self, backend: Backend) -> None:
        image = BinaryImage(data=b'\x89PNG' + b'\0' * 200_000, media_type='image/png')
        conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart(['Look', image])])])

        await backend.store.save(conversation)

        assert await backend.store.load(conversation.conversation_id) == conversation
        assert await backend.stored_message_bytes() < 10_000

    async def test_a_history_with_a_missing_entry_fails_to_load(self, backend: Backend) -> None:
        conversation = Conversation(messages=[_prompt('one'), _prompt('two')])
        record = await _saved(backend.store, conversation)
        assert record.head_id is not None
        root = (await backend.store.read_entry(record.head_id)).parent_id
        assert root is not None

        await backend.delete_entry(root)

        with pytest.raises(LookupError, match='references entries or messages that are missing'):
            await backend.store.load(conversation.conversation_id)
        await backend.delete_entry(record.head_id)
        with pytest.raises(LookupError, match='references entries or messages that are missing'):
            await backend.store.load(conversation.conversation_id)

    async def test_a_history_with_a_missing_message_fails_to_load(self, backend: Backend) -> None:
        conversation = Conversation(messages=[_prompt('one')])
        record = await _saved(backend.store, conversation)
        assert record.head_id is not None

        await backend.delete_messages()

        with pytest.raises(LookupError, match='references entries or messages that are missing'):
            await backend.store.load(conversation.conversation_id)
        with pytest.raises(LookupError):
            await backend.store.read_entry(record.head_id)

    async def test_a_write_continuing_from_a_released_entry_conflicts(self, backend: Backend) -> None:
        if not backend.atomic:
            pytest.skip('A store without transactions never releases entries.')
        entry = ConversationEntry(parent_id='released', message=_prompt('one'))

        with pytest.raises(ConversationConflict):
            await backend.store.write_record(
                ConversationRecord(conversation_id='c', head_id=entry.id), new_entries=[entry], expected_revision=None
            )

    async def test_deleting_a_conversation_saved_since_it_was_read_conflicts(self, backend: Backend) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one')])
        stale = await _saved(store, conversation)
        await store.save(conversation)

        with pytest.raises(ConversationConflict):
            await store.delete_record(conversation.conversation_id, expected_revision=stale.revision)

        assert await store.load(conversation.conversation_id) == conversation

    async def test_a_history_reads_back_its_last_entries(self, backend: Backend) -> None:
        messages = [_prompt('one'), ModelResponse(parts=[TextPart('two')]), _prompt('three')]
        record = await _saved(backend.store, Conversation(messages=messages))

        async def last(limit: int | None) -> list[ModelMessage]:
            return [entry.message for entry in await backend.store.read_history(record.head_id, limit=limit)]

        assert await last(None) == messages
        assert await last(4) == messages
        assert await last(2) == messages[1:]
        assert await last(1) == messages[2:]
        assert await last(0) == []

    async def test_deleting_a_conversation_keeps_a_branch_a_fork_left(self, backend: Backend) -> None:
        store = backend.store
        original = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')])])
        await store.save(original)
        fork = await store.fork(original.conversation_id)
        tip = (await _saved(store, fork)).head_id
        assert tip is not None
        fork = await store.branch(fork.conversation_id, at=1, expected_revision=fork.revision)

        await store.delete(original.conversation_id)

        back = await store.branch(fork.conversation_id, from_entry=tip, expected_revision=fork.revision)
        assert back.messages == original.messages
        assert await backend.count_entries() == 2

    async def test_deleting_a_conversation_deletes_the_branches_it_left(self, backend: Backend) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')])])
        await store.save(conversation)
        await store.branch(conversation.conversation_id, at=1, expected_revision=conversation.revision)

        await store.delete(conversation.conversation_id)

        expected = (0, 0) if backend.collects_garbage else (2, 2)
        assert (await backend.count_entries(), await backend.count_messages()) == expected

    async def test_deleting_the_conversation_a_longer_fork_continues_keeps_the_fork_whole(
        self, backend: Backend
    ) -> None:
        store = backend.store
        original = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')])])
        await store.save(original)
        fork = await store.fork(original.conversation_id)
        fork.messages.extend([_prompt('three'), ModelResponse(parts=[TextPart('four')])])
        await store.save(fork)

        await store.delete(original.conversation_id)

        assert await store.load(original.conversation_id) is None
        assert await store.load(fork.conversation_id) == fork
        assert await backend.count_entries() == 4

    async def test_deleting_a_conversation_keeps_the_prefix_a_fork_branched_off(self, backend: Backend) -> None:
        store = backend.store
        original = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')]), _prompt('three')])
        await store.save(original)
        fork = await store.fork(original.conversation_id, at=2)
        fork.messages.append(_prompt('the fork'))
        await store.save(fork)

        await store.delete(original.conversation_id)

        assert await store.load(fork.conversation_id) == fork
        if backend.collects_garbage:
            assert await backend.count_entries() == 3

    async def test_two_copies_saved_at_once_let_exactly_one_win(
        self, backend: Backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both writers read the stored revision before either writes, as two app workers would."""
        store = backend.store
        original = Conversation(messages=[_prompt('one')])
        await store.save(original)
        copies = [await store.load(original.conversation_id) for _ in range(2)]

        _hold_writes_until_both_read(store, monkeypatch)
        outcomes: dict[int, str] = {}

        async def save(writer: int, copy: Conversation | None) -> None:
            assert copy is not None
            copy.messages.append(ModelResponse(parts=[TextPart(f'writer {writer}')]))
            try:
                await store.save(copy)
            except ConversationConflict:
                outcomes[writer] = 'conflict'
            else:
                outcomes[writer] = 'saved'

        async with anyio.create_task_group() as tg:
            for writer, copy in enumerate(copies):
                tg.start_soon(save, writer, copy)

        assert sorted(outcomes.values()) == ['conflict', 'saved']
        (winner,) = [writer for writer, outcome in outcomes.items() if outcome == 'saved']
        assert await store.load(original.conversation_id) == copies[winner]
        assert await backend.count_entries() == (2 if backend.atomic else 3)

    async def test_two_new_conversations_created_at_once_under_one_id_let_exactly_one_win(
        self, backend: Backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both writers find no record before either inserts, so the backend's insert decides the winner."""
        store = backend.store
        candidates = [Conversation(messages=[_prompt(f'writer {i}')], conversation_id='same') for i in range(2)]
        _hold_writes_until_both_read(store, monkeypatch)
        outcomes: dict[int, str] = {}

        async def save(writer: int) -> None:
            try:
                await store.save(candidates[writer])
            except ConversationConflict:
                outcomes[writer] = 'conflict'
            else:
                outcomes[writer] = 'saved'

        async with anyio.create_task_group() as tg:
            for writer in range(2):
                tg.start_soon(save, writer)

        assert sorted(outcomes.values()) == ['conflict', 'saved']
        (winner,) = [writer for writer, outcome in outcomes.items() if outcome == 'saved']
        assert await store.load('same') == candidates[winner]
