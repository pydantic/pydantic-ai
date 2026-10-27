"""What every Harness `ConversationStore` backend must do, run against each by a subclass.

A subclass provides the `backend` fixture: the store under test plus counts of the entries and
message bodies it keeps, and a way to delete one of either to simulate a damaged store.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio
import pytest

from pydantic_ai import (
    Agent,
    BinaryImage,
    Conversation,
    DeferredToolRequests,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.conversation_store import (
    ConversationConflict,
    ConversationEntry,
    ConversationRecord,
    ConversationStore,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel


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


# How long a writer waits at a barrier for the other one, so a race test fails rather than hangs.
_READINESS_WAIT_TIMEOUT = 5.0


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
        with anyio.fail_after(_READINESS_WAIT_TIMEOUT):
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

    async def test_a_rewritten_history_branches_off_the_last_shared_message(self, backend: Backend) -> None:
        messages = [_prompt('one'), ModelResponse(parts=[TextPart('two')]), _prompt('three')]
        conversation = Conversation(messages=messages)
        await backend.store.save(conversation)
        compacted = Conversation(
            messages=[messages[0], ModelResponse(parts=[TextPart('summary')])],
            conversation_id=conversation.conversation_id,
            revision=conversation.revision,
        )

        await backend.store.save(compacted)

        assert await backend.store.load(conversation.conversation_id) == compacted
        # The replaced tail stays stored: a rewrite may only trim what the model sees.
        assert await backend.count_entries() == 4

    async def test_a_save_from_a_stale_copy_conflicts(self, backend: Backend) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one')])
        await store.save(conversation)
        mine = await store.load(conversation.conversation_id)
        theirs = await store.load(conversation.conversation_id)
        assert mine is not None and theirs is not None
        mine.messages.append(ModelResponse(parts=[TextPart('mine')]))
        theirs.messages.append(ModelResponse(parts=[TextPart('theirs')]))
        await store.save(mine)

        with pytest.raises(ConversationConflict):
            await store.save(theirs)

        assert await store.load(conversation.conversation_id) == mine
        assert await backend.count_entries() == 2

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

    async def test_saving_a_new_conversation_under_a_taken_id_conflicts(self, backend: Backend) -> None:
        await backend.store.save(Conversation(messages=[_prompt('one')], conversation_id='taken'))

        with pytest.raises(ConversationConflict):
            await backend.store.save(Conversation(messages=[_prompt('two')], conversation_id='taken'))

    async def test_a_paused_conversation_keeps_its_scope_and_usage(self, backend: Backend) -> None:
        agent = Agent(TestModel())
        conversation = (await agent.run('Hello')).conversation

        first = await _saved(backend.store, conversation, scope='user-1')
        second = await _saved(backend.store, conversation)

        assert (first.scope, second.scope) == ('user-1', 'user-1')
        assert first.revision != second.revision == conversation.revision
        record = await backend.store.read_record(conversation.conversation_id)
        assert record is not None
        assert record.usage == conversation.usage
        assert record.head_id == second.head_id

    async def test_a_conversation_with_no_messages_round_trips(self, backend: Backend) -> None:
        conversation = Conversation()

        record = await _saved(backend.store, conversation)

        assert record.head_id is None
        assert await backend.store.load(conversation.conversation_id) == conversation

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

        async def listed(**kwargs: Any) -> list[str]:
            return [record.conversation_id for record in await store.list_conversations(**kwargs)]

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
        await store.delete(fork.conversation_id)
        if backend.collects_garbage:
            assert (await backend.count_entries(), await backend.count_messages()) == (0, 0)
        else:
            assert (await backend.count_entries(), await backend.count_messages()) == (2, 2)

    async def test_a_truncated_history_keeps_the_branch_it_replaced(self, backend: Backend) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')])])
        await store.save(conversation)
        conversation.messages = conversation.messages[:1]

        await store.save(conversation)

        assert await store.load(conversation.conversation_id) == conversation
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

    async def test_deleting_a_conversation_saved_since_it_was_read_conflicts(self, backend: Backend) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one')])
        stale = await _saved(store, conversation)
        await store.save(conversation)

        with pytest.raises(ConversationConflict):
            await store.delete_record(conversation.conversation_id, expected_revision=stale.revision)
        with pytest.raises(ConversationConflict):
            await store.delete_record('unknown', expected_revision=stale.revision)

        assert await store.load(conversation.conversation_id) == conversation

    async def test_a_history_reads_back_its_last_entries(self, backend: Backend) -> None:
        messages = [_prompt('one'), ModelResponse(parts=[TextPart('two')]), _prompt('three')]
        record = await _saved(backend.store, Conversation(messages=messages))

        async def last(limit: int | None) -> list[Any]:
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

    # Edge cases: how `save` diffs a history, what it carries, and what fork and delete keep.

    async def test_a_truncated_history_extended_again_branches_off_where_it_was_cut(self, backend: Backend) -> None:
        store = backend.store
        one, two, three = _prompt('one'), ModelResponse(parts=[TextPart('two')]), _prompt('three')
        conversation = Conversation(messages=[one, two, three])
        await store.save(conversation)

        conversation.messages = [one]
        assert await store.save(conversation) == []
        truncated = await store.read_record(conversation.conversation_id)
        assert truncated is not None
        assert await backend.count_entries() == 3
        assert [entry.message for entry in await store.read_history(truncated.head_id)] == [one]

        conversation.messages = [one, ModelResponse(parts=[TextPart('two, again')])]
        await store.save(conversation)

        assert await store.load(conversation.conversation_id) == conversation
        assert await backend.count_entries() == 4

    async def test_an_empty_history_saved_over_a_stored_one_loads_empty(self, backend: Backend) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')])])
        await store.save(conversation)

        conversation.messages = []
        assert await store.save(conversation) == []

        record = await store.read_record(conversation.conversation_id)
        assert record is not None and record.head_id is None
        assert await store.load(conversation.conversation_id) == conversation
        assert await backend.count_entries() == 2

    async def test_saving_an_unchanged_conversation_stores_no_entries_but_advances_its_revision(
        self, backend: Backend
    ) -> None:
        store = backend.store
        conversation = Conversation(messages=[_prompt('one')])
        await store.save(conversation)
        first = await store.read_record(conversation.conversation_id)
        assert first is not None

        assert await store.save(conversation) == []

        second = await store.read_record(conversation.conversation_id)
        assert second is not None
        assert await backend.count_entries() == 1
        assert second.head_id == first.head_id
        assert second.revision != first.revision
        assert second.updated_at >= first.updated_at

    async def test_a_message_repeated_in_one_history_gets_an_entry_per_position(self, backend: Backend) -> None:
        store = backend.store
        again = _prompt('again')
        ok = ModelResponse(parts=[TextPart('ok')])
        conversation = Conversation(messages=[again, ok, again, ok])
        await store.save(conversation)

        assert await store.load(conversation.conversation_id) == conversation
        assert (await backend.count_entries(), await backend.count_messages()) == (4, 2)

        # Dropping the second pair is a truncation, not a match against the first.
        conversation.messages = conversation.messages[:2]
        await store.save(conversation)
        assert await store.load(conversation.conversation_id) == conversation
        assert await backend.count_entries() == 4

    async def test_a_message_removed_or_inserted_mid_history_rewrites_from_there(self, backend: Backend) -> None:
        store = backend.store
        messages: list[ModelMessage] = [
            _prompt('one'),
            ModelResponse(parts=[TextPart('two')]),
            _prompt('three'),
            ModelResponse(parts=[TextPart('four')]),
        ]
        conversation = Conversation(messages=list(messages))
        await store.save(conversation)

        conversation.messages = [messages[0], messages[2], messages[3]]
        await store.save(conversation)
        assert await store.load(conversation.conversation_id) == conversation
        # `one` is shared; `three` and `four` get new entries after it, beside the original four.
        assert (await backend.count_entries(), await backend.count_messages()) == (6, 4)

        inserted = ModelResponse(parts=[TextPart('inserted')])
        conversation.messages = [messages[0], inserted, messages[2], messages[3]]
        await store.save(conversation)
        assert await store.load(conversation.conversation_id) == conversation
        assert (await backend.count_entries(), await backend.count_messages()) == (9, 5)

    async def test_a_history_of_tool_calls_and_retries_continued_in_memory_stores_only_new_messages(
        self, backend: Backend
    ) -> None:
        """A conversation kept in memory across runs is saved without storing what the store has again.

        Every message, a retried call's validation errors included, round-trips equal, and a later run's
        new messages are appended from where the stored ones end.
        """

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls = sum(isinstance(part, ToolCallPart) for message in messages for part in message.parts)
            if calls == 0:
                return ModelResponse(parts=[ToolCallPart('lookup', {'x': 'not a number'})])
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('lookup', {'x': 1}), ToolCallPart('flaky', {})])
            if calls == 3:
                return ModelResponse(parts=[ToolCallPart('flaky', {})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(model))
        flaky_calls = 0

        @agent.tool_plain
        def lookup(x: int) -> dict[str, int]:
            return {'x': x}

        @agent.tool_plain
        def flaky() -> str:
            nonlocal flaky_calls
            flaky_calls += 1
            if flaky_calls == 1:
                raise ModelRetry('try again')
            return 'ok'

        store = backend.store
        conversation = (await agent.run('go')).conversation
        returns = [
            part for message in conversation.messages for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert [part.outcome for part in returns] == ['retried', 'success', 'retried', 'success']
        await store.save(conversation)
        assert await store.load(conversation.conversation_id) == conversation

        continued = (await agent.run('again', conversation=conversation)).conversation
        stored = await store.save(continued)

        assert [entry.message for entry in stored] == continued.messages[len(conversation.messages) :]
        assert await store.load(conversation.conversation_id) == continued
        assert await backend.count_entries() == len(continued.messages)

    async def test_a_conversation_paused_on_deferred_tools_resumes_from_the_store(self, backend: Backend) -> None:
        agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

        @agent.tool_plain(requires_approval=True)
        def delete_file(path: str) -> str:
            return f'deleted {path}'

        store = backend.store
        paused = (await agent.run('Delete it')).conversation
        await store.save(paused, scope='user-1')

        loaded = await store.load(paused.conversation_id)
        assert loaded == paused
        assert loaded is not None and loaded.deferred_tool_requests is not None
        (call,) = loaded.deferred_tool_requests.approvals
        results = loaded.deferred_tool_requests.build_results(approvals={call.tool_call_id: True})
        resumed = (await agent.run(conversation=loaded, deferred_tool_results=results)).conversation
        assert resumed.deferred_tool_requests is None
        stored = await store.save(resumed)

        assert [entry.message for entry in stored] == resumed.messages[len(paused.messages) :]
        record = await store.read_record(paused.conversation_id)
        assert record is not None
        assert record.deferred_tool_requests is None
        assert record.scope == 'user-1'
        assert await store.load(paused.conversation_id) == resumed
        assert await backend.count_entries() == len(resumed.messages)

    async def test_usage_accumulates_across_runs_and_saves(self, backend: Backend) -> None:
        store = backend.store
        agent = Agent(TestModel())

        for _ in range(3):
            conversation = await store.load('usage') or Conversation(conversation_id='usage')
            await store.save((await agent.run('Hello', conversation=conversation)).conversation)

        loaded = await store.load('usage')
        assert loaded is not None
        assert loaded.usage.requests == 3

    async def test_a_partial_fork_starts_its_usage_over_and_a_whole_one_copies_it(self, backend: Backend) -> None:
        store = backend.store
        conversation = (await Agent(TestModel()).run('Hello')).conversation
        conversation.usage.details['reasoning_tokens'] = 7
        await store.save(conversation)

        partial = await store.fork(conversation.conversation_id, at=1)
        whole = await store.fork(conversation.conversation_id)
        assert whole.usage == conversation.usage
        whole.usage.details['reasoning_tokens'] = 100
        whole.usage.requests = 100
        await store.save(whole)

        assert (partial.usage.requests, partial.usage.details) == (0, {})
        source = await store.load(conversation.conversation_id)
        assert source is not None
        assert source.usage == conversation.usage
        assert source.usage.details == {'reasoning_tokens': 7}

    async def test_a_fork_of_a_fork_records_the_conversation_and_entry_it_branched_off(self, backend: Backend) -> None:
        store = backend.store
        original = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')]), _prompt('three')])
        await store.save(original)
        original_record = await store.read_record(original.conversation_id)
        assert original_record is not None
        child = await store.fork(original.conversation_id, at=2)
        child.messages.append(_prompt('child'))
        await store.save(child)
        child_record = await store.read_record(child.conversation_id)
        assert child_record is not None
        grandchild = await store.fork(child.conversation_id, at=1)
        empty = await store.fork(original.conversation_id, at=0)

        history = await store.read_history(original_record.head_id)
        # `forked_from` names the conversation and `forked_at` the entry, and both survive later saves of the fork.
        assert (child_record.forked_from, child_record.forked_at) == (original.conversation_id, history[1].id)
        grandchild_record = await store.read_record(grandchild.conversation_id)
        assert grandchild_record is not None
        assert (grandchild_record.forked_from, grandchild_record.forked_at) == (child.conversation_id, history[0].id)
        # A fork that keeps nothing records no entry, but still says what it was copied from.
        empty_record = await store.read_record(empty.conversation_id)
        assert empty_record is not None
        assert (empty_record.forked_from, empty_record.forked_at) == (original.conversation_id, None)
        assert await backend.count_entries() == 4

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

    async def test_deleting_a_conversation_the_store_does_not_have_does_nothing(self, backend: Backend) -> None:
        store = backend.store
        kept = Conversation(messages=[_prompt('one')])
        await store.save(kept)

        await store.delete('unknown')
        await store.delete('unknown')

        assert await store.load(kept.conversation_id) == kept
        assert await backend.count_entries() == 1

    async def test_forking_onto_a_taken_id_or_the_source_itself_conflicts_and_changes_nothing(
        self, backend: Backend
    ) -> None:
        store = backend.store
        source = Conversation(messages=[_prompt('one'), ModelResponse(parts=[TextPart('two')])])
        await store.save(source)
        taken = Conversation(messages=[_prompt('taken')], conversation_id='taken')
        await store.save(taken)

        for new_conversation_id in ('taken', source.conversation_id):
            with pytest.raises(ConversationConflict) as conflict:
                await store.fork(source.conversation_id, at=1, new_conversation_id=new_conversation_id)
            assert conflict.value.conversation_id == new_conversation_id

        assert await store.load(source.conversation_id) == source
        assert await store.load('taken') == taken

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
