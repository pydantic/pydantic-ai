"""Storing conversations as a log of messages, each stored once.

A [`ConversationStore`][pydantic_ai.conversation_store.ConversationStore] keeps a conversation as a
chain of [`ConversationEntry`][pydantic_ai.conversation_store.ConversationEntry]s, each holding one
message and pointing at the entry before it, plus a
[`ConversationRecord`][pydantic_ai.conversation_store.ConversationRecord] that points at the newest
entry and carries what the messages can't: usage, the deferred tool requests a paused run is
waiting on, and the scope the conversation belongs to.

Saving a conversation appends only the messages the store doesn't have yet. When the history was
rewritten instead of extended, by compaction for instance, the new messages branch off the last one
both histories share. A fork points at an existing entry rather than copying the history before it.
"""

from __future__ import annotations as _annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import KW_ONLY
from datetime import datetime
from typing import Literal

from . import messages as _messages, usage as _usage
from ._deferred import DeferredToolRequests
from ._utils import now_utc
from ._uuid import uuid7
from .conversation import Conversation
from .exceptions import UserError

__all__ = (
    'ConversationEntry',
    'ConversationRecord',
    'ConversationConflict',
    'ConversationStore',
    'InMemoryConversationStore',
)


_STORED_TAIL = '_stored_tail'
"""Where `_mark_stored` keeps the message just before the unsaved ones, outside the fields: not compared, serialized or copied."""


def _mark_stored(conversation: Conversation, index: int) -> None:
    """Set `conversation.new_message_index`, remembering the message before it to check that it still marks the boundary."""
    conversation.new_message_index = index
    conversation.__dict__[_STORED_TAIL] = conversation.messages[index - 1] if index else None


def _stored_boundary(conversation: Conversation) -> int | None:
    """`conversation.new_message_index`, if the messages before it are still the ones it was set after."""
    index = conversation.new_message_index
    if index is None or index == 0:
        return index
    if index > len(conversation.messages) or conversation.messages[index - 1] is not conversation.__dict__.get(
        _STORED_TAIL
    ):
        return None
    return index


@dataclasses.dataclass(frozen=True)
class ConversationEntry:
    """One message in a conversation's log, linked to the entry before it."""

    _: KW_ONLY

    id: str = dataclasses.field(default_factory=lambda: str(uuid7()))
    """The entry's identifier, minted when it is stored.

    Not derived from the message: the same message can legitimately appear at two points in a
    history, and the entry before it is part of what makes an entry what it is.
    """

    parent_id: str | None
    """The entry before this one, or `None` for the first entry in a history."""

    message: _messages.ModelMessage
    """The message this entry holds."""

    kind: Literal['message'] = 'message'
    """What the entry holds, so other kinds of entries can join the log without a migration."""


@dataclasses.dataclass
class ConversationRecord:
    """What a store keeps for a conversation besides its log: where its history ends, and what it carries."""

    _: KW_ONLY

    conversation_id: str
    """The conversation this record is for."""

    head_id: str | None
    """The newest entry in the conversation's history, or `None` while it has no messages."""

    usage: _usage.RunUsage = dataclasses.field(default_factory=_usage.RunUsage)
    """Usage accumulated across every run in the conversation."""

    deferred_tool_requests: DeferredToolRequests | None = None
    """The tool calls the conversation is waiting on, if its last run paused for them."""

    scope: str | None = None
    """Whose or where the conversation is, such as a user or a channel, to list conversations by.

    Separate from `conversation_id`, which names the dialogue but says nothing about who it belongs to.
    """

    forked_from: str | None = None
    """The entry a [`fork`][pydantic_ai.conversation_store.ConversationStore.fork] branched this conversation off."""

    revision: str = dataclasses.field(default_factory=lambda: str(uuid7()))
    """A token that changes on every save, so concurrent writers can't overwrite each other unnoticed.

    Opaque: compare it for equality only. A conversation deleted and created again under the same
    ID gets a new one, so a copy loaded before the delete can't be saved over it.
    """

    updated_at: datetime = dataclasses.field(default_factory=now_utc)
    """When the conversation was last saved."""

    format_version: Literal[1] = 1
    """The storage format the record and its entries were written in."""


class ConversationConflict(UserError):
    """Raised when a conversation changed in the store since it was loaded, so saving it would lose those changes.

    It's an expected outcome when more than one writer saves the same conversation: catch it, load the
    conversation again, and retry. It subclasses [`UserError`][pydantic_ai.exceptions.UserError], so a
    broad `except UserError` catches it too.
    """

    conversation_id: str
    """The conversation that changed."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        super().__init__(
            f'Conversation {conversation_id!r} was saved or deleted since it was loaded, '
            'or already exists in the store. Load it again before saving.'
        )


class ConversationStore(ABC):
    """A store for conversations, keeping each one's messages as a log in which each message is stored once.

    [`load`][pydantic_ai.conversation_store.ConversationStore.load] and
    [`save`][pydantic_ai.conversation_store.ConversationStore.save] a
    [`Conversation`][pydantic_ai.conversation.Conversation] around each run:

    ```python {test="skip" lint="skip"}
    conversation = await store.load(conversation_id) or Conversation(conversation_id=conversation_id)
    result = await agent.run(prompt, conversation=conversation)
    await store.save(result.conversation)
    ```

    The stored conversation only ever grows: a save appends the messages a run added, even when a
    capability trimmed or rewrote the history that run was sent. Rewinding it, to regenerate or edit
    an earlier turn, branches off an earlier message and keeps the old branch stored.

    A backend implements the five storage primitives below; the store works out which messages are new,
    where a history branches, and what a fork shares.
    """

    async def load(self, conversation_id: str, *, limit: int | None = None) -> Conversation | None:
        """Load a conversation, or `None` if the store doesn't have it.

        Args:
            conversation_id: The conversation to load.
            limit: Load only its last `limit` messages, for a run that only needs the recent turns.
                Saving the result still appends to the whole stored conversation. The window can
                start in the middle of a turn, say with a tool result whose call it left out; a
                request repairs such a history before it's sent.

        Raises:
            UserError: If `limit` is negative.
        """
        loaded = await self._read(conversation_id, limit=limit)
        if loaded is None:
            return None
        record, history = loaded
        conversation = Conversation(
            messages=[entry.message for entry in history],
            usage=record.usage,
            conversation_id=record.conversation_id,
            deferred_tool_requests=record.deferred_tool_requests,
            revision=record.revision,
        )
        _mark_stored(conversation, len(conversation.messages))
        return conversation

    async def load_entries(self, conversation_id: str, *, limit: int | None = None) -> list[ConversationEntry] | None:
        """Load a conversation's history with each message's entry, or `None` if the store doesn't have it.

        An entry's [`id`][pydantic_ai.conversation_store.ConversationEntry.id] identifies its message
        in the store, so it's what to tie a rating or a billing row to. `save` returns the entries it
        stored.

        Args:
            conversation_id: The conversation to load.
            limit: Load only its last `limit` messages.

        Raises:
            UserError: If `limit` is negative.
        """
        loaded = await self._read(conversation_id, limit=limit)
        return None if loaded is None else loaded[1]

    async def _read(
        self, conversation_id: str, *, limit: int | None
    ) -> tuple[ConversationRecord, list[ConversationEntry]] | None:
        if limit is not None and limit < 0:
            raise UserError('`limit` must not be negative.')
        while True:
            record = await self.read_record(conversation_id)
            if record is None:
                return None
            history = await self._history_of(record, limit=limit)
            if history is not None:
                return record, history

    async def save(self, conversation: Conversation, *, scope: str | None = None) -> list[ConversationEntry]:
        """Save a conversation, storing only the messages the store doesn't have yet.

        When the conversation knows its
        [`new_message_index`][pydantic_ai.conversation.Conversation.new_message_index], as one that
        was loaded, or a run's result from one, does, the messages from that index on are appended
        to the stored conversation. The messages before it aren't compared, so a history a capability
        trimmed or rewrote doesn't change what's stored.

        Otherwise the messages are compared with the stored history: those that extend it are
        appended, and if it was rewritten, the messages from the first difference on branch off the
        last message both share, keeping the old branch stored. Messages are matched with `==`, so a
        change only to a field message equality ignores, like a file's `media_type` or `identifier`,
        isn't stored.

        The save is checked against the conversation's
        [`revision`][pydantic_ai.conversation.Conversation.revision]: it must be the stored one, or
        `None` for a conversation the store doesn't have yet. On success, `revision` is advanced on
        the conversation passed in, so it can be saved again after another run, and so is its
        `new_message_index`.

        Args:
            conversation: The conversation to save, usually a run's
                [`AgentRunResult.conversation`][pydantic_ai.agent.AgentRunResult.conversation].
            scope: Whose or where the conversation is, to list it by. Defaults to the scope it was
                saved with before.

        Returns:
            The entries this save stored, in order, so their
            [`id`][pydantic_ai.conversation_store.ConversationEntry.id]s can be recorded next to the
            messages, say with a billing row or a rating. Empty when there was nothing new.

        Raises:
            ConversationConflict: If the conversation was saved or deleted since it was loaded, or
                one with its `conversation_id` already exists when it wasn't loaded from the store.
        """
        conversation_id = conversation.conversation_id
        # Read once: a concurrent save of the same object advances it while this one awaits the store.
        expected_revision = conversation.revision
        record = await self.read_record(conversation_id)
        if (record.revision if record is not None else None) != expected_revision:
            raise ConversationConflict(conversation_id)
        boundary = _stored_boundary(conversation)
        if record is not None and boundary is not None:
            parent_id = record.head_id
            unsaved = conversation.messages[boundary:]
        else:
            history = await self._history_of(record) if record is not None else []
            if history is None:
                raise ConversationConflict(conversation_id)
            shared = 0
            for entry, message in zip(history, conversation.messages):
                if entry.message != message:
                    break
                shared += 1
            parent_id = history[shared - 1].id if shared else None
            unsaved = conversation.messages[shared:]
        new_entries: list[ConversationEntry] = []
        for message in unsaved:
            entry = ConversationEntry(parent_id=parent_id, message=message)
            new_entries.append(entry)
            parent_id = entry.id
        updated = ConversationRecord(
            conversation_id=conversation_id,
            head_id=parent_id,
            usage=conversation.usage,
            deferred_tool_requests=conversation.deferred_tool_requests,
            scope=scope if scope is not None else record.scope if record is not None else None,
            forked_from=record.forked_from if record is not None else None,
        )
        await self.write_record(updated, new_entries=new_entries, expected_revision=expected_revision)
        conversation.revision = updated.revision
        _mark_stored(conversation, len(conversation.messages))
        return new_entries

    async def rewind(self, conversation_id: str, *, at: int, expected_revision: str | None) -> Conversation:
        """Move a conversation back to an earlier message, to regenerate or edit what came after it.

        The messages after it stay stored as a branch of the log, but the conversation no longer
        loads them. Continue the returned conversation and save it as usual: the new turn branches
        off the kept history.

        Args:
            conversation_id: The conversation to rewind.
            at: How many of its stored messages to keep.
            expected_revision: The [`revision`][pydantic_ai.conversation.Conversation.revision] the
                conversation was loaded with, so a rewind can't undo a turn someone else saved since.

        Raises:
            UserError: If the store doesn't have the conversation, or `at` is out of range.
            ConversationConflict: If the conversation was saved or deleted since it was loaded.
        """
        record = await self.read_record(conversation_id)
        if record is None:
            raise UserError(f'Conversation {conversation_id!r} is not in the store.')
        if record.revision != expected_revision:
            raise ConversationConflict(conversation_id)
        history = await self._history_of(record)
        if history is None:
            raise ConversationConflict(conversation_id)
        if not 0 <= at <= len(history):
            raise UserError(f'`at` must be between 0 and {len(history)}, the number of messages in the conversation.')
        kept = history[:at]
        rewound = dataclasses.replace(
            record,
            head_id=kept[-1].id if kept else None,
            deferred_tool_requests=record.deferred_tool_requests if at == len(history) else None,
            revision=str(uuid7()),
            updated_at=now_utc(),
        )
        await self.write_record(rewound, expected_revision=expected_revision)
        conversation = Conversation(
            messages=[entry.message for entry in kept],
            usage=rewound.usage,
            conversation_id=conversation_id,
            deferred_tool_requests=rewound.deferred_tool_requests,
            revision=rewound.revision,
        )
        _mark_stored(conversation, len(conversation.messages))
        return conversation

    async def fork(
        self,
        conversation_id: str,
        *,
        at: int | None = None,
        new_conversation_id: str | None = None,
        scope: str | None = None,
    ) -> Conversation:
        """Start a new conversation from a stored one's history, sharing its entries rather than copying them.

        Args:
            conversation_id: The conversation to fork.
            at: How many of its messages the fork keeps. Defaults to all of them.
            new_conversation_id: The fork's identifier. Defaults to a new one.
            scope: The fork's scope. Defaults to the forked conversation's.

        Raises:
            UserError: If the store doesn't have the conversation, or `at` is out of range.
        """
        while True:
            record = await self.read_record(conversation_id)
            if record is None:
                raise UserError(f'Conversation {conversation_id!r} is not in the store.')
            history = await self._history_of(record)
            if history is not None:
                break
        keep = len(history) if at is None else at
        if not 0 <= keep <= len(history):
            raise UserError(f'`at` must be between 0 and {len(history)}, the number of messages in the conversation.')
        head_id = history[keep - 1].id if keep else None
        fork = ConversationRecord(
            conversation_id=new_conversation_id if new_conversation_id is not None else str(uuid7()),
            head_id=head_id,
            usage=dataclasses.replace(record.usage) if keep == len(history) else _usage.RunUsage(),
            deferred_tool_requests=record.deferred_tool_requests if keep == len(history) else None,
            scope=scope if scope is not None else record.scope,
            forked_from=head_id,
        )
        await self.write_record(fork, expected_revision=None)
        forked = Conversation(
            messages=[entry.message for entry in history[:keep]],
            usage=fork.usage,
            conversation_id=fork.conversation_id,
            deferred_tool_requests=fork.deferred_tool_requests,
            revision=fork.revision,
        )
        _mark_stored(forked, keep)
        return forked

    async def list_conversations(
        self, *, scope: str | None = None, limit: int | None = None
    ) -> list[ConversationRecord]:
        """The stored conversations, most recently saved first.

        Args:
            scope: Only list the conversations saved with this `scope`.
            limit: List at most this many, the most recently saved.

        Raises:
            UserError: If `limit` is negative.
        """
        if limit is not None and limit < 0:
            raise UserError('`limit` must not be negative.')
        return list(await self.read_records(scope=scope, limit=limit))

    async def delete(self, conversation_id: str) -> None:
        """Delete a conversation, so it no longer loads or lists.

        Whether the entries no other conversation's history uses are deleted with it depends on the
        backend: [`InMemoryConversationStore`][pydantic_ai.conversation_store.InMemoryConversationStore]
        deletes them, and each backend documents what it does. Don't rely on `delete` alone to
        erase a conversation's messages unless your backend says it does.
        """
        await self.delete_record(conversation_id)

    async def _history_of(
        self, record: ConversationRecord, *, limit: int | None = None
    ) -> list[ConversationEntry] | None:
        """`record`'s history, or `None` if another writer replaced or deleted the record while it was read.

        A backend that reads the record and its history separately can find entries gone that a
        concurrent save or delete released. That is only an error when the record is unchanged.
        """
        try:
            return await self.read_history(record.head_id, limit=limit)
        except LookupError:
            current = await self.read_record(record.conversation_id)
            if current is not None and current.revision == record.revision:
                raise
            return None

    async def read_history(self, head_id: str | None, *, limit: int | None = None) -> list[ConversationEntry]:
        """The entries from the first one up to `head_id`, in order, or only the last `limit` of them."""
        entries: list[ConversationEntry] = []
        entry_id = head_id
        while entry_id is not None and (limit is None or len(entries) < limit):
            entry = await self.read_entry(entry_id)
            entries.append(entry)
            entry_id = entry.parent_id
        entries.reverse()
        return entries

    @abstractmethod
    async def read_record(self, conversation_id: str) -> ConversationRecord | None:
        """The record for a conversation, or `None` if the store doesn't have it."""
        raise NotImplementedError

    @abstractmethod
    async def read_records(self, *, scope: str | None = None, limit: int | None = None) -> Sequence[ConversationRecord]:
        """The stored records, or those with the given `scope`, newest `updated_at` first, and at most `limit` of them."""
        raise NotImplementedError

    @abstractmethod
    async def write_record(
        self,
        record: ConversationRecord,
        *,
        new_entries: Sequence[ConversationEntry] = (),
        expected_revision: str | None,
    ) -> None:
        """Store a record and the new entries its history adds, replacing the record with `expected_revision`.

        `new_entries` are in order, each after the one it points at. A backend that can should
        store them and the record together, so a conflict leaves neither behind.

        Raises:
            ConversationConflict: If the stored record's revision isn't `expected_revision`, a record
                exists when `expected_revision` is `None`, or `record.head_id` names an entry that
                neither the store nor `new_entries` has, because a concurrent delete or rewrite of
                the conversation a fork points into released it.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_record(self, conversation_id: str) -> None:
        """Delete a conversation's record, and, where the backend can do so safely, the entries no remaining record's history reaches.

        A backend that keeps those entries, say because it can't tell them from ones a concurrent
        save is about to use, should say so in its documentation.
        """
        raise NotImplementedError

    @abstractmethod
    async def read_entry(self, entry_id: str) -> ConversationEntry:
        """The entry with this identifier.

        Backends that can read a whole history in one query should override
        [`read_history`][pydantic_ai.conversation_store.ConversationStore.read_history] instead of
        relying on one call per entry.
        """
        raise NotImplementedError


class InMemoryConversationStore(ConversationStore):
    """A [`ConversationStore`][pydantic_ai.conversation_store.ConversationStore] that keeps everything in memory, for tests and examples.

    It keeps its own copy of each record and message, as a store that serializes them would, so
    changing the conversation or record it was given or returned doesn't change what was saved.
    """

    def __init__(self) -> None:
        self._records: dict[str, ConversationRecord] = {}
        self._entries: dict[str, ConversationEntry] = {}

    async def read_record(self, conversation_id: str) -> ConversationRecord | None:
        record = self._records.get(conversation_id)
        return deepcopy(record)

    async def read_records(self, *, scope: str | None = None, limit: int | None = None) -> Sequence[ConversationRecord]:
        records = [record for record in self._records.values() if scope is None or record.scope == scope]
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return [deepcopy(record) for record in records[:limit]]

    async def write_record(
        self,
        record: ConversationRecord,
        *,
        new_entries: Sequence[ConversationEntry] = (),
        expected_revision: str | None,
    ) -> None:
        stored = self._records.get(record.conversation_id)
        if (stored.revision if stored is not None else None) != expected_revision:
            raise ConversationConflict(record.conversation_id)
        new_ids = {entry.id for entry in new_entries}
        if record.head_id is not None and record.head_id not in self._entries and record.head_id not in new_ids:
            raise ConversationConflict(record.conversation_id)
        for entry in new_entries:
            self._entries[entry.id] = deepcopy(entry)
        self._records[record.conversation_id] = deepcopy(record)

    async def delete_record(self, conversation_id: str) -> None:
        self._records.pop(conversation_id, None)
        reachable: set[str] = set()
        for record in self._records.values():
            entry_id = record.head_id
            while entry_id is not None and entry_id not in reachable:
                reachable.add(entry_id)
                entry_id = self._entries[entry_id].parent_id
        self._entries = {entry_id: entry for entry_id, entry in self._entries.items() if entry_id in reachable}

    async def read_entry(self, entry_id: str) -> ConversationEntry:
        return deepcopy(self._entries[entry_id])
