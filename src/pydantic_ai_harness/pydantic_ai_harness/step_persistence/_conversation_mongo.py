"""MongoDB-backed `ConversationStore`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NotRequired

from pydantic import TypeAdapter
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError
from typing_extensions import TypedDict

from pydantic_ai.conversation_store import (
    ConversationConflict,
    ConversationEntry,
    ConversationRecord,
    ConversationStore,
)
from pydantic_ai_harness.media import MediaStore
from pydantic_ai_harness.media._mongo import MongoMediaStore
from pydantic_ai_harness.step_persistence._conversation_codec import (
    RECORD_ADAPTER,
    keyed_entry_messages,
    load_message,
    missing_entries_error,
)
from pydantic_ai_harness.step_persistence._mongo import find_message_docs, insert_missing_messages
from pydantic_ai_harness.step_persistence._store import (
    _DEFAULT_MEDIA_THRESHOLD_BYTES,  # pyright: ignore[reportPrivateUsage]
    _AutoMedia,  # pyright: ignore[reportPrivateUsage]
)

_MongoDocument = dict[str, object]


# What a `conversation_entries` document holds; `depth` is added by the `$graphLookup` reading a history.
class _EntryDocument(TypedDict):
    _id: str
    parent_id: str | None
    sha256: str
    depth: NotRequired[int]


_ENTRY_DOCUMENT_ADAPTER: TypeAdapter[_EntryDocument] = TypeAdapter(_EntryDocument)
_ENTRY_DOCUMENTS_ADAPTER: TypeAdapter[list[_EntryDocument]] = TypeAdapter(list[_EntryDocument])


class MongoConversationStore(ConversationStore):
    """A [`ConversationStore`][pydantic_ai.conversation_store.ConversationStore] in MongoDB.

    Pass `client=` (a shared `AsyncMongoClient`) or `db_url=` (the store creates and owns its own
    client; call `aclose()` to release it), plus `database=`, as for
    `MongoStepStore`. Collections:

    - `conversation_records`: `_id` is the conversation id.
    - `conversation_entries`: `_id` is the entry id.
    - `conversation_messages`: `_id` is the SHA-256 of the message's JSON, each stored once.

    Binary and text parts at or above `media_threshold_bytes` (default 64 KiB) are externalized to
    `media_store`, by default a `MongoMediaStore` on the same client and database.

    The store uses no transactions, so a save writes its messages and entries before the record,
    and a save that conflicts leaves its entries unreferenced. Without a transaction, a delete can't
    tell an entry no history uses from one a concurrent save or fork is about to point at, so
    deleting a conversation deletes only its record: its entries and messages stay.
    """

    def __init__(
        self,
        *,
        client: AsyncMongoClient[_MongoDocument] | None = None,
        db_url: str | None = None,
        database: str | None = None,
        media_store: MediaStore | None | _AutoMedia = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
    ) -> None:
        if (client is None) == (db_url is None):
            raise ValueError('provide exactly one of `client=` or `db_url=`')
        if database is None:
            raise ValueError('`database=` is required')
        self._own_client = client is None
        self._client = client if client is not None else AsyncMongoClient[_MongoDocument](db_url)
        self._db: AsyncDatabase[_MongoDocument] = self._client[database]
        if media_store == 'auto':
            media_store = MongoMediaStore(client=self._client, database=database)
        self._media_store: MediaStore | None = media_store
        self._media_threshold_bytes = media_threshold_bytes
        self._indexes_ready = False

    async def aclose(self) -> None:
        """Close the client if this store created it; no-op for a shared client."""
        if self._own_client:
            await self._client.close()

    async def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        await self._db['conversation_records'].create_index([('scope', ASCENDING), ('updated_at', DESCENDING)])
        await self._db['conversation_records'].create_index([('updated_at', DESCENDING)])
        self._indexes_ready = True

    async def read_record(self, conversation_id: str) -> ConversationRecord | None:
        doc = await self._db['conversation_records'].find_one({'_id': conversation_id})
        return RECORD_ADAPTER.validate_json(str(doc['record'])) if doc is not None else None

    async def read_records(self, *, scope: str | None = None, limit: int | None = None) -> Sequence[ConversationRecord]:
        if limit == 0:
            # MongoDB reads a limit of 0 as no limit.
            return []
        query: _MongoDocument = {} if scope is None else {'scope': scope}
        cursor = self._db['conversation_records'].find(query).sort('updated_at', DESCENDING).limit(limit or 0)
        return [RECORD_ADAPTER.validate_json(str(doc['record'])) async for doc in cursor]

    async def write_record(
        self,
        record: ConversationRecord,
        *,
        new_entries: Sequence[ConversationEntry] = (),
        expected_revision: str | None,
    ) -> None:
        await self._ensure_indexes()
        messages = await keyed_entry_messages(
            new_entries, media_store=self._media_store, threshold_bytes=self._media_threshold_bytes
        )
        await insert_missing_messages(self._db['conversation_messages'], dict(messages))
        if new_entries:
            await self._db['conversation_entries'].insert_many(
                [
                    {'_id': entry.id, 'parent_id': entry.parent_id, 'kind': entry.kind, 'sha256': sha256}
                    for entry, (sha256, _) in zip(new_entries, messages, strict=True)
                ]
            )
        doc: _MongoDocument = {
            '_id': record.conversation_id,
            'revision': record.revision,
            'head_id': record.head_id,
            'scope': record.scope,
            'updated_at': record.updated_at,
            'record': RECORD_ADAPTER.dump_json(record).decode(),
        }
        records = self._db['conversation_records']
        if expected_revision is None:
            try:
                await records.insert_one(doc)
            except DuplicateKeyError:
                written = False
            else:
                written = True
        else:
            written = (
                await records.find_one_and_replace({'_id': record.conversation_id, 'revision': expected_revision}, doc)
                is not None
            )
        if not written:
            raise ConversationConflict(record.conversation_id)

    async def delete_record(self, conversation_id: str, *, expected_revision: str) -> None:
        deleted = await self._db['conversation_records'].delete_one(
            {'_id': conversation_id, 'revision': expected_revision}
        )
        if deleted.deleted_count == 0:
            raise ConversationConflict(conversation_id)

    async def read_entry(self, entry_id: str) -> ConversationEntry:
        doc = await self._db['conversation_entries'].find_one({'_id': entry_id})
        if doc is None:
            raise KeyError(entry_id)
        message = await self._db['conversation_messages'].find_one({'_id': doc['sha256']})
        if message is None:
            raise missing_entries_error(entry_id)
        parent_id = doc.get('parent_id')
        return ConversationEntry(
            id=entry_id,
            parent_id=None if parent_id is None else str(parent_id),
            message=await load_message(str(message['message']), media_store=self._media_store),
        )

    async def read_history(self, head_id: str | None, *, limit: int | None = None) -> list[ConversationEntry]:
        """The entries from the first one up to `head_id`, in order, or only the last `limit` of them.

        Read read with one `$graphLookup`.
        """
        if head_id is None or limit == 0:
            return []
        lookup: _MongoDocument = {
            'from': 'conversation_entries',
            'startWith': '$parent_id',
            'connectFromField': 'parent_id',
            'connectToField': '_id',
            'as': 'ancestors',
            'depthField': 'depth',
        }
        if limit is not None:
            # Depth 0 is the head's parent, so `limit - 2` keeps `limit` entries with the head.
            lookup['maxDepth'] = limit - 2
        pipeline: list[_MongoDocument] = [{'$match': {'_id': head_id}}]
        if limit != 1:
            pipeline.append({'$graphLookup': lookup})
        chain: list[_EntryDocument] = []
        async for doc in await self._db['conversation_entries'].aggregate(pipeline):
            ancestors = _ENTRY_DOCUMENTS_ADAPTER.validate_python(doc.pop('ancestors', []))
            chain = [_ENTRY_DOCUMENT_ADAPTER.validate_python(doc), *sorted(ancestors, key=lambda a: a.get('depth', 0))]
        chain.reverse()
        if not chain or (chain[0]['parent_id'] is not None and (limit is None or len(chain) < limit)):
            raise missing_entries_error(head_id)
        stored = {
            str(message['_id']): str(message['message'])
            async for message in find_message_docs(self._db['conversation_messages'], {doc['sha256'] for doc in chain})
        }
        entries: list[ConversationEntry] = []
        for doc in chain:
            text = stored.get(doc['sha256'])
            if text is None:
                raise missing_entries_error(head_id)
            entries.append(
                ConversationEntry(
                    id=doc['_id'],
                    parent_id=doc['parent_id'],
                    message=await load_message(text, media_store=self._media_store),
                )
            )
        return entries
