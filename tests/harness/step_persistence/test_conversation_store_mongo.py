"""Tests for `MongoConversationStore` against an in-memory `mongomock-motor` client.

The fake's `aggregate` returns its cursor directly, where `pymongo`'s async collection returns it
from a coroutine, so `_mock_client` wraps it to match; that is the only shim. What only a real
server shows is covered by `pydantic_ai_harness/integration_tests/mongodb`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from mongomock_motor import AsyncMongoMockClient
from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import InvalidOperation

from pydantic_ai import Conversation
from pydantic_ai_harness.step_persistence import MongoConversationStore
from tests.harness.step_persistence._conversation_store_contract import Backend, ConversationStoreContract


def _mock_client(monkeypatch: pytest.MonkeyPatch) -> AsyncMongoClient[dict[str, object]]:
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoMockClient()  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
    collection_type: type[AsyncCollection[dict[str, object]]] = type(client['t']['conversation_entries'])
    aggregate = collection_type.aggregate

    async def awaitable_aggregate(
        self: AsyncCollection[dict[str, object]], pipeline: list[dict[str, object]]
    ) -> object:
        return aggregate(self, pipeline)

    monkeypatch.setattr(collection_type, 'aggregate', awaitable_aggregate)
    return client


class TestMongoConversationStore(ConversationStoreContract):
    @pytest.fixture
    async def backend(self, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Backend]:
        client = _mock_client(monkeypatch)
        store = MongoConversationStore(client=client, database='t')
        db = client['t']

        async def count_entries() -> int:
            return await db['conversation_entries'].count_documents({})

        async def count_messages() -> int:
            return await db['conversation_messages'].count_documents({})

        async def delete_entry(entry_id: str) -> None:
            await db['conversation_entries'].delete_one({'_id': entry_id})

        async def delete_messages() -> None:
            await db['conversation_messages'].delete_many({})

        async def stored_message_bytes() -> int:
            return sum([len(str(doc['message'])) async for doc in db['conversation_messages'].find()])

        yield Backend(
            store,
            count_entries,
            count_messages,
            delete_entry,
            delete_messages,
            stored_message_bytes,
            collects_garbage=False,
            atomic=False,
        )
        await store.aclose()


def test_needs_exactly_one_of_client_or_db_url_and_a_database(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match='exactly one'):
        MongoConversationStore(database='t')
    with pytest.raises(ValueError, match='exactly one'):
        MongoConversationStore(client=_mock_client(monkeypatch), db_url='mongodb://x', database='t')
    with pytest.raises(ValueError, match='`database=` is required'):
        MongoConversationStore(client=_mock_client(monkeypatch))


async def test_a_store_closes_only_the_client_it_created() -> None:
    store = MongoConversationStore(db_url='mongodb://127.0.0.1:1', database='t', media_store=None)

    await store.aclose()

    with pytest.raises(InvalidOperation, match='after close'):
        await store.save(Conversation())
