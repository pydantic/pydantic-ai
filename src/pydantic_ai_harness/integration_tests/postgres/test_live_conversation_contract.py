"""`ConversationStoreContract` against a live PostgreSQL server.

The unit suite runs the contract against the SQLite stand-in for the pool, which has one
connection and doesn't enforce foreign keys. This runs the same contract against a real server,
through the helpers in `tests/harness/_postgres_live.py` that `test_live_postgres.py` uses: run it
with `make integration-postgres`.

The contract damages the store to check a load fails cleanly, which the foreign keys would
refuse, so the damage runs with their triggers off. That needs the connecting role to be a
superuser or, on PostgreSQL 15+, one granted `SET` on `session_replication_role`; the `postgres`
role of the documented `docker run` is a superuser.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pydantic_ai_harness.step_persistence import PostgresConversationStore
from tests.harness._postgres_live import count_rows, live_tables
from tests.harness.step_persistence._conversation_store_contract import Backend, ConversationStoreContract


class TestLivePostgresConversationStore(ConversationStoreContract):
    """The contract every Harness `ConversationStore` meets, against a real server."""

    @pytest.fixture
    async def backend(self) -> AsyncIterator[Backend]:
        async with live_tables() as (pool, prefix):
            store = PostgresConversationStore(pool, table=prefix)

            async def damage(sql: str, *args: object) -> None:
                async with pool.acquire() as connection, connection.transaction():
                    await connection.execute('SET LOCAL session_replication_role = replica')
                    await connection.execute(sql, *args)

            async def count_entries() -> int:
                return await count_rows(pool, f'{prefix}_conv_entries')

            async def count_messages() -> int:
                return await count_rows(pool, f'{prefix}_conv_messages')

            async def delete_entry(entry_id: str) -> None:
                await damage(f'DELETE FROM {prefix}_conv_entries WHERE id = $1', entry_id)

            async def delete_messages() -> None:
                await damage(f'DELETE FROM {prefix}_conv_messages')

            async def stored_message_bytes() -> int:
                async with pool.acquire() as connection:
                    value = await connection.fetchval(f'SELECT sum(length(message)) FROM {prefix}_conv_messages')
                return int(value) if isinstance(value, int) else 0

            yield Backend(store, count_entries, count_messages, delete_entry, delete_messages, stored_message_bytes)
