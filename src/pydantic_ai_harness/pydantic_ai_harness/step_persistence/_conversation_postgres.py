"""PostgreSQL-backed `ConversationStore`."""

from __future__ import annotations

from collections.abc import Sequence

import anyio

from pydantic_ai.conversation_store import (
    ConversationConflict,
    ConversationEntry,
    ConversationRecord,
    ConversationStore,
)
from pydantic_ai_harness.media import MediaStore, PostgresConnection, PostgresMediaStore, PostgresPool
from pydantic_ai_harness.step_persistence._conversation_codec import (
    RECORD_ADAPTER,
    keyed_entry_messages,
    load_message,
    missing_entries_error,
)
from pydantic_ai_harness.step_persistence._postgres import (
    _MESSAGES_PER_INSERT,  # pyright: ignore[reportPrivateUsage]
    _TABLE_RE,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_harness.step_persistence._store import (
    _DEFAULT_MEDIA_THRESHOLD_BYTES,  # pyright: ignore[reportPrivateUsage]
    _AutoMedia,  # pyright: ignore[reportPrivateUsage]
)


class PostgresConversationStore(ConversationStore):
    """A [`ConversationStore`][pydantic_ai.conversation_store.ConversationStore] in PostgreSQL.

    Takes a caller-owned asyncpg-compatible pool, which the store never closes, like
    `PostgresStepStore`. `table` is a prefix for the store's tables, created on first use, and
    defaults to the step store's, so the two can share one prefix:

    - `{table}_conv_records(conversation_id PK, revision, head_id, scope, updated_at, record)`
    - `{table}_conv_entries(id PK, parent_id, kind, sha256)`
    - `{table}_conv_messages(sha256 PK, message)`: each message's JSON once, however many entries hold it.

    Binary and text parts at or above `media_threshold_bytes` (default 64 KiB) are externalized to
    `media_store`, by default a `PostgresMediaStore` on the same pool with the table `{table}_media`,
    which the step store shares too: media is keyed by content.

    A save writes its new entries and the record in one transaction. Deleting a conversation, or
    saving it with a rewritten or truncated history, also deletes the entries and messages no other
    conversation's history uses; externalized media stays. Foreign keys keep a concurrent save from pointing at an entry or message a delete is
    removing: one of the two fails instead.
    """

    def __init__(
        self,
        pool: PostgresPool,
        *,
        table: str = 'step_persistence',
        media_store: MediaStore | None | _AutoMedia = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
    ) -> None:
        if not _TABLE_RE.fullmatch(table):
            raise ValueError(f'invalid table name: {table!r}')
        self._pool = pool
        self._table = table
        self._records = f'{table}_conv_records'
        self._entries = f'{table}_conv_entries'
        self._messages = f'{table}_conv_messages'
        if media_store == 'auto':
            media_store = PostgresMediaStore(pool, table=f'{table}_media')
        self._media_store: MediaStore | None = media_store
        self._media_threshold_bytes = media_threshold_bytes
        self._schema_ready = False
        self._schema_lock = anyio.Lock()

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as connection, connection.transaction():
                # `CREATE TABLE IF NOT EXISTS` is not race-free in Postgres: concurrent
                # callers can collide on `pg_type`'s unique index.
                await connection.fetchval('SELECT pg_advisory_xact_lock(hashtext($1))', self._table)
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._messages} (sha256 TEXT PRIMARY KEY, message TEXT NOT NULL)'
                )
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._entries} ('
                    'id TEXT PRIMARY KEY, '
                    f'parent_id TEXT REFERENCES {self._entries} (id), '
                    'kind TEXT NOT NULL, '
                    f'sha256 TEXT NOT NULL REFERENCES {self._messages} (sha256))'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._entries}_parent ON {self._entries} (parent_id)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._entries}_sha256 ON {self._entries} (sha256)'
                )
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._records} ('
                    'conversation_id TEXT PRIMARY KEY, '
                    'revision TEXT NOT NULL, '
                    f'head_id TEXT REFERENCES {self._entries} (id), '
                    'scope TEXT, '
                    'updated_at TIMESTAMPTZ NOT NULL, '
                    'record TEXT NOT NULL)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._records}_head ON {self._records} (head_id)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._records}_scope ON {self._records} (scope, updated_at DESC)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._records}_updated_at ON {self._records} (updated_at DESC)'
                )
            self._schema_ready = True

    async def read_record(self, conversation_id: str) -> ConversationRecord | None:
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            text = await connection.fetchval(
                f'SELECT record FROM {self._records} WHERE conversation_id = $1', conversation_id
            )
        return RECORD_ADAPTER.validate_json(str(text)) if text is not None else None

    async def read_records(self, *, scope: str | None = None, limit: int | None = None) -> Sequence[ConversationRecord]:
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            query = f'SELECT record FROM {self._records}'
            args: list[object] = []
            if scope is not None:
                args.append(scope)
                query += f' WHERE scope = ${len(args)}'
            query += ' ORDER BY updated_at DESC'
            if limit is not None:
                args.append(limit)
                query += f' LIMIT ${len(args)}'
            rows = await connection.fetch(query, *args)
        return [RECORD_ADAPTER.validate_json(str(row[0])) for row in rows]

    async def write_record(
        self,
        record: ConversationRecord,
        *,
        new_entries: Sequence[ConversationEntry] = (),
        expected_revision: str | None,
    ) -> None:
        await self._ensure_schema()
        messages = await keyed_entry_messages(
            new_entries, media_store=self._media_store, threshold_bytes=self._media_threshold_bytes
        )
        async with self._pool.acquire() as connection, connection.transaction():
            await self._insert_messages(connection, dict(messages))
            entries = [
                (entry.id, entry.parent_id, entry.kind, sha256)
                for entry, (sha256, _) in zip(new_entries, messages, strict=True)
            ]
            for start in range(0, len(entries), _MESSAGES_PER_INSERT):
                batch = entries[start : start + _MESSAGES_PER_INSERT]
                values = ', '.join(
                    f'(${4 * i + 1}, ${4 * i + 2}, ${4 * i + 3}, ${4 * i + 4})' for i in range(len(batch))
                )
                await connection.execute(
                    f'INSERT INTO {self._entries} (id, parent_id, kind, sha256) VALUES {values}',
                    *(value for row in batch for value in row),
                )
            text = RECORD_ADAPTER.dump_json(record).decode()
            if expected_revision is None:
                written = await connection.fetchval(
                    f'INSERT INTO {self._records} (conversation_id, revision, head_id, scope, updated_at, record) '
                    'VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (conversation_id) DO NOTHING RETURNING conversation_id',
                    record.conversation_id,
                    record.revision,
                    record.head_id,
                    record.scope,
                    record.updated_at,
                    text,
                )
                previous_head = None
            else:
                # Read before the update: a revision is never reused, so if the update below matches it, this was
                # the head it replaces.
                previous_head = await connection.fetchval(
                    f'SELECT head_id FROM {self._records} WHERE conversation_id = $1 AND revision = $2',
                    record.conversation_id,
                    expected_revision,
                )
                written = await connection.fetchval(
                    f'UPDATE {self._records} SET revision = $2, head_id = $3, scope = $4, updated_at = $5, record = $6 '
                    'WHERE conversation_id = $1 AND revision = $7 RETURNING conversation_id',
                    record.conversation_id,
                    record.revision,
                    record.head_id,
                    record.scope,
                    record.updated_at,
                    text,
                    expected_revision,
                )
            if written is None:
                # Raising rolls back the transaction, so the entries above go with it.
                raise ConversationConflict(record.conversation_id)
            # A rewritten or truncated history leaves its old tail behind.
            if previous_head is not None and previous_head != record.head_id:
                await self._release(connection, str(previous_head))

    async def _insert_messages(self, connection: PostgresConnection, messages: dict[str, str]) -> None:
        """Insert the messages `{table}_messages` lacks, in key order so concurrent saves can't deadlock."""
        rows = sorted(messages.items())
        for start in range(0, len(rows), _MESSAGES_PER_INSERT):
            batch = rows[start : start + _MESSAGES_PER_INSERT]
            values = ', '.join(f'(${2 * i + 1}, ${2 * i + 2})' for i in range(len(batch)))
            await connection.execute(
                f'INSERT INTO {self._messages} (sha256, message) VALUES {values} ON CONFLICT (sha256) DO NOTHING',
                *(value for row in batch for value in row),
            )

    async def delete_record(self, conversation_id: str) -> None:
        await self._ensure_schema()
        async with self._pool.acquire() as connection, connection.transaction():
            entry_id = await connection.fetchval(
                f'DELETE FROM {self._records} WHERE conversation_id = $1 RETURNING head_id', conversation_id
            )
            if entry_id is not None:
                await self._release(connection, str(entry_id))

    async def _release(self, connection: PostgresConnection, entry_id: str | None) -> None:
        """Delete `entry_id` and its ancestors, up to the first one another record's history still reaches.

        That's a record's head, or an entry with another child. A message no remaining entry holds goes too.
        """
        while entry_id is not None:
            deleted = await connection.fetchrow(
                f'DELETE FROM {self._entries} WHERE id = $1 '
                f'AND NOT EXISTS (SELECT 1 FROM {self._records} WHERE head_id = $1) '
                f'AND NOT EXISTS (SELECT 1 FROM {self._entries} WHERE parent_id = $1) '
                'RETURNING parent_id, sha256',
                entry_id,
            )
            if deleted is None:
                break
            parent_id, sha256 = deleted
            entry_id = None if parent_id is None else str(parent_id)
            await connection.execute(
                f'DELETE FROM {self._messages} WHERE sha256 = $1 '
                f'AND NOT EXISTS (SELECT 1 FROM {self._entries} WHERE sha256 = $1)',
                sha256,
            )

    async def read_entry(self, entry_id: str) -> ConversationEntry:
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f'SELECT e.parent_id, m.message FROM {self._entries} e '
                f'JOIN {self._messages} m ON m.sha256 = e.sha256 WHERE e.id = $1',
                entry_id,
            )
        if row is None:
            raise KeyError(entry_id)
        parent_id, message = row
        return ConversationEntry(
            id=entry_id,
            parent_id=None if parent_id is None else str(parent_id),
            message=await load_message(str(message), media_store=self._media_store),
        )

    async def read_history(self, head_id: str | None) -> list[ConversationEntry]:
        """The entries from the first one up to `head_id`, in order, read in one query."""
        if head_id is None:
            return []
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                'WITH RECURSIVE chain (id, parent_id, sha256, depth) AS ('
                f'SELECT id, parent_id, sha256, 0 FROM {self._entries} WHERE id = $1 '
                'UNION ALL '
                'SELECT e.id, e.parent_id, e.sha256, chain.depth + 1 '
                f'FROM {self._entries} e JOIN chain ON e.id = chain.parent_id) '
                'SELECT chain.id, chain.parent_id, m.message '
                f'FROM chain LEFT JOIN {self._messages} m ON m.sha256 = chain.sha256 '
                'ORDER BY chain.depth DESC',
                head_id,
            )
        if not rows or rows[0][1] is not None:
            raise missing_entries_error(head_id)
        entries: list[ConversationEntry] = []
        for entry_id, parent_id, message in rows:
            if message is None:
                raise missing_entries_error(head_id)
            entries.append(
                ConversationEntry(
                    id=str(entry_id),
                    parent_id=None if parent_id is None else str(parent_id),
                    message=await load_message(str(message), media_store=self._media_store),
                )
            )
        return entries
