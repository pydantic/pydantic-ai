from __future__ import annotations as _annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import anyio

from . import messages as _messages


class AbstractMemoryStore(Protocol):
    async def load(self, session_id: str) -> list[_messages.ModelMessage]: ...

    async def save(self, session_id: str, messages: Sequence[_messages.ModelMessage]) -> None: ...

    async def clear(self, session_id: str) -> None: ...


@dataclass
class InMemoryStore:
    _store: dict[str, list[_messages.ModelMessage]] = field(default_factory=dict)

    async def load(self, session_id: str) -> list[_messages.ModelMessage]:
        return list(self._store.get(session_id, []))

    async def save(self, session_id: str, messages: Sequence[_messages.ModelMessage]) -> None:
        self._store[session_id] = list(messages)

    async def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


@dataclass
class SQLiteMemoryStore:
    path: str | Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    async def load(self, session_id: str) -> list[_messages.ModelMessage]:
        def _load_sync() -> list[_messages.ModelMessage]:
            conn = sqlite3.connect(self.path)
            try:
                _init_schema(conn)
                cur = conn.execute('SELECT messages FROM pydantic_ai_memory WHERE session_id = ?', (session_id,))
                row = cur.fetchone()
                if row is None:
                    return []
                (messages_json,) = row
                return list(_messages.ModelMessagesTypeAdapter.validate_json(messages_json))
            finally:
                conn.close()

        return await anyio.to_thread.run_sync(_load_sync)

    async def save(self, session_id: str, messages: Sequence[_messages.ModelMessage]) -> None:
        def _save_sync() -> None:
            conn = sqlite3.connect(self.path)
            try:
                _init_schema(conn)
                messages_json = _messages.ModelMessagesTypeAdapter.dump_json(list(messages))
                conn.execute(
                    'INSERT INTO pydantic_ai_memory (session_id, messages) VALUES (?, ?) '
                    'ON CONFLICT(session_id) DO UPDATE SET messages = excluded.messages',
                    (session_id, messages_json),
                )
                conn.commit()
            finally:
                conn.close()

        await anyio.to_thread.run_sync(_save_sync)

    async def clear(self, session_id: str) -> None:
        def _clear_sync() -> None:
            conn = sqlite3.connect(self.path)
            try:
                _init_schema(conn)
                conn.execute('DELETE FROM pydantic_ai_memory WHERE session_id = ?', (session_id,))
                conn.commit()
            finally:
                conn.close()

        await anyio.to_thread.run_sync(_clear_sync)


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        'CREATE TABLE IF NOT EXISTS pydantic_ai_memory ('
        'session_id TEXT PRIMARY KEY, '
        'messages BLOB NOT NULL'
        ')'
    )
