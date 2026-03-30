"""Pluggable tiered memory store for cross-run agent conversation persistence.

This module provides:
- ``AbstractMemoryStore`` – the protocol every backend must satisfy
- ``MemoryScope``         – ergonomic two-level key for multi-user/multi-agent systems
- ``InMemoryStore``       – dict-backed store for testing / single-process apps
- ``SQLiteMemoryStore``   – stdlib SQLite store for local persistence, zero extra deps
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Protocol, runtime_checkable

import anyio.to_thread

from .messages import ModelMessage, ModelMessagesTypeAdapter

# ---------------------------------------------------------------------------
# MemoryScope — structured two-level key
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MemoryScope:
    """Structured two-level key for multi-agent / multi-user memory scoping.

    Composes into a single ``session_id`` string internally so every store
    implementation stays simple, but gives callers an ergonomic API instead
    of manually formatting ``"user-123:agent-abc:conv-456"``.

    Example::

        result = await agent.run(
            "What's my name?",
            memory_scope=MemoryScope(user_id="user-123", conversation_id="conv-456"),
        )
    """

    user_id: str
    """Required. Identifies the end-user across all agents and conversations."""

    agent_id: str | None = None
    """Optional. Scopes memory to a specific agent (useful in multi-agent systems)."""

    conversation_id: str | None = None
    """Optional. Scopes memory to a specific conversation thread."""

    def session_id(self) -> str:
        """Return the canonical colon-separated session key.

        Format: ``user_id[:agent_id][:conversation_id]``
        """
        parts = [self.user_id]
        if self.agent_id:
            parts.append(self.agent_id)
        if self.conversation_id:
            parts.append(self.conversation_id)
        return ':'.join(parts)


# ---------------------------------------------------------------------------
# AbstractMemoryStore — the protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AbstractMemoryStore(Protocol):
    """Protocol for pluggable agent memory backends.

    Implementations must be async-safe. All methods receive a ``session_id``
    string (use ``MemoryScope.session_id()`` to produce one from structured keys).

    Two-tier memory model
    ---------------------
    * **Short-term** – ``load_recent(limit=N)`` returns the last N messages
      verbatim. These are injected directly into the agent's message history.
    * **Long-term** – ``load_summary()`` returns a compressed text block of
      older interactions. This is prepended as a synthetic system-level context
      message before the recent history.
    * **Summarization** – ``summarize()`` is called (usually offline or
      periodically) to compress old messages into the summary. Built-in stores
      ship a no-op stub; subclass to plug in an LLM-based compressor.

    Minimal custom implementation::

        class MyPostgresStore:
            async def load_recent(self, session_id, limit=20):
                ...
            async def load_summary(self, session_id):
                ...
            async def save(self, session_id, messages):
                ...
            async def summarize(self, session_id):
                ...
            async def clear(self, session_id):
                ...
    """

    async def load_recent(self, session_id: str, limit: int = 20) -> list[ModelMessage]:
        """Return the most recent ``limit`` messages for this session.

        Args:
            session_id: The session to load history for.
            limit: Maximum number of recent messages to return (default 20).

        Returns:
            List of messages in chronological order (oldest first).
        """
        ...

    async def load_summary(self, session_id: str) -> str | None:
        """Return a compressed text summary of older messages, or ``None``.

        The summary is injected as a system-level context block before the
        recent message history, giving the agent long-term memory without
        burning context window on verbatim old messages.

        Args:
            session_id: The session to load the summary for.

        Returns:
            A plain-text summary string, or ``None`` if no summary exists yet.
        """
        ...

    async def save(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Persist the full updated message list for this session.

        Called automatically by the agent after every successful run.

        Args:
            session_id: The session to save messages for.
            messages: The complete message list (``result.all_messages()``).
        """
        ...

    async def summarize(self, session_id: str) -> None:
        """Compress older messages into the long-term summary.

        This is a stub in built-in stores — subclass and call an LLM to
        produce a real summary. Typically called periodically or when the
        message count exceeds a threshold, not on every run.

        Args:
            session_id: The session to summarize.
        """
        ...

    async def clear(self, session_id: str) -> None:
        """Delete all stored messages and the summary for this session.

        Args:
            session_id: The session to clear.
        """
        ...


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


class InMemoryStore:
    """In-process dict-backed memory store.

    Ideal for testing, development, and single-process applications.
    Data is lost when the process exits.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai.memory import InMemoryStore

        store = InMemoryStore()
        agent = Agent("openai:gpt-4o", memory=store)

        await agent.run("My name is Arjun", session_id="user-1")
        await agent.run("What is my name?", session_id="user-1")
        # → "Your name is Arjun."
    """

    def __init__(self) -> None:
        self._messages: dict[str, list[ModelMessage]] = {}
        self._summaries: dict[str, str] = {}

    async def load_recent(self, session_id: str, limit: int = 20) -> list[ModelMessage]:
        """Return the last ``limit`` messages for this session."""
        return self._messages.get(session_id, [])[-limit:]

    async def load_summary(self, session_id: str) -> str | None:
        """Return the stored summary string, or ``None``."""
        return self._summaries.get(session_id)

    async def save(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Replace the stored message list for this session."""
        self._messages[session_id] = list(messages)

    async def summarize(self, session_id: str) -> None:
        """No-op stub. Subclass ``InMemoryStore`` to add summarization logic."""
        warnings.warn(
            'InMemoryStore.summarize() is a no-op. '
            'Subclass InMemoryStore and override summarize() to compress old messages.',
            stacklevel=2,
        )

    async def clear(self, session_id: str) -> None:
        """Delete all messages and the summary for this session."""
        self._messages.pop(session_id, None)
        self._summaries.pop(session_id, None)

    def set_summary(self, session_id: str, summary: str) -> None:
        """Directly set a summary (useful in tests or manual seeding)."""
        self._summaries[session_id] = summary


# ---------------------------------------------------------------------------
# SQLiteMemoryStore
# ---------------------------------------------------------------------------


class SQLiteMemoryStore:
    """SQLite-backed persistent memory store using only Python's stdlib.

    Zero extra dependencies — uses ``sqlite3`` from the standard library and
    ``anyio.to_thread.run_sync`` to run blocking I/O off the event loop.

    Data survives process restarts. Each ``SQLiteMemoryStore`` instance manages
    its own connection to the given database file.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai.memory import SQLiteMemoryStore

        agent = Agent(
            "openai:gpt-4o",
            memory=SQLiteMemoryStore("agent_memory.db"),
        )

        await agent.run("My name is Arjun", session_id="user-1")

        # Later — even after a restart:
        await agent.run("What is my name?", session_id="user-1")
        # → "Your name is Arjun."
    """

    def __init__(self, db_path: str) -> None:
        """Create a SQLiteMemoryStore backed by the given file path.

        Args:
            db_path: Path to the SQLite database file. Will be created if it
                does not exist. Use ``":memory:"`` for an in-process SQLite DB
                (note: not shared across instances, unlike ``InMemoryStore``).
        """
        self._db_path = db_path
        self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        conn.execute('PRAGMA journal_mode=WAL')  # concurrent reads + writes
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    session_id  TEXT    NOT NULL,
                    idx         INTEGER NOT NULL,
                    message     TEXT    NOT NULL,
                    PRIMARY KEY (session_id, idx)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id  TEXT PRIMARY KEY,
                    summary     TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()
        self._initialized = True

    async def _init(self) -> None:
        if not self._initialized:
            await anyio.to_thread.run_sync(self._ensure_schema)

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    async def load_recent(self, session_id: str, limit: int = 20) -> list[ModelMessage]:
        """Return the last ``limit`` messages for this session from SQLite."""
        await self._init()

        def _read() -> list[ModelMessage]:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT message FROM messages
                    WHERE session_id = ?
                    ORDER BY idx DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            finally:
                conn.close()
            # rows are newest-first; reverse to restore chronological order
            raw = [json.loads(row[0]) for row in reversed(rows)]
            return ModelMessagesTypeAdapter.validate_python(raw)

        return await anyio.to_thread.run_sync(_read)

    async def load_summary(self, session_id: str) -> str | None:
        """Return the stored summary for this session, or ``None``."""
        await self._init()

        def _read() -> str | None:
            conn = self._connect()
            try:
                row = conn.execute(
                    'SELECT summary FROM summaries WHERE session_id = ?',
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
            return row[0] if row else None

        return await anyio.to_thread.run_sync(_read)

    async def save(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Persist the full message list, replacing any existing rows."""
        await self._init()
        serialized = [json.dumps(m) for m in ModelMessagesTypeAdapter.dump_python(messages, mode='json')]

        def _write() -> None:
            conn = self._connect()
            try:
                conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
                conn.executemany(
                    'INSERT INTO messages (session_id, idx, message) VALUES (?, ?, ?)',
                    [(session_id, i, msg) for i, msg in enumerate(serialized)],
                )
                conn.commit()
            finally:
                conn.close()

        await anyio.to_thread.run_sync(_write)

    async def summarize(self, session_id: str) -> None:
        """No-op stub. Subclass ``SQLiteMemoryStore`` to add LLM summarization."""
        warnings.warn(
            'SQLiteMemoryStore.summarize() is a no-op stub. '
            'Subclass SQLiteMemoryStore and override summarize() to compress '
            'old messages into the summaries table using an LLM.',
            stacklevel=2,
        )

    async def set_summary(self, session_id: str, summary: str) -> None:
        """Directly write a summary (useful for testing or manual seeding)."""
        await self._init()

        def _write() -> None:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO summaries (session_id, summary)
                    VALUES (?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET summary = excluded.summary
                    """,
                    (session_id, summary),
                )
                conn.commit()
            finally:
                conn.close()

        await anyio.to_thread.run_sync(_write)

    async def clear(self, session_id: str) -> None:
        """Delete all messages and the summary for this session."""
        await self._init()

        def _delete() -> None:
            conn = self._connect()
            try:
                conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
                conn.execute('DELETE FROM summaries WHERE session_id = ?', (session_id,))
                conn.commit()
            finally:
                conn.close()

        await anyio.to_thread.run_sync(_delete)
