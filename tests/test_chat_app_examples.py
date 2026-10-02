"""Tests for the executor lifecycle of the chat app example's `Database`.

These are unit tests rather than VCR tests: `Database.connect` only performs local
SQLite operations through a `ThreadPoolExecutor`, so there is no model request to record.
"""

from __future__ import annotations as _annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import get_ident

import pytest

from pydantic_ai import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

from .conftest import try_import

# chat_app builds `Agent('openai:gpt-5.2')` at import time, so importing it requires an API key even
# though these tests only exercise its `Database`; no model request is ever made.
os.environ.setdefault('OPENAI_API_KEY', 'fake-key')

with try_import() as imports_successful:
    from examples.pydantic_ai_examples import chat_app

pytestmark = pytest.mark.skipif(not imports_successful(), reason='extras not installed')


def _track_executors(monkeypatch: pytest.MonkeyPatch) -> list[ThreadPoolExecutor]:
    created_executors: list[ThreadPoolExecutor] = []
    executor_cls = chat_app.ThreadPoolExecutor

    def recording_executor_cls(*, max_workers: int) -> ThreadPoolExecutor:
        executor = executor_cls(max_workers=max_workers)
        created_executors.append(executor)
        return executor

    monkeypatch.setattr(chat_app, 'ThreadPoolExecutor', recording_executor_cls)
    return created_executors


async def test_database_context_exit_shuts_down_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a normal exit, the executor built by `Database.connect` rejects new work."""
    created_executors = _track_executors(monkeypatch)
    async with chat_app.Database.connect(tmp_path / 'messages.sqlite') as database:
        message = ModelRequest(parts=[UserPromptPart(content='Hello')])
        await database.add_messages(ModelMessagesTypeAdapter.dump_json([message]))
        assert await database.get_messages() == [message]

    assert len(created_executors) == 1
    with pytest.raises(RuntimeError):
        created_executors[0].submit(int)


async def test_database_body_exception_shuts_down_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception in the body propagates and still shuts the executor down."""
    created_executors = _track_executors(monkeypatch)
    with pytest.raises(ValueError, match='boom'):
        async with chat_app.Database.connect(tmp_path / 'messages.sqlite') as database:
            message = ModelRequest(parts=[UserPromptPart(content='Hello')])
            await database.add_messages(ModelMessagesTypeAdapter.dump_json([message]))
            assert await database.get_messages() == [message]
            raise ValueError('boom')

    assert len(created_executors) == 1
    with pytest.raises(RuntimeError):
        created_executors[0].submit(int)


async def test_database_shutdown_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The executor shutdown runs away from the event-loop thread."""
    shutdown_thread_ids: list[int] = []

    class TrackingThreadPoolExecutor(ThreadPoolExecutor):
        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            shutdown_thread_ids.append(get_ident())
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(chat_app, 'ThreadPoolExecutor', TrackingThreadPoolExecutor)
    event_loop_thread_id = get_ident()

    async with chat_app.Database.connect(tmp_path / 'messages.sqlite'):
        pass

    assert len(shutdown_thread_ids) == 1
    assert shutdown_thread_ids[0] != event_loop_thread_id


async def test_database_connect_failure_shuts_down_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the SQLite connection fails, the executor built for it is still shut down."""
    created_executors = _track_executors(monkeypatch)
    context_manager = chat_app.Database.connect(
        tmp_path / 'missing-directory' / 'messages.sqlite'
    )
    with pytest.raises(sqlite3.OperationalError, match='unable to open database file'):
        await context_manager.__aenter__()

    assert len(created_executors) == 1
    with pytest.raises(RuntimeError):
        created_executors[0].submit(int)
