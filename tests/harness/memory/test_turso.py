"""`SqliteMemoryStore` against a real Turso connection.

Turso's Python client provides the SQLite connection extensions the store uses, so it can be
passed as a caller-owned `connection=`. Unlike the step store, nothing here had to change beyond
the annotation; these tests demonstrate the compatible surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import turso

from pydantic_ai_harness.memory import MemoryConflictError, MemoryOperation, SqliteMemoryStore


@pytest.fixture
def turso_store(tmp_path: Path) -> Iterator[SqliteMemoryStore]:
    connection = turso.connect(str(tmp_path / 'memory.db'))
    yield SqliteMemoryStore(connection=connection)
    connection.close()


async def test_compare_and_set_and_search(turso_store: SqliteMemoryStore) -> None:
    created = await turso_store.write('notes/main.md', 'the sky is blue', expected_version=None)
    assert created.version is not None
    assert not created.existed

    stored = await turso_store.read('notes/main.md', max_chars=1_000)
    assert stored is not None
    assert stored.content == 'the sky is blue'
    assert await turso_store.list_paths(limit=10) == ['notes/main.md']

    found = await turso_store.search('', 'sky', limit=5, max_files=10, max_chars=200, max_file_chars=200)
    assert [match.path for match in found.matches] == ['notes/main.md']

    with pytest.raises(MemoryConflictError):
        await turso_store.write('notes/main.md', 'stale', expected_version=None)

    updated = await turso_store.write('notes/main.md', 'the sky is grey', expected_version=created.version)
    assert updated.existed


async def test_operation_receipts_survive_a_replay(turso_store: SqliteMemoryStore) -> None:
    operation = MemoryOperation(id='op-1', fingerprint='f-1')

    first = await turso_store.write('notes/a.md', 'one', expected_version=None, operation=operation)
    replay = await turso_store.write('notes/a.md', 'one', expected_version=None, operation=operation)

    assert not first.replayed
    assert replay.replayed
    assert replay.version == first.version
