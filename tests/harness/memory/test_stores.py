"""Contract and backend tests for memory stores."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Protocol

import anyio
import pytest

from pydantic_ai_harness.memory import (
    FileStore,
    InMemoryStore,
    MemoryConflictError,
    MemoryOperation,
    MemoryOperationConflictError,
    MemoryStore,
    PostgresConnection,
    PostgresMemoryStore,
    PostgresPool,
    SearchableMemoryStore,
    SqliteMemoryStore,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class Store(MemoryStore, SearchableMemoryStore, Protocol):
    """Combined contract implemented by the bundled stores."""


def _local_stores(tmp_path: Path) -> list[Store]:
    return [
        InMemoryStore(),
        FileStore(tmp_path / 'files'),
        SqliteMemoryStore(database=tmp_path / 'memory.sqlite3'),
    ]


@pytest.mark.parametrize('index', range(3))
async def test_local_store_compare_and_set_contract(tmp_path: Path, index: int) -> None:
    store = _local_stores(tmp_path)[index]

    created = await store.write('notes/main.md', 'one', expected_version=None)
    assert created.version is not None
    assert not created.replayed
    assert not created.existed
    file = await store.read('notes/main.md', max_chars=1_000)
    assert file is not None
    assert file.content == 'one'
    assert file.version == created.version
    assert file.operation_id is None

    updated = await store.write('notes/main.md', 'two', expected_version=file.version)
    assert updated.version is not None
    assert updated.version != created.version
    assert updated.existed
    with pytest.raises(MemoryConflictError):
        await store.write('notes/main.md', 'stale', expected_version=file.version)
    with pytest.raises(MemoryConflictError):
        await store.delete('notes/main.md', expected_version=file.version)

    deleted = await store.delete('notes/main.md', expected_version=updated.version)
    assert deleted.version is None
    assert deleted.existed
    assert await store.read('notes/main.md', max_chars=1_000) is None


@pytest.mark.parametrize('index', range(3))
async def test_local_store_versions_do_not_repeat_after_delete_and_recreate(tmp_path: Path, index: int) -> None:
    store = _local_stores(tmp_path)[index]
    first = await store.write('main.md', 'same', expected_version=None)
    await store.delete('main.md', expected_version=first.version)
    recreated = await store.write('main.md', 'same', expected_version=None)

    assert recreated.version != first.version
    with pytest.raises(MemoryConflictError):
        await store.write('main.md', 'stale', expected_version=first.version)
    with pytest.raises(MemoryConflictError):
        await store.delete('main.md', expected_version=first.version)


@pytest.mark.parametrize('index', range(3))
async def test_local_store_read_and_listing_bounds(tmp_path: Path, index: int) -> None:
    store = _local_stores(tmp_path)[index]
    created = await store.write('a.md', '0123456789', expected_version=None)
    await store.write('b.md', 'b', expected_version=None)
    await store.write('c.md', 'c', expected_version=None)

    bounded = await store.read('a.md', max_chars=4)
    assert bounded is not None
    assert bounded.content == '0123'
    assert bounded.version == created.version
    assert bounded.truncated
    complete = await store.read('a.md', max_chars=20)
    assert complete is not None
    assert complete.version == created.version
    assert not complete.truncated
    assert await store.list_paths(limit=2) == ['a.md', 'b.md']


@pytest.mark.parametrize('index', range(3))
async def test_local_store_operation_receipts(tmp_path: Path, index: int) -> None:
    store = _local_stores(tmp_path)[index]
    operation = MemoryOperation(id='run-1:call-1', fingerprint='write:notes/main.md:one')

    first = await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
    replay = await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
    assert replay.version == first.version
    assert replay.replayed
    assert not replay.existed
    assert await store.get_operation(operation) == replay
    file = await store.read('notes/main.md', max_chars=1_000)
    assert file is not None
    assert file.operation_id == operation.id

    with pytest.raises(MemoryOperationConflictError):
        await store.get_operation(MemoryOperation(id=operation.id, fingerprint='different'))

    delete_operation = MemoryOperation(id='run-1:call-2', fingerprint='delete:missing.md')
    deleted = await store.delete('missing.md', expected_version=None, operation=delete_operation)
    assert not deleted.existed
    assert not deleted.replayed
    assert (await store.delete('missing.md', expected_version=None, operation=delete_operation)).replayed


@pytest.mark.parametrize('index', range(3))
async def test_local_store_scoped_bounded_search(tmp_path: Path, index: int) -> None:
    store = _local_stores(tmp_path)[index]
    for path, content in (
        ('tenant-a/main/alpha.md', 'alpha alpha'),
        ('tenant-a/main/beta.md', 'alpha'),
        ('tenant-a/main/other.md', 'unrelated'),
        ('tenant-b/main/private.md', 'alpha alpha alpha'),
    ):
        await store.write(path, content, expected_version=None)

    result = await store.search('tenant-a/main/', 'alpha', limit=10, max_files=10, max_chars=80, max_file_chars=1_000)
    assert [match.path for match in result.matches] == [
        'tenant-a/main/alpha.md',
        'tenant-a/main/beta.md',
    ]
    assert result.scanned == 3
    assert not result.truncated
    assert sum(len(match.path) + len(match.snippet) for match in result.matches) <= 80

    bounded = await store.search('tenant-a/main/', 'alpha', limit=10, max_files=1, max_chars=80, max_file_chars=1_000)
    assert bounded.scanned == 1
    assert bounded.truncated
    tiny = await store.search('tenant-a/main/', 'alpha', limit=10, max_files=10, max_chars=1, max_file_chars=1_000)
    assert tiny.matches == []
    assert tiny.truncated

    for query, limit, max_files, max_chars in (
        ('', 10, 10, 80),
        ('alpha', 0, 10, 80),
        ('alpha', 10, 0, 80),
        ('alpha', 10, 10, 0),
    ):
        empty = await store.search(
            '', query, limit=limit, max_files=max_files, max_chars=max_chars, max_file_chars=1_000
        )
        assert empty.matches == []
        assert empty.scanned == 0
        assert not empty.truncated


@pytest.mark.parametrize('index', range(3))
async def test_local_store_search_snippets_cover_tiny_and_offset_windows(tmp_path: Path, index: int) -> None:
    store = _local_stores(tmp_path)[index]
    await store.write('a', '012345alpha-tail', expected_version=None)

    tiny = await store.search('', 'alpha', limit=1, max_files=1, max_chars=3, max_file_chars=1_000)
    assert len(tiny.matches) == 1
    assert len(tiny.matches[0].snippet) == 2
    offset = await store.search('', 'alpha', limit=1, max_files=1, max_chars=12, max_file_chars=1_000)
    assert len(offset.matches) == 1
    assert offset.matches[0].snippet.startswith('...')


@pytest.mark.parametrize('index', range(3))
async def test_local_store_search_bounds_each_file_and_ignores_namespace_prefix(tmp_path: Path, index: int) -> None:
    store = _local_stores(tmp_path)[index]
    namespace = 'n' * 180
    prefix = f'{namespace}/main/'
    await store.write(f'{prefix}note.md', 'prefix TARGET', expected_version=None)

    bounded = await store.search(prefix, 'target', limit=10, max_files=10, max_chars=100, max_file_chars=6)
    assert bounded.matches == []
    namespace_result = await store.search(prefix, namespace, limit=10, max_files=10, max_chars=100, max_file_chars=100)
    assert namespace_result.matches == []
    visible = await store.search(prefix, 'target', limit=10, max_files=10, max_chars=20, max_file_chars=100)
    assert [match.path for match in visible.matches] == [f'{prefix}note.md']


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('path', ('../escape.md', '/absolute.md', 'a//b.md', 'a/../../b.md', 'a b.md'))
async def test_local_stores_reject_unsafe_paths(tmp_path: Path, index: int, path: str) -> None:
    store = _local_stores(tmp_path)[index]
    with pytest.raises(ValueError):
        await store.read(path, max_chars=1_000)


def test_bundled_stores_implement_public_protocols(tmp_path: Path) -> None:
    for store in _local_stores(tmp_path):
        assert isinstance(store, MemoryStore)
        assert isinstance(store, SearchableMemoryStore)


@pytest.mark.parametrize('index', range(3))
async def test_local_stores_reject_non_positive_read_and_listing_bounds(tmp_path: Path, index: int) -> None:
    store = _local_stores(tmp_path)[index]
    with pytest.raises(ValueError, match='max_chars'):
        await store.read('main.md', max_chars=0)
    with pytest.raises(ValueError, match='limit'):
        await store.list_paths(limit=0)


async def test_in_memory_store_copies_and_indexes_initial_files() -> None:
    initial = {'a.md': 'a', 'b.md': 'b'}
    store = InMemoryStore(files=initial)
    initial['c.md'] = 'c'

    assert (await store.read('a.md', max_chars=10)) is not None
    assert await store.list_paths('a', limit=10) == ['a.md']
    assert await store.read('c.md', max_chars=10) is None
    assert await store.list_paths(limit=10) == ['a.md', 'b.md']


def test_in_memory_store_files_view_is_read_only() -> None:
    store = InMemoryStore(files={'a.md': 'a'})

    with pytest.raises(TypeError):
        exec("files['b.md'] = 'b'", {}, {'files': store.files})
    assert dict(store.files) == {'a.md': 'a'}


async def test_file_store_keeps_markdown_plain_and_hides_journal(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    await store.write('notes/main.md', '# Memory', expected_version=None)

    assert (tmp_path / 'notes/main.md').read_text() == '# Memory'
    assert (tmp_path / '.memory-store.sqlite3').is_file()
    assert await store.list_paths(limit=100) == ['notes/main.md']
    with pytest.raises(ValueError, match='reserved'):
        await store.read('.memory-store.sqlite3', max_chars=1_000)


async def test_file_store_detects_external_edits(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('main.md', 'original', expected_version=None)
    assert created.version is not None

    (tmp_path / 'main.md').write_text('external')
    changed = await store.read('main.md', max_chars=1_000)
    assert changed is not None
    assert changed.version != created.version
    with pytest.raises(MemoryConflictError):
        await store.write('main.md', 'replacement', expected_version=created.version)


async def test_file_store_bounds_multi_megabyte_external_reads(tmp_path: Path) -> None:
    content = 'x' * (3 * 1024 * 1024)
    (tmp_path / 'large.md').write_text(content)

    file = await FileStore(tmp_path).read('large.md', max_chars=1_024)
    assert file is not None
    assert file.content == content[:1_024]
    assert file.truncated


@pytest.mark.parametrize('iteration', range(5))
async def test_file_store_migrates_legacy_journal_concurrently(tmp_path: Path, iteration: int) -> None:
    root = tmp_path / f'legacy-files-{iteration}'
    root.mkdir()
    (root / 'main.md').write_text('content')
    connection = sqlite3.connect(root / '.memory-store.sqlite3')
    try:
        connection.execute('CREATE TABLE file_state (path TEXT PRIMARY KEY, last_operation_id TEXT)')
        connection.execute(
            'CREATE TABLE memory_operations ('
            'id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL, kind TEXT NOT NULL, '
            'path TEXT NOT NULL, expected_version TEXT, new_content TEXT, result_version TEXT, existed INTEGER NOT NULL)'
        )
        connection.commit()
    finally:
        connection.close()
    stores = [FileStore(root), FileStore(root)]
    start = anyio.Event()

    async def read(store: FileStore) -> None:
        await start.wait()
        assert await store.read('main.md', max_chars=100) is not None

    async with anyio.create_task_group() as task_group:
        for store in stores:
            task_group.start_soon(read, store)
        start.set()

    connection = sqlite3.connect(root / '.memory-store.sqlite3')
    try:
        columns = {str(row[1]) for row in connection.execute('PRAGMA table_info(file_state)').fetchall()}
    finally:
        connection.close()
    assert {'version', 'fingerprint'} <= columns


async def test_file_store_rolls_back_failed_journal_migration(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute("CREATE VIEW file_state AS SELECT 'main.md' AS path, NULL AS last_operation_id")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.OperationalError):
        await FileStore(tmp_path).read('main.md', max_chars=100)


async def test_file_store_serializes_cas_across_instances(tmp_path: Path) -> None:
    first = FileStore(tmp_path)
    second = FileStore(tmp_path)
    created = await first.write('main.md', 'initial', expected_version=None)
    outcomes: list[str] = []

    async def update(store: FileStore, content: str) -> None:
        try:
            await store.write('main.md', content, expected_version=created.version)
        except MemoryConflictError:
            outcomes.append('conflict')
        else:
            outcomes.append('written')

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(update, first, 'first')
        task_group.start_soon(update, second, 'second')

    assert sorted(outcomes) == ['conflict', 'written']


async def test_file_store_recovers_a_prepared_write(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('main.md', 'old', expected_version=None)
    assert created.version is not None
    operation = MemoryOperation(id='run-1:call-1', fingerprint='write:main.md:new')
    new_version = str(int(created.version) + 1)
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES (?, ?, 'prepared', 'write', ?, ?, ?, ?, 1)",
            (operation.id, operation.fingerprint, 'main.md', created.version, 'new', new_version),
        )
        connection.execute('UPDATE file_metadata SET generation = ?', (int(new_version),))
        connection.commit()
    finally:
        connection.close()

    recovered_store = FileStore(tmp_path)
    recovered = await recovered_store.read('main.md', max_chars=1_000)
    assert recovered is not None
    assert recovered.content == 'new'
    assert recovered.version == new_version
    assert recovered.operation_id == operation.id
    receipt = await recovered_store.get_operation(operation)
    assert receipt is not None
    assert receipt.replayed
    assert receipt.version == new_version
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        row = connection.execute(
            'SELECT expected_version, new_content FROM memory_operations WHERE id = ?', (operation.id,)
        ).fetchone()
    finally:
        connection.close()
    assert row == (None, None)


async def test_file_store_scrubs_completed_operation_recovery_payloads(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    operation = MemoryOperation(id='run-1:call-1', fingerprint='write:main.md:secret')
    created = await store.write('main.md', 'secret', expected_version=None, operation=operation)
    assert created.version is not None
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'UPDATE memory_operations SET expected_version = ?, new_content = ? WHERE id = ?',
            ('legacy-version', 'legacy-secret', operation.id),
        )
        connection.execute('PRAGMA user_version = 0')
        connection.commit()
    finally:
        connection.close()

    receipt = await FileStore(tmp_path).get_operation(operation)

    assert receipt is not None
    assert receipt.replayed
    assert receipt.version == created.version
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        row = connection.execute(
            'SELECT expected_version, new_content, result_version FROM memory_operations WHERE id = ?', (operation.id,)
        ).fetchone()
    finally:
        connection.close()
    assert row == (None, None, created.version)


async def test_file_store_get_operation_recovers_a_prepared_write(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('main.md', 'old', expected_version=None)
    assert created.version is not None
    operation = MemoryOperation(id='run-1:call-1', fingerprint='write:main.md:new')
    new_version = str(int(created.version) + 1)
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES (?, ?, 'prepared', 'write', ?, ?, ?, ?, 1)",
            (operation.id, operation.fingerprint, 'main.md', created.version, 'new', new_version),
        )
        connection.execute('UPDATE file_metadata SET generation = ?', (int(new_version),))
        connection.commit()
    finally:
        connection.close()

    receipt = await FileStore(tmp_path).get_operation(operation)
    assert receipt is not None
    assert receipt.replayed
    assert (tmp_path / 'main.md').read_text() == 'new'


async def test_file_store_recovery_finalizes_an_already_applied_write(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('main.md', 'old', expected_version=None)
    assert created.version is not None
    operation = MemoryOperation(id='run-1:call-1', fingerprint='write:main.md:new')
    new_version = str(int(created.version) + 1)
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES (?, ?, 'prepared', 'write', ?, ?, ?, ?, 1)",
            (operation.id, operation.fingerprint, 'main.md', created.version, 'new', new_version),
        )
        connection.execute('UPDATE file_metadata SET generation = ?', (int(new_version),))
        connection.commit()
    finally:
        connection.close()
    (tmp_path / 'main.md').write_text('new')

    receipt = await FileStore(tmp_path).get_operation(operation)
    assert receipt is not None
    assert receipt.version == new_version
    assert receipt.replayed


async def test_file_store_recovery_rejects_a_divergent_external_write(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('main.md', 'old', expected_version=None)
    assert created.version is not None
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES ('write-1', 'fingerprint', 'prepared', 'write', 'main.md', ?, 'new', ?, 1)",
            (created.version, str(int(created.version) + 1)),
        )
        connection.commit()
    finally:
        connection.close()
    (tmp_path / 'main.md').write_text('external')

    with pytest.raises(MemoryConflictError, match='blocks recovery'):
        await FileStore(tmp_path).read('main.md', max_chars=1_000)


async def test_file_store_delete_receipt_and_prepared_delete_recovery(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('main.md', 'old', expected_version=None)
    operation = MemoryOperation(id='delete-1', fingerprint='delete:main.md')

    deleted = await store.delete('main.md', expected_version=created.version, operation=operation)
    assert deleted.existed
    assert not deleted.replayed
    assert await store.read('main.md', max_chars=1_000) is None
    assert (await store.delete('main.md', expected_version=created.version, operation=operation)).replayed

    recreated = await store.write('other.md', 'content', expected_version=None)
    prepared = MemoryOperation(id='delete-2', fingerprint='delete:other.md')
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES (?, ?, 'prepared', 'delete', 'other.md', ?, NULL, NULL, 1)",
            (prepared.id, prepared.fingerprint, recreated.version),
        )
        connection.commit()
    finally:
        connection.close()

    receipt = await FileStore(tmp_path).get_operation(prepared)
    assert receipt is not None
    assert receipt.replayed
    assert receipt.existed
    assert not (tmp_path / 'other.md').exists()


async def test_file_store_prepared_delete_rejects_an_external_write(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('main.md', 'old', expected_version=None)
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES ('delete-1', 'fingerprint', 'prepared', 'delete', 'main.md', ?, NULL, NULL, 1)",
            (created.version,),
        )
        connection.commit()
    finally:
        connection.close()
    (tmp_path / 'main.md').write_text('external')

    with pytest.raises(MemoryConflictError, match='blocks recovery'):
        await FileStore(tmp_path).read('main.md', max_chars=1_000)


async def test_file_store_listing_recovers_only_the_requested_tenant(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    tenant_a = await store.write('tenant-a/main.md', 'old', expected_version=None)
    assert tenant_a.version is not None
    await store.write('tenant-b/main.md', 'safe', expected_version=None)
    result_version = str(int(tenant_a.version) + 1)
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES ('write-a', 'fingerprint', 'prepared', 'write', 'tenant-a/main.md', ?, 'new', ?, 1)",
            (tenant_a.version, result_version),
        )
        connection.execute('UPDATE file_metadata SET generation = ?', (int(result_version),))
        connection.commit()
    finally:
        connection.close()
    (tmp_path / 'tenant-a/main.md').write_text('external')

    assert await FileStore(tmp_path).list_paths('tenant-b/', limit=10) == ['tenant-b/main.md']
    with pytest.raises(MemoryConflictError, match='blocks recovery'):
        await FileStore(tmp_path).list_paths('tenant-a/', limit=10)


async def test_file_store_bounded_listing_recovers_deletes_until_page_is_stable(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    versions: dict[str, str] = {}
    for path in ('a.md', 'b.md', 'c.md'):
        mutation = await store.write(path, path, expected_version=None)
        assert mutation.version is not None
        versions[path] = mutation.version
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.executemany(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES (?, ?, 'prepared', 'delete', ?, ?, NULL, NULL, 1)",
            [(f'delete-{path}', f'delete:{path}', path, versions[path]) for path in ('a.md', 'b.md')],
        )
        connection.commit()
    finally:
        connection.close()

    assert await FileStore(tmp_path).list_paths(limit=1) == ['c.md']
    assert not (tmp_path / 'a.md').exists()
    assert not (tmp_path / 'b.md').exists()


async def test_file_store_bounded_listing_recovers_prepared_write_before_page_boundary(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('b.md', 'b', expected_version=None)
    assert created.version is not None
    result_version = str(int(created.version) + 1)
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES ('write-a', 'write:a.md:a', 'prepared', 'write', 'a.md', NULL, 'a', ?, 0)",
            (result_version,),
        )
        connection.execute('UPDATE file_metadata SET generation = ?', (int(result_version),))
        connection.commit()
    finally:
        connection.close()

    assert await FileStore(tmp_path).list_paths(limit=1) == ['a.md']
    assert (tmp_path / 'a.md').read_text() == 'a'


async def test_file_store_bounded_listing_leaves_irrelevant_pending_path_prepared(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    await store.write('a.md', 'a', expected_version=None)
    connection = store._connect()
    try:
        result_version = str(store._next_generation(connection))
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES ('write-z', 'write:z.md:z', 'prepared', 'write', 'z.md', NULL, 'z', ?, 0)",
            (result_version,),
        )
        connection.commit()
    finally:
        connection.close()

    assert await FileStore(tmp_path).list_paths(limit=1) == ['a.md']
    assert not (tmp_path / 'z.md').exists()


async def test_file_store_scoped_listing_recovers_write_that_creates_scope_directory(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    connection = store._connect()
    try:
        result_version = str(store._next_generation(connection))
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES ('write-a', 'write:tenant/a.md:a', 'prepared', 'write', 'tenant/a.md', NULL, 'a', ?, 0)",
            (result_version,),
        )
        connection.commit()
    finally:
        connection.close()

    assert not (tmp_path / 'tenant').exists()
    assert await FileStore(tmp_path).list_paths('tenant/', limit=1) == ['tenant/a.md']
    assert (tmp_path / 'tenant' / 'a.md').read_text() == 'a'


async def test_file_store_page_recovery_uses_bounded_lookahead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileStore(tmp_path)
    versions: dict[str, str] = {}
    for index in range(21):
        path = f'{index:02}.md'
        mutation = await store.write(path, path, expected_version=None)
        assert mutation.version is not None
        versions[path] = mutation.version
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.executemany(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES (?, ?, 'prepared', 'delete', ?, ?, NULL, NULL, 1)",
            [(f'delete-{path}', f'delete:{path}', path, versions[path]) for path in sorted(versions)[:20]],
        )
        connection.commit()
    finally:
        connection.close()
    original_scandir = os.scandir
    scans = 0

    def counted_scandir(path: str | os.PathLike[str]) -> Iterator[os.DirEntry[str]]:
        nonlocal scans
        scans += 1
        return original_scandir(path)

    monkeypatch.setattr(os, 'scandir', counted_scandir)

    assert await FileStore(tmp_path).list_paths(limit=1) == ['20.md']
    assert scans == 2


async def test_file_store_search_skips_a_file_that_disappears_after_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileStore(tmp_path)
    await store.write('a.md', 'alpha', expected_version=None)
    await store.write('b.md', 'alpha', expected_version=None)
    original_resolve = store._resolve

    def disappearing_resolve(path: str) -> Path:
        target = original_resolve(path)
        if path == 'a.md':
            target.unlink(missing_ok=True)
        return target

    monkeypatch.setattr(store, '_resolve', disappearing_resolve)

    result = await store.search('', 'alpha', limit=2, max_files=2, max_chars=100, max_file_chars=100)

    assert [match.path for match in result.matches] == ['b.md']
    assert result.scanned == 1
    assert result.truncated


async def test_file_store_scoped_listing_walks_only_the_requested_tenant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileStore(tmp_path)
    await store.write('tenant-a/unrelated.md', 'a', expected_version=None)
    await store.write('tenant-b/main.md', 'b', expected_version=None)
    await store.write('other.md', 'other', expected_version=None)
    for index in range(200):
        (tmp_path / 'tenant-b' / f'z{index:03}.md').write_text('bounded')
    scanned: list[Path] = []
    original_scandir = os.scandir

    def tracking_scandir(path: str | os.PathLike[str]) -> Iterator[os.DirEntry[str]]:
        scanned.append(Path(path))
        return original_scandir(path)

    monkeypatch.setattr(os, 'scandir', tracking_scandir)

    assert await store.list_paths('tenant-b/', limit=2) == ['tenant-b/main.md', 'tenant-b/z000.md']
    assert scanned == [tmp_path / 'tenant-b']
    assert (await store.list_paths('tenant', limit=2)) == ['tenant-a/unrelated.md', 'tenant-b/main.md']
    assert await store.list_paths('missing/', limit=10) == []


async def test_file_store_recovery_finalizes_an_already_applied_delete(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    created = await store.write('main.md', 'old', expected_version=None)
    operation = MemoryOperation(id='delete-1', fingerprint='delete:main.md')
    connection = sqlite3.connect(tmp_path / '.memory-store.sqlite3')
    try:
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES (?, ?, 'prepared', 'delete', 'main.md', ?, NULL, NULL, 1)",
            (operation.id, operation.fingerprint, created.version),
        )
        connection.commit()
    finally:
        connection.close()
    (tmp_path / 'main.md').unlink()

    receipt = await FileStore(tmp_path).get_operation(operation)
    assert receipt is not None
    assert receipt.existed
    assert receipt.replayed


async def test_file_store_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    outside = tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    os.symlink(outside, root / 'link')

    with pytest.raises(ValueError):
        await FileStore(root).write('link/escape.md', 'secret', expected_version=None)


def test_sqlite_store_requires_one_connection_source() -> None:
    with pytest.raises(ValueError, match='exactly one'):
        SqliteMemoryStore()
    connection = sqlite3.connect(':memory:', check_same_thread=False)
    try:
        with pytest.raises(ValueError, match='exactly one'):
            SqliteMemoryStore(database='memory.sqlite3', connection=connection)
    finally:
        connection.close()
    with pytest.raises(ValueError, match='per-call connections'):
        SqliteMemoryStore(database=':memory:')


async def test_sqlite_store_supports_caller_owned_connection() -> None:
    connection = sqlite3.connect(':memory:', check_same_thread=False)
    try:
        store = SqliteMemoryStore(connection=connection)
        await store.write('main.md', 'content', expected_version=None)
        assert (await store.list_paths(limit=100)) == ['main.md']
    finally:
        connection.close()


async def test_sqlite_store_does_not_commit_a_caller_owned_transaction(tmp_path: Path) -> None:
    database = tmp_path / 'caller-owned.sqlite3'
    connection = sqlite3.connect(database, check_same_thread=False)
    try:
        connection.execute('CREATE TABLE unrelated(value TEXT)')
        connection.commit()
        store = SqliteMemoryStore(connection=connection)
        await store.read('missing.md', max_chars=100)
        connection.execute('BEGIN')
        connection.execute("INSERT INTO unrelated VALUES ('caller-work')")

        with pytest.raises(RuntimeError, match='must be idle'):
            await store.read('missing.md', max_chars=100)

        assert connection.in_transaction
        observer = sqlite3.connect(database)
        try:
            assert observer.execute('SELECT value FROM unrelated').fetchall() == []
        finally:
            observer.close()
        connection.rollback()
    finally:
        connection.close()


async def test_sqlite_store_rolls_back_failed_schema_initialization() -> None:
    connection = sqlite3.connect(':memory:', check_same_thread=False)
    try:
        connection.execute("CREATE VIEW memory_files AS SELECT 'main.md' AS path, 'content' AS content")
        connection.commit()
        store = SqliteMemoryStore(connection=connection)
        with pytest.raises(sqlite3.OperationalError):
            await store.read('main.md', max_chars=1_000)
        assert not connection.in_transaction
    finally:
        connection.close()


async def test_sqlite_store_closes_owned_connection_after_failed_schema_initialization(tmp_path: Path) -> None:
    database = tmp_path / 'invalid.sqlite3'
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE VIEW memory_files AS SELECT 'main.md' AS path, 'content' AS content")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.OperationalError):
        await SqliteMemoryStore(database=database).read('main.md', max_chars=100)
    database.unlink()


@pytest.mark.parametrize('iteration', range(10))
async def test_sqlite_store_migrates_legacy_schema_concurrently(tmp_path: Path, iteration: int) -> None:
    database = tmp_path / f'legacy-{iteration}.sqlite3'
    connection = sqlite3.connect(database)
    try:
        connection.execute('CREATE TABLE memory_files (path TEXT PRIMARY KEY, content TEXT NOT NULL)')
        connection.execute("INSERT INTO memory_files(path, content) VALUES ('legacy.md', 'legacy')")
        connection.commit()
    finally:
        connection.close()

    stores = [SqliteMemoryStore(database=database), SqliteMemoryStore(database=database)]
    files: list[str] = []
    start = anyio.Event()

    async def read(store: SqliteMemoryStore) -> None:
        await start.wait()
        file = await store.read('legacy.md', max_chars=1_000)
        assert file is not None
        files.append(file.version)

    async with anyio.create_task_group() as task_group:
        for store in stores:
            task_group.start_soon(read, store)
        start.set()

    assert files == ['1', '1']
    connection = sqlite3.connect(database)
    try:
        columns = {str(row[1]) for row in connection.execute('PRAGMA table_info(memory_files)').fetchall()}
    finally:
        connection.close()
    assert {'version', 'last_operation_id'} <= columns


async def test_sqlite_store_serializes_cas_across_instances(tmp_path: Path) -> None:
    database = tmp_path / 'memory.sqlite3'
    first = SqliteMemoryStore(database=database)
    second = SqliteMemoryStore(database=database)
    created = await first.write('main.md', 'initial', expected_version=None)
    outcomes: list[str] = []

    async def update(store: SqliteMemoryStore, content: str) -> None:
        try:
            await store.write('main.md', content, expected_version=created.version)
        except MemoryConflictError:
            outcomes.append('conflict')
        else:
            outcomes.append('written')

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(update, first, 'first')
        task_group.start_soon(update, second, 'second')

    assert sorted(outcomes) == ['conflict', 'written']


@dataclass(frozen=True)
class FakePostgresFile:
    content: str
    version: int
    operation_id: str | None


@dataclass(frozen=True)
class FakePostgresReceipt:
    fingerprint: str
    version: str | None
    existed: bool
    completed: bool


@dataclass
class FakePostgresDatabase:
    files: dict[str, FakePostgresFile] = field(default_factory=dict[str, FakePostgresFile])
    receipts: dict[str, FakePostgresReceipt] = field(default_factory=dict[str, FakePostgresReceipt])
    statements: list[str] = field(default_factory=list[str])
    transactions: int = 0
    generation: int = 0
    versions_initialized: bool = False


class FakeTransaction:
    def __init__(self, database: FakePostgresDatabase) -> None:
        self._database = database
        self._files: dict[str, FakePostgresFile] = {}
        self._receipts: dict[str, FakePostgresReceipt] = {}
        self._generation = 0
        self._versions_initialized = False

    async def __aenter__(self) -> object:
        self._files = self._database.files.copy()
        self._receipts = self._database.receipts.copy()
        self._generation = self._database.generation
        self._versions_initialized = self._database.versions_initialized
        self._database.transactions += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            self._database.files = self._files
            self._database.receipts = self._receipts
            self._database.generation = self._generation
            self._database.versions_initialized = self._versions_initialized
        return False


class FakePostgresConnection:
    def __init__(self, database: FakePostgresDatabase) -> None:
        self._database = database

    def transaction(self) -> AbstractAsyncContextManager[object]:
        return FakeTransaction(self._database)

    async def execute(self, query: str, *args: object) -> object:
        await anyio.sleep(0)
        self._database.statements.append(query)
        if query.startswith('UPDATE agent_memory_operations SET'):
            operation_id, version, existed = str(args[0]), args[1], bool(args[2])
            receipt = self._database.receipts[operation_id]
            self._database.receipts[operation_id] = FakePostgresReceipt(
                fingerprint=receipt.fingerprint,
                version=str(version) if version is not None else None,
                existed=existed,
                completed=True,
            )
        elif query.startswith('UPDATE agent_memory SET version'):
            for path, file in self._database.files.items():
                self._database.generation += 1
                self._database.files[path] = FakePostgresFile(
                    file.content, self._database.generation, file.operation_id
                )
        return 'OK'

    async def fetchval(self, query: str, *args: object) -> object:
        self._database.statements.append(query)
        if query.startswith('SELECT nextval'):
            self._database.generation += 1
            return self._database.generation
        if query.startswith('SELECT pg_advisory_xact_lock'):
            return None
        if query.startswith('INSERT INTO agent_memory_metadata'):
            if self._database.versions_initialized:
                return None
            self._database.versions_initialized = True
            return True
        if query.startswith('INSERT INTO agent_memory_operations'):
            operation_id, fingerprint = str(args[0]), str(args[1])
            if operation_id in self._database.receipts:
                return None
            self._database.receipts[operation_id] = FakePostgresReceipt(fingerprint, None, False, False)
            return operation_id
        if query.startswith('SELECT EXISTS'):
            return str(args[0]) in self._database.files
        raise AssertionError(f'unexpected fetchval query: {query}')  # pragma: no cover

    async def fetchrow(self, query: str, *args: object) -> tuple[object, ...] | None:
        self._database.statements.append(query)
        if query.startswith('SELECT fingerprint'):
            receipt = self._database.receipts.get(str(args[0]))
            if receipt is None:
                return None
            return receipt.fingerprint, receipt.version, receipt.existed, receipt.completed
        if query.startswith('SELECT left(content'):
            file = self._database.files.get(str(args[0]))
            if file is None:
                return None
            max_chars = int(str(args[1]))
            return file.content[:max_chars], file.version, file.operation_id, len(file.content)
        if query.startswith('INSERT INTO agent_memory '):
            path, content = str(args[0]), str(args[1])
            if path in self._database.files:
                return None
            self._database.generation += 1
            version = self._database.generation
            self._database.files[path] = FakePostgresFile(
                content, version, str(args[2]) if args[2] is not None else None
            )
            return (version,)
        if query.startswith('UPDATE agent_memory SET'):
            path, content, expected = str(args[0]), str(args[1]), str(args[3])
            file = self._database.files.get(path)
            if file is None or str(file.version) != expected:
                return None
            self._database.generation += 1
            version = self._database.generation
            self._database.files[path] = FakePostgresFile(
                content, version, str(args[2]) if args[2] is not None else None
            )
            return (version,)
        if query.startswith('DELETE FROM agent_memory '):
            path, expected = str(args[0]), str(args[1])
            file = self._database.files.get(path)
            if file is None or str(file.version) != expected:
                return None
            del self._database.files[path]
            return (file.version,)
        raise AssertionError(f'unexpected fetchrow query: {query}')  # pragma: no cover

    async def fetch(self, query: str, *args: object) -> list[tuple[object, ...]]:
        self._database.statements.append(query)
        prefix = str(args[0])
        files = sorted((path, file) for path, file in self._database.files.items() if path.startswith(prefix))
        if query.startswith('SELECT path, left(content'):
            max_chars = int(str(args[1]))
            limit = int(str(args[2]))
            return [(path, file.content[:max_chars], len(file.content)) for path, file in files[:limit]]
        if query.startswith('SELECT path'):
            limit = int(str(args[1]))
            return [(path,) for path, _ in files[:limit]]
        raise AssertionError(f'unexpected fetch query: {query}')  # pragma: no cover


class FakeAcquire:
    def __init__(self, connection: PostgresConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> PostgresConnection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False


class FakePostgresPool:
    def __init__(self) -> None:
        self.database = FakePostgresDatabase()
        self.connection = FakePostgresConnection(self.database)

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        return FakeAcquire(self.connection)


async def test_postgres_store_contract_schema_and_parameterization() -> None:
    pool = FakePostgresPool()
    assert isinstance(pool, PostgresPool)
    store = PostgresMemoryStore(pool)
    assert isinstance(store, MemoryStore)
    assert isinstance(store, SearchableMemoryStore)

    operation = MemoryOperation(id='run-1:call-1', fingerprint='write:tenant/main.md:alpha')
    created = await store.write('tenant/main.md', 'alpha', expected_version=None, operation=operation)
    assert created.version == '1'
    assert (await store.write('tenant/main.md', 'alpha', expected_version=None, operation=operation)).replayed
    with pytest.raises(MemoryConflictError):
        await store.write('tenant/main.md', 'duplicate create', expected_version=None)
    with pytest.raises(MemoryOperationConflictError):
        await store.get_operation(MemoryOperation(id=operation.id, fingerprint='different'))

    file = await store.read('tenant/main.md', max_chars=1_000)
    assert file is not None
    updated = await store.write('tenant/main.md', 'alpha alpha', expected_version=file.version)
    assert updated.version == '2'
    with pytest.raises(MemoryConflictError):
        await store.write('tenant/main.md', 'stale', expected_version=file.version)
    await store.write('other/private.md', 'alpha alpha alpha', expected_version=None)
    result = await store.search('tenant/', 'alpha', limit=10, max_files=10, max_chars=100, max_file_chars=1_000)
    assert [match.path for match in result.matches] == ['tenant/main.md']
    assert await store.list_paths('tenant/', limit=100) == ['tenant/main.md']

    assert pool.database.transactions == 7
    assert len([query for query in pool.database.statements if query.startswith('CREATE TABLE')]) == 3
    assert len([query for query in pool.database.statements if query.startswith('ALTER TABLE')]) == 2
    for query in pool.database.statements:
        if query.startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
            assert '$' in query or query.startswith(
                ('INSERT INTO agent_memory_metadata', 'UPDATE agent_memory SET version')
            )


async def test_postgres_store_concurrent_schema_initialization() -> None:
    pool = FakePostgresPool()
    store = PostgresMemoryStore(pool)
    start = anyio.Event()

    async def read() -> None:
        await start.wait()
        assert await store.read('missing.md', max_chars=1_000) is None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(read)
        task_group.start_soon(read)
        start.set()

    assert len([query for query in pool.database.statements if query.startswith('CREATE TABLE')]) == 3


async def test_postgres_store_missing_receipt_and_delete_contract() -> None:
    store = PostgresMemoryStore(FakePostgresPool())
    operation = MemoryOperation(id='delete-missing', fingerprint='delete:missing.md')

    assert await store.get_operation(operation) is None
    assert await store.read('missing.md', max_chars=1_000) is None
    missing = await store.delete('missing.md', expected_version=None, operation=operation)
    assert not missing.existed
    assert not missing.replayed
    assert (await store.delete('missing.md', expected_version=None, operation=operation)).replayed

    created = await store.write('main.md', 'content', expected_version=None)
    with pytest.raises(MemoryConflictError):
        await store.delete('main.md', expected_version=None)
    with pytest.raises(MemoryConflictError):
        await store.delete('main.md', expected_version='stale')
    deleted = await store.delete('main.md', expected_version=created.version)
    assert deleted.existed
    assert not deleted.replayed
    assert await store.read('main.md', max_chars=1_000) is None


async def test_postgres_store_rejects_non_positive_bounds() -> None:
    store = PostgresMemoryStore(FakePostgresPool())
    with pytest.raises(ValueError, match='max_chars'):
        await store.read('main.md', max_chars=0)
    with pytest.raises(ValueError, match='limit'):
        await store.list_paths(limit=0)
    result = await store.search('', 'query', limit=1, max_files=1, max_chars=1, max_file_chars=0)
    assert result.matches == []


async def test_postgres_store_version_does_not_repeat_after_delete_and_recreate() -> None:
    store = PostgresMemoryStore(FakePostgresPool())
    first = await store.write('main.md', 'same', expected_version=None)
    await store.delete('main.md', expected_version=first.version)
    recreated = await store.write('main.md', 'same', expected_version=None)

    assert recreated.version != first.version
    with pytest.raises(MemoryConflictError):
        await store.write('main.md', 'stale', expected_version=first.version)


async def test_postgres_concurrent_store_initialization_cannot_regress_versions() -> None:
    pool = FakePostgresPool()
    stores = [PostgresMemoryStore(pool), PostgresMemoryStore(pool)]
    start = anyio.Event()
    versions: list[str] = []

    async def write(store: PostgresMemoryStore, path: str) -> None:
        await start.wait()
        mutation = await store.write(path, 'content', expected_version=None)
        assert mutation.version is not None
        versions.append(mutation.version)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(write, stores[0], 'first.md')
        task_group.start_soon(write, stores[1], 'second.md')
        start.set()

    assert len(set(versions)) == 2
    assert not any('setval' in statement for statement in pool.database.statements)
    assert pool.database.statements[0].startswith('SELECT pg_advisory_xact_lock')


@pytest.mark.parametrize(
    'table',
    ('agent-memory', 'agent_memory; DROP TABLE users', '1memory', 'a' * 53),
)
def test_postgres_store_rejects_unsafe_table_names(table: str) -> None:
    with pytest.raises(ValueError, match='invalid table name'):
        PostgresMemoryStore(FakePostgresPool(), table=table)
