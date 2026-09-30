"""Construction tests for `PostgresStepStore`.

These run without a database: the pool is a stub that is never queried. The
behaviour suite against a real server lives in
`pydantic_ai_harness/integration_tests/postgres`.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pytest

import pydantic_ai_harness.media as media_module
import pydantic_ai_harness.step_persistence as step_persistence_module
from pydantic_ai_harness.media import DiskMediaStore, PostgresMediaStore
from pydantic_ai_harness.step_persistence import (
    PostgresConnection,
    PostgresPool,
    PostgresStepStore,
    StepStore,
)


class _UnusedPool:
    """Satisfies `PostgresPool` structurally; construction never acquires."""

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        raise AssertionError('the pool is not queried')  # pragma: no cover


class TestPostgresStepStoreConstruction:
    @pytest.mark.parametrize('table', ['step-store', 'x; DROP TABLE users', '1steps', 'a' * 41, ''])
    def test_rejects_invalid_table_name(self, table: str) -> None:
        with pytest.raises(ValueError, match='invalid table name'):
            PostgresStepStore(_UnusedPool(), table=table)

    def test_accepts_forty_character_prefix(self) -> None:
        table = 'a' * 40
        store = PostgresStepStore(_UnusedPool(), table=table)
        assert store._table == table  # pyright: ignore[reportPrivateUsage]

    def test_rejects_invalid_max_snapshots_per_run(self) -> None:
        with pytest.raises(ValueError, match='must be an int >= 1 or None'):
            PostgresStepStore(_UnusedPool(), max_snapshots_per_run=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match='must be >= 1 or None'):
            PostgresStepStore(_UnusedPool(), max_snapshots_per_run=0)

    def test_auto_media_store_shares_pool(self) -> None:
        pool = _UnusedPool()
        store = PostgresStepStore(pool, table='steps')
        media_store = store._media_store  # pyright: ignore[reportPrivateUsage]
        assert isinstance(media_store, PostgresMediaStore)
        assert media_store._pool is pool  # pyright: ignore[reportPrivateUsage]
        assert media_store._table == 'steps_media'  # pyright: ignore[reportPrivateUsage]

    def test_media_store_none_disables_media(self) -> None:
        store = PostgresStepStore(_UnusedPool(), media_store=None)
        assert store._media_store is None  # pyright: ignore[reportPrivateUsage]

    def test_custom_media_store_is_kept(self, tmp_path: Path) -> None:
        media_store = DiskMediaStore(tmp_path / 'media')
        store = PostgresStepStore(_UnusedPool(), media_store=media_store)
        assert store._media_store is media_store  # pyright: ignore[reportPrivateUsage]


class TestPostgresStepStoreProtocol:
    def test_satisfies_step_store_protocol(self) -> None:
        assert isinstance(PostgresStepStore(_UnusedPool()), StepStore)

    def test_stub_pool_satisfies_pool_protocol(self) -> None:
        assert isinstance(_UnusedPool(), PostgresPool)


class TestPostgresStepStoreExports:
    def test_exported_from_step_persistence(self) -> None:
        for name in ('PostgresConnection', 'PostgresPool', 'PostgresStepStore'):
            assert name in step_persistence_module.__all__
        assert step_persistence_module.PostgresStepStore is PostgresStepStore

    def test_pool_protocols_are_the_media_ones(self) -> None:
        assert PostgresPool is media_module.PostgresPool
        assert PostgresConnection is media_module.PostgresConnection
