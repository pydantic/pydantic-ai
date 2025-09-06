"""SQLite-based state persistence backend for pydantic-graph.

This module provides a persistence implementation backed by a SQLite
database.  It offers similar semantics to the existing file-based
implementation
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pydantic

from .. import _utils as _graph_utils
from ..nodes import BaseNode, End
from . import (
    BaseStatePersistence,
    EndSnapshot,
    NodeSnapshot,
    RunEndT,
    Snapshot,
    SnapshotStatus,
    StateT,
    _utils,
    build_snapshot_list_type_adapter,
)


@dataclass
class SQLiteStatePersistence(BaseStatePersistence[StateT, RunEndT]):
    """Persist each snapshot in its own row; use SQL to atomically fetch/update rows."""

    db_file: Path
    run_id: str
    _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_file)
        try:
            with conn:
                conn.execute(
                    'CREATE TABLE IF NOT EXISTS snapshots ('
                    'run_id TEXT NOT NULL,'
                    'id TEXT NOT NULL,'
                    'data TEXT NOT NULL,'
                    'status TEXT NOT NULL,'
                    'start_ts TEXT,'
                    'duration REAL,'
                    'PRIMARY KEY (run_id, id)'
                    ')'
                )
                conn.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_run_id_status ON snapshots (run_id, status)')
        finally:
            conn.close()

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        snapshot = NodeSnapshot(state=state, node=next_node)
        await self._insert_snapshot(snapshot, status='created')

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        """Insert a NodeSnapshot only if one with this id and run_id does not already exist."""
        if await self._exists_snapshot(snapshot_id):
            return  # do nothing if snapshot already exists
        snapshot = NodeSnapshot(state=state, node=next_node, id=snapshot_id)
        await self._insert_snapshot(snapshot, status='created')

    async def _exists_snapshot(self, snapshot_id: str) -> bool:
        return await _graph_utils.run_in_executor(self._exists_snapshot_sync, snapshot_id)

    def _exists_snapshot_sync(self, snapshot_id: str) -> bool:
        conn = sqlite3.connect(self.db_file)
        try:
            cur = conn.execute(
                'SELECT 1 FROM snapshots WHERE run_id = ? AND id = ?',
                (self.run_id, snapshot_id),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        snapshot = EndSnapshot(state=state, result=end)
        await self._insert_snapshot(snapshot, status='created')

    def should_set_types(self) -> bool:
        """Whether types need to be set."""
        return self._snapshots_type_adapter is None

    def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
        self._snapshots_type_adapter = build_snapshot_list_type_adapter(state_type, run_end_type)

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        # First check whether the snapshot exists
        if not await self._exists_snapshot(snapshot_id):
            raise LookupError(f"No snapshot found with id='{snapshot_id}'")

        # Set status to 'running' and record start timestamp atomically
        start_iso: str = _utils.now_utc().isoformat()
        await self._update_fields_by_id(snapshot_id, status='running', start_ts=start_iso)
        start = perf_counter()
        try:
            yield
        except Exception:
            duration = perf_counter() - start
            await self._update_fields_by_id(snapshot_id, status='error', duration=duration)
            raise
        else:
            duration = perf_counter() - start
            await self._update_fields_by_id(snapshot_id, status='success', duration=duration)

    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        # Atomically select the next created snapshot and mark it pending
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        row = await self._pop_next_snapshot()
        if row is None:
            return None
        data_json, status, start_ts, duration, _ = row
        snapshot: Snapshot[StateT, RunEndT] = self._snapshots_type_adapter.validate_json(
            f'[{data_json}]', by_alias=True, by_name=True
        )[0]
        assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        snapshot.status = status
        if start_ts is not None:
            try:
                snapshot.start_ts = datetime.fromisoformat(start_ts)
            except Exception:
                snapshot.start_ts = datetime.now()
        snapshot.duration = duration
        return snapshot

    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        return await _graph_utils.run_in_executor(self._load_sync)

    def _load_sync(self) -> list[Snapshot[StateT, RunEndT]]:
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        conn = sqlite3.connect(self.db_file)
        try:
            cur = conn.execute(
                'SELECT data, status, start_ts, duration FROM snapshots WHERE run_id = ? ORDER BY rowid ASC',
                (self.run_id,),
            )
            rows = cur.fetchall()
            snapshots: list[Snapshot[StateT, RunEndT]] = []
            for data_json, status, start_ts, duration in rows:
                snap = self._snapshots_type_adapter.validate_json(f'[{data_json}]', by_alias=True, by_name=True)[0]
                # assert isinstance(snap, NodeSnapshot), 'Only NodeSnapshot can be recorded'
                if isinstance(snap, NodeSnapshot):
                    snap.status = status
                    if start_ts is not None:
                        try:
                            snap.start_ts = datetime.fromisoformat(start_ts)
                        except Exception:
                            snap.start_ts = start_ts
                    snap.duration = duration
                snapshots.append(snap)
            return snapshots
        finally:
            conn.close()

    async def _insert_snapshot(self, snapshot: Snapshot[StateT, RunEndT], *, status: str = 'created') -> None:
        """Insert a snapshot asynchronously by delegating to _insert_sync."""
        await _graph_utils.run_in_executor(self._insert_sync, snapshot, status)

    def _insert_sync(self, snapshot: Snapshot[StateT, RunEndT], status: str = 'created') -> None:
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        py = json.loads(self._snapshots_type_adapter.dump_json([snapshot], warnings=False).decode())[0]
        data_json = json.dumps(py)
        conn = sqlite3.connect(self.db_file)
        try:
            with conn:
                conn.execute(
                    'INSERT OR IGNORE INTO snapshots '
                    '(run_id, id, data, status, start_ts, duration) '
                    'VALUES (?, ?, ?, ?, NULL, NULL)',
                    (self.run_id, snapshot.id, data_json, status),
                )
        finally:
            conn.close()

    async def _update_fields_by_id(
        self,
        snapshot_id: str,
        *,
        status: SnapshotStatus | None = None,
        start_ts: str | None = None,
        duration: float | None = None,
    ) -> None:
        await _graph_utils.run_in_executor(
            lambda: self._update_fields_by_id_sync(snapshot_id, status=status, start_ts=start_ts, duration=duration)
        )

    def _update_fields_by_id_sync(
        self,
        snapshot_id: str,
        *,
        status: SnapshotStatus | None = None,
        start_ts: str | None = None,
        duration: float | None = None,
    ) -> None:
        conn = sqlite3.connect(self.db_file)
        try:
            with conn:
                updates: list[str] = []
                params: list[Any] = []
                if status is not None:
                    updates.append('status = ?')
                    params.append(status)
                if start_ts is not None:
                    updates.append('start_ts = ?')
                    params.append(start_ts)
                if duration is not None:
                    updates.append('duration = ?')
                    params.append(duration)
                if updates:
                    sql = 'UPDATE snapshots SET ' + ', '.join(updates) + ' WHERE run_id = ? AND id = ?'
                    params.append(self.run_id)
                    params.append(snapshot_id)
                    conn.execute(sql, params)
        finally:
            conn.close()

    async def _pop_next_snapshot(self) -> tuple[str, SnapshotStatus, str | None, float | None, str] | None:
        """Asynchronously call _pop_next_snapshot_sync using the thread pool."""
        return await _graph_utils.run_in_executor(self._pop_next_snapshot_sync)

    # Helper to atomically pop the next created snapshot
    def _pop_next_snapshot_sync(self) -> tuple[str, SnapshotStatus, str | None, float | None, str] | None:
        conn = sqlite3.connect(self.db_file)
        try:
            with conn:
                row = conn.execute(
                    'UPDATE snapshots '
                    "SET status = 'pending' "
                    'WHERE rowid = ('
                    '    SELECT rowid FROM snapshots '
                    "    WHERE run_id = ? AND status = 'created' "
                    '    ORDER BY rowid '
                    '    LIMIT 1'
                    ') '
                    'RETURNING data, status, start_ts, duration, id',
                    (self.run_id,),
                ).fetchone()
                return row
        finally:
            conn.close()
