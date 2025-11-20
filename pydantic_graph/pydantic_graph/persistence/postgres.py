from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter

from pydantic_graph import End
from pydantic_graph.exceptions import GraphRuntimeError
from pydantic_graph.nodes import BaseNode
from pydantic_graph.persistence import (
    BaseStatePersistence,
    EndSnapshot,
    NodeSnapshot,
    Snapshot,
    build_snapshot_list_type_adapter,
)

StateT = TypeVar('StateT')
RunEndT = TypeVar('RunEndT')


class PostgresStatePersistence(
    BaseStatePersistence[StateT, RunEndT],
    Generic[StateT, RunEndT],
):
    """PostgreSQL-backed implementation of state persistence for graph runs.

    Stores full snapshots (NodeSnapshot / EndSnapshot) as JSONB and tracks status and timings.
    """

    def __init__(self, pool: Any, run_id: str) -> None:
        self.pool = pool
        self.run_id = run_id
        self._snapshot_adapter: TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = None

    def should_set_types(self) -> bool:
        return self._snapshot_adapter is None

    def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
        self._snapshot_adapter = build_snapshot_list_type_adapter(state_type, run_end_type)

    def _dump_snapshot(self, snapshot: Snapshot[StateT, RunEndT]) -> dict[str, Any]:
        """Encode a single snapshot to a JSON-serializable dict using the list adapter."""
        assert self._snapshot_adapter is not None, 'Persistence types not set'
        return self._snapshot_adapter.dump_python([snapshot], mode='json')[0]

    def _load_snapshot(self, data: dict[str, Any]) -> Snapshot[StateT, RunEndT]:
        """Decode a single snapshot from a JSON-compatible dictionary."""
        assert self._snapshot_adapter is not None, 'Persistence types not set'

        if isinstance(data, str):
            data = json.loads(data)

        return self._snapshot_adapter.validate_python([data])[0]

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        """Snapshot a node when it is scheduled by the graph."""
        snapshot = NodeSnapshot(state=state, node=next_node)
        payload = self._dump_snapshot(snapshot)
        node_id = next_node.get_node_def(local_ns=None).node_id

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_snapshots (
                    run_id, snapshot_id, kind, status, node_id, snapshot
                )
                VALUES ($1, $2, 'node', 'queued', $3, $4::jsonb)
                ON CONFLICT (run_id, snapshot_id) DO UPDATE
                SET snapshot = EXCLUDED.snapshot,
                    node_id = EXCLUDED.node_id,
                    status   = EXCLUDED.status
                """,
                self.run_id,
                snapshot.id,
                node_id,
                json.dumps(payload),
            )

    async def snapshot_node_if_new(
        self,
        snapshot_id: str,
        state: StateT,
        next_node: BaseNode[StateT, Any, RunEndT],
    ) -> None:
        """Snapshot a node only if the given snapshot_id does not already exist."""
        snapshot = NodeSnapshot(state=state, node=next_node, id=snapshot_id)
        payload = self._dump_snapshot(snapshot)
        node_id = next_node.get_node_def(local_ns=None).node_id

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_snapshots (
                    run_id, snapshot_id, kind, status, node_id, snapshot
                )
                VALUES ($1, $2, 'node', 'queued', $3, $4::jsonb)
                ON CONFLICT (run_id, snapshot_id) DO NOTHING
                """,
                self.run_id,
                snapshot_id,
                node_id,
                json.dumps(payload),
            )

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        """Snapshot the graph end state and update the run result."""
        snapshot = EndSnapshot(state=state, result=end)
        payload = self._dump_snapshot(snapshot)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_snapshots (
                    run_id, snapshot_id, kind, status, node_id, snapshot
                )
                VALUES ($1, $2, 'end', NULL, 'End', $3::jsonb)
                ON CONFLICT (run_id, snapshot_id) DO UPDATE
                SET snapshot = EXCLUDED.snapshot
                """,
                self.run_id,
                snapshot.id,
                json.dumps(payload),
            )

            await conn.execute(
                """
                UPDATE graph_runs
                SET finished_at = now(),
                    status      = 'success',
                    result      = $2::jsonb
                WHERE id = $1
                """,
                self.run_id,
                json.dumps(payload),
            )

    @asynccontextmanager
    async def record_run(self, snapshot_id: str):
        """Record execution status and timing for a single node run.

        Called by Graph around the actual execution of a node.

        We:
        - assert the node is not already running or finished
        - mark it as running
        - on success or error, update status and timing
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT status
                FROM graph_snapshots
                WHERE run_id = $1
                  AND snapshot_id = $2
                  AND kind = 'node'
                """,
                self.run_id,
                snapshot_id,
            )
            if row is None:
                raise LookupError(f'Snapshot {snapshot_id!r} not found for run {self.run_id}')

            current_status = row['status']
            if current_status not in ('queued', 'pending'):
                raise GraphRuntimeError(
                    f'Snapshot {snapshot_id!r} already in status {current_status!r}',
                )

            now = datetime.now(timezone.utc)
            await conn.execute(
                """
                UPDATE graph_snapshots
                SET status = 'running',
                    started_at = $3
                WHERE run_id = $1 AND snapshot_id = $2
                """,
                self.run_id,
                snapshot_id,
                now,
            )

        start = datetime.now(timezone.utc)

        try:
            yield
        except Exception:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE graph_snapshots
                    SET status        = 'error',
                        finished_at   = now(),
                        duration_secs = $3
                    WHERE run_id = $1 AND snapshot_id = $2
                    """,
                    self.run_id,
                    snapshot_id,
                    duration,
                )
                await conn.execute(
                    """
                    UPDATE graph_runs
                    SET status = 'error',
                        finished_at = COALESCE(finished_at, now())
                    WHERE id = $1
                    """,
                    self.run_id,
                )
            raise
        else:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE graph_snapshots
                    SET status        = 'success',
                        finished_at   = now(),
                        duration_secs = $3
                    WHERE run_id = $1 AND snapshot_id = $2
                    """,
                    self.run_id,
                    snapshot_id,
                    duration,
                )

    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        """Pop the next queued or pending node snapshot for this run."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT snapshot_id, snapshot
                    FROM graph_snapshots
                    WHERE run_id = $1
                      AND kind = 'node'
                      AND status IN ('queued', 'pending')
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """,
                    self.run_id,
                )
                if row is None:
                    return None

                snapshot_id = row['snapshot_id']
                await conn.execute(
                    """
                    UPDATE graph_snapshots
                    SET status = 'pending'
                    WHERE run_id = $1
                      AND snapshot_id = $2
                    """,
                    self.run_id,
                    snapshot_id,
                )

        snapshot = self._load_snapshot(row['snapshot'])
        if not isinstance(snapshot, NodeSnapshot):
            raise TypeError(f'Expected NodeSnapshot, got {type(snapshot)}')
        return snapshot

    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        """Load all snapshots for this run in creation order."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT snapshot
                FROM graph_snapshots
                WHERE run_id = $1
                ORDER BY created_at
                """,
                self.run_id,
            )

        raw_payloads = [r['snapshot'] for r in rows]
        payloads = [json.loads(p) if isinstance(p, str) else p for p in raw_payloads]
        assert self._snapshot_adapter is not None, 'Persistence types not set'
        return self._snapshot_adapter.validate_python(payloads)
    