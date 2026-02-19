from __future__ import annotations as _annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import anyio
import pydantic

from .. import _utils as _graph_utils, exceptions
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
class FileStatePersistence(BaseStatePersistence[StateT, RunEndT]):
    """File based state persistence that hold graph run state in a JSON file."""

    json_file: Path
    """Path to the JSON file where the snapshots are stored.

    You should use a different file for each graph run, but a single file should be reused for multiple
    steps of the same run.

    For example if you have a run ID of the form `run_123abc`, you might create a `FileStatePersistence` thus:

    ```py
    from pathlib import Path

    from pydantic_graph import FullStatePersistence

    run_id = 'run_123abc'
    persistence = FullStatePersistence(Path('runs') / f'{run_id}.json')
    ```
    """
    _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
        default=None, init=False, repr=False
    )

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        await self._append_save(NodeSnapshot(state=state, node=next_node))

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        async with self._lock():
            snapshots = await self.load_all()
            if not any(s.id == snapshot_id for s in snapshots):  # pragma: no branch
                await self._append_save(NodeSnapshot(state=state, node=next_node), lock=False)

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        await self._append_save(EndSnapshot(state=state, result=end))

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        async with self._lock():
            snapshots = await self.load_all()
            try:
                snapshot = next(s for s in snapshots if s.id == snapshot_id)
            except StopIteration as e:
                raise LookupError(f'No snapshot found with id={snapshot_id!r}') from e

            assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
            exceptions.GraphNodeStatusError.check(snapshot.status)
            snapshot.status = 'running'
            snapshot.start_ts = _utils.now_utc()
            await self._save(snapshots)

        start = perf_counter()
        try:
            yield
        except Exception:
            duration = perf_counter() - start
            async with self._lock():
                await _graph_utils.run_in_executor(self._after_run_sync, snapshot_id, duration, 'error')
            raise
        else:
            snapshot.duration = perf_counter() - start
            async with self._lock():
                await _graph_utils.run_in_executor(self._after_run_sync, snapshot_id, snapshot.duration, 'success')

    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        async with self._lock():
            snapshots = await self.load_all()
            if snapshot := next((s for s in snapshots if isinstance(s, NodeSnapshot) and s.status == 'created'), None):
                snapshot.status = 'pending'
                await self._save(snapshots)
                return snapshot

    def should_set_types(self) -> bool:
        """Whether types need to be set."""
        return self._snapshots_type_adapter is None

    def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
        self._snapshots_type_adapter = build_snapshot_list_type_adapter(state_type, run_end_type)

    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        return await _graph_utils.run_in_executor(self._load_sync)

    def _load_sync(self) -> list[Snapshot[StateT, RunEndT]]:
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        try:
            content = self.json_file.read_bytes()
        except FileNotFoundError:
            return []
        else:
            return self._snapshots_type_adapter.validate_json(content)

    def _after_run_sync(self, snapshot_id: str, duration: float, status: SnapshotStatus) -> None:
        snapshots = self._load_sync()
        snapshot = next(s for s in snapshots if s.id == snapshot_id)
        assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        snapshot.duration = duration
        snapshot.status = status
        self._save_sync(snapshots)

    async def _save(self, snapshots: list[Snapshot[StateT, RunEndT]]) -> None:
        await _graph_utils.run_in_executor(self._save_sync, snapshots)

    def _save_sync(self, snapshots: list[Snapshot[StateT, RunEndT]]) -> None:
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        self.json_file.write_bytes(self._snapshots_type_adapter.dump_json(snapshots, indent=2))

    async def _append_save(self, snapshot: Snapshot[StateT, RunEndT], *, lock: bool = True) -> None:
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        async with AsyncExitStack() as stack:
            if lock:
                await stack.enter_async_context(self._lock())
            snapshots = await self.load_all()
            snapshots.append(snapshot)
            await self._save(snapshots)

    @asynccontextmanager
    async def _lock(self, *, timeout: float = 1.0) -> AsyncIterator[None]:
        """Lock a file by checking and writing a `.pydantic-graph-persistence-lock` to it.

        Args:
            timeout: how long to wait for the lock

        Returns: an async context manager that holds the lock
        """
        lock_file = self.json_file.parent / f'{self.json_file.name}.pydantic-graph-persistence-lock'
        lock_id = secrets.token_urlsafe().encode()

        with anyio.fail_after(timeout):
            while not await _file_append_check(lock_file, lock_id):
                await anyio.sleep(0.01)

        try:
            yield
        finally:
            await _graph_utils.run_in_executor(lock_file.unlink, missing_ok=True)


async def _file_append_check(file: Path, content: bytes) -> bool:
    path = anyio.Path(file)
    if await path.exists():
        return False

    async with await anyio.open_file(path, mode='ab') as f:
        await f.write(content + b'\n')

    return (await path.read_bytes()).startswith(content)
