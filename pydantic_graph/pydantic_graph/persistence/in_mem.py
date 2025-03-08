"""In memory state persistence.

This module provides simple in memory state persistence for graphs.
"""

from __future__ import annotations as _annotations

import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import pydantic
from typing_extensions import TypeVar

from ..nodes import BaseNode, End
from . import (
    EndSnapshot,
    NodeSnapshot,
    Snapshot,
    SnapshotStatus,
    StatePersistence,
    _utils,
    build_snapshot_list_type_adapter,
)

StateT = TypeVar('StateT', default=Any)
RunEndT = TypeVar('RunEndT', default=Any)


@dataclass
class SimpleStatePersistence(StatePersistence[StateT, RunEndT]):
    """Simple in memory state persistence that just hold the latest snapshot.

    If no state persistence implementation is provided when running a graph, this is used by default.
    """

    deep_copy: bool = False
    """Whether to deep copy the state and nodes when storing them.

    Defaults to `False` so you can use nodes that don't support deep copying.
    """
    last_snapshot: Snapshot[StateT, RunEndT] | None = None
    """The last snapshot."""

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        self.last_snapshot = NodeSnapshot(
            state=self._prep_state(state),
            node=next_node.deep_copy() if self.deep_copy else next_node,
        )

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        self.last_snapshot = EndSnapshot(
            state=self._prep_state(state),
            result=end.deep_copy_data() if self.deep_copy else end,
        )

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        assert self.last_snapshot is not None, 'No snapshot to record'
        assert isinstance(self.last_snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        assert snapshot_id == self.last_snapshot.id, (
            f'snapshot_id must match the last snapshot ID: {snapshot_id!r} != {self.last_snapshot.id!r}'
        )
        self.last_snapshot.status = 'running'
        self.last_snapshot.start_ts = _utils.now_utc()

        start = perf_counter()
        try:
            yield
        except Exception:
            self.last_snapshot.duration = perf_counter() - start
            self.last_snapshot.status = 'error'
            raise
        else:
            self.last_snapshot.duration = perf_counter() - start
            self.last_snapshot.status = 'success'

    async def restore(self) -> Snapshot[StateT, RunEndT] | None:
        return self.last_snapshot

    async def get_node_snapshot(
        self, snapshot_id: str, status: SnapshotStatus | None = None
    ) -> NodeSnapshot[StateT, RunEndT] | None:
        if isinstance(self.last_snapshot, NodeSnapshot) and self.last_snapshot.id == snapshot_id:
            if status and self.last_snapshot.status != status:
                return None
            else:
                return self.last_snapshot

    def _prep_state(self, state: StateT) -> StateT:
        """Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
        if not self.deep_copy or state is None:
            return state
        else:
            return copy.deepcopy(state)


@dataclass
class FullStatePersistence(StatePersistence[StateT, RunEndT]):
    """In memory state persistence that hold a history of nodes."""

    deep_copy: bool = True
    """Whether to deep copy the state and nodes when storing them.

    Defaults to `True` so even if nodes or state are modified after the snapshot is taken,
    the persistence history will record the value at the time of the snapshot.
    """
    history: list[Snapshot[StateT, RunEndT]] = field(default_factory=list)
    """List of snapshots taken during the graph run."""
    _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
        default=None, init=False, repr=False
    )

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        snapshot = NodeSnapshot(
            state=self._prep_state(state),
            node=next_node.deep_copy() if self.deep_copy else next_node,
        )
        self.history.append(snapshot)

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        snapshot = EndSnapshot(
            state=self._prep_state(state),
            result=end.deep_copy_data() if self.deep_copy else end,
        )
        self.history.append(snapshot)

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        try:
            snapshot = next(s for s in self.history if s.id == snapshot_id)
        except StopIteration as e:
            raise LookupError(f'No snapshot found with id={snapshot_id}') from e

        assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        snapshot.status = 'running'
        snapshot.start_ts = _utils.now_utc()
        start = perf_counter()
        try:
            yield
        except Exception:
            snapshot.duration = perf_counter() - start
            snapshot.status = 'error'
            raise
        else:
            snapshot.duration = perf_counter() - start
            snapshot.status = 'success'

    async def restore(self) -> Snapshot[StateT, RunEndT] | None:
        if self.history:
            return self.history[-1]

    async def get_node_snapshot(
        self, snapshot_id: str, status: SnapshotStatus | None = None
    ) -> Snapshot[StateT, RunEndT] | None:
        for snapshot in self.history:
            if (
                isinstance(snapshot, NodeSnapshot)
                and snapshot.id == snapshot_id
                and (status is None or snapshot.status == status)
            ):
                return snapshot

    def set_types(self, get_types: Callable[[], tuple[type[StateT], type[RunEndT]]]) -> None:
        if self._snapshots_type_adapter is None:
            state_t, run_end_t = get_types()
            self._snapshots_type_adapter = build_snapshot_list_type_adapter(state_t, run_end_t)

    def dump_json(self, *, indent: int | None = None) -> bytes:
        """Dump the history to JSON bytes."""
        assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `dump_json`'
        return self._snapshots_type_adapter.dump_json(self.history, indent=indent)

    def load_json(self, json_data: str | bytes | bytearray) -> None:
        """Load the history from JSON."""
        assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `load_json`'
        self.history = self._snapshots_type_adapter.validate_json(json_data)

    def _prep_state(self, state: StateT) -> StateT:
        """Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
        if not self.deep_copy or state is None:
            return state
        else:
            return copy.deepcopy(state)
