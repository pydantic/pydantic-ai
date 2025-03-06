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
    NodeRunId,
    NodeSnapshot,
    Snapshot,
    StatePersistence,
    build_snapshot_list_type_adapter,
)

StateT = TypeVar('StateT', default=Any)
RunEndT = TypeVar('RunEndT', default=Any)


@dataclass
class SimpleStatePersistence(StatePersistence[StateT, RunEndT]):
    """Simple in memory state persistence that just hold the latest snapshot."""

    deep_copy: bool = False
    last_snapshot: Snapshot[StateT, RunEndT] | None = None

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> NodeRunId:
        self.last_snapshot = NodeSnapshot(
            state=self.prep_state(state),
            node=next_node.deep_copy() if self.deep_copy else next_node,
        )
        return self.last_snapshot.run_id

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> NodeRunId:
        self.last_snapshot = EndSnapshot(
            state=self.prep_state(state),
            result=end.deep_copy_data() if self.deep_copy else end,
        )
        return self.last_snapshot.run_id

    @asynccontextmanager
    async def record_run(self, run_id: NodeRunId) -> AsyncIterator[None]:
        assert self.last_snapshot is not None, 'No snapshot to record'
        assert isinstance(self.last_snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        assert run_id == self.last_snapshot.run_id, 'run_id must match the last snapshot run_id'
        self.last_snapshot.status = 'running'
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

    def prep_state(self, state: StateT) -> StateT:
        """Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
        if not self.deep_copy or state is None:
            return state
        else:
            return copy.deepcopy(state)


@dataclass
class FullStatePersistence(StatePersistence[StateT, RunEndT]):
    """In memory state persistence that hold a history of nodes that were executed."""

    deep_copy: bool = True
    history: list[Snapshot[StateT, RunEndT]] = field(default_factory=list)
    _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
        default=None, init=False, repr=False
    )

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> NodeRunId:
        last_snapshot = NodeSnapshot(
            state=self.prep_state(state),
            node=next_node.deep_copy() if self.deep_copy else next_node,
        )
        self.history.append(last_snapshot)
        return last_snapshot.run_id

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> NodeRunId:
        end = EndSnapshot(
            state=self.prep_state(state),
            result=end.deep_copy_data() if self.deep_copy else end,
        )
        self.history.append(end)
        return end.run_id

    @asynccontextmanager
    async def record_run(self, run_id: NodeRunId) -> AsyncIterator[None]:
        try:
            snapshot = next(s for s in self.history if s.run_id == run_id)
        except StopIteration as e:
            raise LookupError(f'No snapshot found for run_id {run_id}') from e

        assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        snapshot.status = 'running'
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

    def dump_json(self, *, indent: int | None = None) -> bytes:
        assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `dump_json`'
        return self._snapshots_type_adapter.dump_json(self.history, indent=indent)

    def load_json(self, json_data: str | bytes | bytearray) -> None:
        assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `load_json`'
        self.history = self._snapshots_type_adapter.validate_json(json_data)

    def set_types(self, get_types: Callable[[], tuple[type[StateT], type[RunEndT]]]) -> None:
        if self._snapshots_type_adapter is None:
            state_t, run_end_t = get_types()
            self._snapshots_type_adapter = build_snapshot_list_type_adapter(state_t, run_end_t)

    def prep_state(self, state: StateT) -> StateT:
        """Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
        if not self.deep_copy or state is None:
            return state
        else:
            return copy.deepcopy(state)
