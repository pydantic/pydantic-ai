from __future__ import annotations as _annotations

import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, TypeVar

from ..nodes import BaseNode, End, RunEndT
from . import EndSnapshot, NodeSnapshot, Snapshot, StatePersistence, StateT, _utils

S = TypeVar('S')
R = TypeVar('R')


@dataclass
class SimpleStatePersistence(StatePersistence[StateT, RunEndT]):
    """Simple in memory state persistence that just hold the latest snapshot."""

    deep_copy: bool = True
    last_snapshot: Snapshot[StateT, RunEndT] | None = None

    @classmethod
    def from_types(cls, state_type: type[S], run_end_type: type[R]) -> SimpleStatePersistence[S, R]:
        """No-op init method that help type checkers."""
        return SimpleStatePersistence()

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        self.last_snapshot = NodeSnapshot(
            state=self.prep_state(state),
            node=next_node.deep_copy() if self.deep_copy else next_node,
        )

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        self.last_snapshot = EndSnapshot(
            state=self.prep_state(state),
            result=end.deep_copy_data() if self.deep_copy else end,
        )

    @asynccontextmanager
    async def record_run(self) -> AsyncIterator[None]:
        last_snapshot = await self.restore()
        if not isinstance(last_snapshot, NodeSnapshot):
            yield
            return

        last_snapshot.start_ts = _utils.now_utc()
        start = perf_counter()
        try:
            yield
        finally:
            last_snapshot.duration = perf_counter() - start

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

    @classmethod
    def from_types(cls, state_type: type[S], run_end_type: type[R]) -> FullStatePersistence[S, R]:
        """No-op init method that help type checkers."""
        return FullStatePersistence()

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        self.history.append(
            NodeSnapshot(
                state=self.prep_state(state),
                node=next_node.deep_copy() if self.deep_copy else next_node,
            )
        )

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        self.history.append(
            EndSnapshot(
                state=self.prep_state(state),
                result=end.deep_copy_data() if self.deep_copy else end,
            )
        )

    @asynccontextmanager
    async def record_run(self) -> AsyncIterator[None]:
        last_snapshot = await self.restore()
        if not isinstance(last_snapshot, NodeSnapshot):
            yield
            return

        last_snapshot.start_ts = _utils.now_utc()
        start = perf_counter()
        try:
            yield
        finally:
            last_snapshot.duration = perf_counter() - start

    async def restore(self) -> Snapshot[StateT, RunEndT] | None:
        if self.history:
            return self.history[-1]

    def prep_state(self, state: StateT) -> StateT:
        """Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
        if not self.deep_copy or state is None:
            return state
        else:
            return copy.deepcopy(state)
