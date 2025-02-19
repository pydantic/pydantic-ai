import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import pydantic

from ..nodes import BaseNode, End, RunEndT
from . import EndSnapshot, NodeSnapshot, Snapshot, StatePersistence, StateT, _utils


@dataclass
class LatestMemoryStatePersistence(StatePersistence[StateT, RunEndT]):
    deep_copy: bool = True
    last_snapshot: Snapshot[StateT, RunEndT] | None = None

    async def snapshot_node(
        self,
        state: StateT,
        next_node: BaseNode[StateT, Any, RunEndT],
        node_type_adapter: pydantic.TypeAdapter[BaseNode[StateT, Any, RunEndT]],
    ) -> NodeSnapshot[StateT, RunEndT]:
        self.last_snapshot = snapshot = NodeSnapshot(
            state=self.prep_state(state),
            node=next_node.deep_copy() if self.deep_copy else next_node,
        )
        return snapshot

    @asynccontextmanager
    async def record_run(self) -> AsyncIterator[None]:
        last_snapshot = await self.restore_node_snapshot()
        last_snapshot.start_ts = _utils.now_utc()
        start = perf_counter()
        try:
            yield
        finally:
            last_snapshot.duration = perf_counter() - start

    async def snapshot_end(
        self, state: StateT, end: End[RunEndT], end_data_type_adapter: pydantic.TypeAdapter[RunEndT]
    ) -> None:
        self.last_snapshot = EndSnapshot(
            state=self.prep_state(state),
            result=end.deep_copy_data() if self.deep_copy else end.data,
        )

    async def restore(self) -> Snapshot[StateT, RunEndT] | None:
        return self.last_snapshot

    def prep_state(self, state: StateT) -> StateT:
        """Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
        if not self.deep_copy or state is None:
            return state
        else:
            return copy.deepcopy(state)


@dataclass
class HistoryMemoryStatePersistence(StatePersistence[StateT, RunEndT]):
    deep_copy: bool = True
    history: list[Snapshot[StateT, RunEndT]] = field(default_factory=list)

    async def snapshot_node(
        self,
        state: StateT,
        next_node: BaseNode[StateT, Any, RunEndT],
        node_type_adapter: pydantic.TypeAdapter[BaseNode[StateT, Any, RunEndT]],
    ) -> NodeSnapshot[StateT, RunEndT]:
        snapshot = NodeSnapshot(
            state=self.prep_state(state),
            node=next_node.deep_copy() if self.deep_copy else next_node,
        )
        self.history.append(snapshot)
        return snapshot

    @asynccontextmanager
    async def record_run(self) -> AsyncIterator[None]:
        last_snapshot = await self.restore_node_snapshot()
        last_snapshot.start_ts = _utils.now_utc()
        start = perf_counter()
        try:
            yield
        finally:
            last_snapshot.duration = perf_counter() - start

    async def snapshot_end(
        self, state: StateT, end: End[RunEndT], end_data_type_adapter: pydantic.TypeAdapter[RunEndT]
    ) -> None:
        self.history.append(
            EndSnapshot(
                state=self.prep_state(state),
                result=end.deep_copy_data() if self.deep_copy else end.data,
            )
        )

    async def restore(self) -> Snapshot[StateT, RunEndT] | None:
        if self.history:
            return self.history[-1]

    def prep_state(self, state: StateT) -> StateT:
        """Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
        if not self.deep_copy or state is None:
            return state
        else:
            return copy.deepcopy(state)
