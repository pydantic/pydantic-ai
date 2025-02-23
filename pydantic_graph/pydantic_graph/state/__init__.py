from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Callable, Generic, Literal, Union

import pydantic
from typing_extensions import TypeVar

from .. import exceptions
from ..nodes import BaseNode, End, RunEndT
from . import _utils

__all__ = 'StateT', 'NodeSnapshot', 'EndSnapshot', 'Snapshot', 'StatePersistence', 'build_nodes_type_adapter'

StateT = TypeVar('StateT', default=None)
"""Type variable for the state in a graph."""


@dataclass
class NodeSnapshot(Generic[StateT, RunEndT]):
    """History step describing the execution of a node in a graph."""

    state: StateT
    """The state of the graph before the node is run."""
    node: BaseNode[StateT, Any, RunEndT]
    """The node to run next."""
    start_ts: datetime | None = None
    """The timestamp when the node started running, `None` until the run starts."""
    duration: float | None = None
    """The duration of the node run in seconds, if the node has been run."""
    kind: Literal['node'] = 'node'
    """The kind of history step, can be used as a discriminator when deserializing history."""


@dataclass
class EndSnapshot(Generic[StateT, RunEndT]):
    """History step describing the end of a graph run."""

    state: StateT
    """The state of the graph at the end of the run."""
    result: End[RunEndT]
    """The result of the graph run."""
    ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the graph run ended."""
    kind: Literal['end'] = 'end'
    """The kind of history step, can be used as a discriminator when deserializing history."""

    @property
    def node(self) -> End[RunEndT]:
        """Shim to get the [`result`][pydantic_graph.state.EndSnapshot.result].

        Useful to allow `[snapshot.node for snapshot in persistence.history]`.
        """
        return self.result


Snapshot = Union[NodeSnapshot[StateT, RunEndT], EndSnapshot[StateT, RunEndT]]
"""A step in the history of a graph run.

[`Graph.run`][pydantic_graph.graph.Graph.run] returns a list of these steps describing the execution of the graph,
together with the run return value.
"""


class StatePersistence(ABC, Generic[StateT, RunEndT]):
    """Abstract base class for storing the state of a graph."""

    @abstractmethod
    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        """Snapshot the state of a graph before a node is run.

        Args:
            state: The state of the graph.
            next_node: The next node to run or end if the graph has ended

        Returns:
            The snapshot
        """
        raise NotImplementedError

    @abstractmethod
    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        """Snapshot the state of a graph before a node is run.

        Args:
            state: The state of the graph.
            end: data from the end of the run.
        """
        raise NotImplementedError

    @abstractmethod
    @asynccontextmanager
    async def record_run(self) -> AsyncIterator[None]:
        """Record the run of the node.

        In particular this should set [`NodeSnapshot.start_ts`][pydantic_graph.state.NodeSnapshot.start_ts]
        and [`NodeSnapshot.duration`][pydantic_graph.state.NodeSnapshot.duration].
        """
        yield
        raise NotImplementedError

    @abstractmethod
    async def restore(self) -> Snapshot[StateT, RunEndT] | None:
        """Retrieve the latest snapshot.

        Returns:
            The most recent [`Snapshot`][pydantic_graph.state.Snapshot] of the run.
        """
        raise NotImplementedError

    def set_type_adapters(
        self,
        *,
        get_node_type_adapter: Callable[[], pydantic.TypeAdapter[BaseNode[StateT, Any, RunEndT]]],
        get_end_data_type_adapter: Callable[[], pydantic.TypeAdapter[RunEndT]],
        get_snapshot_type_adapter: Callable[[], pydantic.TypeAdapter[Snapshot[StateT, RunEndT]]],
    ):
        pass

    async def restore_node_snapshot(self) -> NodeSnapshot[StateT, RunEndT]:
        snapshot = await self.restore()
        if snapshot is None:
            raise exceptions.GraphRuntimeError('Unable to restore snapshot from state persistence.')
        elif not isinstance(snapshot, NodeSnapshot):
            raise exceptions.GraphRuntimeError('Snapshot returned from persistence indicates the graph has ended.')
        return snapshot


def build_nodes_type_adapter(  # noqa: D103
    nodes: Sequence[type[BaseNode[Any, Any, Any]]], state_t: type[StateT], end_t: type[RunEndT]
) -> pydantic.TypeAdapter[BaseNode[StateT, Any, RunEndT]]:
    return pydantic.TypeAdapter(
        Annotated[BaseNode[state_t, Any, end_t], _utils.CustomNodeSchema(nodes)],
        config=pydantic.ConfigDict(defer_build=True),
    )
