from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, Union

import pydantic
from typing_extensions import TypeVar

from ..nodes import BaseNode, End
from . import _utils

if TYPE_CHECKING:
    from .. import Graph

__all__ = (
    'StateT',
    'NodeSnapshot',
    'EndSnapshot',
    'Snapshot',
    'BaseStatePersistence',
    'SnapshotStatus',
    'build_snapshot_list_type_adapter',
)

StateT = TypeVar('StateT', default=Any)
RunEndT = TypeVar('RunEndT', default=Any)

UNSET_SNAPSHOT_ID = '__unset__'
SnapshotStatus = Literal['created', 'pending', 'running', 'success', 'error']
"""The status of a snapshot.

- `'created'`: The snapshot has been created but not yet run.
- `'pending'`: The snapshot has been retrieved with
  [`load_next`][pydantic_graph.persistence.BaseStatePersistence.load_next] but not yet run.
- `'running'`: The snapshot is currently running.
- `'success'`: The snapshot has been run successfully.
- `'error'`: The snapshot has been run but an error occurred.
"""


@dataclass
class NodeSnapshot(Generic[StateT, RunEndT]):
    """History step describing the execution of a node in a graph."""

    state: StateT
    """The state of the graph before the node is run."""
    node: Annotated[BaseNode[StateT, Any, RunEndT], _utils.CustomNodeSchema()]
    """The node to run next."""
    start_ts: datetime | None = None
    """The timestamp when the node started running, `None` until the run starts."""
    duration: float | None = None
    """The duration of the node run in seconds, if the node has been run."""
    status: SnapshotStatus = 'created'
    """The status of the snapshot."""
    kind: Literal['node'] = 'node'
    """The kind of history step, can be used as a discriminator when deserializing history."""

    id: str = UNSET_SNAPSHOT_ID
    """Unique ID of the snapshot."""

    def __post_init__(self) -> None:
        if self.id == UNSET_SNAPSHOT_ID:
            self.id = self.node.get_snapshot_id()


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

    id: str = UNSET_SNAPSHOT_ID
    """Unique ID of the snapshot."""

    def __post_init__(self) -> None:
        if self.id == UNSET_SNAPSHOT_ID:
            self.id = self.node.get_snapshot_id()

    @property
    def node(self) -> End[RunEndT]:
        """Shim to get the [`result`][pydantic_graph.persistence.EndSnapshot.result].

        Useful to allow `[snapshot.node for snapshot in persistence.history]`.
        """
        return self.result


Snapshot = Union[NodeSnapshot[StateT, RunEndT], EndSnapshot[StateT, RunEndT]]
"""A step in the history of a graph run.

[`Graph.run`][pydantic_graph.graph.Graph.run] returns a list of these steps describing the execution of the graph,
together with the run return value.
"""


class BaseStatePersistence(ABC, Generic[StateT, RunEndT]):
    """Abstract base class for storing the state of a graph run.

    Each instance of a `BaseStatePersistence` subclass should be used for a single graph run.
    """

    @abstractmethod
    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        """Snapshot the state of a graph, when the next step is to run a node.

        This method should add a [`NodeSnapshot`][pydantic_graph.persistence.NodeSnapshot] to persistence.

        Args:
            state: The state of the graph.
            next_node: The next node to run.
        """
        raise NotImplementedError

    @abstractmethod
    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        """Snapshot the state of a graph if the snapshot ID doesn't already exist in persistence.

        This method will generally call [`snapshot_node`][pydantic_graph.persistence.BaseStatePersistence.snapshot_node]
        but should do so in an atomic way.

        Args:
            snapshot_id: The ID of the snapshot to check.
            state: The state of the graph.
            next_node: The next node to run.
        """
        raise NotImplementedError

    @abstractmethod
    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        """Snapshot the state of a graph when the graph has ended.

        This method should add an [`EndSnapshot`][pydantic_graph.persistence.EndSnapshot] to persistence.

        Args:
            state: The state of the graph.
            end: data from the end of the run.
        """
        raise NotImplementedError

    @abstractmethod
    def record_run(self, snapshot_id: str) -> AbstractAsyncContextManager[None]:
        """Record the run of the node, or error if the node is already running.

        Args:
            snapshot_id: The ID of the snapshot to record.

        Raises:
            GraphNodeRunningError: if the node status it not `'created'` or `'pending'`.
            LookupError: if the snapshot ID is not found in persistence.

        Returns:
            An async context manager that records the run of the node.

        In particular this should set:

        - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'running'` and
          [`NodeSnapshot.start_ts`][pydantic_graph.persistence.NodeSnapshot.start_ts] when the run starts.
        - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'success'` or `'error'` and
          [`NodeSnapshot.duration`][pydantic_graph.persistence.NodeSnapshot.duration] when the run finishes.
        """
        raise NotImplementedError

    @abstractmethod
    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        """Retrieve a node snapshot with status `'created`' and set its status to `'pending'`.

        This is used by [`Graph.iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence]
        to get the next node to run.

        Returns: The snapshot, or `None` if no snapshot with status `'created`' exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        """Load the entire history of snapshots.

        `load_all` is not used by pydantic-graph itself, instead it's provided to make it convenient to
        get all [snapshots][pydantic_graph.persistence.Snapshot] from persistence.

        Returns: The list of snapshots.
        """
        raise NotImplementedError

    def set_graph_types(self, graph: Graph[StateT, Any, RunEndT]) -> None:
        """Set the types of the state and run end from a graph.

        You generally won't need to customise this method, instead implement
        [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types] and
        [`should_set_types`][pydantic_graph.persistence.BaseStatePersistence.should_set_types].
        """
        if self.should_set_types():
            with _utils.set_nodes_type_context(graph.get_nodes()):
                self.set_types(*graph.inferred_types)

    def should_set_types(self) -> bool:
        """Whether types need to be set.

        Implementations should override this method to return `True` when types have not been set if they are needed.
        """
        return False

    def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
        """Set the types of the state and run end.

        This can be used to create [type adapters][pydantic.TypeAdapter] for serializing and deserializing snapshots,
        e.g. with [`build_snapshot_list_type_adapter`][pydantic_graph.persistence.build_snapshot_list_type_adapter].

        Args:
            state_type: The state type.
            run_end_type: The run end type.
        """
        pass


def build_snapshot_list_type_adapter(
    state_t: type[StateT], run_end_t: type[RunEndT]
) -> pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]]:
    """Build a type adapter for a list of snapshots.

    This method should be called from within
    [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types]
    where context variables will be set such that Pydantic can create a schema for
    [`NodeSnapshot.node`][pydantic_graph.persistence.NodeSnapshot.node].
    """
    return pydantic.TypeAdapter(list[Annotated[Snapshot[state_t, run_end_t], pydantic.Discriminator('kind')]])
