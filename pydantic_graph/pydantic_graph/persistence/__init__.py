from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Callable, Generic, Literal, NewType, Union
from uuid import uuid4

import pydantic
from typing_extensions import TypeVar

from .. import exceptions
from ..nodes import BaseNode, End, RunEndT
from . import _utils

__all__ = 'StateT', 'NodeRunId', 'NodeSnapshot', 'EndSnapshot', 'Snapshot', 'StatePersistence', 'set_nodes_type_context'

StateT = TypeVar('StateT', default=None)
"""Type variable for the state in a graph."""
NodeRunId = NewType('NodeRunId', str)
"""Unique ID for a node run."""


def new_run_id() -> NodeRunId:
    return NodeRunId(uuid4().hex)


@dataclass
class NodeSnapshot(Generic[StateT, RunEndT]):
    """History step describing the execution of a node in a graph."""

    state: StateT
    """The state of the graph before the node is run."""
    node: Annotated[BaseNode[StateT, Any, RunEndT], _utils.CustomNodeSchema()]
    """The node to run next."""
    run_id: NodeRunId = field(default_factory=new_run_id)
    """Unique ID of the node run."""
    start_ts: datetime | None = None
    """The timestamp when the node started running, `None` until the run starts."""
    duration: float | None = None
    """The duration of the node run in seconds, if the node has been run."""
    status: Literal['not_started', 'pending', 'running', 'success', 'error'] = 'not_started'
    kind: Literal['node'] = 'node'
    """The kind of history step, can be used as a discriminator when deserializing history."""


@dataclass
class EndSnapshot(Generic[StateT, RunEndT]):
    """History step describing the end of a graph run."""

    state: StateT
    """The state of the graph at the end of the run."""
    result: End[RunEndT]
    """The result of the graph run."""
    run_id: NodeRunId = field(default_factory=new_run_id)
    """Unique ID for the end of the graph run."""
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
    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> NodeRunId:
        """Snapshot the state of a graph, when the next step is to run a node.

        In particular this should set [`NodeSnapshot.duration`][pydantic_graph.state.NodeSnapshot.duration]
        when the run finishes.

        Note: although the node

        Args:
            state: The state of the graph.
            next_node: The next node to run or end if the graph has ended

        Returns: an async context manager that wraps the run of the node.
        """
        raise NotImplementedError

    @abstractmethod
    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> NodeRunId:
        """Snapshot the state of a graph after a node has run, when the graph has ended.

        Args:
            state: The state of the graph.
            end: data from the end of the run.
        """
        raise NotImplementedError

    @abstractmethod
    def record_run(self, run_id: NodeRunId) -> AbstractAsyncContextManager[None]:
        """Record the run of the node.

        In particular this should set:

        - [`NodeSnapshot.status`][pydantic_graph.state.NodeSnapshot.status] to `'running'` and
          [`NodeSnapshot.start_ts`][pydantic_graph.state.NodeSnapshot.start_ts] when the run starts.
        - [`NodeSnapshot.status`][pydantic_graph.state.NodeSnapshot.status] to `'success'` or `'error'` and
          [`NodeSnapshot.duration`][pydantic_graph.state.NodeSnapshot.duration] when the run finishes.
        """
        raise NotImplementedError

    @abstractmethod
    async def restore(self) -> Snapshot[StateT, RunEndT] | None:
        """Retrieve the latest snapshot.

        Returns:
            The most recent [`Snapshot`][pydantic_graph.state.Snapshot] of the run.
        """
        raise NotImplementedError

    def set_types(self, get_types: Callable[[], tuple[type[StateT], type[RunEndT]]]) -> None:
        pass

    async def restore_node_snapshot(self) -> NodeSnapshot[StateT, RunEndT]:
        snapshot = await self.restore()
        if snapshot is None:
            raise exceptions.GraphRuntimeError('Unable to restore snapshot from state persistence.')
        elif not isinstance(snapshot, NodeSnapshot):
            raise exceptions.GraphRuntimeError('Snapshot returned from persistence indicates the graph has ended.')
        return snapshot


@contextmanager
def set_nodes_type_context(nodes: Sequence[type[BaseNode[Any, Any, Any]]]) -> Iterator[None]:  # noqa: D103
    token = _utils.nodes_type_context.set(nodes)
    try:
        yield
    finally:
        _utils.nodes_type_context.reset(token)


def build_snapshot_list_type_adapter(
    state_t: type[StateT], run_end_t: type[RunEndT]
) -> pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]]:
    return pydantic.TypeAdapter(list[Annotated[Snapshot[state_t, run_end_t], pydantic.Discriminator('kind')]])


def build_snapshot_single_type_adapter(
    state_t: type[StateT], run_end_t: type[RunEndT]
) -> pydantic.TypeAdapter[Snapshot[StateT, RunEndT]]:
    return pydantic.TypeAdapter(Annotated[Snapshot[state_t, run_end_t], pydantic.Discriminator('kind')])
