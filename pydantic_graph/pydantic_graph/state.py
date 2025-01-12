from __future__ import annotations as _annotations

import copy
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Generic, Literal, Union

from typing_extensions import TypeVar

from . import _utils

__all__ = 'StateT', 'NodeStep', 'EndStep', 'HistoryStep', 'deep_copy_state'

if TYPE_CHECKING:
    from .nodes import BaseNode, End, RunEndT
else:
    RunEndT = TypeVar('RunEndT', default=None)


StateT = TypeVar('StateT', default=None)
"""Type variable for the state in a graph."""


def deep_copy_state(state: StateT) -> StateT:
    """Default method for snapshotting the state in a graph run, uses [`copy.deepcopy`][copy.deepcopy]."""
    if state is None:
        return state
    else:
        return copy.deepcopy(state)


@dataclass
class NodeStep(Generic[StateT, RunEndT]):
    """History step describing the execution of a node in a graph."""

    state: StateT
    """The state of the graph before the node is run."""
    node: BaseNode[StateT, RunEndT]
    """The node that was run."""
    start_ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the node started running."""
    duration: float | None = None
    """The duration of the node run in seconds."""
    kind: Literal['node'] = 'node'
    """The kind of history step, can be used as a discriminator when deserializing history."""
    snapshot_state: InitVar[Callable[[StateT], StateT]] = deep_copy_state
    """Function to snapshot the state of the graph."""

    def __post_init__(self, snapshot_state: Callable[[StateT], StateT]):
        # Copy the state to prevent it from being modified by other code
        self.state = snapshot_state(self.state)

    def data_snapshot(self) -> BaseNode[StateT, RunEndT]:
        """Returns a deep copy of [`self.node`][pydantic_graph.state.NodeStep.node].

        Useful for summarizing history.
        """
        return copy.deepcopy(self.node)


@dataclass
class EndStep(Generic[StateT, RunEndT]):
    """History step describing the end of a graph run."""

    state: StateT
    """The state of the graph after the run."""
    result: End[RunEndT]
    """The result of the graph run."""
    ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the graph run ended."""
    kind: Literal['end'] = 'end'
    """The kind of history step, can be used as a discriminator when deserializing history."""
    snapshot_state: InitVar[Callable[[StateT], StateT]] = deep_copy_state
    """Function to snapshot the state of the graph."""

    def __post_init__(self, snapshot_state: Callable[[StateT], StateT]):
        # Copy the state to prevent it from being modified by other code
        self.state = snapshot_state(self.state)

    def data_snapshot(self) -> End[RunEndT]:
        """Returns a deep copy of [`self.result`][pydantic_graph.state.EndStep.result].

        Useful for summarizing history.
        """
        return copy.deepcopy(self.result)


HistoryStep = Union[NodeStep[StateT, RunEndT], EndStep[StateT, RunEndT]]
"""A step in the history of a graph run.

[`Graph.run`][pydantic_graph.graph.Graph.run] returns a list of these steps describing the execution of the graph,
together with the run return value.
"""
