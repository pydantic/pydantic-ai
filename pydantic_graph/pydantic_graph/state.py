from __future__ import annotations as _annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Generic, Literal, Union

from typing_extensions import Self, TypeVar

from . import _utils

__all__ = 'AbstractState', 'StateT', 'NodeStep', 'EndStep', 'HistoryStep'

if TYPE_CHECKING:
    from .nodes import BaseNode, End, RunEndT
else:
    RunEndT = TypeVar('RunEndT', default=None)


class AbstractState(ABC):
    """Abstract class for a state object."""

    @abstractmethod
    def serialize(self) -> bytes | None:
        """Serialize the state object."""
        raise NotImplementedError

    def deep_copy(self) -> Self:
        """Create a deep copy of the state object."""
        return copy.deepcopy(self)


StateT = TypeVar('StateT', bound=Union[None, AbstractState], default=None)
"""Type variable for the state in a graph."""


@dataclass
class NodeStep(Generic[StateT, RunEndT]):
    """History step describing the execution of a node in a graph."""

    state: StateT
    """The state of the graph after the node has run."""
    node: BaseNode[StateT, RunEndT]
    """The node that was run."""
    start_ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the node started running."""
    duration: float | None = None
    """The duration of the node run in seconds."""
    kind: Literal['node'] = 'node'
    """The kind of history step, can be used as a discriminator when deserializing history."""

    def __post_init__(self):
        # Copy the state to prevent it from being modified by other code
        self.state = _deep_copy_state(self.state)

    def summary(self) -> str:
        return str(self.node)


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

    def __post_init__(self):
        # Copy the state to prevent it from being modified by other code
        self.state = _deep_copy_state(self.state)

    def summary(self) -> str:
        return str(self.result)


def _deep_copy_state(state: StateT) -> StateT:
    if state is None:
        return state
    else:
        return state.deep_copy()


HistoryStep = Union[NodeStep[StateT, RunEndT], EndStep[StateT, RunEndT]]
"""A step in the history of a graph run."""
