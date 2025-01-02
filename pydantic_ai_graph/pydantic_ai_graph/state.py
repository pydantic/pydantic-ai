from __future__ import annotations as _annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Generic, Literal, Union

from typing_extensions import Never, TypeVar

__all__ = 'AbstractState', 'StateT', 'Step', 'EndEvent', 'StepOrEnd'

if TYPE_CHECKING:
    from pydantic_ai_graph import BaseNode
    from pydantic_ai_graph.nodes import End


class AbstractState(ABC):
    """Abstract class for a state object."""

    @abstractmethod
    def serialize(self) -> bytes | None:
        """Serialize the state object."""
        raise NotImplementedError


RunEndT = TypeVar('RunEndT', default=None)
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)
StateT = TypeVar('StateT', bound=Union[None, AbstractState], default=None)


@dataclass
class Step(Generic[StateT, RunEndT]):
    """History item describing the execution of a step of a graph."""

    state: StateT
    node: BaseNode[StateT, RunEndT]
    start_ts: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    duration: float | None = None

    kind: Literal['start_step'] = 'start_step'

    def __post_init__(self):
        # Copy the state to prevent it from being modified by other code
        self.state = copy.deepcopy(self.state)


@dataclass
class EndEvent(Generic[StateT, RunEndT]):
    """History item describing the end of a graph run."""

    state: StateT
    result: End[RunEndT]
    ts: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    kind: Literal['end'] = 'end'

    def __post_init__(self):
        # Copy the state to prevent it from being modified by other code
        self.state = copy.deepcopy(self.state)


StepOrEnd = Union[Step[StateT, RunEndT], EndEvent[StateT, RunEndT]]
