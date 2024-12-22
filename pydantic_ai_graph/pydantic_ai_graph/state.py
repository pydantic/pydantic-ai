from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from typing_extensions import TypeVar

__all__ = 'AbstractState', 'StateT', 'Snapshot'


class AbstractState(ABC):
    """Abstract class for a state object."""

    @abstractmethod
    def serialize(self) -> bytes | None:
        """Serialize the state object."""
        raise NotImplementedError


StateT = TypeVar('StateT', None, AbstractState, default=None)


@dataclass
class Snapshot:
    """Snapshot of a graph."""

    last_node_id: str
    next_node_id: str | None
    start_ts: datetime
    duration: float
    state: bytes | None = None

    @classmethod
    def from_state(
        cls, last_node_id: str, next_node_id: str | None, start_ts: datetime, duration: float, state: StateT
    ) -> Snapshot:
        return cls(
            last_node_id=last_node_id,
            next_node_id=next_node_id,
            start_ts=start_ts,
            duration=duration,
            state=state.serialize() if state is not None else None,
        )
