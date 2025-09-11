from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

from pydantic_graph.v2.id_types import ForkId, NodeId

OutputT = TypeVar('OutputT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)


class StartNode(Generic[OutputT]):
    """A start node."""

    id = ForkId(NodeId('__start__'))


class EndNode(Generic[InputT]):
    """An end node."""

    id = NodeId('__end__')

    def _force_variance(self, inputs: InputT) -> None:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

    # def _force_variance(self) -> InputT:
    #     raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class Fork(Generic[InputT, OutputT]):
    """A fork."""

    id: ForkId

    is_spread: bool  # if is_spread is True, InputT must be Sequence[OutputT]; otherwise InputT must be OutputT

    def _force_variance(self, inputs: InputT) -> OutputT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')
