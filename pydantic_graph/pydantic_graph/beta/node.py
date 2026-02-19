"""Core node types for graph construction and execution.

This module defines the fundamental node types used to build execution graphs,
including start/end nodes and fork nodes for parallel execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

from pydantic_graph.beta.id_types import ForkID, JoinID, NodeID

StateT = TypeVar('StateT', infer_variance=True)
"""Type variable for graph state."""

OutputT = TypeVar('OutputT', infer_variance=True)
"""Type variable for node output data."""

InputT = TypeVar('InputT', infer_variance=True)
"""Type variable for node input data."""


class StartNode(Generic[OutputT]):
    """Entry point node for graph execution.

    The StartNode represents the beginning of a graph execution flow.
    """

    id = NodeID('__start__')
    """Fixed identifier for the start node."""


class EndNode(Generic[InputT]):
    """Terminal node representing the completion of graph execution.

    The EndNode marks the successful completion of a graph execution flow
    and can collect the final output data.
    """

    id = NodeID('__end__')
    """Fixed identifier for the end node."""

    def _force_variance(self, inputs: InputT) -> None:  # pragma: no cover
        """Force type variance for proper generic typing.

        This method exists solely for type checking purposes and should never be called.

        Args:
            inputs: Input data of type InputT.

        Raises:
            RuntimeError: Always, as this method should never be executed.
        """
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class Fork(Generic[InputT, OutputT]):
    """Fork node that creates parallel execution branches.

    A Fork node splits the execution flow into multiple parallel branches,
    enabling concurrent execution of downstream nodes. It can either map
    a sequence across multiple branches or duplicate data to each branch.
    """

    id: ForkID
    """Unique identifier for this fork node."""

    is_map: bool
    """Determines fork behavior.

    If True, InputT must be Sequence[OutputT] and each element is sent to a separate branch.
    If False, InputT must be OutputT and the same data is sent to all branches.
    """
    downstream_join_id: JoinID | None
    """Optional identifier of a downstream join node that should be jumped to if mapping an empty iterable."""

    def _force_variance(self, inputs: InputT) -> OutputT:  # pragma: no cover
        """Force type variance for proper generic typing.

        This method exists solely for type checking purposes and should never be called.

        Args:
            inputs: Input data to be forked.

        Returns:
            Output data type (never actually returned).

        Raises:
            RuntimeError: Always, as this method should never be executed.
        """
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')
