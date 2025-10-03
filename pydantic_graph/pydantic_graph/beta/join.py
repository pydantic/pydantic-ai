"""Join operations and reducers for graph execution.

This module provides the core components for joining parallel execution paths
in a graph, including various reducer types that aggregate data from multiple
sources into a single output.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Generic, overload

from typing_extensions import TypeVar

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.beta.id_types import ForkId, JoinId
from pydantic_graph.beta.step import StepContext

StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)
T = TypeVar('T', infer_variance=True)
K = TypeVar('K', infer_variance=True)
V = TypeVar('V', infer_variance=True)


@dataclass(kw_only=True)
class Reducer(ABC, Generic[StateT, DepsT, InputT, OutputT]):
    """An abstract base class for reducing data from parallel execution paths.

    Reducers accumulate input data from multiple sources and produce a single
    output when finalized. This is the core mechanism for joining parallel
    execution paths in the graph.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of input data to reduce
        OutputT: The type of the final output after reduction
    """

    def reduce(self, ctx: StepContext[StateT, DepsT, InputT]) -> None:
        """Accumulate input data from a step context into the reducer's internal state.

        This method is called for each input that needs to be reduced. Subclasses
        should override this method to implement their specific reduction logic.

        Args:
            ctx: The step context containing input data to reduce
        """
        pass

    def finalize(self, ctx: StepContext[StateT, DepsT, None]) -> OutputT:
        """Finalize the reduction and return the aggregated output.

        This method is called after all inputs have been reduced to produce
        the final output value.

        Args:
            ctx: The step context for finalization (no input data)

        Returns:
            The final aggregated output from all reduced inputs

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError('Finalize method must be implemented in subclasses.')


@dataclass(kw_only=True)
class NullReducer(Reducer[object, object, object, None]):
    """A reducer that discards all input data and returns None.

    This reducer is useful when you need to join parallel execution paths
    but don't care about collecting their outputs - only about synchronizing
    their completion.
    """

    def finalize(self, ctx: StepContext[object, object, object]) -> None:
        """Return None, ignoring all accumulated inputs.

        Args:
            ctx: The step context for finalization

        Returns:
            Always returns None
        """
        return None


@dataclass(kw_only=True)
class ListReducer(Reducer[object, object, T, list[T]], Generic[T]):
    """A reducer that collects all input values into a list.

    This reducer accumulates each input value in order and returns them
    as a list when finalized.

    Type Parameters:
        T: The type of elements in the resulting list
    """

    items: list[T] = field(default_factory=list)
    """The accumulated list of input items."""

    def reduce(self, ctx: StepContext[object, object, T]) -> None:
        """Append the input value to the list of items.

        Args:
            ctx: The step context containing the input value to append
        """
        self.items.append(ctx.inputs)

    def finalize(self, ctx: StepContext[object, object, None]) -> list[T]:
        """Return the accumulated list of items.

        Args:
            ctx: The step context for finalization

        Returns:
            A list containing all accumulated input values in order
        """
        return self.items


@dataclass(kw_only=True)
class DictReducer(Reducer[object, object, dict[K, V], dict[K, V]], Generic[K, V]):
    """A reducer that merges dictionary inputs into a single dictionary.

    This reducer accumulates dictionary inputs by merging them together,
    with later inputs overriding earlier ones for duplicate keys.

    Type Parameters:
        K: The type of dictionary keys
        V: The type of dictionary values
    """

    data: dict[K, V] = field(default_factory=dict)
    """The accumulated dictionary data."""

    def reduce(self, ctx: StepContext[object, object, dict[K, V]]) -> None:
        """Merge the input dictionary into the accumulated data.

        Args:
            ctx: The step context containing the dictionary to merge
        """
        self.data.update(ctx.inputs)

    def finalize(self, ctx: StepContext[object, object, None]) -> dict[K, V]:
        """Return the accumulated merged dictionary.

        Args:
            ctx: The step context for finalization

        Returns:
            A dictionary containing all merged key-value pairs
        """
        return self.data


@dataclass(kw_only=True)
class EarlyStoppingReducer(Reducer[object, object, T, T | None], Generic[T]):
    """A reducer that returns the first encountered value and cancels all other tasks started by its parent fork.

    Type Parameters:
        T: The type of elements in the resulting list
    """

    result: T | None = None

    def reduce(self, ctx: StepContext[object, object, T]) -> None:
        """Append the input value to the list of items.

        Args:
            ctx: The step context containing the input value to append
        """
        self.result = ctx.inputs
        raise StopIteration

    def finalize(self, ctx: StepContext[object, object, None]) -> T | None:
        """Return the accumulated list of items.

        Args:
            ctx: The step context for finalization

        Returns:
            A list containing all accumulated input values in order
        """
        return self.result


class Join(Generic[StateT, DepsT, InputT, OutputT]):
    """A join operation that synchronizes and aggregates parallel execution paths.

    A join defines how to combine outputs from multiple parallel execution paths
    using a [`Reducer`][pydantic_graph.beta.join.Reducer]. It specifies which fork
    it joins (if any) and manages the creation of reducer instances.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of input data to join
        OutputT: The type of the final joined output
    """

    def __init__(
        self, id: JoinId, reducer_type: type[Reducer[StateT, DepsT, InputT, OutputT]], joins: ForkId | None = None
    ) -> None:
        """Initialize a join operation.

        Args:
            id: Unique identifier for this join
            reducer_type: The type of reducer to use for aggregating inputs
            joins: The fork ID this join synchronizes with, if any
        """
        self.id = id
        """Unique identifier for this join operation."""

        self._reducer_type = reducer_type
        """The reducer type used to aggregate inputs."""

        self.joins = joins
        """The fork ID this join synchronizes with, if any."""

        # self._type_adapter: TypeAdapter[Any] = TypeAdapter(reducer_type)  # needs to be annotated this way for variance

    def create_reducer(self) -> Reducer[StateT, DepsT, InputT, OutputT]:
        """Create a reducer instance for this join operation.

        Returns:
            A new reducer instance initialized with the provided context
        """
        return self._reducer_type()

    # TODO(P3): If we want the ability to snapshot graph-run state, we'll need a way to
    #  serialize/deserialize the associated reducers, something like this:
    # def serialize_reducer(self, instance: Reducer[Any, Any, Any]) -> bytes:
    #     return to_json(instance)
    #
    # def deserialize_reducer(self, serialized: bytes) -> Reducer[InputT, OutputT]:
    #     return self._type_adapter.validate_json(serialized)

    def _force_covariant(self, inputs: InputT) -> OutputT:
        """Force covariant typing for generic parameters.

        This method exists solely for typing purposes and should never be called.

        Args:
            inputs: Input value for typing purposes only

        Returns:
            Output value for typing purposes only

        Raises:
            RuntimeError: Always raised as this method should never be called
        """
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

    @overload
    def as_node(self, inputs: None = None) -> JoinNode[StateT, DepsT]: ...

    @overload
    def as_node(self, inputs: InputT) -> JoinNode[StateT, DepsT]: ...

    def as_node(self, inputs: InputT | None = None) -> JoinNode[StateT, DepsT]:
        """Create a step node with bound inputs.

        Args:
            inputs: The input data to bind to this step, or None

        Returns:
            A [`StepNode`][pydantic_graph.beta.step.StepNode] with this step and the bound inputs
        """
        return JoinNode(self, inputs)


@dataclass
class JoinNode(BaseNode[StateT, DepsT, Any]):
    """A base node that represents a join item with bound inputs.

    JoinNode bridges between the v1 and v2 graph execution systems by wrapping
    a [`Join`][pydantic_graph.beta.join.Join] with bound inputs in a BaseNode interface.
    It is not meant to be run directly but rather used to indicate transitions
    to v2-style steps.
    """

    join: Join[StateT, DepsT, Any, Any]
    """The step to execute."""

    inputs: Any
    """The inputs bound to this step."""

    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
        """Attempt to run the join node.

        Args:
            ctx: The graph execution context

        Returns:
            The result of step execution

        Raises:
            NotImplementedError: Always raised as StepNode is not meant to be run directly
        """
        raise NotImplementedError(
            '`JoinNode` is not meant to be run directly, it is meant to be used in `BaseNode` subclasses to indicate a transition to v2-style steps.'
        )
