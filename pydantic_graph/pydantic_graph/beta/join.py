"""Join operations and reducers for graph execution.

This module provides the core components for joining parallel execution paths
in a graph, including various reducer types that aggregate data from multiple
sources into a single output.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, cast, overload

from typing_extensions import Protocol, Self, TypeAliasType, TypeVar

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.beta.id_types import ForkID, ForkStack, JoinID

StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)
T = TypeVar('T', infer_variance=True)
K = TypeVar('K', infer_variance=True)
V = TypeVar('V', infer_variance=True)


# TODO(P1): I guess we should make this class private, etc.
@dataclass
class JoinState:
    """The state of a join during graph execution associated to a particular fork run."""

    current: Any
    downstream_fork_stack: ForkStack
    cancelled_sibling_tasks: bool = False


@dataclass(init=False)
class ReducerContext(Generic[StateT, DepsT]):
    """Context information passed to reducer functions during graph execution.

    The reducer context provides access to the current graph state and dependencies.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
    """

    _state: StateT
    """The current graph state."""
    _deps: DepsT
    """The dependencies of the current graph run."""
    _join_state: JoinState
    """The JoinState for this reducer context."""

    def __init__(self, *, state: StateT, deps: DepsT, join_state: JoinState):
        self._state = state
        self._deps = deps
        self._join_state = join_state

    @property
    def state(self) -> StateT:
        """The state of the graph run."""
        return self._state

    @property
    def deps(self) -> DepsT:
        """The deps for the graph run."""
        return self._deps

    def cancel_sibling_tasks(self):
        """Cancel all sibling tasks created from the same fork.

        You can call this if you want your join to have early-stopping behavior.
        """
        self._join_state.cancelled_sibling_tasks = True


PlainReducerFunction = TypeAliasType(
    'PlainReducerFunction',
    Callable[[OutputT, InputT], OutputT],
    type_params=(InputT, OutputT),
)
ContextReducerFunction = TypeAliasType(
    'ContextReducerFunction',
    Callable[[ReducerContext[StateT, DepsT], OutputT, InputT], OutputT],
    type_params=(StateT, DepsT, InputT, OutputT),
)
ReducerFunction = TypeAliasType(
    'ReducerFunction',
    ContextReducerFunction[StateT, DepsT, InputT, OutputT] | PlainReducerFunction[InputT, OutputT],
    type_params=(StateT, DepsT, InputT, OutputT),
)
"""
A function used for reducing inputs to a join node.
"""


def reduce_null(current: None, inputs: Any) -> None:
    """A reducer that discards all input data and returns None."""
    return None


def reduce_list_append(current: list[T], inputs: T) -> list[T]:
    """A reducer that appends to a list."""
    current.append(inputs)
    return current


def reduce_list_extend(current: list[T], inputs: Iterable[T]) -> list[T]:
    """A reducer that extends a list."""
    current.extend(inputs)
    return current


def reduce_dict_update(current: dict[K, V], inputs: Mapping[K, V]) -> dict[K, V]:
    """A reducer that updates a dict."""
    current.update(inputs)
    return current


class SupportsSum(Protocol):
    """A protocol for a type that supports adding to itself."""

    @abstractmethod
    def __add__(self, other: Self, /) -> Self:
        pass


NumericT = TypeVar('NumericT', bound=SupportsSum, infer_variance=True)


def reduce_sum(current: NumericT, inputs: NumericT) -> NumericT:
    """A reducer that sums numbers."""
    return current + inputs


@dataclass
class ReduceFirstValue(Generic[T]):
    """A reducer that returns the first value it encounters, and cancels all other tasks."""

    def __call__(self, ctx: ReducerContext[object, object], current: T, inputs: T) -> T:
        """The reducer function."""
        ctx.cancel_sibling_tasks()
        return inputs


@dataclass(init=False)
class Join(Generic[StateT, DepsT, InputT, OutputT]):
    """A join operation that synchronizes and aggregates parallel execution paths.

    A join defines how to combine outputs from multiple parallel execution paths
    using a [`ReducerFunction`][pydantic_graph.beta.join.ReducerFunction]. It specifies which fork
    it joins (if any) and manages the initialization of reducers.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of input data to join
        OutputT: The type of the final joined output
    """

    id: JoinID
    _reducer: ReducerFunction[StateT, DepsT, InputT, OutputT]
    _initial_factory: Callable[[], OutputT]
    parent_fork_id: ForkID | None
    preferred_parent_fork: Literal['closest', 'farthest']

    def __init__(
        self,
        *,
        id: JoinID,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        initial_factory: Callable[[], OutputT],
        parent_fork_id: ForkID | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ):
        self.id = id
        self._reducer = reducer
        self._initial_factory = initial_factory
        self.parent_fork_id = parent_fork_id
        self.preferred_parent_fork = preferred_parent_fork

    @property
    def reducer(self):
        return self._reducer

    @property
    def initial_factory(self):
        return self._initial_factory

    def reduce(self, ctx: ReducerContext[StateT, DepsT], current: OutputT, inputs: InputT) -> OutputT:
        n_parameters = len(inspect.signature(self.reducer).parameters)
        if n_parameters == 2:
            return cast(PlainReducerFunction[InputT, OutputT], self.reducer)(current, inputs)
        else:
            return cast(ContextReducerFunction[StateT, DepsT, InputT, OutputT], self.reducer)(ctx, current, inputs)

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
