"""Step-based graph execution components.

This module provides the core abstractions for step-based graph execution,
including step contexts, step functions, and step nodes that bridge between
the v1 and v2 graph execution systems.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, cast, get_origin, overload

from typing_extensions import TypeVar

from pydantic_graph.beta.id_types import NodeID
from pydantic_graph.nodes import BaseNode, End, GraphRunContext

StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)


@dataclass(init=False)
class StepContext(Generic[StateT, DepsT, InputT]):
    """Context information passed to step functions during graph execution.

    The step context provides access to the current graph state, dependencies, and input data for a step.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
    """

    _state: StateT
    """The current graph state."""
    _deps: DepsT
    """The graph run dependencies."""
    _inputs: InputT
    """The input data for this step."""

    def __init__(self, *, state: StateT, deps: DepsT, inputs: InputT):
        self._state = state
        self._deps = deps
        self._inputs = inputs

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def deps(self) -> DepsT:
        return self._deps

    @property
    def inputs(self) -> InputT:
        """The input data for this step.

        This must be a property to ensure correct variance behavior
        """
        return self._inputs


class StepFunction(Protocol[StateT, DepsT, InputT, OutputT]):
    """Protocol for step functions that can be executed in the graph.

    Step functions are async callables that receive a step context and return a result.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> Awaitable[OutputT]:
        """Execute the step function with the given context.

        Args:
            ctx: The step context containing state, dependencies, and inputs

        Returns:
            An awaitable that resolves to the step's output
        """
        raise NotImplementedError


class StreamFunction(Protocol[StateT, DepsT, InputT, OutputT]):
    """Protocol for stream functions that can be executed in the graph.

    Stream functions are async callables that receive a step context and return an async iterator.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> AsyncIterator[OutputT]:
        """Execute the stream function with the given context.

        Args:
            ctx: The step context containing state, dependencies, and inputs

        Returns:
            An async iterator yielding the streamed output
        """
        raise NotImplementedError
        yield


AnyStepFunction = StepFunction[Any, Any, Any, Any]
"""Type alias for a step function with any type parameters."""


@dataclass(init=False)
class Step(Generic[StateT, DepsT, InputT, OutputT]):
    """A step in the graph execution that wraps a step function.

    Steps represent individual units of execution in the graph, encapsulating
    a step function along with metadata like ID and label.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    id: NodeID
    """Unique identifier for this step."""
    _call: StepFunction[StateT, DepsT, InputT, OutputT]
    """The step function to execute."""
    label: str | None
    """Optional human-readable label for this step."""

    def __init__(self, *, id: NodeID, call: StepFunction[StateT, DepsT, InputT, OutputT], label: str | None = None):
        self.id = id
        self._call = call
        self.label = label

    @property
    def call(self) -> StepFunction[StateT, DepsT, InputT, OutputT]:
        """The step function to execute. This needs to be a property for proper variance inference."""
        return self._call

    @overload
    def as_node(self, inputs: None = None) -> StepNode[StateT, DepsT]: ...

    @overload
    def as_node(self, inputs: InputT) -> StepNode[StateT, DepsT]: ...

    def as_node(self, inputs: InputT | None = None) -> StepNode[StateT, DepsT]:
        """Create a step node with bound inputs.

        Args:
            inputs: The input data to bind to this step, or None

        Returns:
            A [`StepNode`][pydantic_graph.beta.step.StepNode] with this step and the bound inputs
        """
        return StepNode(self, inputs)


@dataclass
class StepNode(BaseNode[StateT, DepsT, Any]):
    """A base node that represents a step with bound inputs.

    StepNode bridges between the v1 and v2 graph execution systems by wrapping
    a [`Step`][pydantic_graph.beta.step.Step] with bound inputs in a BaseNode interface.
    It is not meant to be run directly but rather used to indicate transitions
    to v2-style steps.
    """

    step: Step[StateT, DepsT, Any, Any]
    """The step to execute."""

    inputs: Any
    """The inputs bound to this step."""

    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
        """Attempt to run the step node.

        Args:
            ctx: The graph execution context

        Returns:
            The result of step execution

        Raises:
            NotImplementedError: Always raised as StepNode is not meant to be run directly
        """
        raise NotImplementedError(
            '`StepNode` is not meant to be run directly, it is meant to be used in `BaseNode` subclasses to indicate a transition to v2-style steps.'
        )


# Note: we should make this into a frozen dataclass if https://github.com/python/mypy/issues/17623 gets resolved
# Right now, it cannot be because that breaks variance inference in Python 3.13 due to __replace__
class NodeStep(Step[StateT, DepsT, Any, BaseNode[StateT, DepsT, Any] | End[Any]]):
    """A step that wraps a BaseNode type for execution.

    NodeStep allows v1-style BaseNode classes to be used as steps in the
    v2 graph execution system. It validates that the input is of the expected
    node type and runs it with the appropriate graph context.
    """

    node_type: type[BaseNode[StateT, DepsT, Any]]
    """The BaseNode type this step executes."""

    def __init__(
        self,
        node_type: type[BaseNode[StateT, DepsT, Any]],
        *,
        id: NodeID | None = None,
        label: str | None = None,
    ):
        """Initialize a node step.

        Args:
            node_type: The BaseNode class this step will execute
            id: Optional unique identifier, defaults to the node's get_node_id()
            label: Optional human-readable label for this step
        """
        super().__init__(
            id=id or NodeID(node_type.get_node_id()),
            call=self._call_node,
            label=label,
        )
        # `type[BaseNode[StateT, DepsT, Any]]` could actually be a `typing._GenericAlias` like `pydantic_ai._agent_graph.UserPromptNode[~DepsT, ~OutputT]`,
        # so we get the origin to get to the actual class
        self.node_type = get_origin(node_type) or node_type

    async def _call_node(self, ctx: StepContext[StateT, DepsT, Any]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
        """Execute the wrapped node with the step context.

        Args:
            ctx: The step context containing the node instance to run

        Returns:
            The result of running the node, either another BaseNode or End

        Raises:
            ValueError: If the input node is not of the expected type
        """
        node = ctx.inputs
        if not isinstance(node, self.node_type):
            raise ValueError(f'Node {node} is not of type {self.node_type}')  # pragma: no cover
        node = cast(BaseNode[StateT, DepsT, Any], node)
        return await node.run(GraphRunContext(state=ctx.state, deps=ctx.deps))
