"""Step-based graph execution components.

This module provides the core abstractions for step-based graph execution,
including step contexts, step functions, and step nodes that bridge between
the v1 and v2 graph execution systems.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, cast, get_origin, overload

from typing_extensions import TypeVar

from pydantic_graph.beta.id_types import NodeID
from pydantic_graph.nodes import BaseNode, End, GraphRunContext

StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)


class StepContext(Generic[StateT, DepsT, InputT]):
    """Context information passed to step functions during graph execution.

    The step context provides access to the current graph state, dependencies,
    and input data for a step. This class uses manual property definitions
    instead of dataclass to maintain proper type variance.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
    """

    if TYPE_CHECKING:

        def __init__(self, state: StateT, deps: DepsT, inputs: InputT):
            self._state = state
            self._deps = deps
            self._inputs = inputs

        @property
        def state(self) -> StateT:
            """The current graph state."""
            return self._state

        @property
        def deps(self) -> DepsT:
            """The dependencies available to this step."""
            return self._deps

        @property
        def inputs(self) -> InputT:
            """The input data for this step."""
            return self._inputs
    else:
        state: StateT
        """The current graph state."""

        deps: DepsT
        """The dependencies available to this step."""

        inputs: InputT
        """The input data for this step."""

    def __repr__(self) -> str:
        """Return a string representation of the step context.

        Returns:
            A string showing the class name and inputs
        """
        return f'{self.__class__.__name__}(inputs={self.inputs})'


if not TYPE_CHECKING:
    # TODO: Try dropping inputs from StepContext, it would make for fewer generic params, could help
    StepContext = dataclass(StepContext)


class StepFunction(Protocol[StateT, DepsT, InputT, OutputT]):
    """Protocol for step functions that can be executed in the graph.

    Step functions are async callables that receive a step context and return
    a result. This protocol enables serialization and deserialization of step
    calls similar to how evaluators work.

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


AnyStepFunction = StepFunction[Any, Any, Any, Any]
"""Type alias for a step function with any type parameters."""


class Step(Generic[StateT, DepsT, InputT, OutputT]):
    """A step in the graph execution that wraps a step function.

    Steps represent individual units of execution in the graph, encapsulating
    a step function along with metadata like ID and label. This class uses
    manual initialization instead of dataclass to maintain proper type variance.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    def __init__(
        self,
        id: NodeID,
        call: StepFunction[StateT, DepsT, InputT, OutputT],
        user_label: str | None = None,
    ):
        """Initialize a step.

        Args:
            id: Unique identifier for this step
            call: The step function to execute
            user_label: Optional human-readable label for this step
        """
        self.id = id
        """Unique identifier for this step."""

        self._call = call
        """The step function to execute."""

        self.user_label = user_label
        """Optional human-readable label for this step."""

    async def call(self, ctx: StepContext[StateT, DepsT, InputT]) -> OutputT:
        """The step function to execute.

        Returns:
            The wrapped step function
        """
        result = self._call(ctx)
        if inspect.isawaitable(result):
            return await result
        return result

    # TODO(P3): Consider adding a `bind` method that returns an object that can be used to get something you can return from a BaseNode that allows you to transition to nodes using "new"-form edges

    @property
    def label(self) -> str | None:
        """The human-readable label for this step.

        Returns:
            The user-provided label, or None if no label was set
        """
        return self.user_label

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

    def __repr__(self) -> str:
        """Return a string representation of the step context.

        Returns:
            A string showing the class name and inputs
        """
        return f'Step(id={self.id!r}, call={self._call!r}, user_label={self.user_label!r})'


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


@dataclass
class NodeStep(Step[StateT, DepsT, Any, BaseNode[StateT, DepsT, Any] | End[Any]]):
    """A step that wraps a BaseNode type for execution.

    NodeStep allows v1-style BaseNode classes to be used as steps in the
    v2 graph execution system. It validates that the input is of the expected
    node type and runs it with the appropriate graph context.
    """

    def __init__(
        self,
        node_type: type[BaseNode[StateT, DepsT, Any]],
        *,
        id: NodeID | None = None,
        user_label: str | None = None,
    ):
        """Initialize a node step.

        Args:
            node_type: The BaseNode class this step will execute
            id: Optional unique identifier, defaults to the node's get_node_id()
            user_label: Optional human-readable label for this step
        """
        super().__init__(
            id=id or NodeID(node_type.get_node_id()),
            call=self._call,
            user_label=user_label,
        )
        # `type[BaseNode[StateT, DepsT, Any]]` could actually be a `typing._GenericAlias` like `pydantic_ai._agent_graph.UserPromptNode[~DepsT, ~OutputT]`,
        # so we get the origin to get to the actual class
        self.node_type = get_origin(node_type) or node_type
        """The BaseNode type this step executes."""

    async def _call(self, ctx: StepContext[StateT, DepsT, Any]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
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
            raise ValueError(f'Node {node} is not of type {self.node_type}')
        node = cast(BaseNode[StateT, DepsT, Any], node)
        return await node.run(GraphRunContext(state=ctx.state, deps=ctx.deps))
