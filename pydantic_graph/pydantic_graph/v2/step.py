from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, cast, get_origin, overload

from typing_extensions import TypeVar

from pydantic_graph.nodes import BaseNode, End, GraphRunContext
from pydantic_graph.v2.id_types import NodeId

StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)


class StepContext(Generic[StateT, DepsT, InputT]):
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    if TYPE_CHECKING:

        def __init__(self, state: StateT, deps: DepsT, inputs: InputT):
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
            return self._inputs
    else:
        state: StateT
        deps: DepsT
        inputs: InputT

    def __repr__(self):
        return f'{self.__class__.__name__}(inputs={self.inputs})'


if not TYPE_CHECKING:
    StepContext = dataclass(StepContext)


class StepFunction(Protocol[StateT, DepsT, InputT, OutputT]):
    """The purpose of this is to make it possible to deserialize step calls similar to how Evaluators work."""

    def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> Awaitable[OutputT]:
        raise NotImplementedError


AnyStepFunction = StepFunction[Any, Any, Any, Any]


class Step(Generic[StateT, DepsT, InputT, OutputT]):
    """The main reason this is not a dataclass is that we need appropriate variance in the type parameters."""

    def __init__(
        self,
        id: NodeId,
        call: StepFunction[StateT, DepsT, InputT, OutputT],
        user_label: str | None = None,
        activity: bool = False,
    ):
        self.id = id
        self._call = call
        self.user_label = user_label
        self.activity = activity

    # TODO(P3): Consider replacing this with __call__, so the decorated object can still be called with the same signature
    @property
    def call(self) -> StepFunction[StateT, DepsT, InputT, OutputT]:
        # The use of a property here is necessary to ensure that Step is covariant/contravariant as appropriate.
        return self._call

    # TODO(P3): Consider adding a `bind` method that returns an object that can be used to get something you can return from a BaseNode that allows you to transition to nodes using "new"-form edges

    @property
    def label(self) -> str | None:
        return self.user_label

    @overload
    def as_node(self, inputs: None = None) -> StepNode[StateT, DepsT]: ...

    @overload
    def as_node(self, inputs: InputT) -> StepNode[StateT, DepsT]: ...

    def as_node(self, inputs: InputT | None = None) -> StepNode[StateT, DepsT]:
        return StepNode(self, inputs)


@dataclass
class StepNode(BaseNode[StateT, DepsT, Any]):
    """A `BaseNode` that represents a `Step` plus bound inputs."""

    step: Step[StateT, DepsT, Any, Any]
    inputs: Any

    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
        raise NotImplementedError(
            '`StepNode` is not meant to be run directly, it is meant to be used in `BaseNode` subclasses to indicate a transition to v2-style steps.'
        )


@dataclass
class NodeStep(Step[StateT, DepsT, Any, BaseNode[StateT, DepsT, Any] | End[Any]]):
    """A `Step` that represents a `BaseNode` type."""

    def __init__(
        self,
        node_type: type[BaseNode[StateT, DepsT, Any]],
        *,
        id: NodeId | None = None,
        user_label: str | None = None,
        activity: bool = False,
    ):
        super().__init__(
            id=id or NodeId(node_type.get_node_id()),
            call=self._call,
            user_label=user_label,
            activity=activity,
        )
        self.node_type = get_origin(node_type) or node_type

    async def _call(self, ctx: StepContext[StateT, DepsT, Any]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
        node = ctx.inputs
        if not isinstance(node, self.node_type):
            raise ValueError(f'Node {node} is not of type {self.node_type}')
        node = cast(BaseNode[StateT, DepsT, Any], node)
        return await node.run(GraphRunContext(state=ctx.state, deps=ctx.deps))
