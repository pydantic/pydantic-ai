from __future__ import annotations

import inspect
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Never, Protocol, overload

from .nodes import Node, NodeId, TypeUnion


class Routing[T]:
    """This is an auxiliary class that is purposely not a dataclass, and should not be instantiated.

    It should only be used for its `__class_getitem__` method.
    """

    _force_invariant: Callable[[T], T]


@dataclass
class CallNode[StateT, InputT, OutputT](Node[StateT, InputT, OutputT]):
    id: NodeId
    call: Callable[[StateT, InputT], Awaitable[OutputT]]

    async def run(self, state: StateT, inputs: InputT) -> OutputT:
        return await self.call(state, inputs)


@dataclass
class Interruption[StopT, ResumeT]:
    value: StopT
    next_node: Node[Any, ResumeT, Any]


class EmptyNodeFunction[OutputT](Protocol):
    def __call__(self) -> OutputT:
        raise NotImplementedError


class StateNodeFunction[StateT, OutputT](Protocol):
    def __call__(self, state: StateT) -> OutputT:
        raise NotImplementedError


class InputNodeFunction[InputT, OutputT](Protocol):
    def __call__(self, inputs: InputT) -> OutputT:
        raise NotImplementedError


class FullNodeFunction[StateT, InputT, OutputT](Protocol):
    def __call__(self, state: StateT, inputs: InputT) -> OutputT:
        raise NotImplementedError


@overload
def graph_node[OutputT](
    fn: EmptyNodeFunction[OutputT],
) -> Node[Any, object, OutputT]: ...
@overload
def graph_node[InputT, OutputT](
    fn: InputNodeFunction[InputT, OutputT],
) -> Node[Any, InputT, OutputT]: ...
@overload
def graph_node[StateT, OutputT](
    fn: StateNodeFunction[StateT, OutputT],
) -> Node[StateT, object, OutputT]: ...
@overload
def graph_node[StateT, InputT, OutputT](
    fn: FullNodeFunction[StateT, InputT, OutputT],
) -> Node[StateT, InputT, OutputT]: ...


def graph_node(fn: Callable[..., Any]) -> Node[Any, Any, Any]:
    signature = inspect.signature(fn)
    signature_error = "Function may only make use of parameters 'state' and 'inputs'"
    node_id = NodeId(fn.__name__)
    if 'state' in signature.parameters and 'inputs' in signature.parameters:
        assert len(signature.parameters) == 2, signature_error
        return CallNode(id=node_id, call=fn)
    elif 'state' in signature.parameters:
        assert len(signature.parameters) == 1, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn(state))
    elif 'state' in signature.parameters:
        assert len(signature.parameters) == 1, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn(inputs))
    else:
        assert len(signature.parameters) == 0, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn())


class EdgeStart[GraphStateT, NodeInputT, NodeOutputT](Protocol):
    _make_covariant: Callable[[NodeInputT], NodeInputT]
    _make_invariant: Callable[[NodeOutputT], NodeOutputT]

    @staticmethod
    def __call__[SourceT](
        source: type[SourceT],
    ) -> DecisionBranch[SourceT, GraphStateT, NodeInputT, SourceT]:
        raise NotImplementedError


class Decision[SourceT, EndT]:
    _force_source_invariant: Callable[[SourceT], SourceT]
    _force_end_covariant: Callable[[], EndT]

    def branch[S, E, S2, E2](
        self: Decision[S, E], edge: Decision[S2, E2]
    ) -> Decision[S | S2, E | E2]:
        raise NotImplementedError

    def otherwise[E2](self, edge: Decision[Any, E2]) -> Decision[Any, EndT | E2]:
        raise NotImplementedError


def decision() -> Decision[Never, Never]:
    raise NotImplementedError


@dataclass
class GraphBuilder[StateT, InputT, OutputT]:
    # TODO: Should get the following values from __class_getitem__ somehow;
    #   this would make it possible to use typeforms without type errors
    state_type: type[StateT] = field(init=False)
    input_type: type[InputT] = field(init=False)
    output_type: type[OutputT] = field(init=False)

    # _start_at: Router[StateT, OutputT, InputT, InputT] | Node[StateT, InputT, Any]
    # _simple_edges: list[
    #     tuple[
    #         Node[StateT, Any, Any],
    #         TransformFunction[StateT, Any, Any, Any] | None,
    #         Node[StateT, Any, Any],
    #     ]
    # ] = field(init=False, default_factory=list)
    # _routed_edges: list[
    #     tuple[Node[StateT, Any, Any], Router[StateT, OutputT, Any, Any]]
    # ] = field(init=False, default_factory=list)

    def start_edge[NodeInputT, NodeOutputT](
        self, node: Node[StateT, NodeInputT, NodeOutputT]
    ) -> EdgeStart[StateT, NodeInputT, NodeOutputT]:
        raise NotImplementedError

    def handle[SourceT](
        self,
        source: type[TypeUnion[SourceT]] | type[SourceT],
        # condition: Callable[[Any], bool] | None = None,
    ) -> DecisionBranch[SourceT, StateT, object, SourceT]:
        raise NotImplementedError

    def handle_any(
        self,
        condition: Callable[[Any], bool] | None = None,
    ) -> DecisionBranch[Any, StateT, object, Any]:
        raise NotImplementedError

    def add_edges[T](
        self, start: EdgeStart[StateT, Any, T], decision: Decision[T, OutputT]
    ) -> None:
        raise NotImplementedError

    # def edge[T](
    #     self,
    #     *,
    #     source: Node[StateT, Any, T],
    #     transform: TransformFunction[StateT, Any, Any, T] | None = None,
    #     destination: Node[StateT, T, Any],
    # ):
    #     self._simple_edges.append((source, transform, destination))
    #
    # def edges[SourceInputT, SourceOutputT](
    #     self,
    #     source: Node[StateT, SourceInputT, SourceOutputT],
    #     routing: Router[StateT, OutputT, SourceInputT, SourceOutputT],
    # ):
    #     self._routed_edges.append((source, routing))

    # def build(self) -> Graph[StateT, InputT, OutputT]:
    #     # TODO: Build nodes from edges/decisions
    #     nodes: dict[NodeId, Node[StateT, Any, Any]] = {}
    #     assert self._start_at is not None, (
    #         'You must call `GraphBuilder.start_at` before building the graph.'
    #     )
    #     return Graph[StateT, InputT, OutputT](
    #         nodes=nodes,
    #         start_at=self._start_at,
    #         edges=[(e[0].id, e[1], e[2].id) for e in self._simple_edges],
    #         routed_edges=[(d[0].id, d[1]) for d in self._routed_edges],
    #     )

    def _check_output(self, output: OutputT) -> None:
        raise RuntimeError(
            'This method is only included for type-checking purposes and should not be called directly.'
        )


@dataclass
class Graph[StateT, InputT, OutputT]:
    nodes: dict[NodeId, Node[StateT, Any, Any]]

    # TODO: May need to tweak the following to actually work at runtime...
    # start_at: Router[StateT, OutputT, InputT, InputT] | Node[StateT, InputT, Any]
    # edges: list[tuple[NodeId, Any, NodeId]]
    # routed_edges: list[tuple[NodeId, Router[StateT, OutputT, Any, Any]]]

    @staticmethod
    def builder[S, I, O](
        state_type: type[S],
        input_type: type[I],
        output_type: type[TypeUnion[O]] | type[O],
        # start_at: Node[S, I, Any] | Router[S, O, I, I],
    ) -> GraphBuilder[S, I, O]:
        raise NotImplementedError


#     def run(self, state: StateT, inputs: InputT) -> OutputT:
#         raise NotImplementedError
#
#     def resume[NodeInputT](
#         self,
#         state: StateT,
#         node: Node[StateT, NodeInputT, Any],
#         node_inputs: NodeInputT,
#     ) -> OutputT:
#         raise NotImplementedError


class TransformContext[StateT, InputT, OutputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state: StateT, inputs: InputT, output: OutputT):
        self._state = state
        self._inputs = inputs
        self._output = output

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def inputs(self) -> InputT:
        return self._inputs

    @property
    def output(self) -> OutputT:
        return self._output

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, inputs={self.inputs}, output={self.output})'


class _Transform[StateT, InputT, OutputT, T](Protocol):
    def __call__(self, ctx: TransformContext[StateT, InputT, OutputT]) -> T:
        raise NotImplementedError


type TransformFunction[StateT, SourceInputT, SourceOutputT, DestinationInputT] = (
    _Transform[StateT, SourceInputT, SourceOutputT, DestinationInputT]
)


@dataclass
class DecisionBranch[SourceT, GraphStateT, EdgeInputT, EdgeOutputT]:
    _source_type: type[SourceT]
    _is_instance: Callable[[Any], bool]
    _transforms: tuple[TransformFunction[GraphStateT, EdgeInputT, Any, Any], ...] = (
        field(default=())
    )
    _end: bool = field(init=False, default=False)

    # Note: _route_to must use `Any` instead of `HandleOutputT` in the first argument to keep this type contravariant in
    # HandleOutputT. I _believe_ this is safe because instances of this type should never get mutated after this is set.
    _route_to: Node[GraphStateT, Any, Any] | None = field(init=False, default=None)

    def end(
        self,
    ) -> Decision[SourceT, EdgeOutputT]:
        raise NotImplementedError
        # self._end = True
        # return self._source_type

    def route_to(
        self, node: Node[GraphStateT, EdgeOutputT, Any]
    ) -> Decision[SourceT, Never]:
        raise NotImplementedError

    def route_to_parallel[T](
        self: DecisionBranch[SourceT, GraphStateT, EdgeInputT, Sequence[T]],
        node: Node[GraphStateT, T, Any],
    ) -> Decision[SourceT, Never]:
        raise NotImplementedError

    def transform[T](
        self,
        call: _Transform[GraphStateT, EdgeInputT, EdgeOutputT, T],
    ) -> DecisionBranch[SourceT, GraphStateT, EdgeInputT, T]:
        new_transforms = self._transforms + (call,)
        return DecisionBranch(self._source_type, self._is_instance, new_transforms)

    # def handle_parallel[HandleOutputItemT, T, S](
    #     self: Edge[
    #         SourceT,
    #         GraphStateT,
    #         GraphOutputT,
    #         HandleInputT,
    #         Sequence[HandleOutputItemT],
    #     ],
    #     node: Node[GraphStateT, HandleOutputItemT, T],
    #     reducer: Callable[[GraphStateT, list[T]], S],
    # ) -> Edge[SourceT, GraphStateT, GraphOutputT, HandleInputT, S]:
    #     # This requires you to eagerly declare reduction logic; can't do dynamic joining
    #     raise NotImplementedError
