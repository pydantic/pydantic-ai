from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Never, overload

from typing_extensions import TypeAliasType, TypeVar

from pydantic_graph.v2.decision import Decision, DecisionBranchBuilder
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import ForkId, JoinId, NodeId
from pydantic_graph.v2.join import Join, Reducer
from pydantic_graph.v2.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.v2.node_types import (
    AnyNode,
    DestinationNode,
    SourceNode,
)
from pydantic_graph.v2.parent_forks import ParentFork, ParentForkFinder
from pydantic_graph.v2.paths import (
    BroadcastMarker,
    DestinationMarker,
    EdgePath,
    EdgePathBuilder,
    Path,
    PathBuilder,
    SpreadMarker,
)
from pydantic_graph.v2.step import Step, StepFunction
from pydantic_graph.v2.util import TypeOrTypeExpression, get_callable_name, unpack_type_expression

StateT = TypeVar('StateT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)
SourceT = TypeVar('SourceT', infer_variance=True)
SourceOutputT = TypeVar('SourceOutputT', infer_variance=True)
GraphInputT = TypeVar('GraphInputT', infer_variance=True)
GraphOutputT = TypeVar('GraphOutputT', infer_variance=True)
T = TypeVar('T', infer_variance=True)


# Node building:
@overload
def step(
    *,
    node_id: str | None = None,
    label: str | None = None,
    activity: bool = False,
) -> Callable[[StepFunction[StateT, InputT, OutputT]], Step[StateT, InputT, OutputT]]: ...
@overload
def step(
    call: StepFunction[StateT, InputT, OutputT],
    *,
    node_id: str | None = None,
    label: str | None = None,
    activity: bool = False,
) -> Step[StateT, InputT, OutputT]: ...
def step(
    call: StepFunction[StateT, InputT, OutputT] | None = None,
    *,
    node_id: str | None = None,
    label: str | None = None,
    activity: bool = False,
) -> Step[StateT, InputT, OutputT] | Callable[[StepFunction[StateT, InputT, OutputT]], Step[StateT, InputT, OutputT]]:
    """Get a Step instance from a step function."""
    if call is None:

        def decorator(
            func: StepFunction[StateT, InputT, OutputT],
        ) -> Step[StateT, InputT, OutputT]:
            return step(call=func, node_id=node_id, label=label, activity=activity)

        return decorator

    node_id = node_id or get_callable_name(call)

    return Step[StateT, InputT, OutputT](id=NodeId(node_id), call=call, user_label=label, activity=activity)


@overload
def join(
    *,
    node_id: str | None = None,
) -> Callable[[type[Reducer[StateT, InputT, OutputT]]], Join[StateT, InputT, OutputT]]: ...
@overload
def join(
    reducer_type: type[Reducer[StateT, InputT, OutputT]],
    *,
    node_id: str | None = None,
) -> Join[StateT, InputT, OutputT]: ...
def join(
    reducer_type: type[Reducer[StateT, Any, Any]] | None = None,
    *,
    node_id: str | None = None,
) -> Join[StateT, Any, Any] | Callable[[type[Reducer[StateT, Any, Any]]], Join[StateT, Any, Any]]:
    """Get a Join instance from a reducer type."""
    if reducer_type is None:

        def decorator(
            reducer_type: type[Reducer[StateT, Any, Any]],
        ) -> Join[StateT, Any, Any]:
            return join(reducer_type=reducer_type, node_id=node_id)

        return decorator

    # TODO(P3): Ideally we'd be able to infer this from the parent frame variable assignment or similar
    node_id = node_id or get_callable_name(reducer_type)

    return Join[StateT, Any, Any](
        id=JoinId(NodeId(node_id)),
        reducer_type=reducer_type,
    )


@dataclass
class GraphBuilder(Generic[StateT, GraphInputT, GraphOutputT]):
    """A graph builder."""

    state_type: TypeOrTypeExpression[StateT]
    input_type: TypeOrTypeExpression[GraphInputT]
    output_type: TypeOrTypeExpression[GraphOutputT]

    parallel: bool = True  # if False, allow direct state modification and don't copy state sent to steps, but disallow parallel node execution

    _nodes: dict[NodeId, AnyNode] = field(init=False, default_factory=dict)
    _edges_by_source: dict[NodeId, list[Path]] = field(init=False, default_factory=lambda: defaultdict(list))
    _decision_index: int = field(init=False, default=1)

    Source = TypeAliasType('Source', SourceNode[StateT, OutputT], type_params=(OutputT,))
    Destination = TypeAliasType('Destination', DestinationNode[StateT, InputT], type_params=(InputT,))

    def __post_init__(self):
        self._start_node = StartNode[GraphInputT]()
        self._end_node = EndNode[GraphOutputT]()

    # Node building
    @property
    def start_node(self) -> StartNode[GraphInputT]:
        return self._start_node

    @property
    def end_node(self) -> EndNode[GraphOutputT]:
        return self._end_node

    @overload
    def step(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
        activity: bool = False,
    ) -> Callable[[StepFunction[StateT, InputT, OutputT]], Step[StateT, InputT, OutputT]]: ...
    @overload
    def step(
        self,
        call: StepFunction[StateT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
        activity: bool = False,
    ) -> Step[StateT, InputT, OutputT]: ...
    def step(
        self,
        call: StepFunction[StateT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
        activity: bool = False,
    ) -> (
        Step[StateT, InputT, OutputT] | Callable[[StepFunction[StateT, InputT, OutputT]], Step[StateT, InputT, OutputT]]
    ):
        if call is None:
            return step(node_id=node_id, label=label, activity=activity)
        else:
            return step(call=call, node_id=node_id, label=label, activity=activity)

    @overload
    def join(
        self,
        *,
        node_id: str | None = None,
    ) -> Callable[[type[Reducer[StateT, InputT, OutputT]]], Join[StateT, InputT, OutputT]]: ...
    @overload
    def join(
        self,
        reducer_factory: type[Reducer[StateT, InputT, OutputT]],
        *,
        node_id: str | None = None,
    ) -> Join[StateT, InputT, OutputT]: ...
    def join(
        self,
        reducer_factory: type[Reducer[StateT, Any, Any]] | None = None,
        *,
        node_id: str | None = None,
    ) -> Join[StateT, Any, Any] | Callable[[type[Reducer[StateT, Any, Any]]], Join[StateT, Any, Any]]:
        if reducer_factory is None:
            return join(node_id=node_id)
        else:
            return join(reducer_type=reducer_factory, node_id=node_id)

    # Edge building
    def add(self, *edges: EdgePath[StateT]) -> None:
        def _handle_path(p: Path):
            for item in p.items:
                if isinstance(item, BroadcastMarker):
                    new_node = Fork[Any, Any](id=item.fork_id, is_spread=False)
                    self._insert_node(new_node)
                    for path in item.paths:
                        _handle_path(Path(items=[*path.items]))
                elif isinstance(item, SpreadMarker):
                    new_node = Fork[Any, Any](id=item.fork_id, is_spread=True)
                    self._insert_node(new_node)
                elif isinstance(item, DestinationMarker):
                    pass

        for edge in edges:
            for source_node in edge.sources:
                self._insert_node(source_node)
                self._edges_by_source[source_node.id].append(edge.path)
            for destination_node in edge.destinations:
                self._insert_node(destination_node)

            _handle_path(edge.path)

    def add_edge[T](self, source: Source[T], destination: Destination[T], *, label: str | None = None) -> None:
        builder = self.edge_from(source)
        if label is not None:
            builder = builder.label(label)
        self.add(builder.to(destination))

    def add_spreading_edge[T](
        self,
        source: Source[Iterable[T]],
        spread_to: Destination[T],
        *,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None:
        builder = self.edge_from(source)
        if pre_spread_label is not None:
            builder = builder.label(pre_spread_label)
        builder = builder.spread()
        if post_spread_label is not None:
            builder = builder.label(post_spread_label)
        self.add(builder.to(spread_to))

    # TODO(P2): Support adding subgraphs ... not sure exactly what that looks like yet..
    #  probably similar to a step, but with some tweaks

    def edge_from[SourceOutputT](self, *sources: Source[SourceOutputT]) -> EdgePathBuilder[StateT, SourceOutputT]:
        return EdgePathBuilder[StateT, SourceOutputT](sources=sources, path_builder=PathBuilder(working_items=[]))

    def decision(self, *, note: str | None = None) -> Decision[StateT, Never]:
        return Decision(id=NodeId(self._get_new_decision_id()), branches=[], note=note)

    def match[SourceT](
        self,
        source: TypeOrTypeExpression[SourceT],
        *,
        matches: Callable[[Any], bool] | None = None,
    ) -> DecisionBranchBuilder[StateT, SourceT, SourceT, Never]:
        node_id = NodeId(self._get_new_decision_id())
        decision = Decision[StateT, Never](node_id, branches=[], note=None)
        new_path_builder = PathBuilder[StateT, SourceT](working_items=[])
        return DecisionBranchBuilder(decision=decision, source=source, matches=matches, path_builder=new_path_builder)

    # Helpers
    def _insert_node(self, node: AnyNode) -> None:
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
        elif existing is not node:
            raise ValueError(f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}')

    def _get_new_decision_id(self) -> str:
        node_id = f'decision_{self._decision_index}'
        self._decision_index += 1
        while node_id in self._nodes:
            node_id = f'decision_{self._decision_index}'
            self._decision_index += 1
        return node_id

    def _get_new_broadcast_id(self, from_: str | None = None) -> str:
        prefix = 'broadcast'
        if from_ is not None:
            prefix += f'_from_{from_}'

        node_id = prefix
        index = 2
        while node_id in self._nodes:
            node_id = f'{prefix}_{index}'
            index += 1
        return node_id

    def _get_new_spread_id(self, from_: str | None = None, to: str | None = None) -> str:
        prefix = 'spread'
        if from_ is not None:
            prefix += f'_from_{from_}'
        if to is not None:
            prefix += f'_to_{to}'

        node_id = prefix
        index = 2
        while node_id in self._nodes:
            node_id = f'{prefix}_{index}'
            index += 1
        return node_id

    # Graph building
    def build(self) -> Graph[StateT, GraphInputT, GraphOutputT]:
        # TODO(P2): Warn/error if there is no start node / edges, or end node / edges
        # TODO(P2): Warn/error if the graph is not connected
        # TODO(P2): Warn/error if any non-End node is a dead end
        # TODO(P2): Error if the graph does not meet the every-join-has-a-parent-fork requirement (otherwise can't know when to proceed past joins)
        # TODO(P2): Allow the user to specify the parent forks; only infer them if _not_ specified
        # TODO(P2): Verify that any user-specified parent forks are _actually_ valid parent forks, and if not, generate a helpful error message
        # TODO(P3): Consider doing a deepcopy here to prevent modifications to the underlying nodes and edges
        nodes = self._nodes
        edges_by_source = self._edges_by_source
        nodes, edges_by_source = _normalize_forks(nodes, edges_by_source)
        parent_forks = _collect_dominating_forks(nodes, edges_by_source)

        return Graph[StateT, GraphInputT, GraphOutputT](
            state_type=unpack_type_expression(self.state_type),
            input_type=unpack_type_expression(self.input_type),
            output_type=unpack_type_expression(self.output_type),
            nodes=nodes,
            edges_by_source=edges_by_source,
            parent_forks=parent_forks,
        )


def _normalize_forks(
    nodes: dict[NodeId, AnyNode], edges: dict[NodeId, list[Path]]
) -> tuple[dict[NodeId, AnyNode], dict[NodeId, list[Path]]]:
    """Rework the nodes/edges so that the _only_ nodes with multiple edges coming out are broadcast forks.

    Also, add forks to edges.
    """
    new_nodes = nodes.copy()
    new_edges: dict[NodeId, list[Path]] = {}

    paths_to_handle: list[Path] = []

    for source_id, edges_from_source in edges.items():
        paths_to_handle.extend(edges_from_source)

        node = nodes[source_id]
        if isinstance(node, Fork) and not node.is_spread:
            new_edges[source_id] = edges_from_source
            continue  # broadcast fork; nothing to do
        if len(edges_from_source) == 1:
            new_edges[source_id] = edges_from_source
            continue
        new_fork = Fork[Any, Any](id=ForkId(NodeId(f'{node.id}_broadcast_fork')), is_spread=False)
        new_nodes[new_fork.id] = new_fork
        new_edges[source_id] = [Path(items=[BroadcastMarker(fork_id=new_fork.id, paths=edges_from_source)])]
        new_edges[new_fork.id] = edges_from_source

    while paths_to_handle:
        path = paths_to_handle.pop()
        for item in path.items:
            if isinstance(item, SpreadMarker):
                assert item.fork_id in new_nodes
                new_edges[item.fork_id] = [path.next_path]
            if isinstance(item, BroadcastMarker):
                assert item.fork_id in new_nodes
                # if item.fork_id not in new_nodes:
                #     new_nodes[new_fork.id] = Fork[Any, Any](id=item.fork_id, is_spread=False)
                new_edges[item.fork_id] = [*item.paths]
                paths_to_handle.extend(item.paths)

    return new_nodes, new_edges


def _collect_dominating_forks(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Path]]
) -> dict[JoinId, ParentFork[NodeId]]:
    nodes = set(graph_nodes)
    start_ids: set[NodeId] = {StartNode.id}
    edges: dict[NodeId, list[NodeId]] = defaultdict(list)

    fork_ids: set[NodeId] = set(start_ids)
    for source_id in nodes:
        working_source_id = source_id
        node = graph_nodes.get(source_id)

        if isinstance(node, Fork):
            fork_ids.add(node.id)
            continue

        def _handle_path(path: Path, last_source_id: NodeId):
            for item in path.items:
                if isinstance(item, SpreadMarker):
                    fork_ids.add(item.fork_id)
                    edges[last_source_id].append(item.fork_id)
                    last_source_id = item.fork_id
                elif isinstance(item, BroadcastMarker):
                    fork_ids.add(item.fork_id)
                    edges[last_source_id].append(item.fork_id)
                    for fork in item.paths:
                        _handle_path(Path([*fork.items]), item.fork_id)
                    # Broadcasts should only ever occur as the last item in the list, so no need to update the working_source_id
                elif isinstance(item, DestinationMarker):
                    edges[last_source_id].append(item.destination_id)
                    # Destinations should only ever occur as the last item in the list, so no need to update the working_source_id

        if isinstance(node, Decision):
            for branch in node.branches:
                _handle_path(branch.path, working_source_id)
        else:
            for path in graph_edges_by_source.get(source_id, []):
                _handle_path(path, source_id)

    finder = ParentForkFinder(
        nodes=nodes,
        start_ids=start_ids,
        fork_ids=fork_ids,
        edges=edges,
    )

    join_ids = {node.id for node in graph_nodes.values() if isinstance(node, Join)}
    dominating_forks: dict[JoinId, ParentFork[NodeId]] = {}
    for join_id in join_ids:
        dominating_fork = finder.find_parent_fork(join_id)
        if dominating_fork is None:
            # TODO(P3): Print out the mermaid graph and explain the problem
            raise ValueError(f'Join node {join_id} has no dominating fork')
        dominating_forks[join_id] = dominating_fork

    return dominating_forks
