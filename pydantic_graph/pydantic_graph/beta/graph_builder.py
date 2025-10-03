"""Graph builder for constructing executable graph definitions.

This module provides the GraphBuilder class and related utilities for
constructing typed, executable graph definitions with steps, joins,
decisions, and edge routing.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import NoneType
from typing import Any, Generic, cast, get_origin, get_type_hints, overload

from typing_extensions import Never, TypeAliasType, TypeVar

from pydantic_graph import _utils, exceptions
from pydantic_graph.beta.decision import Decision, DecisionBranch, DecisionBranchBuilder
from pydantic_graph.beta.graph import Graph
from pydantic_graph.beta.id_types import ForkID, JoinID, NodeID
from pydantic_graph.beta.join import Join, JoinNode, Reducer
from pydantic_graph.beta.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.beta.node_types import (
    AnyDestinationNode,
    AnyNode,
    DestinationNode,
    SourceNode,
)
from pydantic_graph.beta.parent_forks import ParentFork, ParentForkFinder
from pydantic_graph.beta.paths import (
    BroadcastMarker,
    DestinationMarker,
    EdgePath,
    EdgePathBuilder,
    Path,
    PathBuilder,
    SpreadMarker,
)
from pydantic_graph.beta.step import NodeStep, Step, StepFunction, StepNode
from pydantic_graph.beta.util import TypeOrTypeExpression, get_callable_name, unpack_type_expression
from pydantic_graph.nodes import BaseNode, End

StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)
SourceT = TypeVar('SourceT', infer_variance=True)
SourceNodeT = TypeVar('SourceNodeT', bound=BaseNode[Any, Any, Any], infer_variance=True)
SourceOutputT = TypeVar('SourceOutputT', infer_variance=True)
GraphInputT = TypeVar('GraphInputT', infer_variance=True)
GraphOutputT = TypeVar('GraphOutputT', infer_variance=True)
T = TypeVar('T', infer_variance=True)


@overload
def join(
    *,
    node_id: str | None = None,
) -> Callable[[type[Reducer[StateT, DepsT, InputT, OutputT]]], Join[StateT, DepsT, InputT, OutputT]]: ...
@overload
def join(
    reducer_type: type[Reducer[StateT, DepsT, InputT, OutputT]],
    *,
    node_id: str | None = None,
) -> Join[StateT, DepsT, InputT, OutputT]: ...
def join(
    reducer_type: type[Reducer[StateT, DepsT, Any, Any]] | None = None,
    *,
    node_id: str | None = None,
) -> Join[StateT, DepsT, Any, Any] | Callable[[type[Reducer[StateT, DepsT, Any, Any]]], Join[StateT, DepsT, Any, Any]]:
    """Create a join node from a reducer type.

    This function can be used as a decorator or called directly to create
    a join node that aggregates data from parallel execution paths.

    Args:
        reducer_type: The reducer class to use for aggregating data
        node_id: Optional ID for the node, defaults to the reducer type name

    Returns:
        Either a Join instance or a decorator function
    """
    if reducer_type is None:

        def decorator(
            reducer_type: type[Reducer[StateT, DepsT, Any, Any]],
        ) -> Join[StateT, DepsT, Any, Any]:
            return join(reducer_type=reducer_type, node_id=node_id)

        return decorator

    # TODO(P3): Ideally we'd be able to infer this from the parent frame variable assignment or similar
    node_id = node_id or get_callable_name(reducer_type)

    return Join[StateT, DepsT, Any, Any](
        id=JoinID(NodeID(node_id)),
        reducer_type=reducer_type,
    )


@dataclass(init=False)
class GraphBuilder(Generic[StateT, DepsT, GraphInputT, GraphOutputT]):
    """A builder for constructing executable graph definitions.

    GraphBuilder provides a fluent interface for defining nodes, edges, and
    routing in a graph workflow. It supports typed state, dependencies, and
    input/output validation.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        GraphInputT: The type of the graph input data
        GraphOutputT: The type of the graph output data
    """

    name: str | None
    """Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method."""

    state_type: TypeOrTypeExpression[StateT]
    """The type of the graph state."""

    deps_type: TypeOrTypeExpression[DepsT]
    """The type of the dependencies."""

    input_type: TypeOrTypeExpression[GraphInputT]
    """The type of the graph input data."""

    output_type: TypeOrTypeExpression[GraphOutputT]
    """The type of the graph output data."""

    auto_instrument: bool
    """Whether to automatically create instrumentation spans."""

    _nodes: dict[NodeID, AnyNode]
    """Internal storage for nodes in the graph."""

    _edges_by_source: dict[NodeID, list[Path]]
    """Internal storage for edges by source node."""

    _decision_index: int
    """Counter for generating unique decision node IDs."""

    Source = TypeAliasType('Source', SourceNode[StateT, DepsT, OutputT], type_params=(OutputT,))
    Destination = TypeAliasType('Destination', DestinationNode[StateT, DepsT, InputT], type_params=(InputT,))

    def __init__(
        self,
        *,
        name: str | None = None,
        state_type: TypeOrTypeExpression[StateT] = NoneType,
        deps_type: TypeOrTypeExpression[DepsT] = NoneType,
        input_type: TypeOrTypeExpression[GraphInputT] = NoneType,
        output_type: TypeOrTypeExpression[GraphOutputT] = NoneType,
        auto_instrument: bool = True,
    ):
        """Initialize a graph builder.

        Args:
            name: Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method.
            state_type: The type of the graph state
            deps_type: The type of the dependencies
            input_type: The type of the graph input data
            output_type: The type of the graph output data
            auto_instrument: Whether to automatically create instrumentation spans
        """
        self.name = name

        self.state_type = state_type
        self.deps_type = deps_type
        self.input_type = input_type
        self.output_type = output_type

        self.auto_instrument = auto_instrument

        self._nodes = {}
        self._edges_by_source = defaultdict(list)
        self._decision_index = 1

        self._start_node = StartNode[GraphInputT]()
        self._end_node = EndNode[GraphOutputT]()

    # Node building
    @property
    def start_node(self) -> StartNode[GraphInputT]:
        """Get the start node for the graph.

        Returns:
            The start node that receives the initial graph input
        """
        return self._start_node

    @property
    def end_node(self) -> EndNode[GraphOutputT]:
        """Get the end node for the graph.

        Returns:
            The end node that produces the final graph output
        """
        return self._end_node

    @overload
    def _step(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]: ...
    @overload
    def _step(
        self,
        call: StepFunction[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, OutputT]: ...
    def _step(
        self,
        call: StepFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, OutputT]
        | Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]
    ):
        """Create a step from a step function (internal implementation).

        This internal method handles the actual step creation logic and
        automatic edge inference from type hints.

        Args:
            call: The step function to wrap
            node_id: Optional ID for the node
            label: Optional human-readable label

        Returns:
            Either a Step instance or a decorator function
        """
        if call is None:

            def decorator(
                func: StepFunction[StateT, DepsT, InputT, OutputT],
            ) -> Step[StateT, DepsT, InputT, OutputT]:
                return self._step(call=func, node_id=node_id, label=label)

            return decorator

        node_id = node_id or get_callable_name(call)

        step = Step[StateT, DepsT, InputT, OutputT](id=NodeID(node_id), call=call, user_label=label)

        return step

    @overload
    def step(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]: ...
    @overload
    def step(
        self,
        call: StepFunction[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, OutputT]: ...
    def step(
        self,
        call: StepFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, OutputT]
        | Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]
    ):
        """Create a step from a step function.

        This method can be used as a decorator or called directly to create
        a step node from an async function.

        Args:
            call: The step function to wrap
            node_id: Optional ID for the node
            label: Optional human-readable label

        Returns:
            Either a Step instance or a decorator function
        """
        if call is None:
            return self._step(node_id=node_id, label=label)
        else:
            return self._step(call=call, node_id=node_id, label=label)

    @overload
    def join(
        self,
        *,
        node_id: str | None = None,
    ) -> Callable[[type[Reducer[StateT, DepsT, InputT, OutputT]]], Join[StateT, DepsT, InputT, OutputT]]: ...
    @overload
    def join(
        self,
        reducer_factory: type[Reducer[StateT, DepsT, InputT, OutputT]],
        *,
        node_id: str | None = None,
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...
    def join(
        self,
        reducer_factory: type[Reducer[StateT, DepsT, Any, Any]] | None = None,
        *,
        node_id: str | None = None,
    ) -> (
        Join[StateT, DepsT, Any, Any]
        | Callable[[type[Reducer[StateT, DepsT, Any, Any]]], Join[StateT, DepsT, Any, Any]]
    ):
        """Create a join node with a reducer.

        This method can be used as a decorator or called directly to create
        a join node that aggregates data from parallel execution paths.

        Args:
            reducer_factory: The reducer class to use for aggregating data
            node_id: Optional ID for the node

        Returns:
            Either a Join instance or a decorator function
        """
        if reducer_factory is None:
            return join(node_id=node_id)
        else:
            return join(reducer_type=reducer_factory, node_id=node_id)

    # Edge building
    def add(self, *edges: EdgePath[StateT, DepsT]) -> None:
        """Add one or more edge paths to the graph.

        This method processes edge paths and automatically creates any necessary
        fork nodes for broadcasts and spreads.

        Args:
            *edges: The edge paths to add to the graph
        """

        def _handle_path(p: Path):
            """Process a path and create necessary fork nodes.

            Args:
                p: The path to process
            """
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

        destinations: list[AnyDestinationNode] = []
        for edge in edges:
            for source_node in edge.sources:
                self._insert_node(source_node)
                self._edges_by_source[source_node.id].append(edge.path)
            for destination_node in edge.destinations:
                destinations.append(destination_node)
                self._insert_node(destination_node)

            _handle_path(edge.path)

        # Automatically create edges from step function return hints including `BaseNode`s
        for destination in destinations:
            if not isinstance(destination, Step) or isinstance(destination, NodeStep):
                continue
            parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
            type_hints = get_type_hints(destination.call, localns=parent_namespace, include_extras=True)
            try:
                return_hint = type_hints['return']
            except KeyError:
                pass
            else:
                edge = self._edge_from_return_hint(destination, return_hint)
                if edge is not None:
                    self.add(edge)

    def add_edge(self, source: Source[T], destination: Destination[T], *, label: str | None = None) -> None:
        """Add a simple edge between two nodes.

        Args:
            source: The source node
            destination: The destination node
            label: Optional label for the edge
        """
        builder = self.edge_from(source)
        if label is not None:
            builder = builder.label(label)
        self.add(builder.to(destination))

    def add_spreading_edge(
        self,
        source: Source[Iterable[T]],
        spread_to: Destination[T],
        *,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
        fork_id: ForkID | None = None,
        downstream_join_id: JoinID | None = None,
    ) -> None:
        """Add an edge that spreads iterable data across parallel paths.

        Args:
            source: The source node that produces iterable data
            spread_to: The destination node that receives individual items
            pre_spread_label: Optional label before the spread operation
            post_spread_label: Optional label after the spread operation
            fork_id: Optional ID for the fork node produced for this spread operation
            downstream_join_id: Optional ID of a join node that will always be downstream of this spread.
                Specifying this ensures correct handling if you try to spread an empty iterable.
        """
        builder = self.edge_from(source)
        if pre_spread_label is not None:
            builder = builder.label(pre_spread_label)
        builder = builder.spread(fork_id=fork_id, downstream_join_id=downstream_join_id)
        if post_spread_label is not None:
            builder = builder.label(post_spread_label)
        self.add(builder.to(spread_to))

    # TODO(P2): Support adding subgraphs ... not sure exactly what that looks like yet..
    #  probably similar to a step, but with some tweaks

    def edge_from(self, *sources: Source[SourceOutputT]) -> EdgePathBuilder[StateT, DepsT, SourceOutputT]:
        """Create an edge path builder starting from the given source nodes.

        Args:
            *sources: The source nodes to start the edge path from

        Returns:
            An EdgePathBuilder for constructing the complete edge path
        """
        return EdgePathBuilder[StateT, DepsT, SourceOutputT](
            sources=sources, path_builder=PathBuilder(working_items=[])
        )

    def decision(self, *, note: str | None = None) -> Decision[StateT, DepsT, Never]:
        """Create a new decision node.

        Args:
            note: Optional note to describe the decision logic

        Returns:
            A new Decision node with no branches
        """
        return Decision(id=NodeID(self._get_new_decision_id()), branches=[], note=note)

    def match(
        self,
        source: TypeOrTypeExpression[SourceT],
        *,
        matches: Callable[[Any], bool] | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, SourceT, Never]:
        """Create a decision branch matcher.

        Args:
            source: The type or type expression to match against
            matches: Optional custom matching function

        Returns:
            A DecisionBranchBuilder for constructing the branch
        """
        node_id = NodeID(self._get_new_decision_id())
        decision = Decision[StateT, DepsT, Never](node_id, branches=[], note=None)
        new_path_builder = PathBuilder[StateT, DepsT, SourceT](working_items=[])
        return DecisionBranchBuilder(decision=decision, source=source, matches=matches, path_builder=new_path_builder)

    def match_node(
        self,
        source: type[SourceNodeT],
        *,
        matches: Callable[[Any], bool] | None = None,
    ) -> DecisionBranch[SourceNodeT]:
        """Create a decision branch for BaseNode subclasses.

        This is similar to match() but specifically designed for matching
        against BaseNode types from the v1 system.

        Args:
            source: The BaseNode subclass to match against
            matches: Optional custom matching function

        Returns:
            A DecisionBranch for the BaseNode type
        """
        path = Path(items=[DestinationMarker(NodeStep(source).id)])
        return DecisionBranch(source=source, matches=matches, path=path)

    def node(
        self,
        node_type: type[BaseNode[StateT, DepsT, GraphOutputT]],
    ) -> EdgePath[StateT, DepsT]:
        """Create an edge path from a BaseNode class.

        This method integrates v1-style BaseNode classes into the v2 graph
        system by analyzing their type hints and creating appropriate edges.

        Args:
            node_type: The BaseNode subclass to integrate

        Returns:
            An EdgePath representing the node and its connections

        Raises:
            GraphSetupError: If the node type is missing required type hints
        """
        parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
        type_hints = get_type_hints(node_type.run, localns=parent_namespace, include_extras=True)
        try:
            return_hint = type_hints['return']
        except KeyError as e:
            raise exceptions.GraphSetupError(
                f'Node {node_type} is missing a return type hint on its `run` method'
            ) from e

        node = NodeStep(node_type)

        edge = self._edge_from_return_hint(node, return_hint)
        if not edge:
            raise exceptions.GraphSetupError(f'Node {node_type} is missing a return type hint on its `run` method')

        return edge

    # Helpers
    def _insert_node(self, node: AnyNode) -> None:
        """Insert a node into the graph, checking for ID conflicts.

        Args:
            node: The node to insert

        Raises:
            ValueError: If a different node with the same ID already exists
        """
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
        elif isinstance(existing, NodeStep) and isinstance(node, NodeStep) and existing.node_type is node.node_type:
            pass
        elif existing is not node:
            raise ValueError(f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}')

    def _get_new_decision_id(self) -> str:
        """Generate a unique ID for a new decision node.

        Returns:
            A unique decision node ID
        """
        node_id = f'decision_{self._decision_index}'
        self._decision_index += 1
        while node_id in self._nodes:
            node_id = f'decision_{self._decision_index}'
            self._decision_index += 1
        return node_id

    def _get_new_broadcast_id(self, from_: str | None = None) -> str:
        """Generate a unique ID for a new broadcast fork.

        Args:
            from_: Optional source identifier to include in the ID

        Returns:
            A unique broadcast fork ID
        """
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
        """Generate a unique ID for a new spread fork.

        Args:
            from_: Optional source identifier to include in the ID
            to: Optional destination identifier to include in the ID

        Returns:
            A unique spread fork ID
        """
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

    def _edge_from_return_hint(
        self, node: SourceNode[StateT, DepsT, Any], return_hint: TypeOrTypeExpression[Any]
    ) -> EdgePath[StateT, DepsT] | None:
        """Create edges from a return type hint.

        This method analyzes return type hints from step functions or node methods
        to automatically create appropriate edges in the graph.

        Args:
            node: The source node
            return_hint: The return type hint to analyze

        Returns:
            An EdgePath if edges can be inferred, None otherwise

        Raises:
            GraphSetupError: If the return type hint is invalid or incomplete
        """
        destinations: list[AnyDestinationNode] = []
        union_args = _utils.get_union_args(return_hint)
        for return_type in union_args:
            return_type, annotations = _utils.unpack_annotated(return_type)
            return_type_origin = get_origin(return_type) or return_type
            if return_type_origin is End:
                destinations.append(self.end_node)
            elif return_type_origin is BaseNode:
                raise exceptions.GraphSetupError(
                    f'Node {node} return type hint includes a plain `BaseNode`. '
                    'Edge inference requires each possible returned `BaseNode` subclass to be listed explicitly.'
                )
            elif return_type_origin is StepNode:
                step = cast(
                    Step[StateT, DepsT, Any, Any] | None,
                    next((a for a in annotations if isinstance(a, Step)), None),  # pyright: ignore[reportUnknownArgumentType]
                )
                if step is None:
                    raise exceptions.GraphSetupError(
                        f'Node {node} return type hint includes a `StepNode` without a `Step` annotation. '
                        'When returning `my_step.as_node()`, use `Annotated[StepNode[StateT, DepsT], my_step]` as the return type hint.'
                    )
                destinations.append(step)
            elif return_type_origin is JoinNode:
                join = cast(
                    Join[StateT, DepsT, Any, Any] | None,
                    next((a for a in annotations if isinstance(a, Join)), None),  # pyright: ignore[reportUnknownArgumentType]
                )
                if join is None:
                    raise exceptions.GraphSetupError(
                        f'Node {node} return type hint includes a `JoinNode` without a `Join` annotation. '
                        'When returning `my_join.as_node()`, use `Annotated[JoinNode[StateT, DepsT], my_join]` as the return type hint.'
                    )
                destinations.append(join)
            elif inspect.isclass(return_type_origin) and issubclass(return_type_origin, BaseNode):
                destinations.append(NodeStep(return_type))

        if len(destinations) < len(union_args):
            # Only build edges if all the return types are nodes
            return None

        edge = self.edge_from(node)
        if len(destinations) == 1:
            return edge.to(destinations[0])
        else:
            decision = self.decision()
            for destination in destinations:
                # We don't actually use this decision mechanism, but we need to build the edges for parent-fork finding
                decision = decision.branch(self.match(NoneType).to(destination))
            return edge.to(decision)

    # Graph building
    def build(self) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]:
        """Build the final executable graph from the accumulated nodes and edges.

        This method performs validation, normalization, and analysis of the graph
        structure to create a complete, executable graph instance.

        Returns:
            A complete Graph instance ready for execution

        Raises:
            ValueError: If the graph structure is invalid (e.g., join without parent fork)
        """
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

        return Graph[StateT, DepsT, GraphInputT, GraphOutputT](
            name=self.name,
            state_type=unpack_type_expression(self.state_type),
            deps_type=unpack_type_expression(self.deps_type),
            input_type=unpack_type_expression(self.input_type),
            output_type=unpack_type_expression(self.output_type),
            nodes=nodes,
            edges_by_source=edges_by_source,
            parent_forks=parent_forks,
            auto_instrument=self.auto_instrument,
        )


def _normalize_forks(
    nodes: dict[NodeID, AnyNode], edges: dict[NodeID, list[Path]]
) -> tuple[dict[NodeID, AnyNode], dict[NodeID, list[Path]]]:
    """Normalize the graph structure so only broadcast forks have multiple outgoing edges.

    This function ensures that any node with multiple outgoing edges is converted
    to use an explicit broadcast fork, simplifying the graph execution model.

    Args:
        nodes: The nodes in the graph
        edges: The edges by source node

    Returns:
        A tuple of normalized nodes and edges
    """
    new_nodes = nodes.copy()
    new_edges: dict[NodeID, list[Path]] = {}

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
        new_fork = Fork[Any, Any](id=ForkID(NodeID(f'{node.id}_broadcast_fork')), is_spread=False)
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
    graph_nodes: dict[NodeID, AnyNode], graph_edges_by_source: dict[NodeID, list[Path]]
) -> dict[JoinID, ParentFork[NodeID]]:
    """Find the dominating fork for each join node in the graph.

    This function analyzes the graph structure to find the parent fork that
    dominates each join node, which is necessary for proper synchronization
    during graph execution.

    Args:
        graph_nodes: All nodes in the graph
        graph_edges_by_source: Edges organized by source node

    Returns:
        A mapping from join IDs to their parent fork information

    Raises:
        ValueError: If any join node lacks a dominating fork
    """
    nodes = set(graph_nodes)
    start_ids: set[NodeID] = {StartNode.id}
    edges: dict[NodeID, list[NodeID]] = defaultdict(list)

    fork_ids: set[NodeID] = set(start_ids)
    for source_id in nodes:
        working_source_id = source_id
        node = graph_nodes.get(source_id)

        if isinstance(node, Fork):
            fork_ids.add(node.id)
            continue

        def _handle_path(path: Path, last_source_id: NodeID):
            """Process a path and collect edges and fork information.

            Args:
                path: The path to process
                last_source_id: The current source node ID
            """
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
    dominating_forks: dict[JoinID, ParentFork[NodeID]] = {}
    for join_id in join_ids:
        dominating_fork = finder.find_parent_fork(join_id)
        if dominating_fork is None:
            # TODO(P3): Print out the mermaid graph and explain the problem
            raise ValueError(f'Join node {join_id} has no dominating fork')
        dominating_forks[join_id] = dominating_fork

    return dominating_forks
