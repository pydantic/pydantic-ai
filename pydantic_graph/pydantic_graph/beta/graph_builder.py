"""Graph builder for constructing executable graph definitions.

This module provides the GraphBuilder class and related utilities for
constructing typed, executable graph definitions with steps, joins,
decisions, and edge routing.
"""

from __future__ import annotations

import inspect
from collections import Counter, defaultdict
from collections.abc import AsyncIterable, Callable, Iterable
from dataclasses import dataclass, replace
from types import NoneType
from typing import Any, Generic, Literal, cast, get_origin, get_type_hints, overload

from typing_extensions import Never, TypeAliasType, TypeVar

from pydantic_graph import _utils, exceptions
from pydantic_graph._utils import UNSET, Unset
from pydantic_graph.beta.decision import Decision, DecisionBranch, DecisionBranchBuilder
from pydantic_graph.beta.graph import Graph
from pydantic_graph.beta.id_types import ForkID, JoinID, NodeID, generate_placeholder_node_id, replace_placeholder_id
from pydantic_graph.beta.join import Join, JoinNode, ReducerFunction
from pydantic_graph.beta.mermaid import build_mermaid_graph
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
    MapMarker,
    Path,
    PathBuilder,
)
from pydantic_graph.beta.step import NodeStep, Step, StepContext, StepFunction, StepNode, StreamFunction
from pydantic_graph.beta.util import TypeOrTypeExpression, get_callable_name, unpack_type_expression
from pydantic_graph.exceptions import GraphBuildingError, GraphValidationError
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

            def decorator(
                func: StepFunction[StateT, DepsT, InputT, OutputT],
            ) -> Step[StateT, DepsT, InputT, OutputT]:
                return self.step(call=func, node_id=node_id, label=label)

            return decorator

        node_id = node_id or get_callable_name(call)

        step = Step[StateT, DepsT, InputT, OutputT](id=NodeID(node_id), call=call, label=label)

        return step

    @overload
    def stream(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[
        [StreamFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
    ]: ...
    @overload
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]: ...
    @overload
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
        | Callable[
            [StreamFunction[StateT, DepsT, InputT, OutputT]],
            Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
        ]
    ): ...
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
        | Callable[
            [StreamFunction[StateT, DepsT, InputT, OutputT]],
            Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
        ]
    ):
        """Create a step from an async iterator (which functions like a "stream").

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

            def decorator(
                func: StreamFunction[StateT, DepsT, InputT, OutputT],
            ) -> Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]:
                return self.stream(call=func, node_id=node_id, label=label)

            return decorator

        # We need to wrap the call so that we can call `await` even though the result is an async iterator
        async def wrapper(ctx: StepContext[StateT, DepsT, InputT]):
            return call(ctx)

        return self.step(call=wrapper, node_id=node_id, label=label)

    @overload
    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial: OutputT,
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...
    @overload
    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial_factory: Callable[[], OutputT],
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...

    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial: OutputT | Unset = UNSET,
        initial_factory: Callable[[], OutputT] | Unset = UNSET,
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]:
        if initial_factory is UNSET:
            initial_factory = lambda: initial  # pyright: ignore[reportAssignmentType]  # noqa E731

        return Join[StateT, DepsT, InputT, OutputT](
            id=JoinID(NodeID(node_id or generate_placeholder_node_id(get_callable_name(reducer)))),
            reducer=reducer,
            initial_factory=cast(Callable[[], OutputT], initial_factory),
            parent_fork_id=ForkID(parent_fork_id) if parent_fork_id is not None else None,
            preferred_parent_fork=preferred_parent_fork,
        )

    # Edge building
    def add(self, *edges: EdgePath[StateT, DepsT]) -> None:  # noqa C901
        """Add one or more edge paths to the graph.

        This method processes edge paths and automatically creates any necessary
        fork nodes for broadcasts and maps.

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
                    new_node = Fork[Any, Any](id=item.fork_id, is_map=False, downstream_join_id=None)
                    self._insert_node(new_node)
                    for path in item.paths:
                        _handle_path(Path(items=[*path.items]))
                elif isinstance(item, MapMarker):
                    new_node = Fork[Any, Any](id=item.fork_id, is_map=True, downstream_join_id=item.downstream_join_id)
                    self._insert_node(new_node)
                elif isinstance(item, DestinationMarker):
                    pass

        def _handle_destination_node(d: AnyDestinationNode):
            if id(d) in destination_ids:
                return  # prevent infinite recursion if there is a cycle of decisions

            destination_ids.add(id(d))
            destinations.append(d)
            self._insert_node(d)
            if isinstance(d, Decision):
                for branch in d.branches:
                    _handle_path(branch.path)
                    for d2 in branch.destinations:
                        _handle_destination_node(d2)

        destination_ids = set[int]()
        destinations: list[AnyDestinationNode] = []
        for edge in edges:
            for source_node in edge.sources:
                self._insert_node(source_node)
                self._edges_by_source[source_node.id].append(edge.path)
            for destination_node in edge.destinations:
                _handle_destination_node(destination_node)
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

    def add_mapping_edge(
        self,
        source: Source[Iterable[T]],
        map_to: Destination[T],
        *,
        pre_map_label: str | None = None,
        post_map_label: str | None = None,
        fork_id: ForkID | None = None,
        downstream_join_id: JoinID | None = None,
    ) -> None:
        """Add an edge that maps iterable data across parallel paths.

        Args:
            source: The source node that produces iterable data
            map_to: The destination node that receives individual items
            pre_map_label: Optional label before the map operation
            post_map_label: Optional label after the map operation
            fork_id: Optional ID for the fork node produced for this map operation
            downstream_join_id: Optional ID of a join node that will always be downstream of this map.
                Specifying this ensures correct handling if you try to map an empty iterable.
        """
        builder = self.edge_from(source)
        if pre_map_label is not None:
            builder = builder.label(pre_map_label)
        builder = builder.map(fork_id=fork_id, downstream_join_id=downstream_join_id)
        if post_map_label is not None:
            builder = builder.label(post_map_label)
        self.add(builder.to(map_to))

    # TODO(DavidM): Support adding subgraphs; I think this behaves like a step with the same inputs/outputs but gets rendered as a subgraph in mermaid

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

    def decision(self, *, note: str | None = None, node_id: str | None = None) -> Decision[StateT, DepsT, Never]:
        """Create a new decision node.

        Args:
            note: Optional note to describe the decision logic
            node_id: Optional ID for the node produced for this decision logic

        Returns:
            A new Decision node with no branches
        """
        return Decision(id=NodeID(node_id or generate_placeholder_node_id('decision')), branches=[], note=note)

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
        # Note, the following node_id really is just a placeholder and shouldn't end up in the final graph
        # This is why we don't expose a way for end users to override the value used here.
        node_id = NodeID(generate_placeholder_node_id('match_decision'))
        decision = Decision[StateT, DepsT, Never](id=node_id, branches=[], note=None)
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
        node = NodeStep(source)
        path = Path(items=[DestinationMarker(node.id)])
        return DecisionBranch(source=source, matches=matches, path=path, destinations=[node])

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
        except KeyError as e:  # pragma: no cover
            raise exceptions.GraphSetupError(
                f'Node {node_type} is missing a return type hint on its `run` method'
            ) from e

        node = NodeStep(node_type)

        edge = self._edge_from_return_hint(node, return_hint)
        if not edge:  # pragma: no cover
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
            raise GraphBuildingError(
                f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}'
            )

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
                raise exceptions.GraphSetupError(  # pragma: no cover
                    f'Node {node} return type hint includes a plain `BaseNode`. '
                    'Edge inference requires each possible returned `BaseNode` subclass to be listed explicitly.'
                )
            elif return_type_origin is StepNode:
                step = cast(
                    Step[StateT, DepsT, Any, Any] | None,
                    next((a for a in annotations if isinstance(a, Step)), None),  # pyright: ignore[reportUnknownArgumentType]
                )
                if step is None:
                    raise exceptions.GraphSetupError(  # pragma: no cover
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
                    raise exceptions.GraphSetupError(  # pragma: no cover
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
    def build(self, validate_graph_structure: bool = True) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]:
        """Build the final executable graph from the accumulated nodes and edges.

        This method performs validation, normalization, and analysis of the graph
        structure to create a complete, executable graph instance.

        Args:
            validate_graph_structure: whether to perform validation of the graph structure
                See the docstring of _validate_graph_structure below for more details.

        Returns:
            A complete Graph instance ready for execution

        Raises:
            ValueError: If the graph structure is invalid (e.g., join without parent fork)
        """
        nodes = self._nodes
        edges_by_source = self._edges_by_source

        nodes, edges_by_source = _replace_placeholder_node_ids(nodes, edges_by_source)
        nodes, edges_by_source = _flatten_paths(nodes, edges_by_source)
        nodes, edges_by_source = _normalize_forks(nodes, edges_by_source)
        if validate_graph_structure:
            _validate_graph_structure(nodes, edges_by_source)
        parent_forks = _collect_dominating_forks(nodes, edges_by_source)
        intermediate_join_nodes = _compute_intermediate_join_nodes(nodes, parent_forks)

        return Graph[StateT, DepsT, GraphInputT, GraphOutputT](
            name=self.name,
            state_type=unpack_type_expression(self.state_type),
            deps_type=unpack_type_expression(self.deps_type),
            input_type=unpack_type_expression(self.input_type),
            output_type=unpack_type_expression(self.output_type),
            nodes=nodes,
            edges_by_source=edges_by_source,
            parent_forks=parent_forks,
            intermediate_join_nodes=intermediate_join_nodes,
            auto_instrument=self.auto_instrument,
        )


def _validate_graph_structure(  # noqa C901
    nodes: dict[NodeID, AnyNode],
    edges_by_source: dict[NodeID, list[Path]],
) -> None:
    """Validate the graph structure for common issues.

    This function raises an error if any of the following criteria are not met:
    1. There are edges from the start node
    2. There are edges to the end node
    3. No non-End node is a dead end (no outgoing edges)
    4. The end node is reachable from the start node
    5. All nodes are reachable from the start node

    Note 1: Under some circumstances it may be reasonable to build a graph that violates one or more of
    the above conditions. We may eventually add support for more granular control over validation,
    but today, if you want to build a graph that violates any of these assumptions you need to pass
    `validate_graph_structure=False` to the call to `GraphBuilder.build`.

    Note 2: Some of the earlier items in the above list are redundant with the later items.
    I've included the earlier items in the list as a reminder to ourselves if/when we add more granular validation
    because you might want to check the earlier items but not the later items, as described in Note 1.

    Args:
        nodes: The nodes in the graph
        edges_by_source: The edges by source node

    Raises:
        GraphBuildingError: If any of the aforementioned structural issues are found.
    """
    how_to_suppress = ' If this is intentional, you can suppress this error by passing `validate_graph_structure=False` to the call to `GraphBuilder.build`.'

    # Extract all destination IDs from edges and decision branches
    all_destinations: set[NodeID] = set()

    def _collect_destinations_from_path(path: Path) -> None:
        for item in path.items:
            if isinstance(item, DestinationMarker):
                all_destinations.add(item.destination_id)

    for paths in edges_by_source.values():
        for path in paths:
            _collect_destinations_from_path(path)

    # Also collect destinations from decision branches
    for node in nodes.values():
        if isinstance(node, Decision):
            for branch in node.branches:
                _collect_destinations_from_path(branch.path)

    # Check 1: Check if there are edges from the start node
    start_edges = edges_by_source.get(StartNode.id, [])
    if not start_edges:
        raise GraphValidationError('The graph has no edges from the start node.' + how_to_suppress)

    # Check 2: Check if there are edges to the end node
    if EndNode.id not in all_destinations:
        raise GraphValidationError('The graph has no edges to the end node.' + how_to_suppress)

    # Check 3: Find all nodes with no outgoing edges (dead ends)
    dead_end_nodes: list[NodeID] = []
    for node_id, node in nodes.items():
        # Skip the end node itself
        if isinstance(node, EndNode):
            continue

        # Check if this node has any outgoing edges
        has_edges = node_id in edges_by_source and len(edges_by_source[node_id]) > 0

        # Also check if it's a decision node with branches
        if isinstance(node, Decision):
            has_edges = has_edges or len(node.branches) > 0

        if not has_edges:
            dead_end_nodes.append(node_id)

    if dead_end_nodes:
        raise GraphValidationError(f'The following nodes have no outgoing edges: {dead_end_nodes}.' + how_to_suppress)

    # Checks 4 and 5: Ensure all nodes (and in particular, the end node) are reachable from the start node
    reachable: set[NodeID] = {StartNode.id}
    to_visit = [StartNode.id]

    while to_visit:
        current_id = to_visit.pop()

        # Add destinations from regular edges
        for path in edges_by_source.get(current_id, []):
            for item in path.items:
                if isinstance(item, DestinationMarker):
                    if item.destination_id not in reachable:
                        reachable.add(item.destination_id)
                        to_visit.append(item.destination_id)

        # Add destinations from decision branches
        current_node = nodes.get(current_id)
        if isinstance(current_node, Decision):
            for branch in current_node.branches:
                for item in branch.path.items:
                    if isinstance(item, DestinationMarker):
                        if item.destination_id not in reachable:
                            reachable.add(item.destination_id)
                            to_visit.append(item.destination_id)

    unreachable_nodes = [node_id for node_id in nodes if node_id not in reachable]
    if unreachable_nodes:
        raise GraphValidationError(
            f'The following nodes are not reachable from the start node: {unreachable_nodes}.' + how_to_suppress
        )


def _flatten_paths(
    nodes: dict[NodeID, AnyNode], edges: dict[NodeID, list[Path]]
) -> tuple[dict[NodeID, AnyNode], dict[NodeID, list[Path]]]:
    new_nodes = nodes.copy()
    new_edges: dict[NodeID, list[Path]] = defaultdict(list)

    paths_to_handle: list[tuple[NodeID, Path]] = []

    def _split_at_first_fork(path: Path) -> tuple[Path, list[tuple[NodeID, Path]]]:
        for i, item in enumerate(path.items):
            if isinstance(item, MapMarker):
                assert item.fork_id in nodes, 'This should have been added to the node during GraphBuilder.add'
                upstream = Path(list(path.items[:i]) + [DestinationMarker(item.fork_id)])
                downstream = Path(path.items[i + 1 :])
                return upstream, [(item.fork_id, downstream)]

            if isinstance(item, BroadcastMarker):
                assert item.fork_id in nodes, 'This should have been added to the node during GraphBuilder.add'
                upstream = Path(list(path.items[:i]) + [DestinationMarker(item.fork_id)])
                return upstream, [(item.fork_id, p) for p in item.paths]
        return path, []

    for node in new_nodes.values():
        if isinstance(node, Decision):
            for branch in node.branches:
                upstream, downstreams = _split_at_first_fork(branch.path)
                branch.path = upstream
                paths_to_handle.extend(downstreams)

    for source_id, edges_from_source in edges.items():
        for path in edges_from_source:
            paths_to_handle.append((source_id, path))

    while paths_to_handle:
        source_id, path = paths_to_handle.pop()
        upstream, downstreams = _split_at_first_fork(path)
        new_edges[source_id].append(upstream)
        paths_to_handle.extend(downstreams)

    return new_nodes, dict(new_edges)


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
        if isinstance(node, Fork) and not node.is_map:
            new_edges[source_id] = edges_from_source
            continue  # broadcast fork; nothing to do
        if len(edges_from_source) == 1:
            new_edges[source_id] = edges_from_source
            continue
        new_fork = Fork[Any, Any](id=ForkID(NodeID(f'{node.id}_broadcast_fork')), is_map=False, downstream_join_id=None)
        new_nodes[new_fork.id] = new_fork
        new_edges[source_id] = [Path(items=[DestinationMarker(new_fork.id)])]
        new_edges[new_fork.id] = edges_from_source

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

        def _handle_path(path: Path, last_source_id: NodeID):
            """Process a path and collect edges and fork information.

            Args:
                path: The path to process
                last_source_id: The current source node ID
            """
            for item in path.items:  # pragma: no branch
                # No need to handle MapMarker or BroadcastMarker here as these should have all been removed
                # by the call to `_flatten_paths`
                if isinstance(item, DestinationMarker):
                    edges[last_source_id].append(item.destination_id)
                    # Destinations should only ever occur as the last item in the list, so no need to update the working_source_id
                    break

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

    joins = [node for node in graph_nodes.values() if isinstance(node, Join)]
    dominating_forks: dict[JoinID, ParentFork[NodeID]] = {}
    for join in joins:
        dominating_fork = finder.find_parent_fork(
            join.id, parent_fork_id=join.parent_fork_id, prefer_closest=join.preferred_parent_fork == 'closest'
        )
        if dominating_fork is None:
            rendered_mermaid_graph = build_mermaid_graph(graph_nodes, graph_edges_by_source).render()
            raise GraphBuildingError(f"""A node in the graph is missing a dominating fork.

For every Join J in the graph, there must be a Fork F between the StartNode and J satisfying:
* Every path from the StartNode to J passes through F
* There are no cycles in the graph including J that don't pass through F.
In this case, F is called a "dominating fork" for J.

This is used to determine when all tasks upstream of this Join are complete and we can proceed with execution.

Mermaid diagram:
{rendered_mermaid_graph}

Join {join.id!r} in this graph has no dominating fork in this graph.""")
        dominating_forks[join.id] = dominating_fork

    return dominating_forks


def _compute_intermediate_join_nodes(
    nodes: dict[NodeID, AnyNode], parent_forks: dict[JoinID, ParentFork[NodeID]]
) -> dict[JoinID, set[JoinID]]:
    """Compute which joins have other joins as intermediate nodes.

    A join J1 is an intermediate node of join J2 if J1 appears in J2's intermediate_nodes
    (as computed relative to J2's parent fork).

    This information is used to determine:
    1. Which joins are "final" (have no other joins in their intermediate_nodes)
    2. When selecting which reducer to proceed with when there are no active tasks

    Args:
        nodes: All nodes in the graph
        parent_forks: Parent fork information for each join

    Returns:
        A mapping from each join to the set of joins that are intermediate to it
    """
    intermediate_join_nodes: dict[JoinID, set[JoinID]] = {}

    for join_id, parent_fork in parent_forks.items():
        intermediate_joins = set[JoinID]()
        for intermediate_node_id in parent_fork.intermediate_nodes:
            # Check if this intermediate node is also a join
            intermediate_node = nodes.get(intermediate_node_id)
            if isinstance(intermediate_node, Join):
                # Add it regardless of whether it has the same parent fork
                intermediate_joins.add(JoinID(intermediate_node_id))
        intermediate_join_nodes[join_id] = intermediate_joins

    return intermediate_join_nodes


def _replace_placeholder_node_ids(nodes: dict[NodeID, AnyNode], edges_by_source: dict[NodeID, list[Path]]):
    node_id_remapping = _build_placeholder_node_id_remapping(nodes)
    replaced_nodes = {
        node_id_remapping.get(name, name): _update_node_with_id_remapping(node, node_id_remapping)
        for name, node in nodes.items()
    }
    replaced_edges_by_source = {
        node_id_remapping.get(source, source): [_update_path_with_id_remapping(p, node_id_remapping) for p in paths]
        for source, paths in edges_by_source.items()
    }
    return replaced_nodes, replaced_edges_by_source


def _build_placeholder_node_id_remapping(nodes: dict[NodeID, AnyNode]) -> dict[NodeID, NodeID]:
    """The determinism of the generated remapping here is dependent on the determinism of the ordering of the `nodes` dict.

    Note: If we want to generate more interesting names, we could try to make use of information about the edges
    into/out of the relevant nodes. I'm not sure if there's a good use case for that though so I didn't bother for now.
    """
    counter = Counter[str]()
    remapping: dict[NodeID, NodeID] = {}
    for node_id in nodes.keys():
        replaced_node_id = replace_placeholder_id(node_id)
        if replaced_node_id == node_id:
            continue
        counter[replaced_node_id] = count = counter[replaced_node_id] + 1
        remapping[node_id] = NodeID(f'{replaced_node_id}_{count}' if count > 1 else replaced_node_id)
    return remapping


def _update_node_with_id_remapping(node: AnyNode, node_id_remapping: dict[NodeID, NodeID]) -> AnyNode:
    # Note: it's a bit awkward that we mutate the provided nodes, but this is necessary to ensure that
    # calls to `.as_node` reference the correct node_ids when relying on compatibility with the v1 API.
    # We only mutate placeholder IDs so I _think_ this should generally be okay. I guess we can
    # rework it more carefully if it causes issues in the future..
    if isinstance(node, Step):
        node.id = node_id_remapping.get(node.id, node.id)
    elif isinstance(node, Join):
        node.id = JoinID(node_id_remapping.get(node.id, node.id))
    elif isinstance(node, Fork):
        node.id = ForkID(node_id_remapping.get(node.id, node.id))
        if node.downstream_join_id is not None:
            node.downstream_join_id = JoinID(node_id_remapping.get(node.downstream_join_id, node.downstream_join_id))
    elif isinstance(node, Decision):
        node.id = node_id_remapping.get(node.id, node.id)
        node.branches = [
            replace(branch, path=_update_path_with_id_remapping(branch.path, node_id_remapping))
            for branch in node.branches
        ]
    return node


def _update_path_with_id_remapping(path: Path, node_id_remapping: dict[NodeID, NodeID]) -> Path:
    # Note: we have already deepcopied the node provided to this function so it should be okay to make mutations,
    # this could change if we change the code surrounding the code paths leading to this function call though.
    for item in path.items:
        if isinstance(item, MapMarker):
            downstream_join_id = item.downstream_join_id
            if downstream_join_id is not None:
                item.downstream_join_id = JoinID(node_id_remapping.get(downstream_join_id, downstream_join_id))
            item.fork_id = ForkID(node_id_remapping.get(item.fork_id, item.fork_id))
        elif isinstance(item, BroadcastMarker):
            item.fork_id = ForkID(node_id_remapping.get(item.fork_id, item.fork_id))
            item.paths = [_update_path_with_id_remapping(p, node_id_remapping) for p in item.paths]
        elif isinstance(item, DestinationMarker):
            item.destination_id = node_id_remapping.get(item.destination_id, item.destination_id)
    return path
