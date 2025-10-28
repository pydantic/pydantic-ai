"""Path and edge definition for graph navigation.

This module provides the building blocks for defining paths through a graph,
including transformations, maps, broadcasts, and routing to destinations.
Paths enable complex data flow patterns in graph execution.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, get_origin

from typing_extensions import Protocol, Self, TypeAliasType, TypeVar

from pydantic_graph import BaseNode
from pydantic_graph.beta.id_types import ForkID, JoinID, NodeID, generate_placeholder_node_id
from pydantic_graph.beta.step import NodeStep, StepContext
from pydantic_graph.exceptions import GraphBuildingError

StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
T = TypeVar('T')

if TYPE_CHECKING:
    from pydantic_graph.beta.node_types import AnyDestinationNode, DestinationNode, SourceNode


class TransformFunction(Protocol[StateT, DepsT, InputT, OutputT]):
    """Protocol for step functions that can be executed in the graph.

    Transform functions are sync callables that receive a step context and return
    a result. This protocol enables serialization and deserialization of step
    calls similar to how evaluators work.

    This is very similar to a StepFunction, but must be sync instead of async.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> OutputT:
        """Execute the step function with the given context.

        Args:
            ctx: The step context containing state, dependencies, and inputs

        Returns:
            An awaitable that resolves to the step's output
        """
        raise NotImplementedError


@dataclass
class TransformMarker:
    """A marker indicating a data transformation step in a path.

    Transform markers wrap step functions that modify data as it flows
    through the graph path.
    """

    transform: TransformFunction[Any, Any, Any, Any]
    """The step function that performs the transformation."""


@dataclass
class MapMarker:
    """A marker indicating that iterable data should be map across parallel paths.

    Spread markers take iterable input and create parallel execution paths
    for each item in the iterable.
    """

    fork_id: ForkID
    """Unique identifier for the fork created by this map operation."""
    downstream_join_id: JoinID | None
    """Optional identifier of a downstream join node that should be jumped to if mapping an empty iterable."""


@dataclass
class BroadcastMarker:
    """A marker indicating that data should be broadcast to multiple parallel paths.

    Broadcast markers create multiple parallel execution paths, sending the
    same input data to each path.
    """

    paths: Sequence[Path]
    """The parallel paths that will receive the broadcast data."""

    fork_id: ForkID
    """Unique identifier for the fork created by this broadcast operation."""


@dataclass
class LabelMarker:
    """A marker providing a human-readable label for a path segment.

    Label markers are used for debugging, visualization, and documentation
    purposes to provide meaningful names for path segments.
    """

    label: str
    """The human-readable label for this path segment."""


@dataclass
class DestinationMarker:
    """A marker indicating the target destination node for a path.

    Destination markers specify where data should be routed at the end
    of a path execution.
    """

    destination_id: NodeID
    """The unique identifier of the destination node."""


PathItem = TypeAliasType('PathItem', TransformMarker | MapMarker | BroadcastMarker | LabelMarker | DestinationMarker)
"""Type alias for any item that can appear in a path sequence."""


@dataclass
class Path:
    """A sequence of path items defining data flow through the graph.

    Paths represent the route that data takes through the graph, including
    transformations, forks, and routing decisions.
    """

    items: list[PathItem]
    """The sequence of path items that define this path."""

    @property
    def last_fork(self) -> BroadcastMarker | MapMarker | None:
        """Get the most recent fork or map marker in this path.

        Returns:
            The last BroadcastMarker or MapMarker in the path, or None if no forks exist
        """
        for item in reversed(self.items):
            if isinstance(item, BroadcastMarker | MapMarker):
                return item
        return None

    @property
    def next_path(self) -> Path:
        """Create a new path with the first item removed.

        Returns:
            A new Path with all items except the first one
        """
        return Path(self.items[1:])


@dataclass
class PathBuilder(Generic[StateT, DepsT, OutputT]):
    """A builder for constructing paths with method chaining.

    PathBuilder provides a fluent interface for creating paths by chaining
    operations like transforms, maps, and routing to destinations.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        OutputT: The type of the current data in the path
    """

    working_items: Sequence[PathItem]
    """The accumulated sequence of path items being built."""

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
        fork_id: str | None = None,
    ) -> Path:
        """Route the path to one or more destination nodes.

        Args:
            destination: The primary destination node
            *extra_destinations: Additional destination nodes (creates a broadcast)
            fork_id: Optional ID for the fork created when multiple destinations are specified

        Returns:
            A complete Path ending at the specified destination(s)
        """
        if extra_destinations:
            next_item = BroadcastMarker(
                paths=[Path(items=[DestinationMarker(d.id)]) for d in (destination,) + extra_destinations],
                fork_id=ForkID(NodeID(fork_id or generate_placeholder_node_id('broadcast'))),
            )
        else:
            next_item = DestinationMarker(destination.id)
        return Path(items=[*self.working_items, next_item])

    def broadcast(self, forks: Sequence[Path], /, *, fork_id: str | None = None) -> Path:
        """Create a fork that broadcasts data to multiple parallel paths.

        Args:
            forks: The sequence of paths to run in parallel
            fork_id: Optional ID for the fork, defaults to a generated value

        Returns:
            A complete Path that forks to the specified parallel paths
        """
        next_item = BroadcastMarker(
            paths=forks, fork_id=ForkID(NodeID(fork_id or generate_placeholder_node_id('broadcast')))
        )
        return Path(items=[*self.working_items, next_item])

    def transform(self, func: TransformFunction[StateT, DepsT, OutputT, T], /) -> PathBuilder[StateT, DepsT, T]:
        """Add a transformation step to the path.

        Args:
            func: The step function that will transform the data

        Returns:
            A new PathBuilder with the transformation added
        """
        next_item = TransformMarker(func)
        return PathBuilder[StateT, DepsT, T](working_items=[*self.working_items, next_item])

    def map(
        self: PathBuilder[StateT, DepsT, Iterable[T]] | PathBuilder[StateT, DepsT, AsyncIterable[T]],
        *,
        fork_id: str | None = None,
        downstream_join_id: str | None = None,
    ) -> PathBuilder[StateT, DepsT, T]:
        """Spread iterable data across parallel execution paths.

        This method can only be called when the current output type is iterable.
        It creates parallel paths for each item in the iterable.

        Args:
            fork_id: Optional ID for the fork, defaults to a generated value
            downstream_join_id: Optional ID of a downstream join node which is involved when mapping empty iterables

        Returns:
            A new PathBuilder that operates on individual items from the iterable
        """
        next_item = MapMarker(
            fork_id=ForkID(NodeID(fork_id or generate_placeholder_node_id('map'))),
            downstream_join_id=JoinID(downstream_join_id) if downstream_join_id is not None else None,
        )
        return PathBuilder[StateT, DepsT, T](working_items=[*self.working_items, next_item])

    def label(self, label: str, /) -> PathBuilder[StateT, DepsT, OutputT]:
        """Add a human-readable label to this point in the path.

        Args:
            label: The label to add for documentation/debugging purposes

        Returns:
            A new PathBuilder with the label added
        """
        next_item = LabelMarker(label)
        return PathBuilder[StateT, DepsT, OutputT](working_items=[*self.working_items, next_item])


@dataclass(init=False)
class EdgePath(Generic[StateT, DepsT]):
    """A complete edge connecting source nodes to destinations via a path.

    EdgePath represents a complete connection in the graph, specifying the
    source nodes, the path that data follows, and the destination nodes.
    """

    _sources: Sequence[SourceNode[StateT, DepsT, Any]]
    """The source nodes that provide data to this edge."""
    path: Path
    """The path that data follows through the graph."""
    destinations: list[AnyDestinationNode]
    """The destination nodes that can be referenced by DestinationMarker in the path."""

    def __init__(
        self, sources: Sequence[SourceNode[StateT, DepsT, Any]], path: Path, destinations: list[AnyDestinationNode]
    ):
        self._sources = sources
        self.path = path
        self.destinations = destinations

    @property
    def sources(self) -> Sequence[SourceNode[StateT, DepsT, Any]]:
        return self._sources


class EdgePathBuilder(Generic[StateT, DepsT, OutputT]):
    """A builder for constructing complete edge paths with method chaining.

    EdgePathBuilder combines source nodes with path building capabilities
    to create complete edge definitions. It cannot use dataclass due to
    type variance issues.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        OutputT: The type of the current data in the path
    """

    def __init__(
        self, sources: Sequence[SourceNode[StateT, DepsT, Any]], path_builder: PathBuilder[StateT, DepsT, OutputT]
    ):
        """Initialize an edge path builder.

        Args:
            sources: The source nodes for this edge path
            path_builder: The path builder for defining the data flow
        """
        self.sources = sources
        self._path_builder = path_builder

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT] | type[BaseNode[StateT, DepsT, Any]],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT] | type[BaseNode[StateT, DepsT, Any]],
        fork_id: str | None = None,
    ) -> EdgePath[StateT, DepsT]:
        """Complete the edge path by routing to destination nodes.

        Args:
            destination: Either a destination node or a function that generates edge paths
            *extra_destinations: Additional destination nodes (creates a broadcast)
            fork_id: Optional ID for the fork created when multiple destinations are specified

        Returns:
            A complete EdgePath connecting sources to destinations
        """
        # `type[BaseNode[StateT, DepsT, Any]]` could actually be a `typing._GenericAlias` like `pydantic_ai._agent_graph.UserPromptNode[~DepsT, ~OutputT]`,
        # so we get the origin to get to the actual class
        destination = get_origin(destination) or destination
        extra_destinations = tuple(get_origin(d) or d for d in extra_destinations)
        destinations = [(NodeStep(d) if inspect.isclass(d) else d) for d in (destination, *extra_destinations)]
        return EdgePath(
            sources=self.sources,
            path=self._path_builder.to(destinations[0], *destinations[1:], fork_id=fork_id),
            destinations=destinations,
        )

    def broadcast(
        self, get_forks: Callable[[Self], Sequence[EdgePath[StateT, DepsT]]], /, *, fork_id: str | None = None
    ) -> EdgePath[StateT, DepsT]:
        """Broadcast this EdgePathBuilder into multiple destinations.

        Args:
            get_forks: The callback that will return a sequence of EdgePaths to broadcast to.
            fork_id: Optional node ID to use for the resulting broadcast fork.

        Returns:
            A completed EdgePath with the specified destinations.
        """
        new_edge_paths = get_forks(self)
        new_paths = [Path(x.path.items) for x in new_edge_paths]
        if not new_paths:
            raise GraphBuildingError(f'The call to {get_forks} returned no branches, but must return at least one.')
        path = self._path_builder.broadcast(new_paths, fork_id=fork_id)
        destinations = [d for ep in new_edge_paths for d in ep.destinations]
        return EdgePath(
            sources=self.sources,
            path=path,
            destinations=destinations,
        )

    def map(
        self: EdgePathBuilder[StateT, DepsT, Iterable[T]] | EdgePathBuilder[StateT, DepsT, AsyncIterable[T]],
        *,
        fork_id: str | None = None,
        downstream_join_id: JoinID | None = None,
    ) -> EdgePathBuilder[StateT, DepsT, T]:
        """Spread iterable data across parallel execution paths.

        Args:
            fork_id: Optional ID for the fork, defaults to a generated value
            downstream_join_id: Optional ID of a downstream join node which is involved when mapping empty iterables

        Returns:
            A new EdgePathBuilder that operates on individual items from the iterable
        """
        if len(self.sources) > 1:
            # The current implementation mishandles this because you get one copy of each edge
            # from the MapMarker to its destination for each source, resulting in unintentional multiple execution.
            # I suspect this is fixable without a major refactor, though it's not clear to me what the ideal behavior
            # would be. But for now, it's definitely easiest to just raise an error for this.
            raise NotImplementedError(
                'Map is not currently supported with multiple source nodes.'
                ' You can work around this by just creating a separate edge for each source.'
            )
        return EdgePathBuilder(
            sources=self.sources,
            path_builder=self._path_builder.map(fork_id=fork_id, downstream_join_id=downstream_join_id),
        )

    def transform(self, func: TransformFunction[StateT, DepsT, OutputT, T], /) -> EdgePathBuilder[StateT, DepsT, T]:
        """Add a transformation step to the edge path.

        Args:
            func: The step function that will transform the data

        Returns:
            A new EdgePathBuilder with the transformation added
        """
        return EdgePathBuilder(sources=self.sources, path_builder=self._path_builder.transform(func))

    def label(self, label: str) -> EdgePathBuilder[StateT, DepsT, OutputT]:
        """Add a human-readable label to this point in the edge path.

        Args:
            label: The label to add for documentation/debugging purposes

        Returns:
            A new EdgePathBuilder with the label added
        """
        return EdgePathBuilder(sources=self.sources, path_builder=self._path_builder.label(label))
