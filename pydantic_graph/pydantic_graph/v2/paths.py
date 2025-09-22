from __future__ import annotations

import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, overload

from typing_extensions import Self, TypeAliasType, TypeVar

from pydantic_graph.v2.id_types import ForkId, NodeId
from pydantic_graph.v2.step import StepFunction

StateT = TypeVar('StateT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)

if TYPE_CHECKING:
    from pydantic_graph.v2.node_types import AnyDestinationNode, DestinationNode, SourceNode


@dataclass
class TransformMarker:
    """A transform marker."""

    transform: StepFunction[Any, Any, Any]


@dataclass
class SpreadMarker:
    """A spread marker."""

    fork_id: ForkId


@dataclass
class BroadcastMarker:
    """A broadcast marker."""

    paths: Sequence[Path]
    fork_id: ForkId


@dataclass
class LabelMarker:
    """A label marker."""

    label: str


@dataclass
class DestinationMarker:
    """A destination marker."""

    destination_id: NodeId


PathItem = TypeAliasType('PathItem', TransformMarker | SpreadMarker | BroadcastMarker | LabelMarker | DestinationMarker)


@dataclass
class Path:
    """A path."""

    items: Sequence[PathItem]

    @property
    def last_fork(self) -> BroadcastMarker | SpreadMarker | None:
        """Returns the last fork or spread marker in the path, if any."""
        for item in reversed(self.items):
            if isinstance(item, BroadcastMarker | SpreadMarker):
                return item
        return None

    @property
    def next_path(self) -> Path:
        return Path(self.items[1:])


@dataclass
class PathBuilder(Generic[StateT, OutputT]):
    """A path builder."""

    working_items: Sequence[PathItem]

    @property
    def last_fork(self) -> BroadcastMarker | SpreadMarker | None:
        """Returns the last fork or spread marker in the path, if any."""
        for item in reversed(self.working_items):
            if isinstance(item, BroadcastMarker | SpreadMarker):
                return item
        return None

    def to(
        self,
        destination: DestinationNode[StateT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, OutputT],
        fork_id: str | None = None,
    ) -> Path:
        if extra_destinations:
            next_item = BroadcastMarker(
                paths=[Path(items=[DestinationMarker(d.id)]) for d in (destination,) + extra_destinations],
                fork_id=ForkId(NodeId(fork_id or 'extra_broadcast_' + secrets.token_hex(8))),
            )
        else:
            next_item = DestinationMarker(destination.id)
        return Path(items=[*self.working_items, next_item])

    def fork(self, forks: Sequence[Path], /, *, fork_id: str | None = None) -> Path:
        next_item = BroadcastMarker(paths=forks, fork_id=ForkId(NodeId(fork_id or 'broadcast_' + secrets.token_hex(8))))
        return Path(items=[*self.working_items, next_item])

    def transform(self, func: StepFunction[StateT, OutputT, Any], /) -> PathBuilder[StateT, Any]:
        next_item = TransformMarker(func)
        return PathBuilder[StateT, Any](working_items=[*self.working_items, next_item])

    def spread(self: PathBuilder[StateT, Iterable[Any]], *, fork_id: str | None = None) -> PathBuilder[StateT, Any]:
        next_item = SpreadMarker(fork_id=ForkId(NodeId(fork_id or 'spread_' + secrets.token_hex(8))))
        return PathBuilder[StateT, Any](working_items=[*self.working_items, next_item])

    def label(self, label: str, /) -> PathBuilder[StateT, OutputT]:
        next_item = LabelMarker(label)
        return PathBuilder[StateT, OutputT](working_items=[*self.working_items, next_item])


@dataclass
class EdgePath(Generic[StateT]):
    """An edge path."""

    sources: Sequence[SourceNode[StateT, Any]]
    path: Path
    destinations: list[AnyDestinationNode]  # can be referenced by DestinationMarker in `path.items`


class EdgePathBuilder(Generic[StateT, OutputT]):
    """This can't be a dataclass due to variance issues.

    It could probably be converted back to one once ReadOnly is available in typing_extensions.
    """

    sources: Sequence[SourceNode[StateT, Any]]

    def __init__(self, sources: Sequence[SourceNode[StateT, Any]], path_builder: PathBuilder[StateT, OutputT]):
        self.sources = sources
        self._path_builder = path_builder

    @property
    def path_builder(self) -> PathBuilder[StateT, OutputT]:
        return self._path_builder

    @property
    def last_fork_id(self) -> ForkId | None:
        last_fork = self._path_builder.last_fork
        if last_fork is None:
            return None
        return last_fork.fork_id

    @overload
    def to(
        self, get_forks: Callable[[Self], Sequence[EdgePath[StateT]]], /, *, fork_id: str | None = None
    ) -> EdgePath[StateT]: ...

    @overload
    def to(
        self, /, *destinations: DestinationNode[StateT, OutputT], fork_id: str | None = None
    ) -> EdgePath[StateT]: ...

    def to(
        self,
        first_item: DestinationNode[StateT, OutputT] | Callable[[Self], Sequence[EdgePath[StateT]]],
        /,
        *extra_destinations: DestinationNode[StateT, OutputT],
        fork_id: str | None = None,
    ) -> EdgePath[StateT]:
        if callable(first_item):
            new_edge_paths = first_item(self)
            path = self.path_builder.fork([Path(x.path.items) for x in new_edge_paths], fork_id=fork_id)
            destinations = [d for ep in new_edge_paths for d in ep.destinations]
            return EdgePath(
                sources=self.sources,
                path=path,
                destinations=destinations,
            )
        else:
            return EdgePath(
                sources=self.sources,
                path=self.path_builder.to(first_item, *extra_destinations, fork_id=fork_id),
                destinations=[first_item, *extra_destinations],
            )

    def spread(
        self: EdgePathBuilder[StateT, Iterable[Any]], fork_id: str | None = None
    ) -> EdgePathBuilder[StateT, Any]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.spread(fork_id=fork_id))

    def transform(self, func: StepFunction[StateT, OutputT, Any], /) -> EdgePathBuilder[StateT, Any]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.transform(func))

    def label(self, label: str) -> EdgePathBuilder[StateT, OutputT]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.label(label))
