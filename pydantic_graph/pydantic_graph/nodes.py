from __future__ import annotations as _annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Any, Generic, get_origin, get_type_hints

from typing_extensions import Never, TypeVar

from . import _utils
from .state import StateT

__all__ = 'GraphContext', 'BaseNode', 'End', 'NodeDef'

RunEndT = TypeVar('RunEndT', default=None)
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)


@dataclass
class GraphContext(Generic[StateT]):
    """Context for a graph."""

    state: StateT


class BaseNode(Generic[StateT, NodeRunEndT]):
    """Base class for a node."""

    @abstractmethod
    async def run(self, ctx: GraphContext[StateT]) -> BaseNode[StateT, Any] | End[NodeRunEndT]: ...

    @classmethod
    @cache
    def get_id(cls) -> str:
        return cls.__name__

    @classmethod
    def get_node_def(cls, local_ns: dict[str, Any] | None) -> NodeDef[StateT, NodeRunEndT]:
        type_hints = get_type_hints(cls.run, localns=local_ns)
        try:
            return_hint = type_hints['return']
        except KeyError:
            raise TypeError(f'Node {cls} is missing a return type hint on its `run` method')

        next_node_ids: set[str] = set()
        returns_end: bool = False
        returns_base_node: bool = False
        for return_type in _utils.get_union_args(return_hint):
            return_type_origin = get_origin(return_type) or return_type
            if return_type_origin is End:
                returns_end = True
            elif return_type_origin is BaseNode:
                # TODO: Should we disallow this?
                returns_base_node = True
            elif issubclass(return_type_origin, BaseNode):
                next_node_ids.add(return_type.get_id())
            else:
                raise TypeError(f'Invalid return type: {return_type}')

        return NodeDef(
            cls,
            cls.get_id(),
            next_node_ids,
            returns_end,
            returns_base_node,
        )


@dataclass
class End(Generic[RunEndT]):
    """Type to return from a node to signal the end of the graph."""

    data: RunEndT


@dataclass
class NodeDef(Generic[StateT, NodeRunEndT]):
    """Definition of a node.

    Used by [`Graph`][pydantic_graph.graph.Graph] to store information about a node, and when generating
    mermaid graphs.
    """

    node: type[BaseNode[StateT, NodeRunEndT]]
    """The node definition itself."""
    node_id: str
    """ID of the node."""
    next_node_ids: set[str]
    """IDs of the nodes that can be called next."""
    returns_end: bool
    """The node definition returns an `End`, hence the node can end the run."""
    returns_base_node: bool
    """The node definition returns a `BaseNode`, hence any node in the next can be called next."""
