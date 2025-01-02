from __future__ import annotations as _annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Any, Generic, get_origin, get_type_hints

from typing_extensions import Never, TypeVar

from . import _utils
from .state import StateT

__all__ = ('GraphContext', 'End', 'BaseNode', 'NodeDef')

RunEndT = TypeVar('RunEndT', default=None)
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)


@dataclass
class GraphContext(Generic[StateT]):
    """Context for a graph."""

    state: StateT


@dataclass
class End(Generic[RunEndT]):
    """Type to return from a node to signal the end of the graph."""

    data: RunEndT


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
        next_node_ids: set[str] = set()
        returns_end: bool = False
        returns_base_node: bool = False
        try:
            return_hint = type_hints['return']
        except KeyError:
            raise TypeError(f'Node {cls} is missing a return type hint on its `run` method')

        for return_type in _utils.get_union_args(return_hint):
            return_type_origin = get_origin(return_type) or return_type
            if return_type_origin is End:
                returns_end = True
            elif return_type_origin is BaseNode:
                # TODO: Should we disallow this? More generally, what do we do about sub-subclasses?
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
class NodeDef(Generic[StateT, NodeRunEndT]):
    """Definition of a node.

    Used by [`Graph`][pydantic_ai_graph.graph.Graph] store information about a node.
    """

    node: type[BaseNode[StateT, NodeRunEndT]]
    node_id: str
    next_node_ids: set[str]
    returns_end: bool
    returns_base_node: bool
