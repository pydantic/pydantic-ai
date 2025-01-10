from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from functools import cache
from typing import Any, ClassVar, Generic, get_origin, get_type_hints

from typing_extensions import Never, TypeVar

from . import _utils, exceptions
from .state import StateT

__all__ = 'GraphContext', 'BaseNode', 'End', 'Edge', 'NodeDef'

RunEndT = TypeVar('RunEndT', default=None)
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)


@dataclass
class GraphContext(Generic[StateT]):
    """Context for a graph."""

    state: StateT


class BaseNode(ABC, Generic[StateT, NodeRunEndT]):
    """Base class for a node."""

    enable_docstring_notes: ClassVar[bool] = True
    """Set to `False` to not generate mermaid diagram notes from the class's docstring."""

    @abstractmethod
    async def run(self, ctx: GraphContext[StateT]) -> BaseNode[StateT, Any] | End[NodeRunEndT]: ...

    @classmethod
    @cache
    def get_id(cls) -> str:
        return cls.__name__

    @classmethod
    def get_note(cls) -> str | None:
        if not cls.enable_docstring_notes:
            return None
        docstring = cls.__doc__
        # dataclasses get an automatic docstring which is just their signature, we don't want that
        if docstring and is_dataclass(cls) and docstring.startswith(f'{cls.__name__}('):
            docstring = None
        if docstring:
            # remove indentation from docstring
            import inspect

            docstring = inspect.cleandoc(docstring)
        return docstring

    @classmethod
    def get_node_def(cls, local_ns: dict[str, Any] | None) -> NodeDef[StateT, NodeRunEndT]:
        type_hints = get_type_hints(cls.run, localns=local_ns, include_extras=True)
        try:
            return_hint = type_hints['return']
        except KeyError as e:
            raise exceptions.GraphSetupError(f'Node {cls} is missing a return type hint on its `run` method') from e

        next_node_edges: dict[str, Edge] = {}
        end_edge: Edge | None = None
        returns_base_node: bool = False
        for return_type in _utils.get_union_args(return_hint):
            return_type, annotations = _utils.unpack_annotated(return_type)
            edge = next((a for a in annotations if isinstance(a, Edge)), Edge(None))
            return_type_origin = get_origin(return_type) or return_type
            if return_type_origin is End:
                end_edge = edge
            elif return_type_origin is BaseNode:
                # TODO: Should we disallow this?
                returns_base_node = True
            elif issubclass(return_type_origin, BaseNode):
                next_node_edges[return_type.get_id()] = edge
            else:
                raise exceptions.GraphSetupError(f'Invalid return type: {return_type}')

        return NodeDef(
            cls,
            cls.get_id(),
            cls.get_note(),
            next_node_edges,
            end_edge,
            returns_base_node,
        )


@dataclass
class End(Generic[RunEndT]):
    """Type to return from a node to signal the end of the graph."""

    data: RunEndT


@dataclass
class Edge:
    """Annotation to apply a label to an edge in a graph."""

    label: str | None


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
    note: str | None
    """Note about the node to render on mermaid charts."""
    next_node_edges: dict[str, Edge]
    """IDs of the nodes that can be called next."""
    end_edge: Edge | None
    """If node definition returns an `End` this is an Edge, indicating the node can end the run."""
    returns_base_node: bool
    """The node definition returns a `BaseNode`, hence any node in the next can be called next."""
