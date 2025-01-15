from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Generic, get_origin, get_type_hints

from typing_extensions import Never, TypeVar

from . import _utils, exceptions

if TYPE_CHECKING:
    from .state import StateT
else:
    StateT = TypeVar('StateT', default=None)

__all__ = 'GraphContext', 'BaseNode', 'End', 'Edge', 'NodeDef', 'RunEndT', 'NodeRunEndT'

RunEndT = TypeVar('RunEndT', default=None)
"""Type variable for the return type of a graph [`run`][pydantic_graph.graph.Graph.run]."""
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)
"""Type variable for the return type of a node [`run`][pydantic_graph.nodes.BaseNode.run]."""


@dataclass
class GraphContext(Generic[StateT]):
    """Context for a graph."""

    state: StateT
    """The state of the graph."""


class BaseNode(ABC, Generic[StateT, NodeRunEndT]):
    """Base class for a node."""

    docstring_notes: ClassVar[bool] = False
    """Set to `True` to generate mermaid diagram notes from the class's docstring.

    While this can add valuable information to the diagram, it can make diagrams harder to view, hence
    it is disabled by default. You can also customise notes overriding the
    [`get_note`][pydantic_graph.nodes.BaseNode.get_note] method.
    """

    @abstractmethod
    async def run(self, ctx: GraphContext[StateT]) -> BaseNode[StateT, Any] | End[NodeRunEndT]:
        """Run the node.

        This is an abstract method that must be implemented by subclasses.

        !!! note "Return types used at runtime"
            The return type of this method are read by `pydantic_graph` at runtime and used to define which
            nodes can be called next in the graph. This is displayed in [mermaid diagrams](mermaid.md)
            and enforced when running the graph.

        Args:
            ctx: The graph context.

        Returns:
            The next node to run or [`End`][pydantic_graph.nodes.End] to signal the end of the graph.
        """
        ...

    @classmethod
    @cache
    def get_id(cls) -> str:
        """Get the ID of the node."""
        return cls.__name__

    @classmethod
    def get_note(cls) -> str | None:
        """Get a note about the node to render on mermaid charts.

        By default, this returns a note only if [`docstring_notes`][pydantic_graph.nodes.BaseNode.docstring_notes]
        is `True`. You can override this method to customise the node notes.
        """
        if not cls.docstring_notes:
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
        """Get the node definition."""
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
    """Data to return from the graph."""


@dataclass
class Edge:
    """Annotation to apply a label to an edge in a graph."""

    label: str | None
    """Label for the edge."""


@dataclass
class NodeDef(Generic[StateT, NodeRunEndT]):
    """Definition of a node.

    This is an internal representation of a node, it shouldn't be necessary to use it directly.

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