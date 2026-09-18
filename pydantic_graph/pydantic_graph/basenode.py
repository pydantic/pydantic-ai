from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Any, Generic

from typing_extensions import Never, TypeVar

__all__ = 'GraphRunContext', 'BaseNode', 'End', 'Edge', 'DepsT', 'StateT', 'RunEndT'


StateT = TypeVar('StateT', default=object)
"""Type variable for the state in a graph."""
RunEndT = TypeVar('RunEndT', covariant=True, default=object)
"""Covariant type variable for the return type of a graph [`run`][pydantic_graph.graph_builder.Graph.run]."""
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)
"""Covariant type variable for the return type of a node [`run`][pydantic_graph.basenode.BaseNode.run]."""
DepsT = TypeVar('DepsT', default=object, contravariant=True)
"""Type variable for the dependencies of a graph and node."""


@dataclass(kw_only=True)
class GraphRunContext(Generic[StateT, DepsT]):
    """Context for a graph."""

    state: StateT
    """The state of the graph."""
    deps: DepsT
    """Dependencies for the graph."""


class BaseNode(ABC, Generic[StateT, DepsT, NodeRunEndT]):
    """Base class for a graph node in a `GraphBuilder`-based state machine.

    Each subclass of `BaseNode` represents a single step in the graph. When executed,
    the graph runtime passes a [`GraphRunContext`][pydantic_graph.GraphRunContext]
    containing the shared state and dependencies to the node's [`run`][pydantic_graph.basenode.BaseNode.run]
    method. The method returns either the **next node** to execute or an
    [`End`][pydantic_graph.basenode.End] marker to signal the graph is complete.

    Nodes are registered in a [`GraphBuilder`][pydantic_graph.GraphBuilder] via
    `g.node(MyNode)` which creates an edge path. The builder reads the return type
    annotation of `run()` at runtime to determine which nodes can follow.

    Type Parameters:
        StateT: The type of the shared state passed through [`GraphRunContext`][pydantic_graph.GraphRunContext].
            Use your state dataclass here.
        DepsT: The type of the dependencies passed through [`GraphRunContext`][pydantic_graph.GraphRunContext].
        NodeRunEndT: The output type carried by the [`End`][pydantic_graph.basenode.End] marker
            returned by this node. Usually the same as `StateT`.

    Example:
        ```python
        from dataclasses import dataclass
        from pydantic_graph import BaseNode, End, GraphRunContext, GraphBuilder

        @dataclass
        class MyState:
            count: int = 0

        @dataclass
        class Increment(BaseNode[MyState, None, MyState]):
            async def run(self, ctx: GraphRunContext[MyState, None]) -> 'End[MyState]':
                ctx.state.count += 1
                return End(MyState(count=ctx.state.count))
        ```
    """

    @abstractmethod
    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[NodeRunEndT]:
        """Run the node.

        This is an abstract method that must be implemented by subclasses.

        !!! note "Return types used at runtime"
            The return type of this method are read by `pydantic_graph` at runtime and used to define which
            nodes can be called next in the graph, and enforced when running the graph.

        Args:
            ctx: The graph context.

        Returns:
            The next node to run or [`End`][pydantic_graph.basenode.End] to signal the end of the graph.
        """
        ...

    @classmethod
    @cache
    def get_node_id(cls) -> str:
        """Get the ID of the node."""
        return cls.__name__


@dataclass
class End(Generic[RunEndT]):
    """Type to return from a node to signal the end of the graph."""

    data: RunEndT
    """Data to return from the graph."""


@dataclass(frozen=True)
class Edge:
    """Annotation to apply a label to an edge in a graph."""

    label: str | None
    """Label for the edge."""
