"""Type definitions for graph node categories.

This module defines type aliases and utilities for categorizing nodes in the
graph execution system. It provides clear distinctions between source nodes,
destination nodes, and middle nodes, along with type guards for validation.
"""

from __future__ import annotations

from typing import Any, TypeGuard

from typing_extensions import TypeAliasType, TypeVar

from pydantic_graph.beta.decision import Decision
from pydantic_graph.beta.join import Join
from pydantic_graph.beta.node import EndNode, Fork, StartNode
from pydantic_graph.beta.step import Step

StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)

MiddleNode = TypeAliasType(
    'MiddleNode',
    Step[StateT, DepsT, InputT, OutputT] | Join[StateT, DepsT, InputT, OutputT] | Fork[InputT, OutputT],
    type_params=(StateT, DepsT, InputT, OutputT),
)
"""Type alias for nodes that can appear in the middle of a graph execution path.

Middle nodes can both receive input and produce output, making them suitable
for intermediate processing steps in the graph.
"""
SourceNode = TypeAliasType(
    'SourceNode', MiddleNode[StateT, DepsT, Any, OutputT] | StartNode[OutputT], type_params=(StateT, DepsT, OutputT)
)
"""Type alias for nodes that can serve as sources in a graph execution path.

Source nodes produce output data and can be the starting point for data flow
in the graph. This includes start nodes and middle nodes configured as sources.
"""
DestinationNode = TypeAliasType(
    'DestinationNode',
    MiddleNode[StateT, DepsT, InputT, Any] | Decision[StateT, DepsT, InputT] | EndNode[InputT],
    type_params=(StateT, DepsT, InputT),
)
"""Type alias for nodes that can serve as destinations in a graph execution path.

Destination nodes consume input data and can be the ending point for data flow
in the graph. This includes end nodes, decision nodes, and middle nodes configured as destinations.
"""

AnySourceNode = TypeAliasType('AnySourceNode', SourceNode[Any, Any, Any])
"""Type alias for source nodes with any type parameters."""

AnyDestinationNode = TypeAliasType('AnyDestinationNode', DestinationNode[Any, Any, Any])
"""Type alias for destination nodes with any type parameters."""

AnyNode = TypeAliasType('AnyNode', AnySourceNode | AnyDestinationNode)
"""Type alias for any node in the graph, regardless of its role or type parameters."""


def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]:
    """Check if a node can serve as a source in the graph.

    Source nodes are capable of producing output data and can be the starting
    point for data flow in graph execution paths.

    Args:
        node: The node to check

    Returns:
        True if the node can serve as a source, False otherwise
    """
    return isinstance(node, StartNode | Step | Join)


def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]:
    """Check if a node can serve as a destination in the graph.

    Destination nodes are capable of consuming input data and can be the ending
    point for data flow in graph execution paths.

    Args:
        node: The node to check

    Returns:
        True if the node can serve as a destination, False otherwise
    """
    return isinstance(node, EndNode | Step | Join | Decision)
