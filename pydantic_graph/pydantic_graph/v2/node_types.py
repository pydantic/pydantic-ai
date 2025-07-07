from __future__ import annotations

from typing import Any

from typing_extensions import TypeGuard

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.join import Join
from pydantic_graph.v2.node import EndNode, Fork, StartNode
from pydantic_graph.v2.step import Step

type MiddleNode[StateT, InputT, OutputT] = (
    Step[StateT, InputT, OutputT] | Join[StateT, InputT, OutputT] | Fork[InputT, OutputT]
)
type SourceNode[StateT, OutputT] = MiddleNode[StateT, Any, OutputT] | StartNode[OutputT]
type DestinationNode[StateT, InputT] = MiddleNode[StateT, InputT, Any] | Decision[StateT, InputT] | EndNode[InputT]

type AnySourceNode = SourceNode[Any, Any]
type AnyDestinationNode = DestinationNode[Any, Any]
type AnyNode = AnySourceNode | AnyDestinationNode


def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]:
    """Checks if the provided node is valid as a source."""
    return isinstance(node, (StartNode, Step, Join))


def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]:
    """Checks if the provided node is valid as a destination."""
    return isinstance(node, (EndNode, Step, Join, Decision))
