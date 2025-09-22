from __future__ import annotations

from typing import Any, TypeGuard

from typing_extensions import TypeAliasType, TypeVar

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.join import Join
from pydantic_graph.v2.node import EndNode, Fork, StartNode
from pydantic_graph.v2.step import Step

StateT = TypeVar('StateT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)

MiddleNode = TypeAliasType(
    'MiddleNode',
    Step[StateT, InputT, OutputT] | Join[StateT, InputT, OutputT] | Fork[InputT, OutputT],
    type_params=(StateT, InputT, OutputT),
)
SourceNode = TypeAliasType(
    'SourceNode', MiddleNode[StateT, Any, OutputT] | StartNode[OutputT], type_params=(StateT, OutputT)
)
DestinationNode = TypeAliasType(
    'DestinationNode',
    MiddleNode[StateT, InputT, Any] | Decision[StateT, InputT] | EndNode[InputT],
    type_params=(StateT, InputT),
)

AnySourceNode = TypeAliasType('AnySourceNode', SourceNode[Any, Any])
AnyDestinationNode = TypeAliasType('AnyDestinationNode', DestinationNode[Any, Any])
AnyNode = TypeAliasType('AnyNode', AnySourceNode | AnyDestinationNode)


def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]:
    """Checks if the provided node is valid as a source."""
    return isinstance(node, StartNode | Step | Join)


def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]:
    """Checks if the provided node is valid as a destination."""
    return isinstance(node, EndNode | Step | Join | Decision)
