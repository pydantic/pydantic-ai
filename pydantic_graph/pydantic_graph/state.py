from __future__ import annotations as _annotations

import copy
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Generic, Literal, Union

import pydantic
from pydantic_core import core_schema

from . import _utils
from .nodes import BaseNode, End, RunEndT

__all__ = 'NodeStep', 'EndStep', 'HistoryStep', 'nodes_schema_var'


@dataclass
class NodeStep(Generic[RunEndT]):
    """History step describing the execution of a node in a graph."""

    node: Annotated[BaseNode[Any, RunEndT], CustomNodeSchema()]
    """The node that was run."""
    start_ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the node started running."""
    duration: float | None = None
    """The duration of the node run in seconds."""
    kind: Literal['node'] = 'node'
    """The kind of history step, can be used as a discriminator when deserializing history."""

    @property
    def data(self) -> BaseNode[Any, RunEndT]:
        """Property to access the [`self.node`][pydantic_graph.state.NodeStep.node].

        Useful for summarizing history, since this property is available on all history steps.
        """
        return self.node


@dataclass
class EndStep(Generic[RunEndT]):
    """History step describing the end of a graph run."""

    result: End[RunEndT]
    """The result of the graph run."""
    ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the graph run ended."""
    kind: Literal['end'] = 'end'
    """The kind of history step, can be used as a discriminator when deserializing history."""

    @property
    def data(self) -> End[RunEndT]:
        """Returns a deep copy of [`self.result`][pydantic_graph.state.EndStep.result].

        Useful for summarizing history, since this property is available on all history steps.
        """
        return copy.deepcopy(self.result)


HistoryStep = Union[NodeStep[RunEndT], EndStep[RunEndT]]
"""A step in the history of a graph run.

[`Graph.run`][pydantic_graph.graph.Graph.run] returns a list of these steps describing the execution of the graph,
together with the run return value.
"""


nodes_schema_var: ContextVar[Sequence[type[BaseNode[Any, Any]]]] = ContextVar('nodes_var')


class CustomNodeSchema:
    def __get_pydantic_core_schema__(
        self, _source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        try:
            nodes = nodes_schema_var.get()
        except LookupError as e:
            raise RuntimeError(
                'Unable to build a Pydantic schema for `NodeStep` or `HistoryStep` without setting `nodes_schema_var`. '
                'You probably want to use '
            ) from e
        if len(nodes) == 1:
            nodes_type = nodes[0]
        else:
            nodes_annotated = [Annotated[node, pydantic.Tag(node.get_id())] for node in nodes]
            nodes_type = Annotated[Union[tuple(nodes_annotated)], pydantic.Discriminator(self._node_discriminator)]

        schema = handler(nodes_type)
        schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(
            function=self._node_serializer,
            return_schema=core_schema.dict_schema(core_schema.str_schema(), core_schema.any_schema()),
        )
        return schema

    @staticmethod
    def _node_discriminator(node_data: Any) -> str:
        return node_data.get('node_id')

    @staticmethod
    def _node_serializer(node: Any, handler: pydantic.SerializerFunctionWrapHandler) -> dict[str, Any]:
        node_dict = handler(node)
        node_dict['node_id'] = node.get_id()
        return node_dict
