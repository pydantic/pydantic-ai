from __future__ import annotations as _annotations

import copy
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Callable, Generic, Literal, Union

import pydantic
from pydantic_core import core_schema
from typing_extensions import TypeVar

from . import _utils
from .nodes import BaseNode, End, RunEndT

__all__ = 'StateT', 'NodeStep', 'EndStep', 'HistoryStep', 'deep_copy_state', 'nodes_schema_var'


StateT = TypeVar('StateT', default=None)
"""Type variable for the state in a graph."""


def deep_copy_state(state: StateT) -> StateT:
    """Default method for snapshotting the state in a graph run, uses [`copy.deepcopy`][copy.deepcopy]."""
    if state is None:
        return state
    else:
        return copy.deepcopy(state)


@dataclass
class NodeStep(Generic[StateT, RunEndT]):
    """History step describing the execution of a node in a graph."""

    state: StateT
    """The state of the graph after the node has been run."""
    node: Annotated[BaseNode[StateT, Any, RunEndT], CustomNodeSchema()]
    """The node that was run."""
    start_ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the node started running."""
    duration: float | None = None
    """The duration of the node run in seconds."""
    kind: Literal['node'] = 'node'
    """The kind of history step, can be used as a discriminator when deserializing history."""
    # TODO waiting for https://github.com/pydantic/pydantic/issues/11264, should be an InitVar
    snapshot_state: Annotated[Callable[[StateT], StateT], pydantic.Field(exclude=True, repr=False)] = field(
        default=deep_copy_state, repr=False
    )
    """Function to snapshot the state of the graph."""

    def __post_init__(self):
        # Copy the state to prevent it from being modified by other code
        self.state = self.snapshot_state(self.state)

    def data_snapshot(self) -> BaseNode[StateT, Any, RunEndT]:
        """Returns a deep copy of [`self.node`][pydantic_graph.state.NodeStep.node].

        Useful for summarizing history.
        """
        return copy.deepcopy(self.node)


@dataclass
class EndStep(Generic[RunEndT]):
    """History step describing the end of a graph run."""

    result: End[RunEndT]
    """The result of the graph run."""
    ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the graph run ended."""
    kind: Literal['end'] = 'end'
    """The kind of history step, can be used as a discriminator when deserializing history."""

    def data_snapshot(self) -> End[RunEndT]:
        """Returns a deep copy of [`self.result`][pydantic_graph.state.EndStep.result].

        Useful for summarizing history.
        """
        return copy.deepcopy(self.result)


HistoryStep = Union[NodeStep[StateT, RunEndT], EndStep[RunEndT]]
"""A step in the history of a graph run.

[`Graph.run`][pydantic_graph.graph.Graph.run] returns a list of these steps describing the execution of the graph,
together with the run return value.
"""


nodes_schema_var: ContextVar[Sequence[type[BaseNode[Any, Any, Any]]]] = ContextVar('nodes_var')


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
