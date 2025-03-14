from __future__ import annotations as _annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Union

import pydantic
from pydantic_core import core_schema

from ..nodes import BaseNode

nodes_type_context: ContextVar[Sequence[type[BaseNode[Any, Any, Any]]]] = ContextVar('nodes_type_context')


@dataclass
class CustomNodeSchema:
    def __get_pydantic_core_schema__(
        self, _source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        try:
            nodes = nodes_type_context.get()
        except LookupError as e:
            raise RuntimeError(
                'Unable to build a Pydantic schema for `BaseNode` without setting `nodes_type_context`. '
                'You should build Pydantic schemas for snapshots using `StatePersistence.set_types()`.'
            ) from e
        if len(nodes) == 1:
            nodes_type = nodes[0]
        else:
            nodes_annotated = [Annotated[node, pydantic.Tag(node.get_node_id())] for node in nodes]
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
        node_dict['node_id'] = node.get_node_id()
        return node_dict


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


@contextmanager
def set_nodes_type_context(nodes: Sequence[type[BaseNode[Any, Any, Any]]]) -> Iterator[None]:
    token = nodes_type_context.set(nodes)
    try:
        yield
    finally:
        nodes_type_context.reset(token)
