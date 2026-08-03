from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import TypeAdapter
from temporalio.api.common.v1 import Payload
from temporalio.contrib.pydantic import (
    PydanticJSONPlainPayloadConverter,
    PydanticPayloadConverter,
    ToJsonOptions,
)
from temporalio.converter import CompositePayloadConverter, DefaultPayloadConverter, JSONPlainPayloadConverter


@cache
def _type_adapter(type_hint: Any) -> TypeAdapter[Any]:
    """Build an adapter once per type hint.

    The cache is replay-safe: a `TypeAdapter` is a pure function of its type hint, so cache hits and
    misses validate identically and cannot change workflow history. It is unbounded because its key
    space is closed: it is the set of annotations reachable from the registered workflows and
    activities, which is fixed by the code, not by traffic. A bounded LRU below that working set
    (the original 128 entries) degrades to a 0% hit rate on large registries — exactly the workers
    the memo exists for — and every decode then rebuilds its adapter. Unhashable hints, the main
    dynamic-construction case, bypass the cache entirely.
    """
    return TypeAdapter(type_hint)


class PydanticAIJSONPlainPayloadConverter(PydanticJSONPlainPayloadConverter):
    """Pydantic JSON converter that reuses `TypeAdapter` instances during deserialization."""

    def from_payload(self, payload: Payload, type_hint: type | None = None) -> Any:
        hint = type_hint if type_hint is not None else Any
        adapter: TypeAdapter[Any]
        try:
            hash(hint)
        except TypeError:
            # Pydantic accepts some unhashable hints; they remain valid but cannot be cached.
            adapter = TypeAdapter(hint)
        else:
            adapter = _type_adapter(hint)
        return adapter.validate_json(payload.data)


class PydanticAIPayloadConverter(PydanticPayloadConverter):
    """Temporal Pydantic payload converter with memoized deserialization adapters.

    Custom payload converters can inherit from this class to retain the adapter cache while replacing
    or extending other conversion behavior.
    """

    def __init__(self, to_json_options: ToJsonOptions | None = None) -> None:
        json_payload_converter = PydanticAIJSONPlainPayloadConverter(to_json_options)
        CompositePayloadConverter.__init__(
            self,
            *(
                converter if not isinstance(converter, JSONPlainPayloadConverter) else json_payload_converter
                for converter in DefaultPayloadConverter.default_encoding_payload_converters
            ),
        )
