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
    """Build an adapter once for each type hint.

    The cache is replay-safe: a `TypeAdapter` is a pure function of its type hint, so cache hits and
    misses validate identically and cannot change workflow history.

    It is deliberately unbounded. Hints reach it from workflow and activity signatures and from
    explicit `result_type=` arguments, all of which are written in application code, so the key space
    is fixed by the code rather than growing with traffic. Bounding it is the wrong tool: an LRU
    smaller than the set of hints a worker cycles through has a 0% hit rate, because each lookup
    evicts the entry needed next, so it pays the full `TypeAdapter` build on every payload — the
    problem this cache exists to solve (#7027). An application that builds new type objects per
    request should reuse them instead, as each distinct hint is retained for the life of the worker.
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
