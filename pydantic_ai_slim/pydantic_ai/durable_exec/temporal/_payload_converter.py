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

from pydantic_ai._event_registry import event_registry_version


@cache
def _type_adapter(type_hint: Any, event_registry_version: int) -> TypeAdapter[Any]:
    """Build an adapter once for each type hint, per event-registry version.

    The cache is replay-safe: a `TypeAdapter` is a pure function of its type hint and of the event
    registries at build time, and both are cache keys, so hits and misses validate identically and
    cannot change workflow history. The registry version is a key because a hint containing
    `AgentStreamEvent` embeds a snapshot of the registered `CustomEvent`/`CapabilityEvent` classes
    (see `event_family_schema`): without it, a worker that built an adapter before the module
    defining an event class was imported would decode that class's events as `Unknown*` for the rest
    of its life, and two workers in one deployment could disagree based on import order alone.

    It is deliberately unbounded. Hints reach it from workflow and activity signatures and from
    explicit `result_type=` arguments, all of which are written in application code, so the key space
    is fixed by the code rather than growing with traffic. Bounding it is the wrong tool: an LRU
    smaller than the set of hints a worker cycles through has a 0% hit rate, because each lookup
    evicts the entry needed next, so it pays the full `TypeAdapter` build on every payload — the
    problem this cache exists to solve (#7027). An application that builds new type objects per
    request should reuse them instead, as each distinct hint is retained for the life of the worker.
    Event classes register at import time, so the version key adds entries only during startup.
    """
    return TypeAdapter(type_hint)


def type_adapter(type_hint: Any) -> TypeAdapter[Any]:
    """The cached adapter for `type_hint`, rebuilt if an event class registered since the last build."""
    return _type_adapter(type_hint, event_registry_version())


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
            adapter = type_adapter(hint)
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
