"""Serialization codec for the durable-execution base assembly (prototype).

A `DurabilityCodec` is the single seam the base uses at every serialization site. It splits the
two engine families cleanly:

- **object-passing** engines (Temporal, DBOS, Prefect) hand live Python objects to their durable
  primitive and let the primitive's own serializer persist them. They use `IDENTITY_CODEC`, which
  returns values unchanged and ignores the type.
- **JSON-journal** engines (Restate, AWS Lambda, Absurd) write JSON bytes to a journal and must
  reduce each value to a JSON-able shape first. They use `JSON_CODEC`, which round-trips through a
  cached `TypeAdapter(tp)`.

The base calls `dump(tp, value)` *inside* the durable unit (so a non-serializable payload fails in
the step, the same in production as in tests) and `load(tp, payload)` *outside* it, mirroring what
the JSON engines do by hand today (dump inside `_inner`, validate outside).
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import TypeAdapter


class DurabilityCodec(Protocol):
    """Reduces a value to a durably-storable payload and rebuilds it on the other side.

    `tp` is a "type form" (a class, or an annotated/aliased type like `CallToolResult`), i.e.
    anything `TypeAdapter` accepts -- so it is typed `Any`, not `type[T]`.
    """

    def dump(self, tp: Any, value: Any) -> Any: ...

    def load(self, tp: Any, payload: Any) -> Any: ...


class _IdentityCodec:
    """Passes values through untouched; the engine's durable primitive owns serialization."""

    def dump(self, tp: Any, value: Any) -> Any:
        return value

    def load(self, tp: Any, payload: Any) -> Any:
        return payload


class _JsonCodec:
    """Round-trips values through a cached `TypeAdapter` so they journal as JSON."""

    def __init__(self) -> None:
        self._adapters: dict[Any, TypeAdapter[Any]] = {}

    def _adapter(self, tp: Any) -> TypeAdapter[Any]:
        adapter = self._adapters.get(tp)
        if adapter is None:
            adapter = TypeAdapter(tp)
            self._adapters[tp] = adapter
        return adapter

    def dump(self, tp: Any, value: Any) -> Any:
        return self._adapter(tp).dump_python(value, mode='json')

    def load(self, tp: Any, payload: Any) -> Any:
        return self._adapter(tp).validate_python(payload)


IDENTITY_CODEC: DurabilityCodec = _IdentityCodec()
JSON_CODEC: DurabilityCodec = _JsonCodec()
