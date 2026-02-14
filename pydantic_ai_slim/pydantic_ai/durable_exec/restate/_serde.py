from pydantic import TypeAdapter

from ._restate_types import Serde, T


class PydanticTypeAdapter(Serde[T]):
    """A serializer/deserializer using Pydantic's `TypeAdapter`."""

    def __init__(self, model_type: type[T]):
        self._type_adapter = TypeAdapter(model_type)

    def deserialize(self, buf: bytes) -> T | None:
        if not buf:
            return None
        return self._type_adapter.validate_json(buf)

    def serialize(self, obj: T | None) -> bytes:
        if obj is None:
            return b''
        return self._type_adapter.dump_json(obj)
