from pydantic import TypeAdapter

from ._restate_types import Serde, T


class PydanticTypeAdapter(Serde[T]):
    """A serializer/deserializer for Pydantic models."""

    def __init__(self, model_type: type[T]):
        """Initializes a new instance of the `PydanticTypeAdapter` class.

        Args:
            model_type (typing.Type[T]): The Pydantic model type to serialize/deserialize.
        """
        self._model_type = TypeAdapter(model_type)

    def deserialize(self, buf: bytes) -> T | None:
        """Deserializes bytes to a Pydantic model.

        Args:
            buf (bytes): The bytes to deserialize.

        Returns:
            typing.Optional[T]: The deserialized Pydantic model.
        """
        if not buf:
            return None
        return self._model_type.validate_json(buf.decode('utf-8'))  # raises if invalid

    def serialize(self, obj: T | None) -> bytes:
        """Serializes a Pydantic model to bytes.

        Args:
            obj (typing.Optional[T]): The Pydantic model to serialize.

        Returns:
            bytes: The serialized bytes.
        """
        if obj is None:
            return b''
        return self._model_type.dump_json(obj)
