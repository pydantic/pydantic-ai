from __future__ import annotations

import warnings
from dataclasses import replace
from functools import lru_cache
from typing import Any

import temporalio.api.common.v1
from pydantic import TypeAdapter
from pydantic_core import to_json
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import (
    CompositePayloadConverter,
    DataConverter,
    DefaultPayloadConverter,
    EncodingPayloadConverter,
    JSONPlainPayloadConverter,
)

from pydantic_ai.messages import FileUrl


@lru_cache(maxsize=128)
def _get_type_adapter(cls: type[Any]) -> TypeAdapter[Any]:
    return TypeAdapter(cls)


class PydanticAIJSONPayloadConverter(EncodingPayloadConverter):
    """JSON payload converter using TypeAdapter for FileUrl to preserve computed fields."""

    @property
    def encoding(self) -> str:
        return 'json/plain'

    def to_payload(self, value: Any) -> temporalio.api.common.v1.Payload | None:
        """FileUrl has computed fields (media_type, identifier) backed by excluded private fields (_media_type, _identifier).

        pydantic_core.to_json() doesn't preserve these computed fields correctly,
        so we need `TypeAdapter.dump_json()` for proper serialization.

        We can't use `TypeAdapter` for all BaseModel/dataclass types
        because it causes hangs in Temporal workflows.

        For example, `CallToolParams` has `tool_args: dict[str, Any]` but at runtime contains
        a Pydantic model. `TypeAdapter.dump_json()` tries to serialize through the dict[str, Any] schema,
        which hangs in Temporal's threading context (works fine in isolation, hangs in workflows).

        So we use `TypeAdapter` only for `FileUrl` (which needs it), and to_json() for everything else.
        """
        if isinstance(value, FileUrl):
            data = _get_type_adapter(type(value)).dump_json(value)
        else:
            data = to_json(value)
        return temporalio.api.common.v1.Payload(metadata={'encoding': self.encoding.encode()}, data=data)

    def from_payload(
        self,
        payload: temporalio.api.common.v1.Payload,
        type_hint: type[Any] | None = None,
    ) -> Any:
        return _get_type_adapter(type_hint if type_hint is not None else Any).validate_json(payload.data)


class PydanticAIPayloadConverter(CompositePayloadConverter):
    """Composite payload converter with Pydantic AI JSON serialization."""

    def __init__(self) -> None:
        json_payload_converter = PydanticAIJSONPayloadConverter()
        super().__init__(
            *(
                c if not isinstance(c, JSONPlainPayloadConverter) else json_payload_converter
                for c in DefaultPayloadConverter.default_encoding_payload_converters
            )
        )


pydantic_ai_data_converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)


def make_data_converter(converter: DataConverter | None) -> DataConverter:
    if converter is None:
        return pydantic_ai_data_converter

    # If the payload converter class is already a subclass of PydanticPayloadConverter,
    # the converter is already compatible with Pydantic AI - return it as-is.
    if issubclass(converter.payload_converter_class, PydanticAIPayloadConverter):
        return converter

    # Preserve custom PydanticPayloadConverter subclasses
    if (
        issubclass(converter.payload_converter_class, PydanticPayloadConverter)
        and converter.payload_converter_class is not PydanticPayloadConverter
    ):
        return converter

    # Upgrade to fix computed field serialization
    if converter.payload_converter_class is PydanticPayloadConverter:
        return replace(converter, payload_converter_class=PydanticAIPayloadConverter)

    # If using a non-Pydantic payload converter, warn and replace just the payload converter class,
    # preserving any custom payload_codec or failure_converter_class.
    if converter.payload_converter_class is not DefaultPayloadConverter:
        warnings.warn(
            'A non-Pydantic Temporal payload converter was used which has been replaced with PydanticAIPayloadConverter. '
            'To suppress this warning, ensure your payload_converter_class inherits from PydanticAIPayloadConverter.'
        )

    return replace(converter, payload_converter_class=PydanticAIPayloadConverter)
