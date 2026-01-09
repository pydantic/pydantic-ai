from __future__ import annotations

import warnings
from dataclasses import is_dataclass, replace
from typing import Any

import temporalio.api.common.v1
from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_json
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import (
    CompositePayloadConverter,
    DataConverter,
    DefaultPayloadConverter,
    EncodingPayloadConverter,
    JSONPlainPayloadConverter,
)


class PydanticAIJSONPayloadConverter(EncodingPayloadConverter):
    """JSON payload converter using TypeAdapter for Pydantic models."""

    @property
    def encoding(self) -> str:
        return 'json/plain'

    def to_payload(self, value: Any) -> temporalio.api.common.v1.Payload | None:
        if isinstance(value, BaseModel) or is_dataclass(value):
            data = TypeAdapter(type(value)).dump_json(value)
        else:
            data = to_json(value)
        return temporalio.api.common.v1.Payload(metadata={'encoding': self.encoding.encode()}, data=data)

    def from_payload(
        self,
        payload: temporalio.api.common.v1.Payload,
        type_hint: type[Any] | None = None,
    ) -> Any:
        return TypeAdapter(type_hint if type_hint is not None else Any).validate_json(payload.data)


class PydanticAIPayloadConverter(CompositePayloadConverter):
    """Composite payload converter with PydanticAI JSON serialization."""

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

    # Preserve codec and failure handler configs
    if converter.payload_converter_class is not DefaultPayloadConverter:
        warnings.warn(
            'A non-Pydantic Temporal payload converter was used which has been replaced with PydanticAIPayloadConverter. '
            'To suppress this warning, ensure your payload_converter_class inherits from PydanticAIPayloadConverter.'
        )

    return replace(converter, payload_converter_class=PydanticAIPayloadConverter)
