from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter
from temporalio.api.common.v1 import Payload
from temporalio.contrib.pydantic import (
    PydanticJSONPlainPayloadConverter,
    PydanticPayloadConverter,
    ToJsonOptions,
)
from temporalio.converter import CompositePayloadConverter, DefaultPayloadConverter, JSONPlainPayloadConverter

from pydantic_ai.messages import _JOURNAL_CONTEXT  # pyright: ignore[reportPrivateUsage]


class PydanticAIJSONPlainPayloadConverter(PydanticJSONPlainPayloadConverter):
    """Pydantic JSON converter that decodes payloads as Pydantic AI's own journal."""

    def from_payload(self, payload: Payload, type_hint: type | None = None) -> Any:
        hint = type_hint if type_hint is not None else Any
        adapter: TypeAdapter[Any] = TypeAdapter(hint)
        return adapter.validate_json(payload.data, context=_JOURNAL_CONTEXT)


class PydanticAIPayloadConverter(PydanticPayloadConverter):
    """Temporal Pydantic payload converter that decodes payloads as Pydantic AI's own journal."""

    def __init__(self, to_json_options: ToJsonOptions | None = None) -> None:
        json_payload_converter = PydanticAIJSONPlainPayloadConverter(to_json_options)
        CompositePayloadConverter.__init__(
            self,
            *(
                converter if not isinstance(converter, JSONPlainPayloadConverter) else json_payload_converter
                for converter in DefaultPayloadConverter.default_encoding_payload_converters
            ),
        )
