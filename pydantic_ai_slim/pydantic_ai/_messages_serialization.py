"""Message-history fields embedded in portable conversation and session state."""

from typing import Annotated, Any, cast

import pydantic

from . import messages as _messages


def _dump_messages_json(messages: list[_messages.ModelMessage], info: pydantic.SerializationInfo) -> Any:
    # The messages adapter's byte configuration must reach Any-typed tool returns too. A parent
    # dataclass's config does not apply to its nested message types. Forward field-local include,
    # exclude, and context so redaction works exactly as on ModelMessagesTypeAdapter itself.
    return _messages.ModelMessagesTypeAdapter.dump_python(
        messages,
        mode='json',
        include=cast(Any, info.include),
        exclude=cast(Any, info.exclude),
        by_alias=info.by_alias,
        exclude_unset=info.exclude_unset,
        exclude_defaults=info.exclude_defaults,
        exclude_none=info.exclude_none,
        exclude_computed_fields=info.exclude_computed_fields,
        round_trip=info.round_trip,
        serialize_as_any=info.serialize_as_any,
        # SerializationInfo has no polymorphic_serialization on the oldest supported Pydantic.
        context=info.context,
    )


MessageHistory = Annotated[
    list[_messages.ModelMessage], pydantic.PlainSerializer(_dump_messages_json, when_used='json')
]
