"""Registry-backed serialization for extensible event families.

Event classes like [`CustomEvent`][pydantic_ai.messages.CustomEvent] are open families: applications
and third-party packages define typed subclasses that must round-trip through the closed
`AgentStreamEvent` discriminated union. Like the native tools union (see
`AbstractNativeTool.__get_pydantic_core_schema__`), the family's inner tagged union is rebuilt from a
registry at schema-generation time, so registered subclasses validate to their own class. Unlike
native tools, an unregistered tag doesn't fail validation: it degrades to an "unknown" envelope
carrying the raw payload in `data`, which re-flattens on serialization so a downstream consumer that
does have the defining module imported recovers the typed event.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import pydantic
import pydantic_core

from ._utils import is_str_dict
from .exceptions import UserError

EVENT_ENVELOPE_FIELDS = frozenset({'event_kind', 'name', 'data', 'tool_call_id', 'tool_name'})
"""Fields that identify and attribute an emitted event, as opposed to carrying its payload."""

_UNKNOWN_TAG = '__unknown__'
_BASE_TAG = '__base__'


def event_family_schema(
    handler: pydantic.GetCoreSchemaHandler,
    *,
    registry: Mapping[str, type[Any]],
    tag_field: str,
    unknown_type: type[Any],
    base_schema: pydantic_core.core_schema.CoreSchema | None = None,
) -> pydantic_core.core_schema.CoreSchema:
    """Build the tagged union over an event registry, degrading unregistered tags to `unknown_type`.

    When `base_schema` is given, a value with an unregistered tag and no non-envelope fields
    validates as the plain base class instead (silently): that's a directly-constructed base event,
    not a degraded typed one.
    """
    # Snapshot the registry: the union's choices are fixed once this schema is built, so a class
    # registered later must degrade to the unknown envelope rather than produce a dangling tag.
    known_tags = frozenset(registry)

    def discriminator(value: Any) -> str | None:
        if is_str_dict(value):
            tag = value.get(tag_field)
            if isinstance(tag, str) and tag in known_tags:
                return tag
            if base_schema is not None and not (value.keys() - EVENT_ENVELOPE_FIELDS):
                return _BASE_TAG
            return _UNKNOWN_TAG
        tag = getattr(value, tag_field, None)
        if isinstance(tag, str) and tag in known_tags:
            return tag
        if isinstance(value, unknown_type):
            return _UNKNOWN_TAG
        return _BASE_TAG if base_schema is not None else None

    unknown_schema = pydantic_core.core_schema.no_info_before_validator_function(
        _gather_unknown_payload(tag_field, unknown_type),
        handler.generate_schema(unknown_type),
        serialization=pydantic_core.core_schema.wrap_serializer_function_ser_schema(_flatten_unknown),
    )
    choices: dict[str, pydantic_core.core_schema.CoreSchema] = {}
    for tag, event_cls in registry.items():
        if not dataclasses.is_dataclass(event_cls):
            raise UserError(  # pragma: no cover
                f'Event class {event_cls.__qualname__} (registered as {tag!r}) must be a dataclass.'
            )
        choices[tag] = handler.generate_schema(event_cls)
    choices[_UNKNOWN_TAG] = unknown_schema
    if base_schema is not None:
        choices[_BASE_TAG] = base_schema
    return pydantic_core.core_schema.tagged_union_schema(choices, discriminator)


def _gather_unknown_payload(tag_field: str, unknown_type: type[Any]) -> Callable[[Any], Any]:
    """Before-validator for the unknown-event envelope: move unrecognized payload fields into `data`."""

    def gather(value: Any) -> Any:
        if is_str_dict(value):
            envelope = {k: v for k, v in value.items() if k in EVENT_ENVELOPE_FIELDS}
            payload = {k: v for k, v in value.items() if k not in EVENT_ENVELOPE_FIELDS}
            if payload:
                if (data := envelope.get('data')) is not None:
                    payload['data'] = data
                envelope['data'] = payload
            warnings.warn(
                f'Unknown event {tag_field} {value.get(tag_field)!r}; validating as {unknown_type.__name__}. '
                f'Is the module that defines this event imported?',
                UserWarning,
                stacklevel=2,
            )
            return envelope
        return value

    return gather


def _flatten_unknown(value: Any, serializer: pydantic_core.core_schema.SerializerFunctionWrapHandler) -> Any:
    """Serializer for the unknown-event envelope: re-flatten `data` so the typed event can be recovered."""
    dumped: Any = serializer(value)
    if is_str_dict(dumped):
        if is_str_dict(data := dumped.pop('data', None)):
            return {**data, **dumped}
        dumped['data'] = data
        return dumped
    return dumped
