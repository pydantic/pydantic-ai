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
import functools
import inspect
import sys
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import pydantic
import pydantic_core

from ._utils import is_str_dict
from .exceptions import UserError

_UNKNOWN_TAG = '__unknown__'

RESERVED_EVENT_TAGS = frozenset({_UNKNOWN_TAG})
"""Tags the family schema uses for its synthetic choices; no event class may register under them."""


def is_redefinition(existing: type, cls: type) -> bool:
    """Whether `cls` is the same class as `existing` being defined again.

    A re-run notebook cell, `importlib.reload`, a re-executed docs example, or the class recreation
    `@dataclass(slots=True)` performs replaces its registration; only genuinely distinct classes
    conflict.
    """
    return existing.__module__ == cls.__module__ and existing.__qualname__ == cls.__qualname__


def guard_post_init(cls: type, base_post_init: Callable[[Any], None]) -> None:
    """Keep the base event guards running when a subclass defines its own `__post_init__`.

    A dataclass-generated `__init__` calls only the most-derived `__post_init__`, so a subclass
    that defines one without calling `super().__post_init__()` would silently skip the family's
    construction guards (base instantiation, per-instance tag overrides). Wrapping at class
    definition makes the guards unbypassable; a cooperative `super()` call just re-runs them,
    which is harmless.
    """
    user_post_init = cls.__dict__.get('__post_init__')
    if user_post_init is None or getattr(user_post_init, '_event_guarded', False):
        return

    @functools.wraps(user_post_init)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> None:
        base_post_init(self)
        user_post_init(self, *args, **kwargs)
        # Re-run the guards afterwards too: the user's `__post_init__` could itself corrupt a
        # protocol field (e.g. reassign `name`), and validation is idempotent.
        base_post_init(self)

    guarded._event_guarded = True  # pyright: ignore[reportAttributeAccessIssue]
    cls.__post_init__ = guarded


def own_annotation_names(cls: type) -> set[str]:
    """The names annotated on the class itself, without forcing lazy (PEP 649) annotations.

    On Python 3.14+ annotations are evaluated lazily, so a payload field referencing a class defined
    later in the module must not be evaluated during `__init_subclass__` — plain dataclasses defer it,
    and event registration has to as well. `Format.FORWARDREF` never raises `NameError`.
    """
    if sys.version_info >= (3, 14):
        import annotationlib

        return set(annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF))
    return set(inspect.get_annotations(cls))


def shadowed_envelope_fields(cls: type, reserved: frozenset[str]) -> str | None:
    """The class's own field names that shadow the family's envelope fields, or `None`."""
    shadowed = own_annotation_names(cls) & reserved
    return ', '.join(sorted(shadowed)) if shadowed else None


def inject_tag_field(cls: type, tag_field: str, tag_value: str) -> None:
    """Redeclare `tag_field` on the subclass so it defaults to (and serializes as) the registered tag.

    The annotation is redeclared on the subclass so `@dataclass` (which runs after
    `__init_subclass__`) picks up the new default. On Python 3.14+ the merge wraps the class's lazy
    `__annotate__` function instead of materializing `__annotations__`, preserving PEP 649 deferred
    evaluation for payload fields that reference names defined later in the module.
    """
    if sys.version_info >= (3, 14):
        import annotationlib

        original_annotate = annotationlib.get_annotate_from_class_namespace(cls.__dict__)
        if original_annotate is not None:

            def annotate(format: int, /) -> dict[str, Any]:
                return {
                    tag_field: str,
                    **annotationlib.call_annotate_function(original_annotate, annotationlib.Format(format), owner=cls),
                }

            cls.__annotate__ = annotate  # pyright: ignore[reportAttributeAccessIssue]
        else:
            # No lazy annotate function: the class body had no annotations, or stored them eagerly
            # (e.g. under `from __future__ import annotations`); merge with whatever it has.
            cls.__annotations__ = {tag_field: str, **cls.__dict__.get('__annotations__', {})}
    else:
        cls.__annotations__ = {tag_field: 'str', **cls.__annotations__}
    setattr(cls, tag_field, dataclasses.field(default=tag_value, kw_only=True))


def event_family_schema(
    handler: pydantic.GetCoreSchemaHandler,
    *,
    registry: Mapping[str, type[Any]],
    tag_field: str,
    unknown_type: type[Any],
    envelope_fields: frozenset[str],
) -> pydantic_core.core_schema.CoreSchema:
    """Build the tagged union over an event registry, degrading unregistered tags to `unknown_type`."""
    # Snapshot the registry: the union's choices are fixed once this schema is built, so a class
    # registered later must degrade to the unknown envelope rather than produce a dangling tag.
    known_tags = frozenset(registry)

    def discriminator(value: Any) -> str | None:
        if is_str_dict(value):
            tag = value.get(tag_field)
            if isinstance(tag, str) and tag in known_tags:
                return tag
            return _UNKNOWN_TAG
        tag = getattr(value, tag_field, None)
        if isinstance(tag, str) and tag in known_tags:
            return tag
        return _UNKNOWN_TAG if isinstance(value, unknown_type) else None

    unknown_schema = pydantic_core.core_schema.no_info_before_validator_function(
        _gather_unknown_payload(tag_field, unknown_type, envelope_fields),
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
    return pydantic_core.core_schema.tagged_union_schema(choices, discriminator)


def _gather_unknown_payload(
    tag_field: str, unknown_type: type[Any], envelope_fields: frozenset[str]
) -> Callable[[Any], Any]:
    """Before-validator for the unknown-event envelope: move unrecognized payload fields into `data`."""

    def gather(value: Any) -> Any:
        if is_str_dict(value):
            envelope = {k: v for k, v in value.items() if k in envelope_fields}
            payload = {k: v for k, v in value.items() if k not in envelope_fields}
            if payload:
                if (data := envelope.get('data')) is not None:
                    payload['data'] = data
                envelope['data'] = payload
            warnings.warn(
                f'Unknown event {tag_field} {value.get(tag_field)!r}; validating as {unknown_type.__name__}. '
                f'Is the module that defines this event imported? (A serializer built before the event '
                f'class was defined also treats it as unknown.)',
                UserWarning,
                stacklevel=2,
            )
            return envelope
        return value

    return gather


def _flatten_unknown(value: Any, serializer: pydantic_core.core_schema.SerializerFunctionWrapHandler) -> Any:
    """Serializer for the unknown-event envelope: re-flatten `data` so the typed event can be recovered."""
    dumped: Any = serializer(value)
    if not is_str_dict(dumped):  # pragma: no cover - the family serializer always produces a dict
        return dumped
    if is_str_dict(data := dumped.pop('data', None)):
        return {**data, **dumped}
    dumped['data'] = data
    return dumped
