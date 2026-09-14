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
import sys
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar, overload

import pydantic
import pydantic_core

from . import _utils
from ._utils import is_str_dict
from .exceptions import UserError

_UNKNOWN_TAG = '__unknown__'

RESERVED_EVENT_TAGS = frozenset({_UNKNOWN_TAG})
"""Tags the family schema uses for its synthetic choices; no event class may register under them."""

_registry_version = 0

_EventClassT = TypeVar('_EventClassT')
_DefaultT = TypeVar('_DefaultT')


class EventRegistry(dict[str, _EventClassT], Generic[_EventClassT]):
    """One event family's tag-to-class registry, versioned so schema caches can detect staleness.

    `event_family_schema` snapshots the registry it builds from, so a schema built before a class
    registered keeps degrading that class's events to the unknown envelope. That's correct for a
    schema built once and thrown away, but a consumer that memoizes adapters for the life of a
    process (e.g. Temporal's payload converter) would keep serving the stale one, making decoding
    depend on import order. Such consumers key their memo on `event_registry_version()`, which the
    mutations below bump, so a later registration rebuilds the adapter instead.

    A registry is only ever mutated by item assignment (registration, from `__init_subclass__`),
    `del`, and `pop` (test cleanup), which is why those are the ones overridden. `dict`'s bulk
    mutators would change the registry without bumping the version, leaving a memoizing consumer on
    a stale adapter; register through class definition rather than reaching for them.
    """

    def __setitem__(self, tag: str, event_cls: _EventClassT) -> None:
        super().__setitem__(tag, event_cls)
        _bump_registry_version()

    def __delitem__(self, tag: str) -> None:
        super().__delitem__(tag)
        _bump_registry_version()

    @overload
    def pop(self, tag: str, /) -> _EventClassT: ...

    @overload
    def pop(self, tag: str, default: _EventClassT | _DefaultT, /) -> _EventClassT | _DefaultT: ...

    def pop(self, tag: str, /, *default: Any) -> Any:
        popped = super().pop(tag, *default)
        _bump_registry_version()
        return popped


def _bump_registry_version() -> None:
    global _registry_version
    _registry_version += 1


def event_registry_version() -> int:
    """A counter bumped whenever any [`EventRegistry`][] changes.

    Cache keys that include this are invalidated by a registration, so an adapter built before an
    event class was imported isn't reused after it is.
    """
    return _registry_version


def is_redefinition(existing: type, cls: type) -> bool:
    """Whether `cls` is the same class as `existing` being defined again.

    A re-run notebook cell, `importlib.reload`, a re-executed docs example, or the class recreation
    `@dataclass(slots=True)` performs replaces its registration; only genuinely distinct classes
    conflict.
    """
    return existing.__module__ == cls.__module__ and existing.__qualname__ == cls.__qualname__


_replay_isolation_guard: Callable[[], bool] | None = None


def set_replay_isolation_guard(guard: Callable[[], bool]) -> None:
    """Declare when class definitions are happening in an isolated re-execution of the app's modules.

    A durable execution runtime may re-execute application modules in an isolated interpreter view
    while sharing `pydantic_ai` itself with the host process — Temporal's workflow sandbox does
    exactly this. The re-executed copy of an event class is a redefinition of the host's, so left
    alone it would take over the registry, and the host would then decode payloads into a class its
    own `isinstance` checks (including `@on_event(MyEvent)` filtering) don't recognize.

    While `guard()` is true, redefinitions keep the class already registered as the family's
    canonical one. Instances of the re-executed copy still serialize and validate normally: the
    family schema canonicalizes them (see `event_family_schema`), so application code holds one
    event class regardless of which side of the boundary it runs on.
    """
    global _replay_isolation_guard
    _replay_isolation_guard = guard


def keeps_canonical_registration() -> bool:
    """Whether a redefinition right now should leave the existing registration in place."""
    return _replay_isolation_guard is not None and _replay_isolation_guard()


def _canonicalize(value: object, event_cls: type) -> Any:
    """Rebuild a re-executed copy of `event_cls` as `event_cls` itself, so serialization stays exact.

    Only a class the registry considers the same one (see `is_redefinition`) is converted. Such a
    copy comes from re-executing the same module source, so it carries the same fields by
    construction, and reading them off it by name is exact.

    This is the exception to the "no `fields()` + `getattr` copying" rule in `agent_docs/index.md`:
    that rule protects *our own* statically-known types, where Pyright can check field existence and
    a rename would silently break the copy. `event_cls` is an application-defined subclass resolved
    at runtime, so there is no static field set to check against, and a rename lands on both copies
    at once — they are the same source, executed twice.
    """
    value_type = type(value)
    if value_type is event_cls or not is_redefinition(event_cls, value_type):
        return value
    fields = dataclasses.fields(event_cls)
    # A field declared `init=False` isn't accepted by the constructor, so it's assigned afterwards
    # rather than dropped: `__post_init__` can only recompute one whose value follows from the init
    # fields, and anything else would silently come back as its default.
    canonical = event_cls(**{f.name: getattr(value, f.name) for f in fields if f.init})
    for f in fields:
        if not f.init:
            setattr(canonical, f.name, getattr(value, f.name))
    return canonical


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


def shadowed_envelope_fields(cls: type, reserved: frozenset[str]) -> str | None:
    """The class's own field names that shadow the family's envelope fields, or `None`."""
    shadowed = set(_utils.own_annotations(cls)) & reserved
    return ', '.join(sorted(shadowed)) if shadowed else None


def inherited_namespace(cls: type, family: type) -> str | None:
    """The namespace `cls` inherits from its nearest base that carries one, or `None`.

    A registered base holds its namespace inside `_registered_kind`; an `abstract=True` base never
    registered, so it keeps the namespace on its own attribute instead. Either way the search stops at
    the first base that has one, so a subclass takes the namespace of the family it was defined in.
    """
    for base in cls.__mro__[1:]:
        if not issubclass(base, family):
            continue
        if base_kind := base.__dict__.get('_registered_kind'):
            return base_kind.rpartition('.')[0]
        if base_namespace := base.__dict__.get('_abstract_namespace'):
            return base_namespace
    return None


def undecorated_field_base(cls: type, family: type) -> type | None:
    """A class between `cls` and `family` that declares payload fields but isn't a dataclass.

    `@dataclass` only collects fields from bases that are dataclasses themselves, so an intermediate
    base holding shared fields silently contributes nothing unless it is decorated too: the fields
    vanish from the payload, the wire, and every consumer, with no error anywhere. Each base is fully
    built by the time a subclass is being registered, so its decoration can be checked reliably here.
    """
    for base in cls.__mro__[1:]:  # pragma: no branch  # `family` is always in the MRO, so this always breaks
        if base is family or not issubclass(base, family):
            break
        if '__dataclass_fields__' in base.__dict__:
            continue
        # A `ClassVar`-only base loses nothing by not being a dataclass: it carries settings, not payload.
        if _utils.declares_dataclass_fields(base):
            return base
    return None


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
        choices[tag] = _canonicalizing_schema(handler, event_cls)
    choices[_UNKNOWN_TAG] = unknown_schema
    return pydantic_core.core_schema.tagged_union_schema(choices, discriminator)


def _canonicalizing_schema(handler: pydantic.GetCoreSchemaHandler, event_cls: type[Any]) -> Any:
    """The event class's own schema, serializing a re-executed copy of it as the registered class.

    Under a `set_replay_isolation_guard`, code on the isolated side holds its own copy of the class
    while the registry keeps the host's. The copy is the same class by every meaning that matters
    here — same module, same qualname, same fields — so serializing it as the registered one is
    exact, and it's what lets both sides use the class they imported.
    """

    def serialize(value: Any, serializer: pydantic_core.core_schema.SerializerFunctionWrapHandler) -> Any:
        return serializer(_canonicalize(value, event_cls))

    return pydantic_core.core_schema.no_info_before_validator_function(
        lambda value: value,
        handler.generate_schema(event_cls),
        serialization=pydantic_core.core_schema.wrap_serializer_function_ser_schema(serialize),
    )


def _gather_unknown_payload(
    tag_field: str, unknown_type: type[Any], envelope_fields: frozenset[str]
) -> Callable[[Any], Any]:
    """Before-validator for the unknown-event envelope: move unrecognized payload fields into `data`.

    The envelope's own `data` slot is synthetic — only this validator ever fills it — so a `data` key
    on the wire is one of the unknown event's payload fields, not the envelope's, and is gathered
    like any other. Treating it as the envelope's would let `_flatten_unknown` promote its entries to
    top-level fields, changing the wire shape of an event whose only payload field is named `data`.
    """
    carried_fields = envelope_fields - {'data'}

    def gather(value: Any) -> Any:
        if is_str_dict(value):
            envelope = {k: v for k, v in value.items() if k in carried_fields}
            payload = {k: v for k, v in value.items() if k not in carried_fields}
            if payload:
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
    # The family serializer always produces a dict.
    if not is_str_dict(dumped):  # pragma: no cover
        return dumped
    if is_str_dict(data := dumped.pop('data', None)):
        return {**data, **dumped}
    dumped['data'] = data
    return dumped
