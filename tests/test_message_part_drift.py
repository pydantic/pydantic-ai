"""Introspection guards against discriminated-union / discriminator-tag drift in `messages.py`.

`messages.py` serializes message history through several hand-maintained discriminated unions and
callable `pydantic.Discriminator`s. Their member sets carry invariants that nothing enforces and that
line coverage can't see — a union member is a type-declaration line, so dropping one neither lowers
coverage nor fails a test until that exact shape happens to be deserialized. These guards reconcile,
at the union level, the several places a part's tag is declared (the `pydantic.Tag` on a union member,
the `_TYPED_PART_TAGS` registries, each class's `part_kind`/`tool_kind` `Literal`, and the
`previous_part_kind`/`next_part_kind` `Literal`s) so drift fails CI loudly.

See issue #5802. The motivating instance is #5721/#5723 (base `ToolReturnPart` missing from
`ModelResponsePart`).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import pydantic
import pytest

from pydantic_ai import _deferred_capabilities, _tool_search, messages
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart, LoadCapabilityReturnPart
from pydantic_ai._tool_search import (
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
)
from pydantic_ai.messages import (
    _TYPED_PART_TAGS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS_BY_TYPE,  # pyright: ignore[reportPrivateUsage]
    AgentStreamEvent,
    BinaryContent,
    CompactionPart,
    FilePart,
    HandleResponseEvent,
    InstructionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponsePartDelta,
    ModelResponseStreamEvent,
    MultiModalContent,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# --- introspection helpers -------------------------------------------------------------------


def _leaf_classes(annotation: Any) -> list[type]:
    """Flatten a (possibly `Annotated`, possibly nested) union into its leaf member classes."""
    if get_origin(annotation) is Annotated:
        return _leaf_classes(get_args(annotation)[0])
    if get_origin(annotation) in (Union, types.UnionType):
        return [cls for arg in get_args(annotation) for cls in _leaf_classes(arg)]
    return [annotation]


def _tagged_union_members(union_type: Any) -> dict[str, type]:
    """For a `Tag`-annotated union (callable discriminator) return `{tag: cls}`."""
    out: dict[str, type] = {}
    for member in get_args(get_args(union_type)[0]):
        cls = get_args(member)[0]
        tag = next(meta.tag for meta in member.__metadata__ if isinstance(meta, pydantic.Tag))
        out[tag] = cls
    return out


def _discriminator_field(union_type: Any) -> str | None:
    """Return the discriminator field name for a string-keyed union, or `None` for callable ones."""
    discriminator = get_args(union_type)[1].discriminator
    return discriminator if isinstance(discriminator, str) else None


def _literal_values(annotation: Any) -> set[str]:
    """Extract the `Literal[...]` member set from a `Literal[...] | None` annotation."""
    candidates = (annotation, *get_args(annotation))
    for candidate in candidates:
        if get_origin(candidate) is Literal:
            return set(get_args(candidate))
    raise AssertionError(f'no Literal found in {annotation!r}')  # pragma: no cover


def _str_attr(cls: type, name: str) -> str:
    """Read a string class attribute (a dataclass field default like `part_kind`) off a bare `type`."""
    value = getattr(cls, name)
    assert isinstance(value, str)
    return value


REQUEST_PART_MEMBERS = _tagged_union_members(ModelRequestPart)
RESPONSE_PART_MEMBERS = _tagged_union_members(ModelResponsePart)


# --- source of truth: which message role(s) each concrete part serializes in -----------------

# Hand-maintained on purpose: the "missing member" failure mode (#5721) can't be derived from the
# union itself without going circular. This map is what catches a part dropped from (or never wired
# into) a union — `_model_response_part_discriminator` would route it nowhere.
PART_EXPECTED_UNIONS: dict[type, set[str]] = {
    SystemPromptPart: {'request'},
    UserPromptPart: {'request'},
    ToolReturnPart: {'request', 'response'},
    ToolSearchReturnPart: {'request'},
    LoadCapabilityReturnPart: {'request'},
    RetryPromptPart: {'request'},
    TextPart: {'response'},
    ThinkingPart: {'response'},
    CompactionPart: {'response'},
    FilePart: {'response'},
    ToolCallPart: {'response'},
    ToolSearchCallPart: {'response'},
    LoadCapabilityCallPart: {'response'},
    NativeToolCallPart: {'response'},
    NativeToolSearchCallPart: {'response'},
    NativeToolReturnPart: {'response'},
    NativeToolSearchReturnPart: {'response'},
}

# Concrete parts that have a `part_kind` but are never serialized as a member of a part union.
NON_SERIALIZED_PARTS: dict[type, str] = {
    InstructionPart: 'rendered into ModelRequest.instructions (a str), never a serialized part',
}


_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)
_TCID = 'test-call-id'
# Non-image so FilePart's `BinaryImage.narrow_type` AfterValidator doesn't re-type content on round-trip.
_PDF = BinaryContent(data=b'%PDF-1.4', media_type='application/pdf')

PART_FACTORIES: dict[type, Any] = {
    SystemPromptPart: lambda: SystemPromptPart(content='sys', timestamp=_TS),
    UserPromptPart: lambda: UserPromptPart(content='hi', timestamp=_TS),
    ToolReturnPart: lambda: ToolReturnPart(tool_name='t', content='ok', tool_call_id=_TCID, timestamp=_TS),
    ToolSearchReturnPart: lambda: ToolSearchReturnPart(
        content={'discovered_tools': []}, tool_call_id=_TCID, timestamp=_TS
    ),
    LoadCapabilityReturnPart: lambda: LoadCapabilityReturnPart(
        content={'instructions': 'x'}, tool_call_id=_TCID, timestamp=_TS
    ),
    RetryPromptPart: lambda: RetryPromptPart(content='retry', tool_call_id=_TCID, timestamp=_TS),
    TextPart: lambda: TextPart(content='hello'),
    ThinkingPart: lambda: ThinkingPart(content='think'),
    CompactionPart: lambda: CompactionPart(),
    FilePart: lambda: FilePart(content=_PDF),
    ToolCallPart: lambda: ToolCallPart(tool_name='t', args={'a': 1}, tool_call_id=_TCID),
    ToolSearchCallPart: lambda: ToolSearchCallPart(args={'queries': ['x']}, tool_call_id=_TCID),
    LoadCapabilityCallPart: lambda: LoadCapabilityCallPart(args={'id': 'cap1'}, tool_call_id=_TCID),
    NativeToolCallPart: lambda: NativeToolCallPart(tool_name='t', args={'a': 1}, tool_call_id=_TCID),
    NativeToolSearchCallPart: lambda: NativeToolSearchCallPart(args={'queries': ['x']}, tool_call_id=_TCID),
    NativeToolReturnPart: lambda: NativeToolReturnPart(tool_name='t', content='ok', tool_call_id=_TCID, timestamp=_TS),
    NativeToolSearchReturnPart: lambda: NativeToolSearchReturnPart(
        content={'discovered_tools': []}, tool_call_id=_TCID, timestamp=_TS
    ),
}


def _discover_concrete_part_classes() -> set[type]:
    """Every dataclass with a `part_kind` field, across the modules that define parts."""
    found: set[type] = set()
    for module in (messages, _tool_search, _deferred_capabilities):
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and dataclasses.is_dataclass(obj)
                and 'part_kind' in {f.name for f in dataclasses.fields(obj)}
            ):
                found.add(obj)
    return found


# --- Guard 2a: every concrete part is wired into the source-of-truth maps --------------------


def test_every_concrete_part_is_declared():
    """A new `*Part` must be added to `PART_EXPECTED_UNIONS` (or allow-listed), forcing the author to
    state which union(s) it belongs to — caught at class-def altitude, before any (de)serialization."""
    discovered = _discover_concrete_part_classes()
    declared = set(PART_EXPECTED_UNIONS) | set(NON_SERIALIZED_PARTS)
    assert discovered == declared


def test_every_serialized_part_has_a_factory():
    assert set(PART_FACTORIES) == set(PART_EXPECTED_UNIONS)


# --- Guard 2b: declared membership matches actual union membership ----------------------------


def test_declared_membership_matches_unions():
    """Each part declared for a role is actually a member of that role's union. This is the precise
    one-line localizer for #5721 (base `ToolReturnPart` missing from `ModelResponsePart`)."""
    for cls, roles in PART_EXPECTED_UNIONS.items():
        if 'request' in roles:
            assert cls in REQUEST_PART_MEMBERS.values(), f'{cls.__name__} missing from ModelRequestPart'
        if 'response' in roles:
            assert cls in RESPONSE_PART_MEMBERS.values(), f'{cls.__name__} missing from ModelResponsePart'


# --- Guard 1 (umbrella): round-trip every concrete part ----------------------------------------


@pytest.mark.parametrize('part_cls', list(PART_FACTORIES))
def test_round_trip_every_part(part_cls: type):
    """Every concrete part survives `ModelMessagesTypeAdapter` dump→validate (python + json) inside
    each message role it's declared for, preserving both `type(...)` and field values."""
    roles = PART_EXPECTED_UNIONS[part_cls]
    for role in roles:
        part = PART_FACTORIES[part_cls]()
        message: ModelMessage = ModelRequest(parts=[part]) if role == 'request' else ModelResponse(parts=[part])

        for dump, validate in (
            (ModelMessagesTypeAdapter.dump_python, ModelMessagesTypeAdapter.validate_python),
            (ModelMessagesTypeAdapter.dump_json, ModelMessagesTypeAdapter.validate_json),
        ):
            round_tripped = validate(dump([message]))[0].parts[0]
            assert type(round_tripped) is part_cls, f'{part_cls.__name__} mis-routed in {role}'
            assert round_tripped == part


# --- Guard 3: discriminator-tag <-> member bijection ------------------------------------------


def test_typed_part_registries_reconcile_with_unions():
    """The three places a part's tag lives — the `Tag` on a union member, the `_TYPED_PART_TAGS`
    registries, and each class's `part_kind`/`tool_kind` — must agree, with no orphan tag and no
    unreachable member."""
    all_members = {**REQUEST_PART_MEMBERS, **RESPONSE_PART_MEMBERS}
    member_classes = set(all_members.values())

    for tag, cls in all_members.items():
        if cls in _TYPED_PART_TAGS_BY_TYPE:
            assert _TYPED_PART_TAGS_BY_TYPE[cls] == tag
            assert _TYPED_PART_TAGS[(_str_attr(cls, 'part_kind'), _str_attr(cls, 'tool_kind'))] == tag
        else:
            assert _str_attr(cls, 'part_kind') == tag

    # No registry entry points outside the unions.
    assert set(_TYPED_PART_TAGS_BY_TYPE) <= member_classes
    assert set(_TYPED_PART_TAGS.values()) <= set(all_members)
    # The dict-deserialization path and the instance path register the same tags.
    assert set(_TYPED_PART_TAGS.values()) == set(_TYPED_PART_TAGS_BY_TYPE.values())


# --- Guard 4: previous/next part-kind Literals == ModelResponsePart tags ----------------------


def test_streaming_part_kind_literals_match_response_union():
    """`PartStartEvent.previous_part_kind` / `PartEndEvent.next_part_kind` are populated from any
    streamed `ModelResponsePart.part_kind` with no filtering, so they must enumerate the full union."""
    response_tags = set(RESPONSE_PART_MEMBERS)
    assert _literal_values(typing.get_type_hints(PartStartEvent)['previous_part_kind']) == response_tags
    assert _literal_values(typing.get_type_hints(PartEndEvent)['next_part_kind']) == response_tags


# --- Guard 5: string-discriminated unions have a present, unique tag per member ---------------


@pytest.mark.parametrize(
    'union_type',
    [
        MultiModalContent,
        ModelMessage,
        ModelResponsePartDelta,
        ModelResponseStreamEvent,
        HandleResponseEvent,
        AgentStreamEvent,
    ],
)
def test_string_discriminated_union_tags_unique(union_type: Any):
    """Every member declares the union's discriminator field, and no two members share a value — a
    collision would silently mis-route on deserialization (failure mode 2 in #5802)."""
    field = _discriminator_field(union_type)
    assert field is not None
    members = _leaf_classes(union_type)
    tags = [getattr(member, field) for member in members]
    assert all(isinstance(tag, str) for tag in tags)
    assert len(tags) == len(set(tags)), f'duplicate {field} across {union_type}'
