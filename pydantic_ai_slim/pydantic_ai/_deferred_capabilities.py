"""Deferred-capability typed message parts.

`load_capability` is a framework-emitted function tool used by deferred capabilities
to gate hidden tools behind an explicit load step. There is no native server-side
counterpart — capability loading is always client-executed.

This module follows the same leaf-module pattern as `_tool_search.py`: it is
late-imported by `pydantic_ai.messages` after the base `ToolCallPart` /
`ToolReturnPart` types are defined, registers its typed subclasses with the
shared narrower and discriminator-tag tables, and is then re-exported via
`messages.py` for public consumption.

User code can match the typed
[`LoadCapabilityCallPart`][pydantic_ai.messages.LoadCapabilityCallPart] /
[`LoadCapabilityReturnPart`][pydantic_ai.messages.LoadCapabilityReturnPart]
subclasses via `isinstance` (e.g. for UI rendering) or synthesize them directly
to inject loads mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Union, cast

import pydantic
from typing_extensions import NotRequired, TypedDict

from ._utils import copy_dataclass_fields

# Mirror `_tool_search.py`'s late-binding pattern: bind only the base parts and
# registration tables we need at class-definition time. The message-level types
# (`ModelMessage`, `ModelRequest`, etc.) are not referenced here — capability
# loading has no cross-provider history translation, so no synthesize/replay
# helpers live in this module.
from .messages import (
    _TOOL_CALL_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _TOOL_RETURN_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS_BY_TYPE,  # pyright: ignore[reportPrivateUsage]
    ToolCallPart,
    ToolReturnPart,
)


class LoadCapabilityArgs(TypedDict):
    """Typed arguments for a load-capability call.

    Carried on
    [`LoadCapabilityCallPart.args`][pydantic_ai.messages.LoadCapabilityCallPart.args]
    as the canonical shape of the framework-emitted `load_capability` function tool.
    """

    id: Annotated[
        str,
        pydantic.Field(
            description='The id of the capability to load, as shown in the available capabilities list.',
        ),
    ]
    """ID of the capability to load, as listed in the agent's capability registry."""


class LoadCapabilityReturn(TypedDict):
    """Typed return value of the framework-managed `load_capability` builtin.

    Carried on
    [`LoadCapabilityReturnPart.content`][pydantic_ai.messages.LoadCapabilityReturnPart.content].

    The loaded capability's id is not echoed here — it's already present on the paired
    [`LoadCapabilityCallPart.args['id']`][pydantic_ai.messages.LoadCapabilityCallPart.args].
    Match call and return by `tool_call_id`.
    """

    instructions: NotRequired[str]
    """Instructions for the model to follow when using the loaded capability. Omitted when the capability declared none."""


@dataclass(repr=False)
class LoadCapabilityCallPart(ToolCallPart):
    """Typed view of a [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] for the framework-emitted `load_capability` function call.

    Emitted when the model invokes `load_capability(id=...)` to load a deferred
    capability. There is no native server-side counterpart — capability loading is
    always client-executed.

    Shadows `args` with the canonical typed shape. The `str` variant covers the
    streaming / partial-args case before parsing completes; once parsed, `args` is
    a [`LoadCapabilityArgs`][pydantic_ai.messages.LoadCapabilityArgs] `TypedDict`.
    """

    tool_name: Literal['load_capability'] = 'load_capability'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Default tool name for the typed subclass. Discrimination drives off `tool_kind`."""

    args: str | LoadCapabilityArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    """Load-capability call payload.

    Narrows the parent's `str | dict[str, Any] | None` to a typed
    [`LoadCapabilityArgs`][pydantic_ai.messages.LoadCapabilityArgs] when parsed.
    Streaming / partial-args still arrive as `str` until they're complete.
    """

    tool_kind: Literal['capability-load'] = 'capability-load'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass (framework-emitted `load_capability` call)."""

    @property
    def typed_args(self) -> LoadCapabilityArgs | None:
        """Typed view of the validated load-capability arguments, or `None` if not yet parseable.

        In non-streaming code (a typed call part on a finalized
        [`ModelResponse`][pydantic_ai.messages.ModelResponse]), this is always
        populated — once a part is narrowed to this typed subclass, its `args`
        have been parsed and validated.

        Returns `None` only in streaming-partial state, where `args` is still an
        in-progress JSON string the model hasn't finished emitting.
        """
        if self.args is None:
            return None
        try:
            return cast('LoadCapabilityArgs', self.args_as_dict(raise_if_invalid=True))
        except (ValueError, AssertionError):
            return None

    @property
    def capability_id(self) -> str | None:
        """Subfield accessor for `typed_args['id']`.

        Returns `None` if args haven't been parsed yet (streaming-partial,
        i.e. `typed_args` is `None`).
        """
        typed = self.typed_args
        if typed is None:
            return None
        return typed.get('id')


@dataclass(repr=False)
class LoadCapabilityReturnPart(ToolReturnPart):
    """Typed view of a [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] for the framework-emitted `load_capability` function return.

    Carries any instructions the loaded capability declared. There is no native
    server-side counterpart — capability loading is always client-executed.

    Shadows `content` with a narrower
    [`LoadCapabilityReturn`][pydantic_ai.messages.LoadCapabilityReturn] `TypedDict`.
    """

    content: LoadCapabilityReturn = field(kw_only=True)
    """Load-capability return payload.

    Narrows the parent's `ToolReturnContent` to a typed
    [`LoadCapabilityReturn`][pydantic_ai.messages.LoadCapabilityReturn].
    """

    tool_name: Literal['load_capability'] = 'load_capability'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Default tool name for the typed subclass. Discrimination drives off `tool_kind`."""

    tool_kind: Literal['capability-load'] = 'capability-load'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass (framework-emitted `load_capability` return)."""

    @property
    def instructions(self) -> str | None:
        """Subfield accessor for `content['instructions']`.

        `None` when the capability declared no instructions (the key is absent from `content`).
        """
        return self.content.get('instructions')


_LOAD_CAPABILITY_CALL_ARGS_TA: pydantic.TypeAdapter[str | LoadCapabilityArgs | None] = pydantic.TypeAdapter(
    Union[str, LoadCapabilityArgs, None]  # noqa: UP007
)
_LOAD_CAPABILITY_RETURN_CONTENT_TA: pydantic.TypeAdapter[LoadCapabilityReturn] = pydantic.TypeAdapter(
    LoadCapabilityReturn
)


def _narrow_load_capability_call(part: ToolCallPart) -> LoadCapabilityCallPart:
    if isinstance(part, LoadCapabilityCallPart):
        return part
    validated_args = _LOAD_CAPABILITY_CALL_ARGS_TA.validate_python(part.args)
    return copy_dataclass_fields(part, LoadCapabilityCallPart, args=validated_args, tool_kind='capability-load')


def _narrow_load_capability_return(part: ToolReturnPart) -> LoadCapabilityReturnPart:
    if isinstance(part, LoadCapabilityReturnPart):
        return part
    validated_content = _LOAD_CAPABILITY_RETURN_CONTENT_TA.validate_python(part.content)
    return copy_dataclass_fields(part, LoadCapabilityReturnPart, content=validated_content, tool_kind='capability-load')


# `load_capability` has no native counterpart — capability loading is always client-executed,
# so only the local-shape narrowers are registered. Narrowers dispatch on `tool_kind` (set by
# the framework when it emits a typed call/return) so user tools that happen to share
# `tool_name` with the typed subclass are not accidentally promoted.
_TOOL_CALL_NARROWERS['capability-load'] = _narrow_load_capability_call
_TOOL_RETURN_NARROWERS['capability-load'] = _narrow_load_capability_return

# Register typed-part discriminator tags so `messages._model_request_part_discriminator` /
# `_model_response_part_discriminator` can route serialized dicts and Python instances to
# the right typed subclass without hard-coded if/elif chains.
_TYPED_PART_TAGS[('tool-call', 'capability-load')] = 'capability-load-call'
_TYPED_PART_TAGS[('tool-return', 'capability-load')] = 'capability-load-return'

_TYPED_PART_TAGS_BY_TYPE[LoadCapabilityCallPart] = 'capability-load-call'
_TYPED_PART_TAGS_BY_TYPE[LoadCapabilityReturnPart] = 'capability-load-return'
