"""Typed message parts for deferred capability loading."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Union, cast

import pydantic
from typing_extensions import NotRequired, TypedDict

from ._utils import copy_dataclass_fields

# Imported late by `messages.py`; avoid imports that would re-enter it.
from .messages import (
    _TOOL_CALL_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _TOOL_RETURN_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS_BY_TYPE,  # pyright: ignore[reportPrivateUsage]
    ToolCallPart,
    ToolReturnPart,
)

if TYPE_CHECKING:
    from .messages import ModelMessage


class LoadCapabilityArgs(TypedDict):
    """Typed arguments for a `load_capability` tool call."""

    id: Annotated[
        str,
        pydantic.Field(
            description='The id of the capability to load.',
        ),
    ]
    """ID of the capability to load."""


class LoadCapabilityReturn(TypedDict):
    """Typed return value for the `load_capability` tool."""

    instructions: NotRequired[str]
    """Instructions for the loaded capability."""


@dataclass(repr=False)
class LoadCapabilityCallPart(ToolCallPart):
    """Typed `ToolCallPart` for the `load_capability` tool."""

    _: KW_ONLY

    tool_name: Literal['load_capability'] = 'load_capability'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Tool name for the typed subclass."""

    args: str | LoadCapabilityArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    """Load-capability call payload."""

    tool_kind: Literal['capability-load'] = 'capability-load'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass."""

    @property
    def typed_args(self) -> LoadCapabilityArgs | None:
        """Parsed load-capability arguments, or `None` for incomplete streaming args."""
        if self.args is None:
            return None
        try:
            return cast('LoadCapabilityArgs', self.args_as_dict(raise_if_invalid=True))
        except (ValueError, AssertionError):
            return None

    @property
    def capability_id(self) -> str | None:
        """Capability id from the parsed args, if available."""
        typed = self.typed_args
        if typed is None:
            return None
        return typed.get('id')


@dataclass(repr=False)
class LoadCapabilityReturnPart(ToolReturnPart):
    """Typed `ToolReturnPart` for the `load_capability` tool."""

    _: KW_ONLY

    content: LoadCapabilityReturn
    """Load-capability return payload.

    Narrows the parent's `ToolReturnContent` to a typed `LoadCapabilityReturn`.
    """

    tool_name: Literal['load_capability'] = 'load_capability'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Tool name for the typed subclass."""

    tool_kind: Literal['capability-load'] = 'capability-load'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass."""

    @property
    def instructions(self) -> str | None:
        """Loaded capability instructions, if any."""
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


# Narrow on `tool_kind` so user tools named `load_capability` are not promoted.
_TOOL_CALL_NARROWERS['capability-load'] = _narrow_load_capability_call
_TOOL_RETURN_NARROWERS['capability-load'] = _narrow_load_capability_return

_TYPED_PART_TAGS[('tool-call', 'capability-load')] = 'capability-load-call'
_TYPED_PART_TAGS[('tool-return', 'capability-load')] = 'capability-load-return'

_TYPED_PART_TAGS_BY_TYPE[LoadCapabilityCallPart] = 'capability-load-call'
_TYPED_PART_TAGS_BY_TYPE[LoadCapabilityReturnPart] = 'capability-load-return'


def parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse visible history to find capabilities loaded via `load_capability`.

    Every [`CompactionPart`][pydantic_ai.messages.CompactionPart] resets the derived
    state at its exact position in a response. This is deliberately provider-agnostic:
    over-counting can expose tools whose load evidence is no longer visible, while
    under-counting once only permitted a redundant, idempotent load. Now that availability
    gates execution, an under-count also *refuses* the call — see
    [`post_compaction_window`][pydantic_ai.messages.post_compaction_window] for when that
    is wrong and what is tracked to fix it.

    Only the [`post_compaction_window`][pydantic_ai.messages.post_compaction_window] is scanned —
    the one definition of the boundary — so only pairs entirely after the boundary count.
    """
    # This module loads while `messages` is still mid-import (see the module-level import note),
    # and `post_compaction_window` is defined after that point, so it can only be imported at call time.
    from .messages import post_compaction_window

    return _parse_loaded_capabilities(post_compaction_window(messages))


def _parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse capability-load evidence from an already-selected message window."""
    call_id_by_tool_call_id: dict[str, str] = {}
    loaded: set[str] = set()
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, LoadCapabilityCallPart):
                if part.capability_id is not None:
                    call_id_by_tool_call_id[part.tool_call_id] = part.capability_id
            elif isinstance(part, LoadCapabilityReturnPart):
                cap_id = call_id_by_tool_call_id.get(part.tool_call_id)
                if cap_id is not None:
                    loaded.add(cap_id)
    return loaded
