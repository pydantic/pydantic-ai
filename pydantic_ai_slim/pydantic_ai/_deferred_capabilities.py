"""Typed message parts for deferred capability loading."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Annotated, Literal, NotRequired

import pydantic
from typing_extensions import TypedDict

# Imported late by `messages.py`; avoid imports that would re-enter it.
from .messages import (
    _TYPED_PART_TAGS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS_BY_TYPE,  # pyright: ignore[reportPrivateUsage]
    ToolCallPart,
    ToolReturnPart,
    TypedArgs,
    TypedContent,
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
class LoadCapabilityCallPart(ToolCallPart, _core=True):
    """Typed `ToolCallPart` for the `load_capability` tool."""

    _: KW_ONLY

    tool_name: Literal['load_capability'] = 'load_capability'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Tool name for the typed subclass."""

    args: str | LoadCapabilityArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    """Load-capability call payload."""

    tool_kind: Literal['capability-load'] = 'capability-load'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass."""

    typed_args = TypedArgs(LoadCapabilityArgs)
    """The validated load-capability arguments, or `None` while they stream in or if they don't fit."""

    @property
    def capability_id(self) -> str | None:
        """Capability id from the parsed args, if available."""
        typed = self.typed_args
        if typed is None:
            return None
        return typed.get('id')


@dataclass(repr=False)
class LoadCapabilityReturnPart(ToolReturnPart, _core=True):
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

    typed_content = TypedContent(LoadCapabilityReturn)
    """The validated load-capability return value, or `None` if it doesn't fit."""

    @property
    def instructions(self) -> str | None:
        """Loaded capability instructions, if any."""
        return self.content.get('instructions')


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


def registered_loaded_capability_ids(messages: Sequence[ModelMessage], capability_ids: Collection[str]) -> set[str]:
    """`parse_loaded_capabilities`, narrowed to capabilities this run actually registered.

    History outlives configuration: a conversation resumed against a smaller capability set still
    carries the load records of capabilities that are no longer configured, and without this
    `RunContext.loaded_capability_ids` — and the `active_capability_ids` that unions it — would
    name a capability the run has no way to act on. Every consumer today starts from a real
    capability or a real `ToolDefinition`, so nothing observes the difference yet; the sets are
    public, though, and should not promise something that isn't there.

    Leans on the registry being seeded once at run start. Capabilities registered mid-run would make
    this a moving target and would need the narrowing reapplied wherever they land.
    """
    return parse_loaded_capabilities(messages) & set(capability_ids)


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
