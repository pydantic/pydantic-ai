"""Typed message parts for deferred capability loading."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal, NotRequired

import pydantic
from typing_extensions import TypedDict

# Imported late by `messages.py`; avoid imports that would re-enter it.
from .messages import (
    _TYPED_PART_TAGS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS_BY_TYPE,  # pyright: ignore[reportPrivateUsage]
    ToolCallPart,
    ToolKind,
    ToolReturnPart,
)

if TYPE_CHECKING:
    from .messages import ModelMessage, ModelRequestPart, ModelResponsePart


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

    message: NotRequired[str]
    """Confirms the load to the model, and says whether the capability's instructions follow.

    Absent from histories recorded by earlier versions.
    """

    instructions: NotRequired[str]
    """Instructions for the loaded capability, as they were when it loaded, in a realtime session only.

    Elsewhere the load returns only its `message`, and the instructions are recorded as
    [`InstructionDeltaPart`][pydantic_ai.messages.InstructionDeltaPart]s for the capability's block,
    `capability:<id>`, on the request after the load. Histories from earlier versions carry them here, and
    those still seed the block's value without being sent again.
    """


CAPABILITY_LOAD_KIND = ToolKind._core(  # pyright: ignore[reportPrivateUsage]
    'capability-load', args=LoadCapabilityArgs, content=LoadCapabilityReturn
)
"""The kind of the `load_capability` tool."""


@dataclass(repr=False)
class LoadCapabilityCallPart(ToolCallPart[LoadCapabilityArgs], kind=CAPABILITY_LOAD_KIND):
    """Typed `ToolCallPart` for the `load_capability` tool."""

    _: KW_ONLY

    tool_name: Literal['load_capability'] = 'load_capability'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Tool name for the typed subclass."""

    args: str | LoadCapabilityArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    """Load-capability call payload."""

    tool_kind: Literal['capability-load'] = 'capability-load'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass."""

    @property
    def capability_id(self) -> str | None:
        """Capability id from the parsed args, if available."""
        typed = self.typed_args
        if typed is None:
            return None
        return typed.get('id')


@dataclass(repr=False)
class LoadCapabilityReturnPart(ToolReturnPart[LoadCapabilityReturn], kind=CAPABILITY_LOAD_KIND):
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
    exchanges = LoadExchanges()
    return {
        capability_id
        for message in messages
        for part in message.parts
        if (capability_id := exchanges.observe(part)) is not None
    }


@dataclass
class LoadExchanges:
    """Pairs `load_capability` calls with their returns: the one rule for which capability a return loads.

    A return loads the capability its call named, when the call was seen first and its arguments parsed. A
    failed load is a plain `ToolReturnPart`, so it loads nothing.
    """

    _calls: dict[str, str] = field(default_factory=dict[str, str])

    def observe(self, part: ModelRequestPart | ModelResponsePart) -> str | None:
        """Fold in `part`, returning the capability it loads if it's a return completing a load."""
        if isinstance(part, LoadCapabilityCallPart):
            if part.capability_id is not None:
                self._calls[part.tool_call_id] = part.capability_id
        elif isinstance(part, LoadCapabilityReturnPart):
            return self._calls.get(part.tool_call_id)
        return None
