"""Typed message parts for the framework-emitted `load_capability` tool.

Mirrors the structure of `pydantic_ai._tool_search` (the typed subclasses for the
`search_tools` builtin) — same dispatch-on-`tool_kind` design, same registry-driven
discriminator. `load_capability` has no native server-side equivalent on any current
provider, so there's only the local-shape typed pair (`LoadCapabilityCallPart` /
`LoadCapabilityReturnPart`); there are no `NativeLoadCapability*Part` siblings.

Return content shape is asymmetric — the success path emits a
[`LoadCapabilityReturn`][pydantic_ai.messages.LoadCapabilityReturn] dict, the
catalog-miss path emits a plain string. The typed return part therefore does NOT
shadow `content` with a narrower type (would reject the string); convenience accessors
(`capability_id`, `instructions`, `error`) handle the dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, cast

import pydantic_core
from pydantic import Field
from typing_extensions import TypedDict

from ._deferred import LOAD_CAPABILITY_TOOL_NAME, LoadCapabilityReturn, extract_load_capability_return
from ._utils import copy_dataclass_fields
from .messages import (
    _TOOL_CALL_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _TOOL_RETURN_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS,  # pyright: ignore[reportPrivateUsage]
    _TYPED_PART_TAGS_BY_TYPE,  # pyright: ignore[reportPrivateUsage]
    ToolCallPart,
    ToolReturnContent,
    ToolReturnPart,
)


class LoadCapabilityArgs(TypedDict):
    """Typed arguments for a `load_capability` call.

    Carried on [`LoadCapabilityCallPart.args`][pydantic_ai.messages.LoadCapabilityCallPart.args]
    as the canonical shape.
    """

    id: Annotated[
        str,
        Field(description='The id of the capability to load, as shown in the available capabilities list.'),
    ]


@dataclass(repr=False)
class LoadCapabilityCallPart(ToolCallPart):
    """Typed view of a `ToolCallPart` for the framework-emitted `load_capability` call.

    To detect a load-capability part regardless of construction path (model-emitted
    tool call vs. synthetic injection), check `part.tool_kind == 'capability-load'`.

    Shadows `args` with the canonical typed shape. The `str` variant covers the
    streaming / partial-args case before parsing completes; once parsed, `args` is
    a [`LoadCapabilityArgs`][pydantic_ai.messages.LoadCapabilityArgs] `TypedDict`.
    """

    tool_name: Literal['load_capability'] = LOAD_CAPABILITY_TOOL_NAME  # pyright: ignore[reportIncompatibleVariableOverride]
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

        Returns `None` only in streaming-partial state, where `args` is still an
        in-progress JSON string the model hasn't finished emitting.
        """
        if self.args is None:
            return None
        if isinstance(self.args, dict):
            return self.args
        try:
            parsed = pydantic_core.from_json(self.args)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        return cast('LoadCapabilityArgs', parsed)

    @property
    def capability_id(self) -> str | None:
        """Subfield accessor for `typed_args['id']`. Returns `None` on streaming-partial."""
        typed = self.typed_args
        return typed.get('id') if typed is not None else None


@dataclass(repr=False)
class LoadCapabilityReturnPart(ToolReturnPart):
    """Typed view of a `ToolReturnPart` for the framework-emitted `load_capability` return.

    `content` is intentionally NOT shadowed with a narrower type. The framework emits
    [`LoadCapabilityReturn`][pydantic_ai.messages.LoadCapabilityReturn] on success and
    a plain string on catalog-miss; the property accessors below dispatch between the
    two so callers don't have to.
    """

    # `kw_only=True` keeps the redeclared `content` valid alongside the subclass's defaulted
    # `tool_name` / `tool_kind` overrides: leaving `content` without a default would otherwise
    # place a non-default field after a default one in the synthesized `__init__`. Mirrors
    # the same trick used in `ToolSearchReturnPart`.
    content: ToolReturnContent = field(kw_only=True)
    """The load-capability return content.

    `content` is intentionally left at the inherited `ToolReturnContent` type so both the
    success-path [`LoadCapabilityReturn`][pydantic_ai.messages.LoadCapabilityReturn] dict
    and the catalog-miss string flow through unchanged. Use the `loaded` / `error`
    accessors to dispatch.
    """

    tool_name: Literal['load_capability'] = LOAD_CAPABILITY_TOOL_NAME  # pyright: ignore[reportIncompatibleVariableOverride]
    """Default tool name for the typed subclass. Discrimination drives off `tool_kind`."""

    tool_kind: Literal['capability-load'] = 'capability-load'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass (framework-emitted `load_capability` return)."""

    @property
    def loaded(self) -> LoadCapabilityReturn | None:
        """Validated success-shape return, or `None` for a catalog miss / malformed payload."""
        return extract_load_capability_return(self.content)

    @property
    def error(self) -> str | None:
        """The catalog-miss error message, or `None` when the load succeeded.

        Returns the raw string content when `content` is a string; otherwise `None`.
        Useful for UI rendering that wants to distinguish success from failure without
        re-parsing.
        """
        return self.content if isinstance(self.content, str) else None


def _narrow_load_capability_call(part: ToolCallPart) -> LoadCapabilityCallPart:
    if isinstance(part, LoadCapabilityCallPart):
        return part
    return copy_dataclass_fields(part, LoadCapabilityCallPart, tool_kind='capability-load')


def _narrow_load_capability_return(part: ToolReturnPart) -> LoadCapabilityReturnPart:
    if isinstance(part, LoadCapabilityReturnPart):
        return part
    return copy_dataclass_fields(part, LoadCapabilityReturnPart, tool_kind='capability-load')


# Dispatch is by `tool_kind` (set by the framework when it emits a typed call/return)
# so user-defined tools that happen to share `tool_name` with the framework's are not
# accidentally promoted.
_TOOL_CALL_NARROWERS['capability-load'] = _narrow_load_capability_call
_TOOL_RETURN_NARROWERS['capability-load'] = _narrow_load_capability_return

_TYPED_PART_TAGS[('tool-call', 'capability-load')] = 'capability-load-call'
_TYPED_PART_TAGS[('tool-return', 'capability-load')] = 'capability-load-return'

_TYPED_PART_TAGS_BY_TYPE[LoadCapabilityCallPart] = 'capability-load-call'
_TYPED_PART_TAGS_BY_TYPE[LoadCapabilityReturnPart] = 'capability-load-return'
