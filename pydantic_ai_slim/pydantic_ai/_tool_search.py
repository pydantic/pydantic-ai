"""Tool-search typed message parts and cross-provider history translation.

Tool search has two execution paths that produce typed message parts:

* **Native server-side** (Anthropic BM25/regex, OpenAI Responses): the provider runs
  the search and emits typed
  [`BuiltinToolSearchCallPart`][pydantic_ai.messages.BuiltinToolSearchCallPart] /
  [`BuiltinToolSearchReturnPart`][pydantic_ai.messages.BuiltinToolSearchReturnPart].
* **Local fallback** (any provider): the model calls the regular `search_tools`
  function tool; the toolset emits typed
  [`ToolSearchCallPart`][pydantic_ai.messages.ToolSearchCallPart] /
  [`ToolSearchReturnPart`][pydantic_ai.messages.ToolSearchReturnPart].

User code can match these typed subclasses via `isinstance` (e.g. for UI rendering)
and synthesize them directly to inject discoveries mid-run.

`synthesize_local_tool_search_messages` translates `BuiltinToolSearch*Part` history
into the local-shape typed parts when the next turn runs against a provider
without native tool-search support, so previously discovered tools remain
accessible across provider boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import pydantic
from typing_extensions import NotRequired, TypedDict, assert_never

from . import messages as _messages

# `messages.py` imports this module before its `ModelMessage` / `ModelRequest` / `ModelResponse`
# types are defined; bind the parts we need at class-definition time directly here, and access
# the message-level types via `_messages.ModelResponse` etc. at function-call time.
from .messages import (
    _BUILTIN_CALL_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _BUILTIN_RETURN_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _TOOL_CALL_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    _TOOL_RETURN_NARROWERS,  # pyright: ignore[reportPrivateUsage]
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ToolCallPart,
    ToolReturnPart,
)
from .usage import RequestUsage

if TYPE_CHECKING:
    from .messages import ModelMessage, ModelRequestPart, ModelResponse, ModelResponsePart


class ToolSearchMatch(TypedDict):
    """A single match in a tool-search result."""

    name: str
    """Name of the discovered tool, as the model will call it."""

    description: str | None
    """Human-readable description, if the tool provided one."""


class ToolSearchArgs(TypedDict):
    """Typed arguments for a tool-search call.

    Carried on
    [`BuiltinToolSearchCallPart.args`][pydantic_ai.messages.BuiltinToolSearchCallPart.args]
    (native server-side path) and
    [`ToolSearchCallPart.args`][pydantic_ai.messages.ToolSearchCallPart.args]
    (local-fallback path) as the canonical cross-provider shape. Each adapter
    normalizes its provider's wire format into this shape on parse, and rebuilds the
    wire format from this shape on emit.
    """

    queries: list[str]
    """Normalized search inputs.

    * Anthropic BM25 / regex: single-item list with the query string.
    * OpenAI server-executed `tool_search`: the list of tool paths the model picked.
    * OpenAI client-execution / local `search_tools` fallback: single-item list with
      the keywords string.
    """


class ToolSearchReturnContent(TypedDict):
    """Typed return value of [`ToolSearchTool`][pydantic_ai.builtin_tools.tool_search.ToolSearchTool].

    Carried on
    [`BuiltinToolSearchReturnPart.content`][pydantic_ai.messages.BuiltinToolSearchReturnPart.content]
    (native server-side path) and
    [`ToolSearchReturnPart.content`][pydantic_ai.messages.ToolSearchReturnPart.content]
    (local-fallback path) as the canonical cross-provider shape.
    """

    discovered_tools: list[ToolSearchMatch]
    """Matches ordered by relevance. An empty list means "search ran, nothing matched"."""

    message: NotRequired[str]
    """Optional text shown to the model when no matches were found.

    Rendered as text on local fallback / Anthropic custom-callable empty-results path.
    Stripped on OpenAI client-execution and Anthropic server-side replay (those carry
    only structural fields).
    """


@dataclass(repr=False)
class BuiltinToolSearchCallPart(BuiltinToolCallPart):
    """Typed view of a [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] for tool search.

    Used on the native server-side tool-search path (Anthropic BM25/regex, OpenAI
    Responses) where the provider executes the search and emits a builtin result.
    The local-fallback path uses
    [`ToolSearchCallPart`][pydantic_ai.messages.ToolSearchCallPart] instead.

    Shadows `args` with a narrower type. The `str` variant covers the
    streaming / partial-args case before parsing completes; once parsed,
    `args` is a [`ToolSearchArgs`][pydantic_ai.messages.ToolSearchArgs]
    `TypedDict`.
    """

    tool_name: Literal['tool_search'] = 'tool_search'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Default tool name for the typed subclass. Discrimination drives off `tool_kind`."""

    args: str | ToolSearchArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    """Tool-search query payload.

    Narrows the parent's `str | dict[str, Any] | None` to a typed
    [`ToolSearchArgs`][pydantic_ai.messages.ToolSearchArgs] when parsed. Streaming /
    partial-args still arrive as `str` until they're complete.
    """

    tool_kind: Literal['tool-search'] = 'tool-search'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass (cross-provider tool-search call)."""


@dataclass(repr=False)
class BuiltinToolSearchReturnPart(BuiltinToolReturnPart):
    """Typed view of a [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] for tool search.

    Used on the native server-side tool-search path (Anthropic BM25/regex, OpenAI
    Responses) where the provider executes the search and emits a builtin result.
    The local-fallback path uses
    [`ToolSearchReturnPart`][pydantic_ai.messages.ToolSearchReturnPart] instead.

    Shadows `content` with a narrower
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]
    `TypedDict`.
    """

    tool_name: Literal['tool_search'] = 'tool_search'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Default tool name for the typed subclass. Discrimination drives off `tool_kind`."""

    content: ToolSearchReturnContent | None = None
    """Discovered-tools payload.

    Narrows the parent's `ToolReturnContent` to a typed
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent].
    """

    tool_kind: Literal['tool-search'] = 'tool-search'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass (cross-provider tool-search return)."""


@dataclass(repr=False)
class ToolSearchCallPart(ToolCallPart):
    """Typed view of a [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] for the local `search_tools` function call.

    Used on the local-fallback path (and as the synthetic-injection target on
    non-native providers receiving cross-provider history). The native server-side
    path uses
    [`BuiltinToolSearchCallPart`][pydantic_ai.messages.BuiltinToolSearchCallPart]
    instead.

    Shadows `args` with the canonical typed shape. The `str` variant covers the
    streaming / partial-args case before parsing completes; once parsed,
    `args` is a [`ToolSearchArgs`][pydantic_ai.messages.ToolSearchArgs]
    `TypedDict`.
    """

    tool_name: Literal['search_tools'] = 'search_tools'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Default tool name for the typed subclass. Discrimination drives off `tool_kind`."""

    args: str | ToolSearchArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    """Tool-search query payload.

    Narrows the parent's `str | dict[str, Any] | None` to a typed
    [`ToolSearchArgs`][pydantic_ai.messages.ToolSearchArgs] when parsed. Streaming /
    partial-args still arrive as `str` until they're complete.
    """

    tool_kind: Literal['tool-search'] = 'tool-search'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass (framework-emitted `search_tools` call)."""


@dataclass(repr=False)
class ToolSearchReturnPart(ToolReturnPart):
    """Typed view of a [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] for the local `search_tools` function return.

    Used on the local-fallback path (and as the synthetic-injection target on
    non-native providers receiving cross-provider history). The native server-side
    path uses
    [`BuiltinToolSearchReturnPart`][pydantic_ai.messages.BuiltinToolSearchReturnPart]
    instead.

    Shadows `content` with a narrower
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]
    `TypedDict`.
    """

    tool_name: Literal['search_tools'] = 'search_tools'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Default tool name for the typed subclass. Discrimination drives off `tool_kind`."""

    content: ToolSearchReturnContent | None = None
    """Discovered-tools payload.

    Narrows the parent's `ToolReturnContent` to a typed
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent].
    """

    tool_kind: Literal['tool-search'] = 'tool-search'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Discriminator for the typed subclass (framework-emitted `search_tools` return)."""


_TOOL_SEARCH_CALL_ARGS_TA: pydantic.TypeAdapter[str | ToolSearchArgs | None] = pydantic.TypeAdapter(
    Union[str, ToolSearchArgs, None]  # noqa: UP007
)
_TOOL_SEARCH_RETURN_CONTENT_TA: pydantic.TypeAdapter[ToolSearchReturnContent | None] = pydantic.TypeAdapter(
    Union[ToolSearchReturnContent, None]  # noqa: UP007
)


def _copy_dataclass_fields(src: Any, dst_cls: type, **overrides: Any) -> Any:
    """Construct a new dataclass instance from `src`'s fields, overriding selected ones.

    Lets typed-part narrowers stay maintainable when fields are added to the base
    class — `BaseToolCallPart` / `BaseToolReturnPart` field changes flow through automatically
    instead of needing every narrower to be updated by hand.
    """
    field_values: dict[str, Any] = {f.name: getattr(src, f.name) for f in fields(src)}
    field_values.update(overrides)
    return dst_cls(**field_values)


def _narrow_builtin_tool_search_call(part: BuiltinToolCallPart) -> BuiltinToolSearchCallPart:
    if isinstance(part, BuiltinToolSearchCallPart):
        return part
    validated_args = _TOOL_SEARCH_CALL_ARGS_TA.validate_python(part.args)
    return _copy_dataclass_fields(part, BuiltinToolSearchCallPart, args=validated_args, tool_kind='tool-search')


def _narrow_builtin_tool_search_return(part: BuiltinToolReturnPart) -> BuiltinToolSearchReturnPart:
    if isinstance(part, BuiltinToolSearchReturnPart):
        return part
    validated_content = _TOOL_SEARCH_RETURN_CONTENT_TA.validate_python(part.content)
    return _copy_dataclass_fields(part, BuiltinToolSearchReturnPart, content=validated_content, tool_kind='tool-search')


def _narrow_tool_search_call(part: ToolCallPart) -> ToolSearchCallPart:
    if isinstance(part, ToolSearchCallPart):
        return part
    validated_args = _TOOL_SEARCH_CALL_ARGS_TA.validate_python(part.args)
    return _copy_dataclass_fields(part, ToolSearchCallPart, args=validated_args, tool_kind='tool-search')


def _narrow_tool_search_return(part: ToolReturnPart) -> ToolSearchReturnPart:
    if isinstance(part, ToolSearchReturnPart):
        return part
    validated_content = _TOOL_SEARCH_RETURN_CONTENT_TA.validate_python(part.content)
    return _copy_dataclass_fields(part, ToolSearchReturnPart, content=validated_content, tool_kind='tool-search')


# Narrowers dispatch on `tool_kind` (set by the framework when it emits a typed call/return)
# so user-defined tools that happen to share `tool_name` with a typed subclass are not
# accidentally promoted.
_BUILTIN_CALL_NARROWERS['tool-search'] = _narrow_builtin_tool_search_call
_BUILTIN_RETURN_NARROWERS['tool-search'] = _narrow_builtin_tool_search_return
_TOOL_CALL_NARROWERS['tool-search'] = _narrow_tool_search_call
_TOOL_RETURN_NARROWERS['tool-search'] = _narrow_tool_search_return


def _split_response(original: ModelResponse, parts: list[ModelResponsePart], *, first: bool) -> ModelResponse:
    """Build a split-off `ModelResponse` carrying a subset of `original`'s parts.

    `first=True` keeps the original's identity-level metadata (provider response id,
    usage, etc.). `first=False` blanks `provider_response_id` and zeroes `usage` so
    downstream consumers don't double-count usage or find two responses for one API
    call. Other contextual fields (model name, provider name, timestamp) carry over
    unchanged — they're informational on a synthetic split.
    """
    if first:
        return replace(original, parts=parts)
    return replace(
        original,
        parts=parts,
        provider_response_id=None,
        usage=RequestUsage(),
    )


def synthesize_local_from_builtin_call(part: BuiltinToolSearchCallPart) -> ToolSearchCallPart:
    """Translate a server-side tool-search call to a local function-tool call.

    Preserves `tool_call_id` so the matching return part links up; drops
    `provider_*` because the local-shape part is provider-agnostic.
    """
    return ToolSearchCallPart(
        args=part.args,
        tool_call_id=part.tool_call_id,
    )


def synthesize_local_from_builtin_return(part: BuiltinToolSearchReturnPart) -> ToolSearchReturnPart:
    """Translate a server-side tool-search return to a local function-tool return.

    Preserves `tool_call_id`, `content` (the typed
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]),
    and `metadata`; drops `provider_*` because the local-shape part is
    provider-agnostic.
    """
    return ToolSearchReturnPart(
        content=part.content,
        tool_call_id=part.tool_call_id,
        metadata=part.metadata,
        timestamp=part.timestamp,
        outcome=part.outcome,
    )


def synthesize_local_tool_search_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Translate any `BuiltinToolSearch*Part` instances in the message history into local equivalents.

    Returns a new list with translated copies of any messages that contain
    `BuiltinToolSearch*Part`s; messages without such parts are returned
    unchanged (no copy). Suitable for non-native adapters that don't support
    native tool search but need to honor discovered-tool state from prior turns
    on different providers.

    A native server-side tool-search exchange is a single `ModelResponse` carrying both
    `BuiltinToolSearchCallPart` (the call) and `BuiltinToolSearchReturnPart` (the inline
    server-side result). Local function-tool execution shapes the same exchange as a pair
    of messages — `ModelResponse(parts=[ToolSearchCallPart(...)])` followed by
    `ModelRequest(parts=[ToolSearchReturnPart(...)])` — because the model produces the
    call and the framework produces the return in a separate request turn.

    Each `BuiltinToolSearchReturnPart` acts as a flush boundary when splitting: parts
    before it (text, the search call itself) become a `ModelResponse`, the return becomes
    a `ModelRequest`, and any parts after it (downstream tool calls, more text) become a
    fresh `ModelResponse`. This preserves the natural turn order — e.g. a native turn
    `[Text, SearchCall, SearchReturn, ToolCall(weather)]` translates to four messages
    where the weather call sits on its own response after the search return, matching
    what the model would have emitted across two turns on a non-native provider.

    Identity-level metadata (`provider_response_id`, `usage`) is kept on the first split
    response only; subsequent splits get blank/zero values so downstream consumers don't
    double-count usage or treat one API call as two distinct responses.
    """
    out: list[ModelMessage] = []

    for msg in messages:
        if isinstance(msg, _messages.ModelResponse):
            buffer: list[ModelResponsePart] = []
            split_emitted = False  # Tracks whether we've emitted a response from this msg already.
            changed = False
            for part in msg.parts:
                if isinstance(part, BuiltinToolSearchCallPart):
                    buffer.append(synthesize_local_from_builtin_call(part))
                    changed = True
                elif isinstance(part, BuiltinToolSearchReturnPart):
                    # Flush the buffered parts as a `ModelResponse` (skip if empty), then
                    # emit the search return as its own `ModelRequest`. Subsequent parts
                    # start a fresh buffer that becomes the next `ModelResponse`.
                    if buffer:
                        out.append(_split_response(msg, buffer, first=not split_emitted))
                        split_emitted = True
                    out.append(
                        _messages.ModelRequest(
                            parts=cast('list[ModelRequestPart]', [synthesize_local_from_builtin_return(part)]),
                        ),
                    )
                    buffer = []
                    changed = True
                else:
                    buffer.append(part)
            if changed:
                if buffer:
                    out.append(_split_response(msg, buffer, first=not split_emitted))
            else:
                out.append(msg)
        elif isinstance(msg, _messages.ModelRequest):
            # Translate any framework-emitted `ToolReturnPart` with `tool_kind='tool-search'`
            # on requests — covers fresh code paths that constructed a base `ToolReturnPart`
            # directly while still flagging it as framework-emitted. Dispatching on `tool_kind`
            # rather than `tool_name` means a user tool literally named `search_tools` is left
            # alone as a base `ToolReturnPart`.
            request_changed = False
            new_request_parts: list[ModelRequestPart] = []
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and not isinstance(part, ToolSearchReturnPart)
                    and part.tool_kind == 'tool-search'
                ):
                    promoted = ToolReturnPart.narrow_type(part)
                    if isinstance(promoted, ToolSearchReturnPart):  # pragma: no branch
                        new_request_parts.append(promoted)
                        request_changed = True
                        continue
                new_request_parts.append(part)
            if request_changed:
                out.append(replace(msg, parts=new_request_parts))
            else:
                out.append(msg)
        else:
            assert_never(msg)

    return out
