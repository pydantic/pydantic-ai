"""Cross-provider tool-search history translation.

Tool search has two execution paths:

* **Native server-side**: provider runs the search and emits typed
  [`BuiltinToolSearchCallPart`][pydantic_ai.messages.BuiltinToolSearchCallPart] /
  [`BuiltinToolSearchReturnPart`][pydantic_ai.messages.BuiltinToolSearchReturnPart].
* **Local fallback**: model calls the regular `search_tools` function tool; the
  toolset emits typed
  [`ToolSearchCallPart`][pydantic_ai.messages.ToolSearchCallPart] /
  [`ToolSearchReturnPart`][pydantic_ai.messages.ToolSearchReturnPart].

When a user runs a turn against a native provider (e.g. Anthropic) and then a
follow-up turn against a non-native provider (e.g. Google), the history carries
`BuiltinToolSearch*Part` instances. Non-native adapters can't ship those on the
wire — they don't have a tool-search builtin to attach them to. Instead, they
call [`synthesize_local_tool_search_messages`][pydantic_ai._tool_search_synthetic.synthesize_local_tool_search_messages]
at the start of message-prep to translate any `BuiltinToolSearch*Part` carried
over from a prior native turn into the local-shape typed parts. The model then
sees a normal function-call exchange against `search_tools`, and the toolset's
`_parse_discovered_tools` picks up the discoveries via the discriminated-union
dispatch — restoring access to previously-discovered tools across provider
boundaries.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from .builtin_tools.tool_search import TOOL_SEARCH_FUNCTION_TOOL_NAME
from .messages import (
    BuiltinToolSearchCallPart,
    BuiltinToolSearchReturnPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ToolReturnPart,
    ToolSearchArgs,
    ToolSearchCallPart,
    ToolSearchReturnContent,
    ToolSearchReturnPart,
)
from .usage import RequestUsage


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
    args = part.args
    if isinstance(args, dict):
        normalized: ToolSearchArgs | None = {'queries': list(args.get('queries', []))}
    elif isinstance(args, str):
        normalized = args  # pyright: ignore[reportAssignmentType] -- streaming-string passthrough
    else:
        normalized = None
    return ToolSearchCallPart(
        args=normalized,
        tool_call_id=part.tool_call_id,
    )


def synthesize_local_from_builtin_return(part: BuiltinToolSearchReturnPart) -> ToolSearchReturnPart:
    """Translate a server-side tool-search return to a local function-tool return.

    Preserves `tool_call_id`, `content` (the typed
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]),
    and `metadata`; drops `provider_*` because the local-shape part is
    provider-agnostic.
    """
    content: ToolSearchReturnContent | str | None
    if isinstance(part.content, (dict, str)):
        content = part.content
    else:
        content = {'discovered_tools': []}
    return ToolSearchReturnPart(
        content=content,
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
        if isinstance(msg, ModelResponse):
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
                        ModelRequest(
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
        elif isinstance(msg, ModelRequest):
            # Also translate any direct `ToolReturnPart('search_tools', ...)` on requests —
            # covers fresh code paths that constructed a base `ToolReturnPart` with the local
            # typed args/content. (Pydantic deserialization auto-promotes via the
            # discriminated-union dispatch.)
            request_changed = False
            new_request_parts: list[ModelRequestPart] = []
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and not isinstance(part, ToolSearchReturnPart)
                    and part.tool_name == TOOL_SEARCH_FUNCTION_TOOL_NAME
                ):
                    promoted = ToolReturnPart.narrow_type(part)
                    # The registered narrower for `search_tools` always returns a
                    # `ToolSearchReturnPart`; the isinstance guard is defensive in case
                    # the registry is mutated at runtime.
                    if isinstance(promoted, ToolSearchReturnPart):  # pragma: no branch
                        new_request_parts.append(promoted)
                        request_changed = True
                        continue
                new_request_parts.append(part)
            if request_changed:
                out.append(replace(msg, parts=new_request_parts))
            else:
                out.append(msg)
        else:  # pragma: no cover - exhaustive over ModelMessage union
            out.append(msg)

    return out
