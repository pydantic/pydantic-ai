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
    of messages instead — `ModelResponse(parts=[ToolSearchCallPart(...)])` followed by
    `ModelRequest(parts=[ToolSearchReturnPart(...)])` — because the model produces the
    call and the framework produces the return in a separate request turn. The translation
    here splits the original response accordingly: builtin call parts stay on the response
    (as local `ToolSearchCallPart`), builtin return parts are lifted onto a fresh trailing
    `ModelRequest` immediately after the response.
    """
    out: list[ModelMessage] = []

    for msg in messages:
        if isinstance(msg, ModelResponse):
            new_parts: list[ModelResponsePart] = []
            lifted_returns: list[ToolSearchReturnPart] = []
            changed = False
            for part in msg.parts:
                if isinstance(part, BuiltinToolSearchCallPart):
                    new_parts.append(synthesize_local_from_builtin_call(part))
                    changed = True
                elif isinstance(part, BuiltinToolSearchReturnPart):
                    # Lift the return out of the response into a fresh trailing request.
                    lifted_returns.append(synthesize_local_from_builtin_return(part))
                    changed = True
                else:
                    new_parts.append(part)
            if changed:
                # Only emit a response if there's anything left after lifting returns.
                if new_parts:
                    out.append(replace(msg, parts=new_parts))
                if lifted_returns:
                    out.append(ModelRequest(parts=cast('list[ModelRequestPart]', list(lifted_returns))))
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
