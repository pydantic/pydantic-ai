"""Tool search: hosted and client-executed calls, their outputs, and their replay."""

from __future__ import annotations as _annotations

from collections import deque
from typing import Any, Literal, cast

from openai.types import responses
from openai.types.responses.response_tool_search_call import ResponseToolSearchCall
from openai.types.responses.response_tool_search_output_item_param_param import ResponseToolSearchOutputItemParamParam
from typing_extensions import Protocol

from ... import UnexpectedModelBehavior
from ...messages import (
    NativeToolCallPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
)
from ...native_tools._tool_search import TOOL_SEARCH_FUNCTION_TOOL_NAME, ToolSearchArgs, ToolSearchMatch, ToolSearchTool
from ...tools import ToolDefinition
from .. import ModelRequestParameters, ToolVisibility


def tool_search_namespace_for_synthesis(tool_name: str, model_request_parameters: ModelRequestParameters) -> str | None:
    """Return the synthetic OpenAI namespace for a cross-provider replay, or `None`.

    OpenAI-origin calls round-trip `provider_details['namespace']`. Non-OpenAI history lacks that
    field, but OpenAI rejects a replayed call to a tool that isn't in the default namespace when the
    call doesn't say so (`Missing namespace for function_call '...'. It does not exist in the default
    namespace.`). For the flat deferred function tools this adapter emits, OpenAI-generated calls use
    `namespace == tool_name` — verified by live probe against a capability owning multiple deferred
    tools.

    What decides is the tool's placement on *this* request's wire, not what kind of tool it is: a
    name declared through an `additional_tools` item, or occupying a `tools` entry with its schema
    withheld behind `defer_loading`, lives outside the default namespace, while a plain `tools` entry
    is default-namespace even if stored history happens to mention its name — tagging that one would
    be exactly the mismatch the error above complains about, from the other direction.
    """
    if tool_name not in model_request_parameters.revealed_tool_names:
        return None
    for tool in model_request_parameters.function_tools:
        if tool.name == tool_name and model_request_parameters.visibility_of(tool.name) in (
            'deferred',
            'via_history',
        ):
            return tool_name
    return None


def has_tool_search(model_request_parameters: ModelRequestParameters) -> bool:
    """Whether the current run carries any `ToolSearchTool` builtin.

    Used to gate two related behaviors:

    * Filter the local `search_tools` function tool out of the wire `tools[]` (it's
      represented by the builtin instead).
    * Promote local-shape `ToolSearch*Part` history into native `tool_search_call` /
      `tool_search_output` items (`execution='client'`), unlocking previously-discovered
      `defer_loading=True` tools without forcing the model to re-search.

    Both behaviors apply regardless of the active strategy (default native, named
    native, or custom callable) — the gate is "tool search is active in some form".
    """
    return any(isinstance(t, ToolSearchTool) for t in model_request_parameters.native_tools)


def find_search_tool_definition(
    model_request_parameters: ModelRequestParameters,
) -> ToolDefinition | None:
    """Locate the local `search_tools` function-tool definition in the current request.

    In custom-callable tool search mode, `ToolSearchToolset` leaves its `search_tools`
    function tool in `function_tools` (no `unless_native`), so we look it up by its `tool_kind`.
    """
    return next(
        (t for t in model_request_parameters.function_tools if t.tool_kind == 'tool-search'),
        None,
    )


def normalize_tool_search_args(raw: Any) -> ToolSearchArgs:
    """Translate an OpenAI `tool_search_call.arguments` payload into `ToolSearchArgs`.

    OpenAI's wire shape varies by execution mode:

    * **Server-executed `tool_search`**: the result carries the picked tool paths under
      `paths`. Fold those into the canonical `queries` slot for cross-provider history.
    * **Client-executed `tool_search` (`execution='client'`)**: arguments mirror the
      schema we registered for the local `search_tools` function tool — already
      `{"queries": list[str]}` — so pass through unchanged.

    Empty `arguments={}` (the streaming-mid first-event case) returns `{'queries': []}`.
    Any other shape we don't recognize raises `UnexpectedModelBehavior` — schema drift
    in the OpenAI SDK should surface loudly at the parse boundary, not silently degrade.
    """
    if isinstance(raw, dict):
        if not raw:
            return {'queries': []}
        queries = raw.get('queries')  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(queries, list):
            return {'queries': [q for q in queries if isinstance(q, str)]}  # pyright: ignore[reportUnknownVariableType]
        paths = raw.get('paths')  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(paths, list):
            return {'queries': [p for p in paths if isinstance(p, str)]}  # pyright: ignore[reportUnknownVariableType]
    raise UnexpectedModelBehavior(f'Unrecognized tool_search arguments shape: {raw!r}')


def tool_search_call_ids_by_output(response: responses.Response) -> dict[str, str]:
    """Pair hosted tool-search outputs with calls, using response order when IDs are absent."""
    pending_call_ids = deque(
        item.id
        for item in response.output
        if isinstance(item, responses.ResponseToolSearchCall) and item.execution == 'server' and item.call_id is None
    )
    server_call_ids = {
        item.call_id or item.id
        for item in response.output
        if isinstance(item, responses.ResponseToolSearchCall) and item.execution == 'server'
    }
    output_call_ids: dict[str, str] = {}
    for item in response.output:
        if not isinstance(item, responses.ResponseToolSearchOutputItem) or item.execution != 'server':
            continue
        if item.call_id is not None and item.call_id not in server_call_ids:
            continue
        if (call_id := match_tool_search_output_to_call(item.call_id, pending_call_ids)) is not None:
            output_call_ids[item.id] = call_id
    return output_call_ids


def match_tool_search_output_to_call(output_call_id: str | None, pending_call_ids: deque[str]) -> str | None:
    """Match an output to a call and remove that call from the pending queue."""
    if output_call_id is not None:
        # The provider's explicit association is authoritative.
        if output_call_id in pending_call_ids:
            pending_call_ids.remove(output_call_id)
        return output_call_id
    if pending_call_ids:
        # Anonymous hosted-search outputs follow their calls in FIFO order.
        return pending_call_ids.popleft()
    return None


def map_tool_search_call(item: ResponseToolSearchCall, provider_name: str) -> NativeToolSearchCallPart:
    """Map an OpenAI server-executed tool-search call without fabricating its output."""
    call_id = item.call_id or item.id
    return NativeToolSearchCallPart(
        provider_name=provider_name,
        args=normalize_tool_search_args(item.arguments),
        tool_call_id=call_id,
        id=item.id,
        provider_details={'call_id': item.call_id, 'execution': item.execution, 'status': item.status},
    )


def tool_search_replay_details(
    part: NativeToolCallPart | NativeToolSearchReturnPart,
) -> tuple[str | None, Literal['in_progress', 'completed', 'incomplete'], dict[str, Any]]:
    """Read provider-native tool-search identity and status with backward-compatible defaults."""
    details = part.provider_details or {}
    call_id = details.get('call_id', part.tool_call_id)
    if not isinstance(call_id, str | None):
        call_id = part.tool_call_id
    status = details.get('status')
    if status not in ('in_progress', 'completed', 'incomplete'):
        status = 'completed'
    return call_id, status, details


def lacks_tool_search_output_identity(part: NativeToolSearchReturnPart) -> bool:
    """Identify return parts that did not come from a preserved `tool_search_output` item.

    Parts built from a real output item always carry its `id` in `provider_details`
    (see `build_tool_search_return_part`). Parts without it are pre-fix history
    (only `status` was stashed) or metadata-stripped round-trips. For those, replay
    falls back to sending the call only, the shape pre-fix code sent and the only
    one proven against the live API for such histories; fabricating an output item
    risks colliding with server-side state the call already references.
    """
    output_id = (part.provider_details or {}).get('id')
    return not (isinstance(output_id, str) and output_id)


class _MapToolDefinition(Protocol):
    def __call__(self, f: ToolDefinition, *, visibility: ToolVisibility) -> responses.FunctionToolParam: ...


def build_tool_search_output_param(
    part: ToolSearchReturnPart | NativeToolSearchReturnPart,
    call_id: str | None,
    execution: Literal['server', 'client'],
    status: Literal['in_progress', 'completed', 'incomplete'],
    model_request_parameters: ModelRequestParameters,
    map_tool_definition: _MapToolDefinition,
) -> ResponseToolSearchOutputItemParamParam:
    """Build a `tool_search_output` replay param from normalized discovery state.

    Looks up each discovered tool name in `function_tools` and emits a
    `FunctionToolParam` so OpenAI sees the same `Tool` definitions it originally
    loaded in the prior turn. The shape is `ResponseToolSearchOutputItemParamParam`.
    Reads the typed
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]
    off of `part.content`; the tool-return value is the contract. Uses the same
    `map_tool_definition` the model used at initial send-time, so `strict` honors
    the user's preference and the model profile rather than a replay-only override.
    """
    discovered = [match['name'] for match in part.discovered_tools]
    tool_defs_by_name = {t.name: t for t in model_request_parameters.function_tools}
    tool_params: list[responses.tool_param.ToolParam] = [
        cast(
            'responses.tool_param.ToolParam',
            map_tool_definition(tool_def, visibility=model_request_parameters.visibility_of(name)),
        )
        for name in discovered
        if (tool_def := tool_defs_by_name.get(name)) is not None
    ]
    return {
        'type': 'tool_search_output',
        'execution': execution,
        'tools': tool_params,
        'call_id': call_id,
        'status': status,
    }


def map_client_tool_search_call(item: ResponseToolSearchCall, provider_name: str) -> ToolSearchCallPart:
    """Map a client-executed OpenAI `tool_search_call` into a typed `ToolSearchCallPart`.

    With `ToolSearchToolParam(execution='client')`, OpenAI still emits the call wrapped
    as a `tool_search_call` item but leaves execution to us: the standard agent-graph
    tool-execution path runs the local `search_tools` function and produces the
    matching `ToolSearchReturnPart`.

    OpenAI's wire shape for `tool_search_call.arguments` mirrors the schema we registered
    with the builtin — `{"queries": list[str]}` matching the cross-provider
    [`ToolSearchArgs`][pydantic_ai.messages.ToolSearchArgs] — so we forward it
    unchanged. Streaming-mid item events carry `arguments={}` (empty), which validates
    fine as `{"queries": []}` per `normalize_tool_search_args`.

    Emits the typed [`ToolSearchCallPart`][pydantic_ai.messages.ToolSearchCallPart]
    subclass directly — the wire shape is distinct enough that the adapter can identify
    this as a tool-search call without going through the framework's post-hoc
    `_narrow_tool_call_parts` pass.
    """
    call_id = item.call_id or item.id
    args = normalize_tool_search_args(item.arguments)
    return ToolSearchCallPart(
        tool_name=TOOL_SEARCH_FUNCTION_TOOL_NAME,
        args=args,
        tool_call_id=call_id,
        id=item.id,
        provider_name=provider_name,
        tool_kind='tool-search',
    )


def build_tool_search_return_part(
    call_id: str,
    output_item: responses.ResponseToolSearchOutputItem,
    provider_name: str,
) -> NativeToolSearchReturnPart:
    """Build the typed return part for an actual OpenAI tool-search output.

    Writes the cross-provider
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]
    to `content` (carrying only the matched tool names — the full
    [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] is injected on the
    next request via defer-loading, so description is redundant here) and
    stashes the provider item identity and state on `provider_details` for replay.
    """
    matches: list[ToolSearchMatch] = []
    # `output_item.tools` is a union of OpenAI Responses tool variants; only
    # function tools carry a stable `name` field. Other variants (file_search,
    # image generation, etc.) can't appear here in practice but aren't
    # statically excluded from the union, so we filter by type.
    for t in output_item.tools:
        if isinstance(t, responses.FunctionTool):
            matches.append({'name': t.name})
    return NativeToolSearchReturnPart(
        provider_name=provider_name,
        content={'discovered_tools': matches},
        tool_call_id=call_id,
        provider_details={
            'id': output_item.id,
            'call_id': output_item.call_id,
            'execution': output_item.execution,
            'status': output_item.status,
        },
    )
