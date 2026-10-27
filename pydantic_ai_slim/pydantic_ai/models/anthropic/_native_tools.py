"""Mapping native tool blocks between Pydantic AI parts and the Anthropic wire format."""

from __future__ import annotations as _annotations

from dataclasses import replace
from typing import Any, Literal, TypeAlias, assert_never, cast

import pydantic_core
from anthropic.types.beta import (
    BetaAdvisorTool20260301Param,
    BetaAdvisorToolResultBlock,
    BetaBashCodeExecutionToolResultBlock,
    BetaBashCodeExecutionToolResultBlockParam,
    BetaCacheControlEphemeralParam,
    BetaCodeExecutionTool20250825Param,
    BetaCodeExecutionTool20260120Param,
    BetaCodeExecutionToolResultBlock,
    BetaCodeExecutionToolResultBlockContent,
    BetaCodeExecutionToolResultBlockParam,
    BetaCodeExecutionToolResultBlockParamContentParam,
    BetaDirectCaller,
    BetaMCPToolResultBlock,
    BetaMCPToolUseBlock,
    BetaServerToolCaller,
    BetaServerToolCaller20260120,
    BetaServerToolUseBlock,
    BetaServerToolUseBlockParam,
    BetaTextEditorCodeExecutionToolResultBlock,
    BetaTextEditorCodeExecutionToolResultBlockParam,
    BetaToolReferenceBlockParam,
    BetaToolSearchToolResultBlock,
    BetaToolSearchToolResultBlockParam,
    BetaToolSearchToolResultErrorParam,
    BetaToolSearchToolSearchResultBlockParam,
    BetaToolUnionParam,
    BetaWebFetchToolResultBlock,
    BetaWebFetchToolResultBlockParam,
    BetaWebSearchToolResultBlock,
    BetaWebSearchToolResultBlockContent,
    BetaWebSearchToolResultBlockParam,
)
from anthropic.types.beta.beta_advisor_tool_result_block import Content as AdvisorToolResultBlockContent
from anthropic.types.beta.beta_bash_code_execution_tool_result_block import (
    Content as BashCodeExecutionToolResultBlockContent,
)
from anthropic.types.beta.beta_bash_code_execution_tool_result_block_param import (
    Content as BashCodeExecutionToolResultBlockParamContent,
)
from anthropic.types.beta.beta_text_editor_code_execution_tool_result_block import (
    Content as TextEditorCodeExecutionToolResultBlockContent,
)
from anthropic.types.beta.beta_text_editor_code_execution_tool_result_block_param import (
    Content as TextEditorCodeExecutionToolResultBlockParamContent,
)
from pydantic import TypeAdapter

from ... import _utils
from ...messages import (
    NativeToolCallPart,
    NativeToolReturnPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    ToolReturnPart,
    ToolSearchReturnPart,
)
from ...native_tools import AdvisorTool, CodeExecutionTool, MCPServerTool, WebFetchTool, WebSearchTool
from ...native_tools._tool_search import ToolSearchArgs, ToolSearchMatch
from ...profiles.anthropic import AnthropicCodeExecutionToolVersion

_AnthropicCodeExecutionToolName: TypeAlias = Literal[
    'code_execution', 'bash_code_execution', 'text_editor_code_execution'
]


_AnthropicCodeExecutionProviderDetailToolName: TypeAlias = Literal['bash_code_execution', 'text_editor_code_execution']


_ANTHROPIC_CODE_EXECUTION_TOOL_NAMES: tuple[_AnthropicCodeExecutionToolName, ...] = (
    'code_execution',
    'bash_code_execution',
    'text_editor_code_execution',
)


_ANTHROPIC_CODE_EXECUTION_TOOL_NAME_DETAIL = 'anthropic_tool_name'


_ANTHROPIC_SERVER_TOOL_CALLER_DETAIL = 'anthropic_caller'


def build_custom_tool_search_replay_blocks(
    request_part: ToolReturnPart, deferred_tools_active: bool, declared_tool_names: set[str]
) -> tuple[list[BetaToolReferenceBlockParam] | None, str | None]:
    """Tool-search replay payload for the Anthropic `tool_result` block.

    Reads the typed [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]
    off `part.content` (the local `search_tools` return shape) and unpacks it into:

    * `tool_references`: matched tools, ready to be wrapped in `BetaToolReferenceBlockParam`s.
    * `empty_message`: fallback text to send when there is no reference to send (Anthropic
      rejects an empty `tool_result` content list): the typed return's own message when the
      search matched nothing, or a line naming the matches when none of them is declared on
      this request.

    Returns `(None, None)` when the current request withholds no tool schemas — nothing for a
    reference to unhide — or when this isn't a typed `search_tools` return; the caller then falls
    through to the default text-formatting path. Fires for any active tool-search strategy (default
    native, named native, or custom callable), so cross-provider history (e.g. a prior local turn on
    Google) gets re-shaped into Anthropic's "client-side" tool_search wire when the current turn runs
    on Anthropic. Both flavors live in one helper because the wire shape is the same: `tool_use` +
    `tool_result` with `tool_reference` content blocks.

    `declared_tool_names` filters the references against the tools currently in
    `function_tools` on the wire — Anthropic rejects `tool_reference` entries for
    tools not in the request's `tools` list (e.g. an MCP server that failed to
    register this turn).
    """
    if not deferred_tools_active:
        return None, None
    if not isinstance(request_part, ToolSearchReturnPart):
        return None, None
    refs = [
        BetaToolReferenceBlockParam(tool_name=match['name'], type='tool_reference')
        for match in request_part.discovered_tools
        if match['name'] in declared_tool_names
    ]
    if refs or not request_part.discovered_tools:
        return refs, request_part.message
    # The search matched, but none of its matches is declared on this request: the tool was folded
    # into a code-execution tool, or its toolset failed to produce it this turn. The toolset's
    # no-matches text would tell the model the search found nothing, so name what it found instead.
    names = ', '.join(match['name'] for match in request_part.discovered_tools)
    return refs, f'Discovered tools: {names}. None is declared as a top-level tool in this request.'


def build_code_execution_tool_result_block(
    part: NativeToolReturnPart, tool_use_id: str
) -> (
    BetaCodeExecutionToolResultBlockParam
    | BetaBashCodeExecutionToolResultBlockParam
    | BetaTextEditorCodeExecutionToolResultBlockParam
):
    match get_anthropic_code_execution_tool_name(part):
        case 'code_execution':
            return BetaCodeExecutionToolResultBlockParam(
                tool_use_id=tool_use_id,
                type='code_execution_tool_result',
                content=cast(BetaCodeExecutionToolResultBlockParamContentParam, part.content),
            )
        case 'bash_code_execution':
            return BetaBashCodeExecutionToolResultBlockParam(
                tool_use_id=tool_use_id,
                type='bash_code_execution_tool_result',
                content=cast(BashCodeExecutionToolResultBlockParamContent, part.content),
            )
        case 'text_editor_code_execution':  # pragma: no branch
            return BetaTextEditorCodeExecutionToolResultBlockParam(
                tool_use_id=tool_use_id,
                type='text_editor_code_execution_tool_result',
                content=cast(TextEditorCodeExecutionToolResultBlockParamContent, part.content),
            )


def build_tool_search_replay_block(
    response_part: NativeToolSearchReturnPart, tool_use_id: str, declared_tool_names: set[str]
) -> BetaToolSearchToolResultBlockParam:
    """Reconstruct an Anthropic tool-search result block for history replay.

    Reads the cross-provider
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent] off
    `content` and any error fields the parse-time mapper stashed on `provider_details`.

    `declared_tool_names` filters references against the tools currently in
    `function_tools` on the wire — Anthropic rejects `tool_reference` entries for
    tools not in the request's `tools` list (e.g. an MCP server that failed to
    register this turn).
    """
    err = response_part.provider_details or {}
    inner: BetaToolSearchToolResultErrorParam | BetaToolSearchToolSearchResultBlockParam
    if err.get('error_code') is not None:
        # `BetaToolSearchToolResultErrorParam` only carries `error_code` (no
        # `error_message`); the parse-time mapper stashes the message for
        # observability but it doesn't make it back onto the wire.
        inner = BetaToolSearchToolResultErrorParam(
            type='tool_search_tool_result_error',
            error_code=err['error_code'],
        )
    else:
        tool_refs = [
            BetaToolReferenceBlockParam(tool_name=match['name'], type='tool_reference')
            for match in response_part.discovered_tools
            if match['name'] in declared_tool_names
        ]
        inner = BetaToolSearchToolSearchResultBlockParam(
            type='tool_search_tool_search_result',
            tool_references=tool_refs,
        )
    return BetaToolSearchToolResultBlockParam(
        tool_use_id=tool_use_id,
        type='tool_search_tool_result',
        content=inner,
    )


_BUILTIN_TOOL_KIND_BY_SERVER_TOOL_USE_NAME: dict[str, str] = {
    'web_search': WebSearchTool.kind,
    'code_execution': CodeExecutionTool.kind,
    'web_fetch': WebFetchTool.kind,
}


def anthropic_code_execution_tool_provider_details(
    tool_name: _AnthropicCodeExecutionProviderDetailToolName,
) -> dict[str, _AnthropicCodeExecutionProviderDetailToolName]:
    return {_ANTHROPIC_CODE_EXECUTION_TOOL_NAME_DETAIL: tool_name}


def anthropic_caller_provider_details(
    caller: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120 | None,
) -> dict[str, Any]:
    # A `direct` caller is the implicit default (the server tool was invoked directly, not from within a
    # code execution tool), so it carries no state worth round-tripping; only non-direct callers (e.g.
    # `code_execution_*`) need to be preserved and re-emitted on replay.
    if caller is None or caller.type == 'direct':
        return {}
    return {_ANTHROPIC_SERVER_TOOL_CALLER_DETAIL: caller.model_dump(mode='json')}


def add_anthropic_caller_param(
    block: BetaServerToolUseBlockParam | BetaWebSearchToolResultBlockParam | BetaWebFetchToolResultBlockParam,
    part: NativeToolCallPart | NativeToolReturnPart,
) -> None:
    if part.provider_details is None:
        return

    caller = part.provider_details.get(_ANTHROPIC_SERVER_TOOL_CALLER_DETAIL)
    if not _utils.is_str_dict(caller):
        return

    # Re-emit whatever caller `anthropic_caller_provider_details` stored. That producer only stores
    # genuine SDK callers, so replaying the stored value as-is keeps the read/write paths symmetric and
    # forward-compatible with new caller types, rather than dropping any not in a hardcoded allow-list.
    block['caller'] = caller  # pyright: ignore[reportGeneralTypeIssues]


def get_anthropic_code_execution_tool_name(
    part: NativeToolCallPart | NativeToolReturnPart,
) -> _AnthropicCodeExecutionToolName:
    if part.provider_details:
        tool_name = part.provider_details.get(_ANTHROPIC_CODE_EXECUTION_TOOL_NAME_DETAIL)
        if isinstance(tool_name, str) and tool_name in _ANTHROPIC_CODE_EXECUTION_TOOL_NAMES:
            return tool_name

    if part.tool_name in ('bash_code_execution', 'text_editor_code_execution'):
        return cast(_AnthropicCodeExecutionToolName, part.tool_name)

    if isinstance(part, NativeToolReturnPart) and _utils.is_str_dict(part.content):
        content_type = part.content.get('type')
        if isinstance(content_type, str):
            if content_type.startswith('bash_code_execution'):
                return 'bash_code_execution'
            elif content_type.startswith('text_editor_code_execution'):
                return 'text_editor_code_execution'

    return 'code_execution'


def map_code_execution_tool(version: AnthropicCodeExecutionToolVersion) -> BetaToolUnionParam:
    match version:
        case '20250825':
            return BetaCodeExecutionTool20250825Param(name='code_execution', type='code_execution_20250825')
        case '20260120':
            return BetaCodeExecutionTool20260120Param(name='code_execution', type='code_execution_20260120')
        case _:
            assert_never(version)


def map_advisor_tool(tool: AdvisorTool) -> BetaAdvisorTool20260301Param:
    param = BetaAdvisorTool20260301Param(type='advisor_20260301', name='advisor', model=tool.model)
    if tool.max_uses is not None:
        param['max_uses'] = tool.max_uses
    if tool.max_tokens is not None:
        param['max_tokens'] = tool.max_tokens
    if tool.caching is not None:
        param['caching'] = BetaCacheControlEphemeralParam(type='ephemeral', ttl=tool.caching)
    return param


def map_server_tool_use_block(item: BetaServerToolUseBlock, provider_name: str) -> NativeToolCallPart:
    tool_args = cast(dict[str, Any], item.input) or None
    if item.name in ('web_search', 'code_execution', 'web_fetch'):
        kind = _BUILTIN_TOOL_KIND_BY_SERVER_TOOL_USE_NAME[item.name]
        part = NativeToolCallPart(
            provider_name=provider_name,
            tool_name=kind,
            args=tool_args,
            tool_call_id=item.id,
            provider_details=anthropic_caller_provider_details(item.caller) or None,
        )
        if item.name == 'code_execution':
            part.otel_metadata = {'code_arg_name': 'code', 'code_arg_language': 'python'}
        return part
    if item.name in ('tool_search_tool_regex', 'tool_search_tool_bm25'):
        # Normalize the wire payload into the cross-provider `{"queries": [...]}` shape
        # carried on the typed call part. bm25 emits `{"query": "..."}`, regex emits
        # `{"pattern": "..."}`. The variant goes on `provider_details` so same-provider
        # replay can pick the original tool name back out.
        normalized_args = normalize_tool_search_args(tool_args, item.name)
        provider_details: dict[str, Any] = {
            'strategy': 'regex' if item.name == 'tool_search_tool_regex' else 'bm25',
            **anthropic_caller_provider_details(item.caller),
        }
        return NativeToolSearchCallPart(
            provider_name=provider_name,
            args=normalized_args,
            tool_call_id=item.id,
            provider_details=provider_details,
        )
    if item.name == 'bash_code_execution':
        return NativeToolCallPart(
            provider_name=provider_name,
            tool_name=CodeExecutionTool.kind,
            args=tool_args,
            tool_call_id=item.id,
            provider_details={
                **anthropic_code_execution_tool_provider_details('bash_code_execution'),
                **anthropic_caller_provider_details(item.caller),
            },
        )
    if item.name == 'text_editor_code_execution':
        return NativeToolCallPart(
            provider_name=provider_name,
            tool_name=CodeExecutionTool.kind,
            args=tool_args,
            tool_call_id=item.id,
            provider_details={
                **anthropic_code_execution_tool_provider_details('text_editor_code_execution'),
                **anthropic_caller_provider_details(item.caller),
            },
        )
    if item.name == 'advisor':
        # The advisor `server_tool_use` block always carries an empty `input` ({}), so `tool_args`
        # is None and `args` stays None on the round-tripped call part.
        return NativeToolCallPart(
            provider_name=provider_name,
            tool_name=AdvisorTool.kind,
            args=tool_args,
            tool_call_id=item.id,
            provider_details=anthropic_caller_provider_details(item.caller) or None,
        )
    assert_never(item.name)


def normalize_tool_search_args(tool_args: dict[str, Any] | None, tool_name: str) -> ToolSearchArgs:
    """Normalize an Anthropic `tool_search_tool_*.input` payload into `ToolSearchArgs`.

    Wire keys differ by variant: bm25 emits `{"query": "..."}`, regex emits
    `{"pattern": "..."}`. Both map to the cross-provider canonical
    `{"queries": [...]}` shape. Used by both the non-streaming `_process_response` path
    (which has the full input at once) and the streaming finalizer (which has only
    accumulated JSON-string deltas to reparse at content_block_stop time).
    """
    wire_key = 'pattern' if tool_name == 'tool_search_tool_regex' else 'query'
    raw = (tool_args or {}).get(wire_key, '')
    queries = [raw] if isinstance(raw, str) else []
    return {'queries': queries}


def finalize_streamed_tool_search_call_part(part: NativeToolSearchCallPart) -> NativeToolSearchCallPart:
    """Finalize a streamed tool-search call's args.

    Converts a `NativeToolSearchCallPart` whose `args` accumulated as a JSON string
    (via `BetaInputJSONDelta`) into the canonical dict shape produced by the
    non-streaming path. Already-canonical dict args (typed `ToolSearchArgs`) pass
    through unchanged; `None` finalizes to an empty `queries` list.
    """
    if isinstance(part.args, dict):
        return part
    if isinstance(part.args, str):
        try:
            parsed: dict[str, Any] | None = cast(dict[str, Any], pydantic_core.from_json(part.args))
        except ValueError:  # pragma: no cover
            # Malformed partial args.
            parsed = None
    else:
        parsed = None
    # `map_server_tool_use_block` stashes the variant on `provider_details['strategy']`
    # at content_block_start; map it back to the wire-shape tool name so the variant's
    # input key (`pattern` vs `query`) is honored.
    strategy = (part.provider_details or {}).get('strategy')
    tool_name = 'tool_search_tool_regex' if strategy == 'regex' else 'tool_search_tool_bm25'
    return replace(part, args=normalize_tool_search_args(parsed, tool_name))


def map_tool_search_tool_result_block(
    item: BetaToolSearchToolResultBlock, provider_name: str
) -> NativeToolSearchReturnPart:
    """Map a tool-search result block into a typed [`NativeToolSearchReturnPart`][pydantic_ai.messages.NativeToolSearchReturnPart].

    Writes a cross-provider [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent] to `content` (no
    provider-shape smuggling) and stashes the Anthropic-specific error fields on
    `provider_details` so we can faithfully reconstruct the original block on replay.
    """
    block = item.content
    provider_details: dict[str, Any] | None = None
    matches: list[ToolSearchMatch] = []
    if block.type == 'tool_search_tool_search_result':
        matches = [{'name': ref.tool_name} for ref in block.tool_references]
    else:  # tool_search_tool_result_error
        provider_details = {'error_code': block.error_code, 'error_message': block.error_message}
    return NativeToolSearchReturnPart(
        provider_name=provider_name,
        content={'discovered_tools': matches},
        tool_call_id=item.tool_use_id,
        provider_details=provider_details,
    )


web_search_tool_result_content_ta: TypeAdapter[BetaWebSearchToolResultBlockContent] = TypeAdapter(
    BetaWebSearchToolResultBlockContent
)


def map_web_search_tool_result_block(item: BetaWebSearchToolResultBlock, provider_name: str) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=WebSearchTool.kind,
        content=web_search_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
        provider_details=anthropic_caller_provider_details(item.caller) or None,
    )


code_execution_tool_result_content_ta: TypeAdapter[BetaCodeExecutionToolResultBlockContent] = TypeAdapter(
    BetaCodeExecutionToolResultBlockContent
)


def map_code_execution_tool_result_block(
    item: BetaCodeExecutionToolResultBlock, provider_name: str
) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        content=code_execution_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
    )


bash_code_execution_tool_result_content_ta: TypeAdapter[BashCodeExecutionToolResultBlockContent] = TypeAdapter(
    BashCodeExecutionToolResultBlockContent
)


def map_bash_code_execution_tool_result_block(
    item: BetaBashCodeExecutionToolResultBlock, provider_name: str
) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        content=bash_code_execution_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
        provider_details=anthropic_code_execution_tool_provider_details('bash_code_execution'),
    )


text_editor_code_execution_tool_result_content_ta: TypeAdapter[TextEditorCodeExecutionToolResultBlockContent] = (
    TypeAdapter(TextEditorCodeExecutionToolResultBlockContent)
)


def map_text_editor_code_execution_tool_result_block(
    item: BetaTextEditorCodeExecutionToolResultBlock, provider_name: str
) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        content=text_editor_code_execution_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
        provider_details=anthropic_code_execution_tool_provider_details('text_editor_code_execution'),
    )


def map_web_fetch_tool_result_block(item: BetaWebFetchToolResultBlock, provider_name: str) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=WebFetchTool.kind,
        # Store just the content field (BetaWebFetchBlock) which has {content, type, url, retrieved_at}
        content=item.content.model_dump(mode='json'),
        tool_call_id=item.tool_use_id,
        provider_details=anthropic_caller_provider_details(item.caller) or None,
    )


advisor_tool_result_content_ta: TypeAdapter[AdvisorToolResultBlockContent] = TypeAdapter(AdvisorToolResultBlockContent)


def map_advisor_tool_result_block(item: BetaAdvisorToolResultBlock, provider_name: str) -> NativeToolReturnPart:
    # Store the discriminated content dict verbatim (plaintext, redacted, or error). Round-tripping it
    # unchanged is what lets the server decrypt a redacted advisor result on the next turn — the client
    # can't read it, and no variant needs special-casing.
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=AdvisorTool.kind,
        content=advisor_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
    )


def map_mcp_server_use_block(item: BetaMCPToolUseBlock, provider_name: str) -> NativeToolCallPart:
    return NativeToolCallPart(
        provider_name=provider_name,
        tool_name=':'.join([MCPServerTool.kind, item.server_name]),
        args={
            'action': 'call_tool',
            'tool_name': item.name,
            'tool_args': cast(dict[str, Any], item.input),
        },
        tool_call_id=item.id,
    )


def map_mcp_server_result_block(
    item: BetaMCPToolResultBlock, call_part: NativeToolCallPart | None, provider_name: str
) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=call_part.tool_name if call_part else MCPServerTool.kind,
        content=item.model_dump(mode='json', include={'content', 'is_error'}),
        tool_call_id=item.tool_use_id,
    )
