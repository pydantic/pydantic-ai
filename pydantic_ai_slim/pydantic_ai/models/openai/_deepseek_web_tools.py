"""DeepSeek Flash web tool declarations, response parts, and signed history for the Responses API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

from openai.types import responses
from pydantic import BaseModel, JsonValue

from ..._utils import is_str_dict
from ...messages import NativeToolCallPart, NativeToolReturnPart
from ...native_tools import WebFetchTool, WebSearchTool

_WEB_TOOL_KEY = 'deepseek_web_tool'


def web_tool_usage(usage: BaseModel | None) -> dict[str, int]:
    counters = (usage.model_extra or {}).get('server_tool_use') if usage else None
    if not is_str_dict(counters):
        return {}
    return {
        'web_search_requests' if key == 'tool_search' else key: count
        for key, count in counters.items()
        if isinstance(count, int) and count > 0
    }


class _WebToolItem(BaseModel):
    type: Literal['web_search_call']
    id: str
    kind: Literal['search', 'deep_search', 'image_search', 'deep_image_search', 'read_link']
    status: Literal['in_progress', 'searching', 'completed', 'failed']
    action: dict[str, JsonValue]
    page: dict[str, JsonValue] | None = None
    signature: str | None = None
    error: dict[str, JsonValue] | None = None


def web_tool_param(tool: WebSearchTool | WebFetchTool) -> responses.ToolParam:
    # The OpenAI SDK serializes the list but doesn't declare DeepSeek's tool types.
    return cast(
        responses.ToolParam, {'type': 'deepseek_read_url' if isinstance(tool, WebFetchTool) else 'deepseek_web_search'}
    )


def web_tool_parts(
    item: responses.ResponseFunctionWebSearch, provider_name: str
) -> tuple[NativeToolCallPart, NativeToolReturnPart] | None:
    wire = item.model_dump(mode='json', exclude_unset=True, warnings=False)
    if 'kind' not in wire:
        return None
    parsed = _WebToolItem.model_validate(wire)
    name = WebFetchTool.kind if parsed.kind == 'read_link' else WebSearchTool.kind
    args = {key: value for key, value in parsed.action.items() if key != 'sources'}
    content = {
        'status': parsed.status,
        'sources': parsed.action.get('sources'),
        'page': parsed.page,
        'error': parsed.error,
    }
    return (
        NativeToolCallPart(
            tool_name=name, args=args, tool_call_id=parsed.id, id=parsed.id, provider_name=provider_name
        ),
        NativeToolReturnPart(
            tool_name=name,
            content={key: value for key, value in content.items() if value is not None},
            tool_call_id=parsed.id,
            outcome='failed' if parsed.status == 'failed' else 'success',
            provider_name=provider_name,
            provider_details={_WEB_TOOL_KEY: parsed.model_dump(include={'kind', 'signature'}, exclude_unset=True)},
        ),
    )


def replayable_web_results(parts: Iterable[NativeToolReturnPart]) -> dict[tuple[str, str], NativeToolReturnPart]:
    # A streamed return can complete out of order. Match it to the call when rebuilding the request.
    return {
        (part.tool_name, part.tool_call_id): part
        for part in parts
        if _WEB_TOOL_KEY in (part.provider_details or {}) and isinstance(part.content, dict)
    }


def web_tool_replay(call: NativeToolCallPart, result: NativeToolReturnPart) -> responses.ResponseInputItemParam:
    metadata = (result.provider_details or {})[_WEB_TOOL_KEY]
    assert is_str_dict(result.content)
    content = result.content
    action = call.args_as_dict()
    if 'sources' in content:
        action = {**action, 'sources': content['sources']}
    # History may be supplied by a client. Only restore fields owned by the web tool.
    item = _WebToolItem(
        type='web_search_call',
        id=call.tool_call_id,
        action=action,
        status=content['status'],
        page=content.get('page'),
        error=content.get('error'),
        kind=metadata['kind'],
        signature=metadata.get('signature'),
    )
    return cast(responses.ResponseInputItemParam, item.model_dump(mode='json', exclude_none=True))
