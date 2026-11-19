"""DeepSeek Flash's extensions to the Responses web tool protocol."""

from __future__ import annotations

from typing import Literal, cast

from openai.types import responses
from pydantic import BaseModel, ConfigDict, JsonValue

from ...messages import NativeToolCallPart, NativeToolReturnPart
from ...native_tools import WebFetchTool, WebSearchTool

_DETAILS_KEY = 'deepseek_responses_web_tool'


class _WebToolItem(BaseModel):
    model_config = ConfigDict(extra='allow')

    type: Literal['web_search_call']
    id: str
    kind: Literal['search', 'deep_search', 'image_search', 'deep_image_search', 'read_link']
    status: Literal['in_progress', 'searching', 'completed', 'failed']
    action: dict[str, JsonValue]
    page: dict[str, JsonValue] | None = None
    signature: str | None = None
    error: dict[str, JsonValue] | None = None


def web_tool_param(tool: WebSearchTool | WebFetchTool) -> responses.ToolParam:
    tool_type = 'deepseek_read_url' if isinstance(tool, WebFetchTool) else 'deepseek_web_search'
    # The OpenAI SDK serializes the list but doesn't declare DeepSeek's tool types.
    return cast(responses.ToolParam, {'type': tool_type})


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
            provider_details={_DETAILS_KEY: parsed.model_dump(mode='json', exclude_unset=True)},
        ),
    )


def web_tool_replay(part: NativeToolReturnPart) -> responses.ResponseInputItemParam | None:
    if (wire := (part.provider_details or {}).get(_DETAILS_KEY)) is None:
        return None
    parsed = _WebToolItem.model_validate(wire)
    # The OpenAI SDK's item union lacks DeepSeek's signed fields.
    return cast(responses.ResponseInputItemParam, parsed.model_dump(mode='json', exclude_unset=True))
