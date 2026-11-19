"""DeepSeek's hosted web tool extensions to the Anthropic Messages protocol."""

from __future__ import annotations

from typing import Literal, cast

from anthropic.types.beta import (
    BetaServerToolUseBlock,
    BetaServerToolUseBlockParam,
    BetaToolUnionParam,
    BetaWebFetchToolResultBlock,
    BetaWebFetchToolResultBlockParam,
    BetaWebSearchToolResultBlock,
    BetaWebSearchToolResultBlockParam,
)
from pydantic import BaseModel, ConfigDict, JsonValue

from ...messages import NativeToolCallPart, NativeToolReturnPart
from ...native_tools import WebFetchTool, WebSearchTool
from .._deepseek_web_tools import DeepSeekWebToolKind, DeepSeekWebToolName, native_web_tool_name, web_tool_name

_DETAILS_KEY = 'deepseek_messages_web_block'


class _WebCallBlock(BaseModel):
    model_config = ConfigDict(extra='allow')

    type: Literal['server_tool_use']
    id: str
    name: DeepSeekWebToolName
    kind: DeepSeekWebToolKind
    input: dict[str, JsonValue]


class _WebReturnBlock(BaseModel):
    model_config = ConfigDict(extra='allow')

    type: Literal['web_search_tool_result', 'web_fetch_tool_result']
    tool_use_id: str
    kind: DeepSeekWebToolKind
    content: dict[str, JsonValue] | list[dict[str, JsonValue]]
    signature: str | None = None


def web_tool_param(tool: WebSearchTool | WebFetchTool) -> BetaToolUnionParam:
    name = web_tool_name(tool)
    # These tools extend the Anthropic SDK's request union.
    return cast(BetaToolUnionParam, {'type': name, 'name': name})


def web_tool_part(block: object, provider: str) -> NativeToolCallPart | NativeToolReturnPart | None:
    if isinstance(block, BetaServerToolUseBlock):
        raw = block.model_dump(mode='json', exclude_unset=True, warnings=False)
        if raw.get('name') not in ('deepseek_web_search', 'deepseek_read_url'):
            return None
        call = _WebCallBlock.model_validate(raw)
        return NativeToolCallPart(
            tool_name=native_web_tool_name(call.kind),
            tool_call_id=call.id,
            args=call.input,
            provider_name=provider,
            provider_details={_DETAILS_KEY: call.model_dump(mode='json', exclude_unset=True)},
        )
    if isinstance(block, BetaWebSearchToolResultBlock | BetaWebFetchToolResultBlock):
        raw = block.model_dump(mode='json', exclude_unset=True, warnings=False)
        if 'kind' not in raw:
            return None
        result = _WebReturnBlock.model_validate(raw)
        contents = result.content if isinstance(result.content, list) else [result.content]
        return NativeToolReturnPart(
            tool_name=native_web_tool_name(result.kind),
            tool_call_id=result.tool_use_id,
            content=result.content,
            outcome='failed' if any('error_code' in item for item in contents) else 'success',
            provider_name=provider,
            provider_details={_DETAILS_KEY: result.model_dump(mode='json', exclude_unset=True)},
        )
    return None


def web_tool_replay(
    part: NativeToolCallPart | NativeToolReturnPart,
) -> BetaServerToolUseBlockParam | BetaWebFetchToolResultBlockParam | BetaWebSearchToolResultBlockParam | None:
    if (raw := (part.provider_details or {}).get(_DETAILS_KEY)) is None:
        return None
    block = (
        _WebCallBlock.model_validate(raw)
        if isinstance(part, NativeToolCallPart)
        else _WebReturnBlock.model_validate(raw)
    )
    # Preserve the DeepSeek-only kind and signed payload through the SDK's request union.
    return cast(
        BetaServerToolUseBlockParam | BetaWebFetchToolResultBlockParam | BetaWebSearchToolResultBlockParam,
        block.model_dump(mode='json', exclude_unset=True),
    )
