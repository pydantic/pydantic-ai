"""DeepSeek Flash's extensions to the Responses web tool protocol."""

from __future__ import annotations

from typing import Literal, cast

from openai.types import chat, responses
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from ..._utils import is_str_dict
from ...messages import NativeToolCallPart, NativeToolReturnPart
from ...native_tools import WebFetchTool, WebSearchTool
from ._errors import response_error

RESPONSES_WEB_TOOL_KEY = 'deepseek_responses_web_tool'
CHAT_STEPS_KEY = 'deepseek_server_tool_calls'
CHAT_STEP_KEY = 'deepseek_reasoning_step'


DeepSeekWebToolName = Literal['deepseek_web_search', 'deepseek_read_url']
DeepSeekWebToolKind = Literal['search', 'deep_search', 'image_search', 'deep_image_search', 'read_link']


def web_tool_name(tool: WebSearchTool | WebFetchTool) -> DeepSeekWebToolName:
    return 'deepseek_read_url' if isinstance(tool, WebFetchTool) else 'deepseek_web_search'


def native_web_tool_name(kind: DeepSeekWebToolKind) -> str:
    return WebFetchTool.kind if kind == 'read_link' else WebSearchTool.kind


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
    model_config = ConfigDict(extra='allow')

    type: Literal['web_search_call']
    id: str
    kind: DeepSeekWebToolKind
    status: Literal['in_progress', 'searching', 'completed', 'failed']
    action: dict[str, JsonValue]
    page: dict[str, JsonValue] | None = None
    signature: str | None = None
    error: dict[str, JsonValue] | None = None


def web_tool_param(tool: WebSearchTool | WebFetchTool) -> responses.ToolParam:
    # The OpenAI SDK serializes the list but doesn't declare DeepSeek's tool types.
    return cast(responses.ToolParam, {'type': web_tool_name(tool)})


def web_tool_parts(
    item: responses.ResponseFunctionWebSearch, provider_name: str
) -> tuple[NativeToolCallPart, NativeToolReturnPart] | None:
    wire = item.model_dump(mode='json', exclude_unset=True, warnings=False)
    if 'kind' not in wire:
        return None
    parsed = _WebToolItem.model_validate(wire)
    name = native_web_tool_name(parsed.kind)
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
            provider_details={
                RESPONSES_WEB_TOOL_KEY: parsed.model_dump(include={'kind', 'signature'}, exclude_unset=True)
            },
        ),
    )


def web_tool_replay(call: NativeToolCallPart, result: NativeToolReturnPart) -> responses.ResponseInputItemParam:
    metadata = (result.provider_details or {})[RESPONSES_WEB_TOOL_KEY]
    assert is_str_dict(result.content)
    content = result.content.copy()
    action = call.args_as_dict()
    if 'sources' in content:
        action = {**action, 'sources': content.pop('sources')}
    # DeepSeek extends the SDK's web_search_call with the signed page and tool kind.
    return cast(
        responses.ResponseInputItemParam,
        {'type': 'web_search_call', 'id': call.tool_call_id, 'action': action, **content, **metadata},
    )


class ChatWebCall(BaseModel):
    model_config = ConfigDict(extra='allow')

    id: str
    index: int
    type: DeepSeekWebToolName
    kind: DeepSeekWebToolKind
    action: dict[str, JsonValue]
    status: Literal['completed', 'failed'] | None = None
    sources: list[dict[str, JsonValue]] | None = None
    page: dict[str, JsonValue] | None = None
    signature: str | None = None
    error: dict[str, JsonValue] | None = None

    @property
    def tool_name(self) -> str:
        return native_web_tool_name(self.kind)

    def call_part(self, provider: str) -> NativeToolCallPart:
        return NativeToolCallPart(
            tool_name=self.tool_name, tool_call_id=self.id, args=self.action, provider_name=provider
        )

    def return_part(self, provider: str) -> NativeToolReturnPart:
        return NativeToolReturnPart(
            tool_name=self.tool_name,
            tool_call_id=self.id,
            content=self.model_dump(include={'status', 'sources', 'page', 'error'}, exclude_none=True),
            outcome='failed' if self.status == 'failed' else 'success',
            provider_name=provider,
        )


class ChatWebStep(BaseModel):
    model_config = ConfigDict(extra='allow')

    index: int
    reasoning_content: str | None = None
    calls: list[ChatWebCall]


class ChatWebEvent(BaseModel):
    type: Literal['step_reasoning_done', 'call_started', 'call_completed']
    step: int
    call: dict[str, JsonValue] | None = None


chat_web_steps = TypeAdapter(list[ChatWebStep])


def chat_web_step_metadata(steps: list[ChatWebStep]) -> list[object]:
    """Keep wire step boundaries and field presence, with reasoning stored in ThinkingParts."""
    stripped = [
        step.model_copy(update={'reasoning_content': ''}) if step.reasoning_content is not None else step
        for step in steps
    ]
    return cast(list[object], chat_web_steps.dump_python(stripped, mode='json', exclude_unset=True))


def chat_web_tool_param(tool: WebSearchTool | WebFetchTool) -> chat.ChatCompletionToolParam:
    # DeepSeek's declarations are extensions to the SDK's tool union.
    return cast(chat.ChatCompletionToolParam, {'type': web_tool_name(tool)})


def check_chat_web_error(raw: object, model: str, *, in_stream: bool) -> None:
    match raw:
        case {'message': str(message), 'type': str(error_type)}:
            raise response_error(model, None, message, error_type=error_type, in_stream=in_stream)
        case {'message': str(message)}:
            raise response_error(model, None, message, in_stream=in_stream)
        case _:
            return
