"""DeepSeek Flash's extensions to the Responses web tool protocol."""

from __future__ import annotations

from typing import Literal, cast

from openai.types import chat, responses
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from ...messages import NativeToolCallPart, NativeToolReturnPart
from ...native_tools import WebFetchTool, WebSearchTool
from .._deepseek_web_tools import DeepSeekWebToolKind, DeepSeekWebToolName, native_web_tool_name, web_tool_name
from ._errors import response_error

_DETAILS_KEY = 'deepseek_responses_web_tool'
CHAT_STEPS_KEY = 'deepseek_server_tool_calls'
CHAT_REASONING_KEY = 'deepseek_final_reasoning'


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
            provider_details={_DETAILS_KEY: parsed.model_dump(mode='json', exclude_unset=True)},
        ),
    )


def web_tool_replay(part: NativeToolReturnPart) -> responses.ResponseInputItemParam | None:
    if (wire := (part.provider_details or {}).get(_DETAILS_KEY)) is None:
        return None
    parsed = _WebToolItem.model_validate(wire)
    # The OpenAI SDK's item union lacks DeepSeek's signed fields.
    return cast(responses.ResponseInputItemParam, parsed.model_dump(mode='json', exclude_unset=True))


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
