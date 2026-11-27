"""DeepSeek Flash web tool declarations, response parts, and signed history for Chat and Responses."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from openai.types import chat, responses
from pydantic import BaseModel, JsonValue, TypeAdapter

from ..._utils import is_str_dict
from ...messages import ModelResponsePart, NativeToolCallPart, NativeToolReturnPart, ThinkingPart
from ...native_tools import WebFetchTool, WebSearchTool
from ._errors import response_error

_RESPONSES_WEB_TOOL_KEY = 'deepseek_responses_web_tool'
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
                _RESPONSES_WEB_TOOL_KEY: parsed.model_dump(include={'kind', 'signature'}, exclude_unset=True)
            },
        ),
    )


def replayable_web_results(parts: Iterable[NativeToolReturnPart]) -> dict[tuple[str, str], NativeToolReturnPart]:
    # A streamed return can complete out of order. Match it to the call when rebuilding the request.
    return {
        (part.tool_name, part.tool_call_id): part
        for part in parts
        if _RESPONSES_WEB_TOOL_KEY in (part.provider_details or {}) and isinstance(part.content, dict)
    }


def web_tool_replay(call: NativeToolCallPart, result: NativeToolReturnPart) -> responses.ResponseInputItemParam:
    metadata = (result.provider_details or {})[_RESPONSES_WEB_TOOL_KEY]
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


class ChatWebCall(BaseModel):
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

    def result_content(self) -> dict[str, JsonValue]:
        return self.model_dump(include={'status', 'sources', 'page', 'error'}, exclude_none=True)

    def return_part(self, provider: str) -> NativeToolReturnPart:
        return NativeToolReturnPart(
            tool_name=self.tool_name,
            tool_call_id=self.id,
            content=self.result_content(),
            outcome='failed' if self.status == 'failed' else 'success',
            provider_name=provider,
        )


class ChatWebStep(BaseModel):
    index: int
    reasoning_content: str | None = None
    calls: list[ChatWebCall]

    def parts(self, provider: str) -> list[ModelResponsePart]:
        items: list[ModelResponsePart] = []
        if self.reasoning_content:
            items.append(
                ThinkingPart(
                    id='reasoning_content',
                    content=self.reasoning_content,
                    provider_name=provider,
                    provider_details={CHAT_STEP_KEY: self.index},
                )
            )
        items.extend(call.call_part(provider) for call in self.calls)
        items.extend(call.return_part(provider) for call in self.calls)
        return items


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


def chat_web_parts(
    message: chat.ChatCompletionMessage, provider: str, details: dict[str, Any]
) -> list[ModelResponsePart]:
    parts: list[ModelResponsePart] = []
    if raw_steps := (message.model_extra or {}).get('server_tool_calls'):
        steps = chat_web_steps.validate_python(raw_steps)
        details[CHAT_STEPS_KEY] = chat_web_step_metadata(steps)
        for step in steps:
            parts.extend(step.parts(provider))
    return parts


def replay_chat_web_steps(
    provider_details: dict[str, Any] | None,
    param: chat.ChatCompletionAssistantMessageParam | None,
    parts: Iterable[ModelResponsePart],
    *,
    include_thinking: bool,
) -> chat.ChatCompletionAssistantMessageParam | None:
    if not (wire_steps := (provider_details or {}).get(CHAT_STEPS_KEY)):
        return param
    calls, results, thinking_by_step = _retained_chat_parts(parts, include_thinking)
    steps: list[ChatWebStep] = []
    for step in chat_web_steps.validate_python(wire_steps):
        # A history processor may edit a part while retaining its ID. Only replay intact tool calls.
        step.calls = [
            call
            for call in step.calls
            if (calls.get((call.id, call.tool_name)), results.get((call.id, call.tool_name)))
            == (call.action, call.result_content())
        ]
        if not step.calls:
            continue
        if step.reasoning_content is not None:
            step.reasoning_content = ''.join(thinking_by_step.get(step.index, ()))
        steps.append(step)
    if param is None:
        if not steps:
            return None
        param = chat.ChatCompletionAssistantMessageParam(role='assistant', content=None)
    # The SDK message type lacks these DeepSeek extensions.
    wire = cast(dict[str, object], param)
    if steps:
        wire['server_tool_calls'] = chat_web_steps.dump_python(steps, mode='json', exclude_unset=True)
    wire.pop('reasoning_content', None)
    if final_thinking := thinking_by_step.get(None):
        wire['reasoning_content'] = ''.join(final_thinking)
    return param


def _retained_chat_parts(
    parts: Iterable[ModelResponsePart], include_thinking: bool
) -> tuple[dict[tuple[str, str], object], dict[tuple[str, str], object], dict[int | None, list[str]]]:
    calls: dict[tuple[str, str], object] = {}
    returns: dict[tuple[str, str], object] = {}
    thinking_by_step: dict[int | None, list[str]] = {}
    for part in parts:
        if isinstance(part, NativeToolCallPart):
            calls[(part.tool_call_id, part.tool_name)] = part.args_as_dict()
        elif isinstance(part, NativeToolReturnPart):
            returns[(part.tool_call_id, part.tool_name)] = part.content
        elif isinstance(part, ThinkingPart) and include_thinking:
            index = (part.provider_details or {}).get(CHAT_STEP_KEY)
            thinking_by_step.setdefault(index, []).append(part.content)
    return calls, returns, thinking_by_step
