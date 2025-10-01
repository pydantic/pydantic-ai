"""Shared OpenAI compatibility helpers (in-progress).

This module is a working scaffold. Implementations will be ported in small,
covered steps from `_openai_compat_ref.py` to preserve coverage.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, overload

from pydantic import ValidationError
from typing_extensions import assert_never

from pydantic_ai.messages import (
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)

from .. import UnexpectedModelBehavior, _utils, usage
from .._output import OutputObjectDefinition
from .._thinking_part import split_content_into_text_and_thinking
from .._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc, number_to_datetime
from ..exceptions import UserError
from ..profiles import ModelProfile
from ..profiles.openai import OpenAIModelProfile
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import ModelRequestParameters, StreamedResponse, get_user_agent

try:
    from openai import NOT_GIVEN, APIStatusError, AsyncStream
    from openai.types import chat
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionMessageCustomToolCall,
        ChatCompletionMessageFunctionToolCall,
    )
    from openai.types.chat.completion_create_params import ResponseFormat, WebSearchOptions
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'OpenAICompatStreamedResponse',
    'completions_create',
    'map_messages',
    'map_tool_definition',
    'map_usage',
    'process_response',
    'process_streamed_response',
)


def _map_tool_call(t: ToolCallPart) -> Any:
    """Map a ToolCallPart to OpenAI ChatCompletionMessageFunctionToolCallParam."""
    return {
        'id': _guard_tool_call_id(t=t),
        'type': 'function',
        'function': {'name': t.tool_name, 'arguments': t.args_as_json_str()},
    }


def map_tool_definition(model_profile: ModelProfile, f: ToolDefinition) -> Any:
    """Map a ToolDefinition to OpenAI ChatCompletionToolParam."""
    tool_param: dict[str, Any] = {
        'type': 'function',
        'function': {
            'name': f.name,
            'description': f.description or '',
            'parameters': f.parameters_json_schema,
        },
    }
    if f.strict and OpenAIModelProfile.from_profile(model_profile).openai_supports_strict_tool_definition:
        tool_param['function']['strict'] = f.strict
    return tool_param


async def map_messages(model: Any, messages: list[ModelMessage]) -> list[Any]:
    """Async mapping of internal ModelMessage list to OpenAI chat messages."""
    openai_messages: list[Any] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            async for item in model._map_user_message(message):
                openai_messages.append(item)
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[Any] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(_map_tool_call(item))
            message_param: dict[str, Any] = {'role': 'assistant'}
            if texts:
                message_param['content'] = '\n\n'.join(texts)
            else:
                message_param['content'] = None
            if tool_calls:
                message_param['tool_calls'] = tool_calls
            openai_messages.append(message_param)
        else:
            assert_never(message)

    return openai_messages


def get_tools(model_profile: ModelProfile, tool_defs: dict[str, ToolDefinition]) -> list[Any]:
    """Get OpenAI tools from tool definitions."""
    return [map_tool_definition(model_profile, r) for r in tool_defs.values()]


def _map_json_schema(model_profile: ModelProfile, o: OutputObjectDefinition) -> ResponseFormat:
    """Map an OutputObjectDefinition to OpenAI ResponseFormatJSONSchema."""
    response_format_param: ResponseFormat = {
        'type': 'json_schema',
        'json_schema': {'name': o.name or 'output', 'schema': o.json_schema},
    }
    if o.description:
        response_format_param['json_schema']['description'] = o.description
    profile = OpenAIModelProfile.from_profile(model_profile)
    if profile.openai_supports_strict_tool_definition:  # pragma: no branch
        response_format_param['json_schema']['strict'] = bool(o.strict)
    return response_format_param


def _get_web_search_options(model_profile: ModelProfile, builtin_tools: list[Any]) -> WebSearchOptions | None:
    """Extract WebSearchOptions from builtin_tools if WebSearchTool is present."""
    for tool in builtin_tools:
        if tool.__class__.__name__ == 'WebSearchTool':
            if not OpenAIModelProfile.from_profile(model_profile).openai_chat_supports_web_search:
                raise UserError(
                    f'WebSearchTool is not supported with `OpenAIChatModel` and model {getattr(model_profile, "model_name", None) or "<unknown>"!r}. '
                    f'Please use `OpenAIResponsesModel` instead.'
                )
            if tool.user_location:
                from openai.types.chat.completion_create_params import (
                    WebSearchOptionsUserLocation,
                    WebSearchOptionsUserLocationApproximate,
                )

                return WebSearchOptions(
                    search_context_size=tool.search_context_size,
                    user_location=WebSearchOptionsUserLocation(
                        type='approximate',
                        approximate=WebSearchOptionsUserLocationApproximate(**tool.user_location),
                    ),
                )
            return WebSearchOptions(search_context_size=tool.search_context_size)
        else:
            raise UserError(
                f'`{tool.__class__.__name__}` is not supported by `OpenAIChatModel`. If it should be, please file an issue.'
            )
    return None


@overload
async def completions_create(
    model: Any,
    messages: list[ModelMessage],
    stream: Literal[True],
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> AsyncStream[ChatCompletionChunk]: ...


@overload
async def completions_create(
    model: Any,
    messages: list[ModelMessage],
    stream: Literal[False],
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> chat.ChatCompletion: ...


async def completions_create(
    model: Any,
    messages: list[ModelMessage],
    stream: bool,
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
    """Create a chat completion using OpenAI SDK with compat helpers.

    Handles tool mapping, response-format mapping, unsupported-setting pruning,
    and SDK invocation with error translation.
    """
    tools = get_tools(model.profile, model_request_parameters.tool_defs)
    web_search_options = _get_web_search_options(model.profile, model_request_parameters.builtin_tools)

    if not tools:
        tool_choice: Literal['none', 'required', 'auto'] | None = None
    elif (
        not model_request_parameters.allow_text_output
        and OpenAIModelProfile.from_profile(model.profile).openai_supports_tool_choice_required
    ):
        tool_choice = 'required'
    else:
        tool_choice = 'auto'

    openai_messages = await map_messages(model, messages)

    response_format: ResponseFormat | None = None
    if model_request_parameters.output_mode == 'native':
        output_object = model_request_parameters.output_object
        assert output_object is not None
        response_format = _map_json_schema(model.profile, output_object)
    elif (
        model_request_parameters.output_mode == 'prompted' and model.profile.supports_json_object_output
    ):  # pragma: no branch
        response_format = {'type': 'json_object'}

    unsupported_model_settings = OpenAIModelProfile.from_profile(model.profile).openai_unsupported_model_settings
    for setting in unsupported_model_settings:
        model_settings.pop(setting, None)

    try:
        extra_headers = model_settings.get('extra_headers', {})
        extra_headers.setdefault('User-Agent', get_user_agent())
        return await model.client.chat.completions.create(
            model=model._model_name,
            messages=openai_messages,
            parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
            tools=tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            stream_options={'include_usage': True} if stream else NOT_GIVEN,
            stop=model_settings.get('stop_sequences', NOT_GIVEN),
            max_completion_tokens=model_settings.get('max_tokens', NOT_GIVEN),
            timeout=model_settings.get('timeout', NOT_GIVEN),
            response_format=response_format or NOT_GIVEN,
            seed=model_settings.get('seed', NOT_GIVEN),
            reasoning_effort=model_settings.get('openai_reasoning_effort', NOT_GIVEN),
            user=model_settings.get('openai_user', NOT_GIVEN),
            web_search_options=web_search_options or NOT_GIVEN,
            service_tier=model_settings.get('openai_service_tier', NOT_GIVEN),
            prediction=model_settings.get('openai_prediction', NOT_GIVEN),
            temperature=model_settings.get('temperature', NOT_GIVEN),
            top_p=model_settings.get('top_p', NOT_GIVEN),
            presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
            frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
            logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
            logprobs=model_settings.get('openai_logprobs', NOT_GIVEN),
            top_logprobs=model_settings.get('openai_top_logprobs', NOT_GIVEN),
            extra_headers=extra_headers,
            extra_body=model_settings.get('extra_body'),
        )
    except APIStatusError as e:
        if (status_code := e.status_code) >= 400:
            from .. import ModelHTTPError

            raise ModelHTTPError(status_code=status_code, model_name=model.model_name, body=e.body) from e
        raise  # pragma: lax no cover


def process_response(
    model: Any,
    response: chat.ChatCompletion | str,
    *,
    map_usage_fn: Callable[[chat.ChatCompletion], usage.RequestUsage],
    finish_reason_map: Mapping[str, FinishReason],
) -> ModelResponse:
    """Process a non-streamed chat completion response into a ModelResponse."""
    if not isinstance(response, chat.ChatCompletion):
        raise UnexpectedModelBehavior('Invalid response from OpenAI chat completions endpoint, expected JSON data')

    if response.created:
        timestamp = number_to_datetime(response.created)
    else:
        timestamp = _now_utc()
        response.created = int(timestamp.timestamp())

    # Workaround for local Ollama which sometimes returns a `None` finish reason.
    if response.choices and (choice := response.choices[0]) and choice.finish_reason is None:  # pyright: ignore[reportUnnecessaryComparison]
        choice.finish_reason = 'stop'

    try:
        response = chat.ChatCompletion.model_validate(response.model_dump())
    except ValidationError as e:  # pragma: no cover
        raise UnexpectedModelBehavior(f'Invalid response from OpenAI chat completions endpoint: {e}') from e

    choice = response.choices[0]
    items: list[ModelResponsePart] = []

    # OpenRouter uses 'reasoning', OpenAI previously used 'reasoning_content' (removed Feb 2025)
    reasoning_content = getattr(choice.message, 'reasoning', None) or getattr(choice.message, 'reasoning_content', None)
    if reasoning_content:
        items.append(ThinkingPart(id='reasoning_content', content=reasoning_content, provider_name=model.system))

    vendor_details: dict[str, Any] = {}

    if choice.logprobs is not None and choice.logprobs.content:
        vendor_details['logprobs'] = [
            {
                'token': lp.token,
                'bytes': lp.bytes,
                'logprob': lp.logprob,
                'top_logprobs': [
                    {'token': tlp.token, 'bytes': tlp.bytes, 'logprob': tlp.logprob} for tlp in lp.top_logprobs
                ],
            }
            for lp in choice.logprobs.content
        ]

    if choice.message.content is not None:
        items.extend(
            (replace(part, id='content', provider_name=model.system) if isinstance(part, ThinkingPart) else part)
            for part in split_content_into_text_and_thinking(choice.message.content, model.profile.thinking_tags)
        )

    if choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                part = ToolCallPart(tool_call.function.name, tool_call.function.arguments, tool_call_id=tool_call.id)
            elif isinstance(tool_call, ChatCompletionMessageCustomToolCall):  # pragma: no cover
                raise RuntimeError('Custom tool calls are not supported')
            else:
                assert_never(tool_call)
            part.tool_call_id = _guard_tool_call_id(part)
            items.append(part)

    raw_finish_reason = choice.finish_reason
    vendor_details['finish_reason'] = raw_finish_reason
    finish_reason = finish_reason_map.get(raw_finish_reason)

    return ModelResponse(
        parts=items,
        usage=map_usage_fn(response),
        model_name=response.model,
        timestamp=timestamp,
        provider_details=vendor_details or None,
        provider_response_id=response.id,
        provider_name=model.system,
        finish_reason=finish_reason,
    )


async def process_streamed_response(
    model: Any,
    response: AsyncStream[ChatCompletionChunk],
    model_request_parameters: ModelRequestParameters,
    *,
    map_usage_fn: Callable[[ChatCompletionChunk], usage.RequestUsage],
    finish_reason_map: Mapping[str, FinishReason],
) -> OpenAICompatStreamedResponse:
    """Wrap a streamed chat completion response with compat handling."""
    peekable_response = _utils.PeekableAsyncStream(response)
    first_chunk = await peekable_response.peek()
    if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
        raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

    model_name = first_chunk.model or model.model_name

    return OpenAICompatStreamedResponse(
        model_request_parameters=model_request_parameters,
        _model_name=model_name,
        _model_profile=model.profile,
        _response=peekable_response,
        _timestamp=number_to_datetime(first_chunk.created),
        _provider_name=model.system,
        _map_usage_fn=map_usage_fn,
        _finish_reason_map=finish_reason_map,
    )


@dataclass
class OpenAICompatStreamedResponse(StreamedResponse):
    """Streaming response wrapper for OpenAI chat completions."""

    model_request_parameters: ModelRequestParameters
    _model_name: str
    _model_profile: ModelProfile
    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime
    _provider_name: str
    _map_usage_fn: Callable[[ChatCompletionChunk], usage.RequestUsage] = field(repr=False)
    _finish_reason_map: Mapping[str, FinishReason] = field(repr=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage += self._map_usage_fn(chunk)

            if chunk.id:  # pragma: no branch
                self.provider_response_id = chunk.id

            if chunk.model:
                self._model_name = chunk.model

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            if choice.delta is None:  # pyright: ignore[reportUnnecessaryComparison]
                continue

            if raw_finish_reason := choice.finish_reason:
                self.provider_details = {'finish_reason': raw_finish_reason}
                self.finish_reason = self._finish_reason_map.get(raw_finish_reason)

            content = choice.delta.content
            if content is not None:
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id='content',
                    content=content,
                    thinking_tags=self._model_profile.thinking_tags,
                    ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
                )
                if maybe_event is not None:
                    if isinstance(maybe_event, PartStartEvent) and isinstance(maybe_event.part, ThinkingPart):
                        maybe_event.part.id = 'content'
                        maybe_event.part.provider_name = self.provider_name
                    yield maybe_event

            # OpenRouter uses 'reasoning', OpenAI previously used 'reasoning_content' (removed Feb 2025)
            reasoning_content = getattr(choice.delta, 'reasoning', None) or getattr(
                choice.delta, 'reasoning_content', None
            )
            if reasoning_content:
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id='reasoning_content',
                    id='reasoning_content',
                    content=reasoning_content,
                    provider_name=self.provider_name,
                )

            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event is not None:
                    yield maybe_event

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        return self._timestamp


def map_usage(
    response: chat.ChatCompletion | ChatCompletionChunk,
) -> usage.RequestUsage:
    response_usage = response.usage
    if response_usage is None:
        return usage.RequestUsage()
    else:
        details = {
            key: value
            for key, value in response_usage.model_dump(
                exclude_none=True,
                exclude={'prompt_tokens', 'completion_tokens', 'total_tokens'},
            ).items()
            if isinstance(value, int)
        }
        result = usage.RequestUsage(
            input_tokens=response_usage.prompt_tokens,
            output_tokens=response_usage.completion_tokens,
            details=details,
        )
        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))
            result.output_audio_tokens = response_usage.completion_tokens_details.audio_tokens or 0
        if response_usage.prompt_tokens_details is not None:
            result.input_audio_tokens = response_usage.prompt_tokens_details.audio_tokens or 0
            result.cache_read_tokens = response_usage.prompt_tokens_details.cached_tokens or 0
        return result
