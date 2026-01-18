from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, overload

import httpx

from pydantic_ai.exceptions import ModelAPIError

from .. import ModelHTTPError, UnexpectedModelBehavior
from .._run_context import RunContext
from .._thinking_part import split_content_into_text_and_thinking
from .._utils import guard_tool_call_id, now_utc, number_to_datetime
from ..messages import (
    FinishReason,
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from ..profiles import ModelProfileSpec
from ..profiles.openai import OpenAIModelProfile
from ..providers import Provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import (
    ModelRequestParameters,
    OpenAIChatCompatibleProvider,
    StreamedResponse,
    check_allow_model_requests,
)
from .openai import OpenAIChatModel, OpenAIChatModelSettings

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AsyncStream
    from openai.types import chat
    from openai.types.chat import ChatCompletionChunk
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the Databricks model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


__all__ = ('DatabricksModel',)


@dataclass(init=False)
class DatabricksModel(OpenAIChatModel):
    """A model that uses the Databricks API.

    This class subclasses `OpenAIChatModel` to handle Databricks-specific behavior,
    specifically the `content` field in responses which can be a list of content items
    instead of a string.
    """

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenAIChatCompatibleProvider
        | Literal['databricks', 'gateway']
        | Provider[AsyncOpenAI] = 'databricks',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        super().__init__(
            model_name=model_name,
            provider=provider,
            profile=profile,
            settings=settings,
        )

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | ModelResponse: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk] | ModelResponse:
        # This method is primarily used by the base class `request` method.
        # For streaming, we override `request_stream` directly, so this usually won't be called with stream=True
        # unless `OpenAIChatModel` logic changes.

        payload = await self._prepare_request_payload(messages, stream, model_settings, model_request_parameters)
        extra_headers = model_settings.get('extra_headers', {}).copy()

        if stream:
            # Should not happen via our request_stream override, but for safety/completeness:
            # We can't return a raw generator here because OpenAIChatModel expects an AsyncStream context manager.
            # So we raise or return a dummy.
            # Let's just raise strict error or try to support it?
            # Supporting it requires wrapping in a context manager.
            raise NotImplementedError('Use `request_stream` for streaming with DatabricksModel')
        else:
            try:
                response = await self.client.post(
                    '/chat/completions',
                    body=payload,
                    options={'headers': extra_headers, 'timeout': model_settings.get('timeout')},
                    cast_to=object,  # return raw dict/object
                )
            except APIStatusError as e:
                if (status_code := e.status_code) >= 400:
                    raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
                raise
            except APIConnectionError as e:
                raise ModelAPIError(model_name=self.model_name, message=e.message) from e

            if not isinstance(response, dict):
                raise UnexpectedModelBehavior(
                    f'Invalid response from {self.system} chat completions endpoint, expected JSON data'
                )

            return self._process_databricks_response(cast('dict[str, Any]', response))

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )

        payload = await self._prepare_request_payload(
            messages, True, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )
        model_settings = model_settings or {}
        extra_headers = model_settings.get('extra_headers', {}).copy()
        timeout = model_settings.get('timeout')

        response_iterator = self._stream_completions(payload, extra_headers, timeout)

        # We pass the iterator directly. _process_streamed_response handles PeekableAsyncStream wrapping.
        yield await self._process_streamed_response(
            cast('AsyncStream[ChatCompletionChunk]', response_iterator), model_request_parameters
        )

    async def _stream_completions(
        self, payload: dict[str, Any], extra_headers: dict[str, Any], timeout: float | httpx.Timeout | None
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Manually stream completions to handle non-standard chunk content."""
        response = await self.client.post(
            '/chat/completions',
            body=payload,
            options={'headers': extra_headers, 'timeout': timeout},
            stream=True,
            cast_to=httpx.Response,
        )

        try:
            if response.status_code >= 400:
                response.read()
                raise ModelHTTPError(
                    status_code=response.status_code,
                    model_name=self.model_name,
                    body=response.content,
                )

            async for line in response.aiter_lines():
                if not line.startswith('data: '):
                    continue
                data_str = line[6:].strip()
                if data_str == '[DONE]':
                    break

                try:
                    data = json.loads(data_str)
                    chunk = self._parse_chunk(data)
                    yield chunk
                except json.JSONDecodeError:
                    continue
                except Exception:
                    raise
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e
        finally:
            await response.aclose()

    def _parse_chunk(self, data: dict[str, Any]) -> ChatCompletionChunk:
        """Parse a dictionary into a ChatCompletionChunk, flattening list content if necessary."""
        choices = cast('list[dict[str, Any]]', data.get('choices', []))
        for choice in choices:
            delta = choice.get('delta', {})
            content: list[str | dict] = delta.get('content')
            if isinstance(content, list):
                # Flatten list content to string
                text_content = ''
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_content += str(item.get('text', ''))
                    elif isinstance(item, str):
                        text_content += item
                delta['content'] = text_content

        # Now it should satisfy Pydantic validation
        return ChatCompletionChunk.model_validate(data)

    async def _prepare_request_payload(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> dict[str, Any]:
        tools = self._get_tools(model_request_parameters)
        web_search_options = self._get_web_search_options(model_request_parameters)

        if not tools:
            tool_choice = None
        elif (
            not model_request_parameters.allow_text_output
            and OpenAIModelProfile.from_profile(self.profile).openai_supports_tool_choice_required
        ):
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = await self._map_messages(messages, model_request_parameters)

        response_format = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output:
            response_format = {'type': 'json_object'}

        payload: dict[str, Any] = {
            'model': self.model_name,
            'messages': openai_messages,
            'stream': stream,
        }
        if stream:
            payload['stream_options'] = {'include_usage': True}

        if tools:
            payload['tools'] = tools
        if tool_choice:
            payload['tool_choice'] = tool_choice
        if response_format:
            payload['response_format'] = response_format

        # Map common settings
        for key, payload_key in [
            ('parallel_tool_calls', 'parallel_tool_calls'),
            ('stop_sequences', 'stop'),
            ('seed', 'seed'),
            ('openai_reasoning_effort', 'reasoning_effort'),
            ('openai_user', 'user'),
            ('openai_service_tier', 'service_tier'),
            ('openai_prediction', 'prediction'),
            ('presence_penalty', 'presence_penalty'),
            ('frequency_penalty', 'frequency_penalty'),
            ('logit_bias', 'logit_bias'),
            ('openai_logprobs', 'logprobs'),
            ('openai_top_logprobs', 'top_logprobs'),
            ('openai_prompt_cache_key', 'prompt_cache_key'),
            ('openai_prompt_cache_retention', 'prompt_cache_retention'),
            ('max_tokens', 'max_completion_tokens'),
            ('temperature', 'temperature'),
            ('top_p', 'top_p'),
        ]:
            if val := model_settings.get(key):
                payload[payload_key] = val

        if web_search_options:
            payload['web_search_options'] = web_search_options

        if extra_body := model_settings.get('extra_body'):
            payload.update(extra_body)

        return payload

    def _process_databricks_response(self, response: dict[str, Any]) -> ModelResponse:
        # Helper to get timestamp
        created = response.get('created')
        timestamp = number_to_datetime(created) if isinstance(created, (int, float)) else now_utc()

        choices = cast('list[dict[str, Any]]', response.get('choices', []))
        if not choices:
            raise UnexpectedModelBehavior('Response ended without choices')

        choice = choices[0]
        message = choice.get('message', {})

        # Handle content
        content = message.get('content')
        items: list[ModelResponsePart] = []

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        items.append(TextPart(content=str(item.get('text', ''))))
                elif isinstance(item, str):
                    items.append(TextPart(content=item))
        elif isinstance(content, str):
            items.extend(
                (replace(part, id='content', provider_name=self.system) if isinstance(part, ThinkingPart) else part)
                for part in split_content_into_text_and_thinking(content, self.profile.thinking_tags)
            )

        # Handle tool calls
        tool_calls = cast('list[dict[str, Any]] | None', message.get('tool_calls'))
        if tool_calls:
            for c in tool_calls:
                if c.get('type') == 'function':
                    func = cast('dict[str, Any]', c.get('function', {}))
                    part = ToolCallPart(
                        tool_name=cast(str, func.get('name')),
                        args=func.get('arguments'),
                        tool_call_id=cast('str | None', c.get('id')),
                    )
                    part.tool_call_id = guard_tool_call_id(part)
                    items.append(part)

        provider_details = {}
        if created:
            provider_details['timestamp'] = timestamp

        finish_reason = cast('str | None', choice.get('finish_reason'))
        if finish_reason:
            provider_details['finish_reason'] = finish_reason

        usage_dict = cast('dict[str, int]', response.get('usage', {}))
        usage = RequestUsage(
            input_tokens=usage_dict.get('prompt_tokens', 0),
            output_tokens=usage_dict.get('completion_tokens', 0),
        )

        _CHAT_FINISH_REASON_MAP: dict[str, FinishReason] = {
            'stop': 'stop',
            'length': 'length',
            'tool_calls': 'tool_call',
            'content_filter': 'content_filter',
            'function_call': 'tool_call',
        }
        mapped_finish_reason = _CHAT_FINISH_REASON_MAP.get(finish_reason) if finish_reason else None

        return ModelResponse(
            parts=items,
            usage=usage,
            model_name=cast('str', response.get('model', self.model_name)),
            timestamp=timestamp,
            provider_details=provider_details or None,
            provider_response_id=cast('str | None', response.get('id')),
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            finish_reason=mapped_finish_reason,
        )
