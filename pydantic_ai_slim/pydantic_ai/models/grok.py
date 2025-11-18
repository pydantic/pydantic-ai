"""Grok model implementation using xAI SDK."""

import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import xai_sdk.chat as chat_types

# Import xai_sdk components
from xai_sdk import AsyncClient
from xai_sdk.chat import assistant, system, tool, tool_result, user

from .._run_context import RunContext
from .._utils import now_utc
from ..messages import (
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from ..settings import ModelSettings
from ..usage import RequestUsage


class GrokModel(Model):
    """A model that uses the xAI SDK to interact with Grok."""

    _model_name: str
    _api_key: str
    _client: AsyncClient | None

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        client: AsyncClient | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize the Grok model.

        Args:
            model_name: The name of the Grok model to use (e.g., "grok-3", "grok-4-fast-non-reasoning")
            api_key: The xAI API key. If not provided, uses XAI_API_KEY environment variable.
            client: Optional AsyncClient instance for testing. If provided, api_key is ignored.
            settings: Optional model settings.
        """
        super().__init__(settings=settings)
        self._model_name = model_name
        self._client = client
        if client is None:
            self._api_key = api_key or os.getenv('XAI_API_KEY') or ''
            if not self._api_key:
                raise ValueError('XAI API key is required')
        else:
            self._api_key = api_key or ''

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return 'xai'

    def _map_messages(self, messages: list[ModelMessage]) -> list[chat_types.chat_pb2.Message]:
        """Convert pydantic_ai messages to xAI SDK messages."""
        xai_messages: list[chat_types.chat_pb2.Message] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                xai_messages.extend(self._map_request_parts(message.parts))
            elif isinstance(message, ModelResponse):
                if response_msg := self._map_response_parts(message.parts):
                    xai_messages.append(response_msg)

        return xai_messages

    def _map_request_parts(self, parts: Sequence[ModelRequestPart]) -> list[chat_types.chat_pb2.Message]:
        """Map ModelRequest parts to xAI messages."""
        xai_messages: list[chat_types.chat_pb2.Message] = []

        for part in parts:
            if isinstance(part, SystemPromptPart):
                xai_messages.append(system(part.content))
            elif isinstance(part, UserPromptPart):
                if user_msg := self._map_user_prompt(part):
                    xai_messages.append(user_msg)
            elif isinstance(part, ToolReturnPart):
                xai_messages.append(tool_result(part.model_response_str()))

        return xai_messages

    def _map_user_prompt(self, part: UserPromptPart) -> chat_types.chat_pb2.Message | None:
        """Map a UserPromptPart to an xAI user message."""
        if isinstance(part.content, str):
            return user(part.content)

        # Handle complex content (images, etc.)
        text_parts: list[str] = [item for item in part.content if isinstance(item, str)]
        if text_parts:
            return user(' '.join(text_parts))

        return None

    def _map_response_parts(self, parts: Sequence[ModelResponsePart]) -> chat_types.chat_pb2.Message | None:
        """Map ModelResponse parts to an xAI assistant message."""
        content_parts: list[str] = [part.content for part in parts if isinstance(part, TextPart)]

        if content_parts:
            return assistant(' '.join(content_parts))

        return None

    def _map_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat_types.chat_pb2.Tool]:
        """Convert pydantic_ai tool definitions to xAI SDK tools."""
        tools: list[chat_types.chat_pb2.Tool] = []
        for tool_def in model_request_parameters.tool_defs.values():
            xai_tool = tool(
                name=tool_def.name,
                description=tool_def.description or '',
                parameters=tool_def.parameters_json_schema,
            )
            tools.append(xai_tool)
        return tools

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the Grok model."""
        # Use injected client or create one in the current async context
        client = self._client or AsyncClient(api_key=self._api_key)

        # Convert messages to xAI format
        xai_messages = self._map_messages(messages)

        # Convert tools if any
        tools = self._map_tools(model_request_parameters) if model_request_parameters.tool_defs else None

        # Filter model settings to only include xAI SDK compatible parameters
        xai_settings: dict[str, Any] = {}
        if model_settings:
            # Map pydantic_ai settings to xAI SDK parameters
            if 'temperature' in model_settings:
                xai_settings['temperature'] = model_settings['temperature']
            if 'top_p' in model_settings:
                xai_settings['top_p'] = model_settings['top_p']
            if 'max_tokens' in model_settings:
                xai_settings['max_tokens'] = model_settings['max_tokens']
            if 'stop_sequences' in model_settings:
                xai_settings['stop'] = model_settings['stop_sequences']
            if 'seed' in model_settings:
                xai_settings['seed'] = model_settings['seed']

        # Create chat instance
        chat = client.chat.create(model=self._model_name, messages=xai_messages, tools=tools, **xai_settings)

        # Sample the response
        response = await chat.sample()

        # Convert response to pydantic_ai format
        return self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the Grok model."""
        # Use injected client or create one in the current async context
        client = self._client or AsyncClient(api_key=self._api_key)

        # Convert messages to xAI format
        xai_messages = self._map_messages(messages)

        # Convert tools if any
        tools = self._map_tools(model_request_parameters) if model_request_parameters.tool_defs else None

        # Filter model settings to only include xAI SDK compatible parameters
        xai_settings: dict[str, Any] = {}
        if model_settings:
            # Map pydantic_ai settings to xAI SDK parameters
            if 'temperature' in model_settings:
                xai_settings['temperature'] = model_settings['temperature']
            if 'top_p' in model_settings:
                xai_settings['top_p'] = model_settings['top_p']
            if 'max_tokens' in model_settings:
                xai_settings['max_tokens'] = model_settings['max_tokens']
            if 'stop_sequences' in model_settings:
                xai_settings['stop'] = model_settings['stop_sequences']
            if 'seed' in model_settings:
                xai_settings['seed'] = model_settings['seed']

        # Create chat instance
        chat = client.chat.create(model=self._model_name, messages=xai_messages, tools=tools, **xai_settings)

        # Stream the response
        response_stream = chat.stream()
        streamed_response = GrokStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _response=response_stream,
            _timestamp=now_utc(),
            _provider_name='xai',
        )
        yield streamed_response

    def _process_response(self, response: chat_types.Response) -> ModelResponse:
        """Convert xAI SDK response to pydantic_ai ModelResponse."""
        from typing import cast

        parts: list[ModelResponsePart] = []

        # Add text content
        if response.content:
            parts.append(TextPart(content=response.content))

        # Add tool calls
        for tool_call in response.tool_calls:
            parts.append(
                ToolCallPart(
                    tool_name=tool_call.function.name,
                    args=tool_call.function.arguments,
                    tool_call_id=tool_call.id,
                )
            )

        # Convert usage - try to access attributes, default to 0 if not available
        input_tokens = getattr(response.usage, 'input_tokens', 0)
        output_tokens = getattr(response.usage, 'output_tokens', 0)
        usage = RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        # Map finish reason
        finish_reason_map = {
            'stop': 'stop',
            'length': 'length',
            'content_filter': 'content_filter',
            'max_output_tokens': 'length',
            'cancelled': 'error',
            'failed': 'error',
        }
        raw_finish_reason = response.finish_reason
        mapped_reason = (
            finish_reason_map.get(raw_finish_reason, 'stop') if isinstance(raw_finish_reason, str) else 'stop'
        )
        finish_reason = cast(FinishReason, mapped_reason)

        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=self._model_name,
            timestamp=now_utc(),
            provider_name='xai',
            finish_reason=finish_reason,
        )


@dataclass
class GrokStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for xAI SDK."""

    _model_name: str
    _response: Any  # xai_sdk chat stream
    _timestamp: Any
    _provider_name: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Iterate over streaming events from xAI SDK."""
        from typing import cast

        async for response, chunk in self._response:
            # Update usage if available
            if hasattr(response, 'usage'):
                input_tokens = getattr(response.usage, 'input_tokens', 0)
                output_tokens = getattr(response.usage, 'output_tokens', 0)
                self._usage = RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)

            # Set provider response ID
            if hasattr(response, 'id') and self.provider_response_id is None:
                self.provider_response_id = response.id

            # Handle finish reason
            if hasattr(response, 'finish_reason') and response.finish_reason:
                finish_reason_map = {
                    'stop': 'stop',
                    'length': 'length',
                    'content_filter': 'content_filter',
                    'max_output_tokens': 'length',
                    'cancelled': 'error',
                    'failed': 'error',
                }
                mapped_reason = finish_reason_map.get(response.finish_reason, 'stop')
                self.finish_reason = cast(FinishReason, mapped_reason)

            # Handle text content
            if hasattr(chunk, 'content') and chunk.content:
                event = self._parts_manager.handle_text_delta(
                    vendor_part_id='content',
                    content=chunk.content,
                )
                if event is not None:
                    yield event

            # Handle tool calls
            # Note: We use the accumulated Response tool calls, not the Chunk deltas,
            # because pydantic validation needs complete JSON, not partial deltas
            if hasattr(response, 'tool_calls'):
                for tool_call in response.tool_calls:
                    if hasattr(tool_call.function, 'name') and tool_call.function.name:
                        yield self._parts_manager.handle_tool_call_part(
                            vendor_part_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            args=tool_call.function.arguments,
                            tool_call_id=tool_call.id,
                        )

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self):
        """Get the timestamp of the response."""
        return self._timestamp
