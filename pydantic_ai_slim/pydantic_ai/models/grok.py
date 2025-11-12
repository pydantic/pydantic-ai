"""Grok model implementation using xAI SDK."""

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from .._run_context import RunContext
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    ToolReturnPart,
    TextPart,
    ToolCallPart,
    FinishReason,
)
from ..models import (
    Model,
    ModelRequestParameters,
    ModelSettings,
    StreamedResponse,
)
from ..usage import RequestUsage
from .._utils import now_utc

# Import xai_sdk components
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user, assistant, tool, tool_result
import xai_sdk.chat as chat_types


class GrokModel(Model):
    """A model that uses the xAI SDK to interact with Grok."""

    _model_name: str
    _api_key: str

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize the Grok model.

        Args:
            model_name: The name of the Grok model to use (e.g., "grok-3", "grok-4-fast-non-reasoning")
            api_key: The xAI API key. If not provided, uses XAI_API_KEY environment variable.
            settings: Optional model settings.
        """
        super().__init__(settings=settings)
        self._model_name = model_name
        self._api_key = api_key or os.getenv("XAI_API_KEY") or ""
        if not self._api_key:
            raise ValueError("XAI API key is required")

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return "xai"

    def _map_messages(self, messages: list[ModelMessage]) -> list[chat_types.chat_pb2.Message]:
        """Convert pydantic_ai messages to xAI SDK messages."""
        xai_messages = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        xai_messages.append(system(part.content))
                    elif isinstance(part, UserPromptPart):
                        # Handle user prompt content
                        if isinstance(part.content, str):
                            xai_messages.append(user(part.content))
                        else:
                            # Handle complex content (images, etc.)
                            # For now, just concatenate text content
                            text_parts = []
                            for item in part.content:
                                if isinstance(item, str):
                                    text_parts.append(item)
                            if text_parts:
                                xai_messages.append(user(" ".join(text_parts)))
                    elif isinstance(part, ToolReturnPart):
                        xai_messages.append(tool_result(part.model_response_str()))
            elif isinstance(message, ModelResponse):
                content_parts = []
                for part in message.parts:
                    if isinstance(part, TextPart):
                        content_parts.append(part.content)
                    elif isinstance(part, ToolCallPart):
                        # Tool calls will be handled separately in the response processing
                        pass

                if content_parts:
                    xai_messages.append(assistant(" ".join(content_parts)))

        return xai_messages

    def _map_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[chat_types.chat_pb2.Tool]:
        """Convert pydantic_ai tool definitions to xAI SDK tools."""
        tools = []
        for tool_def in model_request_parameters.tool_defs.values():
            xai_tool = tool(
                name=tool_def.name,
                description=tool_def.description or "",
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
        # Create client in the current async context to avoid event loop issues
        client = AsyncClient(api_key=self._api_key)

        # Convert messages to xAI format
        xai_messages = self._map_messages(messages)

        # Convert tools if any
        tools = (
            self._map_tools(model_request_parameters)
            if model_request_parameters.tool_defs
            else None
        )

        # Filter model settings to only include xAI SDK compatible parameters
        xai_settings = {}
        if model_settings:
            # Map pydantic_ai settings to xAI SDK parameters
            if "temperature" in model_settings:
                xai_settings["temperature"] = model_settings["temperature"]
            if "top_p" in model_settings:
                xai_settings["top_p"] = model_settings["top_p"]
            if "max_tokens" in model_settings:
                xai_settings["max_tokens"] = model_settings["max_tokens"]
            if "stop_sequences" in model_settings:
                xai_settings["stop"] = model_settings["stop_sequences"]
            if "seed" in model_settings:
                xai_settings["seed"] = model_settings["seed"]

        # Create chat instance
        chat = client.chat.create(
            model=self._model_name, messages=xai_messages, tools=tools, **xai_settings
        )

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
        # Create client in the current async context to avoid event loop issues
        client = AsyncClient(api_key=self._api_key)

        # Convert messages to xAI format
        xai_messages = self._map_messages(messages)

        # Convert tools if any
        tools = (
            self._map_tools(model_request_parameters)
            if model_request_parameters.tool_defs
            else None
        )

        # Filter model settings to only include xAI SDK compatible parameters
        xai_settings = {}
        if model_settings:
            # Map pydantic_ai settings to xAI SDK parameters
            if "temperature" in model_settings:
                xai_settings["temperature"] = model_settings["temperature"]
            if "top_p" in model_settings:
                xai_settings["top_p"] = model_settings["top_p"]
            if "max_tokens" in model_settings:
                xai_settings["max_tokens"] = model_settings["max_tokens"]
            if "stop_sequences" in model_settings:
                xai_settings["stop"] = model_settings["stop_sequences"]
            if "seed" in model_settings:
                xai_settings["seed"] = model_settings["seed"]

        # Create chat instance
        chat = client.chat.create(
            model=self._model_name, messages=xai_messages, tools=tools, **xai_settings
        )

        # Stream the response
        response_stream = chat.stream()
        streamed_response = GrokStreamedResponse(model_request_parameters)
        streamed_response._model_name = self._model_name
        streamed_response._response = response_stream
        streamed_response._timestamp = now_utc()
        streamed_response._provider_name = "xai"
        yield streamed_response

    def _process_response(self, response: chat_types.Response) -> ModelResponse:
        """Convert xAI SDK response to pydantic_ai ModelResponse."""
        from typing import cast

        parts = []

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
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        usage = RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        # Map finish reason
        finish_reason_map = {
            "stop": "stop",
            "length": "length",
            "content_filter": "content_filter",
            "max_output_tokens": "length",
            "cancelled": "error",
            "failed": "error",
        }
        raw_finish_reason = response.finish_reason
        mapped_reason = (
            finish_reason_map.get(raw_finish_reason, "stop")
            if isinstance(raw_finish_reason, str)
            else "stop"
        )
        finish_reason = cast(FinishReason, mapped_reason)

        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=self._model_name,
            timestamp=now_utc(),
            provider_name="xai",
            finish_reason=finish_reason,
        )


class GrokStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for xAI SDK."""

    _model_name: str
    _response: Any  # xai_sdk chat stream
    _timestamp: Any
    _provider_name: str
    _usage: RequestUsage
    provider_response_id: str | None
    finish_reason: Any

    async def _get_event_iterator(self):
        """Iterate over streaming events from xAI SDK."""
        from typing import cast

        async for response, chunk in self._response:
            # Update usage if available
            if hasattr(response, "usage"):
                input_tokens = getattr(response.usage, "input_tokens", 0)
                output_tokens = getattr(response.usage, "output_tokens", 0)
                self._usage = RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)

            # Set provider response ID
            if hasattr(response, "id") and self.provider_response_id is None:
                self.provider_response_id = response.id

            # Handle finish reason
            if hasattr(response, "finish_reason") and response.finish_reason:
                finish_reason_map = {
                    "stop": "stop",
                    "length": "length",
                    "content_filter": "content_filter",
                    "max_output_tokens": "length",
                    "cancelled": "error",
                    "failed": "error",
                }
                mapped_reason = finish_reason_map.get(response.finish_reason, "stop")
                self.finish_reason = cast(FinishReason, mapped_reason)

            # Handle text content
            if hasattr(chunk, "content") and chunk.content:
                yield self._parts_manager.handle_text_delta(
                    vendor_part_id="content",
                    content=chunk.content,
                )

            # Handle tool calls
            if hasattr(chunk, "tool_calls"):
                for tool_call in chunk.tool_calls:
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
