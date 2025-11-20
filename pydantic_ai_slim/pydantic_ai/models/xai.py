"""xAI model implementation using xAI SDK."""

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

from typing_extensions import assert_never

try:
    import xai_sdk.chat as chat_types

    # Import xai_sdk components
    from xai_sdk import AsyncClient
    from xai_sdk.chat import assistant, image, system, tool, tool_result, user
    from xai_sdk.tools import code_execution, get_tool_call_type, mcp, web_search  # x_search not yet supported
except ImportError as _import_error:
    raise ImportError(
        'Please install `xai-sdk` to use the xAI model, '
        'you can use the `xai` optional group — `pip install "pydantic-ai-slim[xai]"`'
    ) from _import_error

from .._run_context import RunContext
from .._utils import now_utc
from ..builtin_tools import CodeExecutionTool, MCPServerTool, WebSearchTool
from ..exceptions import UserError
from ..messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from ..profiles import ModelProfileSpec
from ..providers import Provider, infer_provider
from ..providers.grok import GrokModelName
from ..settings import ModelSettings
from ..usage import RequestUsage

# Type alias for consistency
XaiModelName = GrokModelName


class XaiModel(Model):
    """A model that uses the xAI SDK to interact with xAI models."""

    _model_name: str
    _provider: Provider[AsyncClient]

    def __init__(
        self,
        model_name: XaiModelName,
        *,
        provider: Literal['xai'] | Provider[AsyncClient] = 'xai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize the xAI model.

        Args:
            model_name: The name of the xAI model to use (e.g., "grok-4-1-fast-non-reasoning")
            provider: The provider to use for API calls. Defaults to `'xai'`.
            profile: Optional model profile specification. Defaults to a profile picked by the provider based on the model name.
            settings: Optional model settings.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

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
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    # Retry prompt as user message
                    xai_messages.append(user(part.model_response()))
                else:
                    # Retry prompt as tool result
                    xai_messages.append(tool_result(part.model_response()))
            else:
                assert_never(part)

        return xai_messages

    def _map_user_prompt(self, part: UserPromptPart) -> chat_types.chat_pb2.Message | None:
        """Map a UserPromptPart to an xAI user message."""
        if isinstance(part.content, str):
            return user(part.content)

        # Handle complex content (images, text, etc.)
        content_items: list[chat_types.Content] = []

        for item in part.content:
            if isinstance(item, str):
                content_items.append(item)
            elif isinstance(item, ImageUrl):
                # Get detail from vendor_metadata if available
                detail: chat_types.ImageDetail = 'auto'
                if item.vendor_metadata and 'detail' in item.vendor_metadata:
                    detail = item.vendor_metadata['detail']
                content_items.append(image(item.url, detail=detail))
            elif isinstance(item, BinaryContent):
                if item.is_image:
                    # Convert binary content to data URI and use image()
                    content_items.append(image(item.data_uri, detail='auto'))
                else:
                    # xAI SDK doesn't support non-image binary content yet
                    pass

        if content_items:
            return user(*content_items)

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

    def _get_builtin_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat_types.chat_pb2.Tool]:
        """Convert pydantic_ai built-in tools to xAI SDK server-side tools."""
        tools: list[chat_types.chat_pb2.Tool] = []
        for builtin_tool in model_request_parameters.builtin_tools:
            if isinstance(builtin_tool, WebSearchTool):
                tools.append(web_search())
            elif isinstance(builtin_tool, CodeExecutionTool):
                tools.append(code_execution())
            elif isinstance(builtin_tool, MCPServerTool):
                tools.append(
                    mcp(
                        server_url=builtin_tool.url,
                        server_label=builtin_tool.id,
                        server_description=builtin_tool.description,
                        allowed_tool_names=builtin_tool.allowed_tools,
                        authorization=builtin_tool.authorization_token,
                        extra_headers=builtin_tool.headers,
                    )
                )
            else:
                raise UserError(
                    f'`{builtin_tool.__class__.__name__}` is not supported by `XaiModel`. '
                    f'Supported built-in tools: WebSearchTool, CodeExecutionTool, MCPServerTool. '
                    f'If XSearchTool should be supported, please file an issue.'
                )
        return tools

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the xAI model."""
        client = self._provider.client

        # Convert messages to xAI format
        xai_messages = self._map_messages(messages)

        # Convert tools: combine built-in (server-side) tools and custom (client-side) tools
        tools: list[chat_types.chat_pb2.Tool] = []
        if model_request_parameters.builtin_tools:
            tools.extend(self._get_builtin_tools(model_request_parameters))
        if model_request_parameters.tool_defs:
            tools.extend(self._map_tools(model_request_parameters))
        tools_param = tools if tools else None

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
            if 'parallel_tool_calls' in model_settings:
                xai_settings['parallel_tool_calls'] = model_settings['parallel_tool_calls']
            if 'presence_penalty' in model_settings:
                xai_settings['presence_penalty'] = model_settings['presence_penalty']
            if 'frequency_penalty' in model_settings:
                xai_settings['frequency_penalty'] = model_settings['frequency_penalty']

        # Create chat instance
        chat = client.chat.create(model=self._model_name, messages=xai_messages, tools=tools_param, **xai_settings)

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
        """Make a streaming request to the xAI model."""
        client = self._provider.client

        # Convert messages to xAI format
        xai_messages = self._map_messages(messages)

        # Convert tools: combine built-in (server-side) tools and custom (client-side) tools
        tools: list[chat_types.chat_pb2.Tool] = []
        if model_request_parameters.builtin_tools:
            tools.extend(self._get_builtin_tools(model_request_parameters))
        if model_request_parameters.tool_defs:
            tools.extend(self._map_tools(model_request_parameters))
        tools_param = tools if tools else None

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
            if 'parallel_tool_calls' in model_settings:
                xai_settings['parallel_tool_calls'] = model_settings['parallel_tool_calls']
            if 'presence_penalty' in model_settings:
                xai_settings['presence_penalty'] = model_settings['presence_penalty']
            if 'frequency_penalty' in model_settings:
                xai_settings['frequency_penalty'] = model_settings['frequency_penalty']

        # Create chat instance
        chat = client.chat.create(model=self._model_name, messages=xai_messages, tools=tools_param, **xai_settings)

        # Stream the response
        response_stream = chat.stream()
        streamed_response = XaiStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _response=response_stream,
            _timestamp=now_utc(),
            _provider_name='xai',
        )
        yield streamed_response

    def _process_response(self, response: chat_types.Response) -> ModelResponse:
        """Convert xAI SDK response to pydantic_ai ModelResponse."""
        parts: list[ModelResponsePart] = []

        # Add reasoning/thinking content first if present
        if response.reasoning_content:
            # reasoning_content is the human-readable summary
            parts.append(
                ThinkingPart(
                    content=response.reasoning_content,
                    signature=None,
                    provider_name='xai',
                )
            )
        elif response.encrypted_content:
            # encrypted_content is a signature that can be sent back for reasoning continuity
            parts.append(
                ThinkingPart(
                    content='',  # No readable content for encrypted-only reasoning
                    signature=response.encrypted_content,
                    provider_name='xai',
                )
            )

        # Add tool calls (both client-side and server-side) first
        # For server-side tools, these were executed before generating the final content
        for tool_call in response.tool_calls:
            # Try to determine if this is a server-side tool
            # In real responses, we can use get_tool_call_type()
            # In mock responses, we default to client-side tools
            is_server_side_tool = False
            try:
                tool_type = get_tool_call_type(tool_call)
                # If it's not a client-side tool, it's a server-side tool
                is_server_side_tool = tool_type != 'client_side_tool'
            except Exception:
                # If we can't determine the type, treat as client-side
                pass

            if is_server_side_tool:
                # Server-side tools are executed by xAI, so we add both call and return parts
                # The final result is in response.content
                parts.append(
                    BuiltinToolCallPart(
                        tool_name=tool_call.function.name,
                        args=tool_call.function.arguments,
                        tool_call_id=tool_call.id,
                        provider_name='xai',
                    )
                )
                # Always add the return part for server-side tools since they're already executed
                parts.append(
                    BuiltinToolReturnPart(
                        tool_name=tool_call.function.name,
                        content={'status': 'completed'},
                        tool_call_id=tool_call.id,
                        provider_name='xai',
                    )
                )
            else:
                # Client-side tool call (or mock)
                parts.append(
                    ToolCallPart(
                        tool_name=tool_call.function.name,
                        args=tool_call.function.arguments,
                        tool_call_id=tool_call.id,
                    )
                )

        # Add text content after tool calls (for server-side tools, this is the final result)
        if response.content:
            parts.append(TextPart(content=response.content))

        # Convert usage with detailed token information
        usage = self._map_usage(response)

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
            provider_response_id=response.id,
            finish_reason=finish_reason,
        )

    def _map_usage(self, response: chat_types.Response) -> RequestUsage:
        """Extract usage information from xAI SDK response, including reasoning tokens and cache tokens."""
        return XaiModel.extract_usage(response)

    @staticmethod
    def extract_usage(response: chat_types.Response) -> RequestUsage:
        """Extract usage information from xAI SDK response.

        Extracts token counts and additional usage details including:
        - reasoning_tokens: Tokens used for model reasoning/thinking
        - cache_read_tokens: Tokens read from prompt cache
        - server_side_tools_used: Count of server-side (built-in) tools executed
        """
        if not response.usage:
            return RequestUsage()

        usage_obj = response.usage

        prompt_tokens = usage_obj.prompt_tokens or 0
        completion_tokens = usage_obj.completion_tokens or 0

        # Build details dict for additional usage metrics
        details: dict[str, int] = {}

        # Add reasoning tokens if available (optional attribute)
        reasoning_tokens = getattr(usage_obj, 'reasoning_tokens', None)
        if reasoning_tokens:
            details['reasoning_tokens'] = reasoning_tokens

        # Add cached prompt tokens if available (optional attribute)
        cached_tokens = getattr(usage_obj, 'cached_prompt_text_tokens', None)
        if cached_tokens:
            details['cache_read_tokens'] = cached_tokens

        # Add server-side tools used count if available (optional attribute)
        server_side_tools = getattr(usage_obj, 'server_side_tools_used', None)
        if server_side_tools:
            # server_side_tools_used is a repeated field (list-like) in the real SDK
            # but may be an int in mocks for simplicity
            if isinstance(server_side_tools, int):
                tools_count = server_side_tools
            else:
                tools_count = len(server_side_tools)
            if tools_count:
                details['server_side_tools_used'] = tools_count

        if details:
            return RequestUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                details=details,
            )
        else:
            return RequestUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )


@dataclass
class XaiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for xAI SDK."""

    _model_name: str
    _response: Any  # xai_sdk chat stream
    _timestamp: Any
    _provider_name: str

    def _update_response_state(self, response: Any) -> None:
        """Update response state including usage, response ID, and finish reason."""
        # Update usage
        if response.usage:
            self._usage = XaiModel.extract_usage(response)

        # Set provider response ID
        if response.id and self.provider_response_id is None:
            self.provider_response_id = response.id

        # Handle finish reason
        if response.finish_reason:
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

    def _handle_reasoning_content(self, response: Any, reasoning_handled: bool) -> Iterator[ModelResponseStreamEvent]:
        """Handle reasoning content (both readable and encrypted)."""
        if reasoning_handled:
            return

        if response.reasoning_content:
            # reasoning_content is the human-readable summary
            thinking_part = ThinkingPart(
                content=response.reasoning_content,
                signature=None,
                provider_name='xai',
            )
            yield self._parts_manager.handle_part(vendor_part_id='reasoning', part=thinking_part)
        elif response.encrypted_content:
            # encrypted_content is a signature that can be sent back for reasoning continuity
            thinking_part = ThinkingPart(
                content='',  # No readable content for encrypted-only reasoning
                signature=response.encrypted_content,
                provider_name='xai',
            )
            yield self._parts_manager.handle_part(vendor_part_id='encrypted_reasoning', part=thinking_part)

    def _handle_text_delta(self, chunk: Any) -> Iterator[ModelResponseStreamEvent]:
        """Handle text content delta from chunk."""
        if chunk.content:
            event = self._parts_manager.handle_text_delta(
                vendor_part_id='content',
                content=chunk.content,
            )
            if event is not None:
                yield event

    def _handle_single_tool_call(self, tool_call: Any) -> Iterator[ModelResponseStreamEvent]:
        """Handle a single tool call, routing to server-side or client-side handler."""
        if not tool_call.function.name:
            return

        # Determine if this is a server-side (built-in) tool
        is_server_side_tool = False
        try:
            tool_type = get_tool_call_type(tool_call)
            is_server_side_tool = tool_type != 'client_side_tool'
        except Exception:
            pass  # Treat as client-side if we can't determine

        if is_server_side_tool:
            # Server-side tools - create BuiltinToolCallPart and BuiltinToolReturnPart
            # These tools are already executed by xAI's infrastructure
            call_part = BuiltinToolCallPart(
                tool_name=tool_call.function.name,
                args=tool_call.function.arguments,
                tool_call_id=tool_call.id,
                provider_name='xai',
            )
            yield self._parts_manager.handle_part(vendor_part_id=tool_call.id, part=call_part)

            # Immediately yield the return part since the tool was already executed
            return_part = BuiltinToolReturnPart(
                tool_name=tool_call.function.name,
                content={'status': 'completed'},
                tool_call_id=tool_call.id,
                provider_name='xai',
            )
            yield self._parts_manager.handle_part(vendor_part_id=f'{tool_call.id}_return', part=return_part)
        else:
            # Client-side tools - use standard handler
            yield self._parts_manager.handle_tool_call_part(
                vendor_part_id=tool_call.id,
                tool_name=tool_call.function.name,
                args=tool_call.function.arguments,
                tool_call_id=tool_call.id,
            )

    def _handle_tool_calls(self, response: Any) -> Iterator[ModelResponseStreamEvent]:
        """Handle tool calls (both client-side and server-side)."""
        if not response.tool_calls:
            return

        for tool_call in response.tool_calls:
            yield from self._handle_single_tool_call(tool_call)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Iterate over streaming events from xAI SDK."""
        reasoning_handled = False  # Track if we've already handled reasoning content

        async for response, chunk in self._response:
            self._update_response_state(response)

            # Handle reasoning content (only emit once)
            reasoning_events = list(self._handle_reasoning_content(response, reasoning_handled))
            if reasoning_events:
                reasoning_handled = True
                for event in reasoning_events:
                    yield event

            # Handle text content delta
            for event in self._handle_text_delta(chunk):
                yield event

            # Handle tool calls
            for event in self._handle_tool_calls(response):
                yield event

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
