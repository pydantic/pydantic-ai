"""xAI model implementation using xAI SDK."""

import json
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

from typing_extensions import assert_never

try:
    import xai_sdk.chat as chat_types
    from xai_sdk import AsyncClient
    from xai_sdk.chat import assistant, file, image, system, tool, tool_result, user
    from xai_sdk.proto.v6 import chat_pb2, usage_pb2
    from xai_sdk.tools import code_execution, get_tool_call_type, mcp, web_search  # x_search not yet supported
except ImportError as _import_error:
    raise ImportError(
        'Please install `xai-sdk` to use the xAI model, '
        'you can use the `xai` optional group — `pip install "pydantic-ai-slim[xai]"`'
    ) from _import_error

from .. import _utils
from .._output import OutputObjectDefinition
from .._run_context import RunContext
from ..builtin_tools import CodeExecutionTool, MCPServerTool, WebSearchTool
from ..exceptions import UnexpectedModelBehavior, UserError
from ..messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
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
    VideoUrl,
)
from ..models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
    download_item,
)
from ..profiles import ModelProfileSpec
from ..profiles.grok import GrokModelProfile
from ..providers import Provider, infer_provider
from ..providers.xai import XaiModelName
from ..settings import ModelSettings
from ..usage import RequestUsage

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    'stop': 'stop',
    'length': 'length',
    'content_filter': 'content_filter',
    'max_output_tokens': 'length',
    'cancelled': 'error',
    'failed': 'error',
}


class XaiModelSettings(ModelSettings, total=False):
    """Settings specific to xAI models.

    See [xAI SDK documentation](https://docs.x.ai/docs) for more details on these parameters.
    """

    logprobs: bool
    """Whether to return log probabilities of the output tokens or not."""

    top_logprobs: int
    """An integer between 0 and 20 specifying the number of most likely tokens to return at each position."""

    use_encrypted_content: bool
    """Whether to use encrypted content for reasoning continuity."""

    store_messages: bool
    """Whether to store messages on xAI's servers for conversation continuity."""

    user: str
    """A unique identifier representing your end-user, which can help xAI to monitor and detect abuse."""


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

    async def _map_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[str | None, list[chat_types.chat_pb2.Message]]:
        """Convert pydantic_ai messages to xAI SDK messages.

        Returns:
            A tuple of (instructions, xai_messages) where instructions may be None.
        """
        xai_messages: list[chat_types.chat_pb2.Message] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                xai_messages.extend(await self._map_request_parts(message.parts))
            elif isinstance(message, ModelResponse):
                if response_msg := self._map_response_parts(message.parts):
                    xai_messages.append(response_msg)
            else:
                assert_never(message)

        instructions = self._get_instructions(messages, model_request_parameters)
        return instructions, xai_messages

    async def _map_request_parts(self, parts: Sequence[ModelRequestPart]) -> list[chat_types.chat_pb2.Message]:
        """Map ModelRequest parts to xAI messages."""
        xai_messages: list[chat_types.chat_pb2.Message] = []

        for part in parts:
            if isinstance(part, SystemPromptPart):
                xai_messages.append(system(part.content))
            elif isinstance(part, UserPromptPart):
                if user_msg := await self._map_user_prompt(part):
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

    def _map_response_parts(self, parts: Sequence[ModelResponsePart]) -> chat_types.chat_pb2.Message | None:
        """Map ModelResponse parts to an xAI assistant message."""
        # Collect content from response parts
        texts: list[str] = []
        reasoning_texts: list[str] = []
        tool_calls: list[chat_types.chat_pb2.ToolCall] = []

        # Track builtin tool calls to update their status with corresponding return parts
        code_execution_tool_call: chat_types.chat_pb2.ToolCall | None = None
        web_search_tool_call: chat_types.chat_pb2.ToolCall | None = None

        for item in parts:
            if isinstance(item, TextPart):
                texts.append(item.content)
            elif isinstance(item, ThinkingPart):
                # xAI models (grok) support reasoning_content directly
                reasoning_texts.append(item.content)
            elif isinstance(item, ToolCallPart):
                tool_calls.append(self._map_tool_call(item))
            elif isinstance(item, BuiltinToolCallPart):
                # Map builtin tool calls with appropriate status
                builtin_call = self._map_builtin_tool_call_part(item)
                if builtin_call:
                    tool_calls.append(builtin_call)
                    # Track specific tool calls for status updates
                    if item.tool_name == CodeExecutionTool.kind:
                        code_execution_tool_call = builtin_call
                    elif item.tool_name == WebSearchTool.kind:
                        web_search_tool_call = builtin_call
            elif isinstance(item, BuiltinToolReturnPart):
                # Update tool call status based on return part
                self._update_builtin_tool_status(item, code_execution_tool_call, web_search_tool_call)
            elif isinstance(item, FilePart):  # pragma: no cover
                # Files generated by models (e.g., from CodeExecutionTool) are not sent back
                pass
            else:
                assert_never(item)

        # Create assistant message with content, reasoning_content, and tool_calls
        return self._build_assistant_message(texts, reasoning_texts, tool_calls)

    def _map_tool_call(self, tool_call_part: ToolCallPart) -> chat_types.chat_pb2.ToolCall:
        """Map a ToolCallPart to an xAI SDK ToolCall."""
        return chat_types.chat_pb2.ToolCall(
            id=tool_call_part.tool_call_id,
            function=chat_types.chat_pb2.FunctionCall(
                name=tool_call_part.tool_name,
                arguments=tool_call_part.args_as_json_str(),
            ),
        )

    def _map_builtin_tool_call_part(self, item: BuiltinToolCallPart) -> chat_types.chat_pb2.ToolCall | None:
        """Map a BuiltinToolCallPart to an xAI SDK ToolCall with appropriate type and status."""
        if not item.tool_call_id:
            return None

        if item.tool_name == CodeExecutionTool.kind:
            return chat_types.chat_pb2.ToolCall(
                id=item.tool_call_id,
                type=chat_types.chat_pb2.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
                status=chat_types.chat_pb2.TOOL_CALL_STATUS_COMPLETED,
                function=chat_types.chat_pb2.FunctionCall(
                    name=CodeExecutionTool.kind,
                    arguments=item.args_as_json_str(),
                ),
            )
        elif item.tool_name == WebSearchTool.kind:
            return chat_types.chat_pb2.ToolCall(
                id=item.tool_call_id,
                type=chat_types.chat_pb2.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
                status=chat_types.chat_pb2.TOOL_CALL_STATUS_COMPLETED,
                function=chat_types.chat_pb2.FunctionCall(
                    name=WebSearchTool.kind,
                    arguments=item.args_as_json_str(),
                ),
            )
        elif item.tool_name.startswith(MCPServerTool.kind):
            return chat_types.chat_pb2.ToolCall(
                id=item.tool_call_id,
                type=chat_types.chat_pb2.TOOL_CALL_TYPE_MCP_TOOL,
                status=chat_types.chat_pb2.TOOL_CALL_STATUS_COMPLETED,
                function=chat_types.chat_pb2.FunctionCall(
                    name=item.tool_name,
                    arguments=item.args_as_json_str(),
                ),
            )
        return None

    def _update_builtin_tool_status(
        self,
        item: BuiltinToolReturnPart,
        code_execution_tool_call: chat_types.chat_pb2.ToolCall | None,
        web_search_tool_call: chat_types.chat_pb2.ToolCall | None,
    ) -> None:
        """Update the status of builtin tool calls based on their return parts."""
        if not isinstance(item.content, dict):
            return

        content = cast(dict[str, Any], item.content)
        status = content.get('status')

        # Update status if it failed or has an error
        if status == 'failed' or 'error' in content:
            if item.tool_name == CodeExecutionTool.kind and code_execution_tool_call is not None:
                code_execution_tool_call.status = chat_types.chat_pb2.TOOL_CALL_STATUS_FAILED
                if error_msg := content.get('error'):
                    code_execution_tool_call.error_message = str(error_msg)
            elif item.tool_name == WebSearchTool.kind and web_search_tool_call is not None:
                web_search_tool_call.status = chat_types.chat_pb2.TOOL_CALL_STATUS_FAILED
                if error_msg := content.get('error'):
                    web_search_tool_call.error_message = str(error_msg)

    def _build_assistant_message(
        self,
        texts: list[str],
        reasoning_texts: list[str],
        tool_calls: list[chat_types.chat_pb2.ToolCall],
    ) -> chat_types.chat_pb2.Message | None:
        """Build an assistant message from collected parts."""
        if not (texts or reasoning_texts or tool_calls):
            return None

        # Simple text-only message
        if texts and not (reasoning_texts or tool_calls):
            return assistant('\n\n'.join(texts))

        # Message with reasoning and/or tool calls
        if texts:
            msg = assistant('\n\n'.join(texts))
        else:
            msg = chat_types.chat_pb2.Message(role=chat_types.chat_pb2.MessageRole.ROLE_ASSISTANT)

        if reasoning_texts:
            msg.reasoning_content = '\n\n'.join(reasoning_texts)
        if tool_calls:
            msg.tool_calls.extend(tool_calls)

        return msg

    async def _upload_file_to_xai(self, data: bytes, filename: str) -> str:
        """Upload a file to xAI files API and return the file ID.

        Args:
            data: The file content as bytes
            filename: The filename to use for the upload

        Returns:
            The file ID from xAI
        """
        uploaded_file = await self._provider.client.files.upload(data, filename=filename)
        return uploaded_file.id

    async def _map_user_prompt(self, part: UserPromptPart) -> chat_types.chat_pb2.Message | None:  # noqa: C901
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
                elif item.is_audio:
                    raise NotImplementedError('AudioUrl/BinaryContent with audio is not supported by xAI SDK')
                elif item.is_document:
                    # Upload document to xAI files API and reference it
                    filename = item.identifier or f'document.{item.format}'
                    file_id = await self._upload_file_to_xai(item.data, filename)
                    content_items.append(file(file_id))
                else:  # pragma: no cover
                    raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
            elif isinstance(item, AudioUrl):
                raise NotImplementedError('AudioUrl is not supported by xAI SDK')
            elif isinstance(item, DocumentUrl):
                # Download and upload to xAI files API
                downloaded = await download_item(item, data_format='bytes')
                filename = item.identifier or 'document'
                if 'data_type' in downloaded and downloaded['data_type']:
                    filename = f'{filename}.{downloaded["data_type"]}'

                file_id = await self._upload_file_to_xai(downloaded['data'], filename)
                content_items.append(file(file_id))
            elif isinstance(item, VideoUrl):
                raise NotImplementedError('VideoUrl is not supported by xAI SDK')
            elif isinstance(item, CachePoint):
                # xAI doesn't support prompt caching via CachePoint, so we filter it out
                pass
            else:
                assert_never(item)

        if content_items:
            return user(*content_items)

        return None

    async def _create_chat(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        """Create an xAI chat instance with common setup for both request and stream.

        Returns:
            The xAI SDK chat object, ready to call .sample() or .stream() on.
        """
        # Convert messages to xAI format
        instructions, xai_messages = await self._map_messages(messages, model_request_parameters)

        # Insert instructions as a system message at the beginning if present
        if instructions:
            xai_messages.insert(0, system(instructions))

        # Convert tools: combine built-in (server-side) tools and custom (client-side) tools
        tools: list[chat_types.chat_pb2.Tool] = []
        if model_request_parameters.builtin_tools:
            tools.extend(self._get_builtin_tools(model_request_parameters))
        if model_request_parameters.tool_defs:
            tools.extend(self._map_tools(model_request_parameters))
        tools_param = tools if tools else None

        # Set tool_choice based on whether tools are available and text output is allowed
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif (
            not model_request_parameters.allow_text_output
            and GrokModelProfile.from_profile(self.profile).grok_supports_tool_choice_required
        ):
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        # Set response_format based on the output_mode
        response_format: chat_pb2.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted'
            and not tools
            and self.profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = self._map_json_object()

        # Map model settings to xAI SDK parameters
        xai_settings = self._map_model_settings(model_settings)

        # Create and return chat instance
        return self._provider.client.chat.create(
            model=self._model_name,
            messages=xai_messages,
            tools=tools_param,
            tool_choice=tool_choice,
            response_format=response_format,
            **xai_settings,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the xAI model."""
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )

        chat = await self._create_chat(messages, model_settings, model_request_parameters)
        response = await chat.sample()
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
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )

        chat = await self._create_chat(messages, model_settings, model_request_parameters)
        response_stream = chat.stream()
        yield await self._process_streamed_response(response_stream, model_request_parameters)

    def _process_response(self, response: chat_types.Response) -> ModelResponse:
        """Convert xAI SDK response to pydantic_ai ModelResponse."""
        parts: list[ModelResponsePart] = []

        # Add reasoning/thinking content first if present
        if response.reasoning_content or response.encrypted_content:
            signature = response.encrypted_content or None
            parts.append(
                ThinkingPart(
                    content=response.reasoning_content or '',  # Empty string if only encrypted
                    signature=signature,
                    provider_name=self.system if signature else None,
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
                builtin_tool_name = XaiModel.get_builtin_tool_name(tool_call)
                parts.append(
                    BuiltinToolCallPart(
                        tool_name=builtin_tool_name,
                        args=tool_call.function.arguments,
                        tool_call_id=tool_call.id,
                        provider_name=self.system,
                    )
                )
                # Always add the return part for server-side tools since they're already executed
                parts.append(
                    BuiltinToolReturnPart(
                        tool_name=builtin_tool_name,
                        content={'status': 'completed'},
                        tool_call_id=tool_call.id,
                        provider_name=self.system,
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
        usage = self.extract_usage(response)

        # Map finish reason
        finish_reason = _FINISH_REASON_MAP.get(response.finish_reason, 'stop')

        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=self._model_name,
            timestamp=response.created,
            provider_name=self.system,
            provider_response_id=response.id,
            finish_reason=finish_reason,
        )

    async def _process_streamed_response(
        self,
        response: AsyncIterator[tuple[chat_types.Response, Any]],
        model_request_parameters: ModelRequestParameters,
    ) -> 'XaiStreamedResponse':
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_item = await peekable_response.peek()
        if isinstance(first_item, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        first_response, _ = first_item

        return XaiStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=first_response.created,
            _provider=self._provider,
        )

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
        if usage_obj.reasoning_tokens:
            details['reasoning_tokens'] = usage_obj.reasoning_tokens

        # Add cached prompt tokens if available (optional attribute)
        if usage_obj.cached_prompt_text_tokens:
            details['cache_read_tokens'] = usage_obj.cached_prompt_text_tokens

        # Aggregate server-side tools used by PydanticAI builtin tool name
        if usage_obj.server_side_tools_used:
            tool_counts: dict[str, int] = defaultdict(int)
            for server_side_tool in usage_obj.server_side_tools_used:
                tool_name = XaiModel._map_server_side_tool_to_builtin_name(server_side_tool)
                tool_counts[tool_name] += 1
            # Add each tool as a separate details entry (server_side_tools must be flattened to comply with details being dict[str, int])
            for tool_name, count in tool_counts.items():
                details[f'server_side_tools_{tool_name}'] = count

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

    @staticmethod
    def get_builtin_tool_name(tool_call: chat_types.chat_pb2.ToolCall) -> str:
        """Get the PydanticAI tool name for an xAI builtin tool call.

        Maps xAI SDK tool call types to PydanticAI builtin tool names.

        Args:
            tool_call: The xAI SDK tool call.

        Returns:
            The PydanticAI tool name (e.g., 'web_search', 'code_execution').
        """
        tool_type = get_tool_call_type(tool_call)

        if tool_type == 'web_search_tool':
            return WebSearchTool.kind
        elif tool_type == 'code_execution_tool':
            return CodeExecutionTool.kind
        elif tool_type == 'mcp_tool':
            # For MCP tools, use the function name which includes the server label/tool name
            return tool_call.function.name
        elif tool_type == 'x_search_tool':
            # X search not currently supported in PydanticAI, use function name as fallback
            return tool_call.function.name
        elif tool_type == 'collections_search_tool':
            # Collections search not currently supported in PydanticAI, use function name as fallback
            return tool_call.function.name
        else:
            # Fallback to function name for unknown types
            return tool_call.function.name

    @staticmethod
    def _map_server_side_tool_to_builtin_name(server_side_tool: usage_pb2.ServerSideTool) -> str:
        """Map xAI SDK ServerSideTool enum to PydanticAI builtin tool name.

        Args:
            server_side_tool: The ServerSideTool enum value.

        Returns:
            The PydanticAI tool name (e.g., 'web_search', 'code_execution').
        """
        if server_side_tool == usage_pb2.SERVER_SIDE_TOOL_WEB_SEARCH:
            return WebSearchTool.kind
        elif server_side_tool == usage_pb2.SERVER_SIDE_TOOL_CODE_EXECUTION:
            return CodeExecutionTool.kind
        elif server_side_tool == usage_pb2.SERVER_SIDE_TOOL_MCP:
            return MCPServerTool.kind
        elif server_side_tool == usage_pb2.SERVER_SIDE_TOOL_X_SEARCH:
            return 'x_search'  # Not yet supported in PydanticAI
        elif server_side_tool == usage_pb2.SERVER_SIDE_TOOL_COLLECTIONS_SEARCH:
            return 'collections_search'  # Not yet supported in PydanticAI
        elif server_side_tool == usage_pb2.SERVER_SIDE_TOOL_VIEW_IMAGE:
            return 'view_image'  # Not yet supported in PydanticAI
        elif server_side_tool == usage_pb2.SERVER_SIDE_TOOL_VIEW_X_VIDEO:
            return 'view_x_video'  # Not yet supported in PydanticAI
        else:
            return 'unknown'

    @staticmethod
    def _map_json_schema(o: OutputObjectDefinition) -> chat_pb2.ResponseFormat:
        """Convert OutputObjectDefinition to xAI ResponseFormat protobuf object."""
        # xAI uses a simpler ResponseFormat structure with format_type and schema (as JSON string)
        return chat_pb2.ResponseFormat(
            format_type=chat_pb2.FORMAT_TYPE_JSON_SCHEMA,
            schema=json.dumps(o.json_schema),
        )

    @staticmethod
    def _map_json_object() -> chat_pb2.ResponseFormat:
        """Create a ResponseFormat for JSON object mode (prompted output)."""
        return chat_pb2.ResponseFormat(format_type=chat_pb2.FORMAT_TYPE_JSON_OBJECT)

    @staticmethod
    def _map_model_settings(model_settings: ModelSettings | None) -> dict[str, Any]:
        """Map pydantic_ai ModelSettings to xAI SDK parameters."""
        if not model_settings:
            return {}

        # Mapping of pydantic_ai setting keys to xAI SDK parameter names
        # Most keys are the same, but 'stop_sequences' maps to 'stop'
        setting_mapping = {
            'temperature': 'temperature',
            'top_p': 'top_p',
            'max_tokens': 'max_tokens',
            'stop_sequences': 'stop',
            'parallel_tool_calls': 'parallel_tool_calls',
            'presence_penalty': 'presence_penalty',
            'frequency_penalty': 'frequency_penalty',
            'logprobs': 'logprobs',
            'top_logprobs': 'top_logprobs',
            'reasoning_effort': 'reasoning_effort',
            'use_encrypted_content': 'use_encrypted_content',
            'store_messages': 'store_messages',
            'user': 'user',
        }

        # Build the settings dict, only including keys that are present in the input
        # TypedDict is just a dict at runtime, so we can iterate over it directly
        return {setting_mapping[key]: value for key, value in model_settings.items() if key in setting_mapping}

    @staticmethod
    def _map_tools(model_request_parameters: ModelRequestParameters) -> list[chat_types.chat_pb2.Tool]:
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

    @staticmethod
    def _get_builtin_tools(model_request_parameters: ModelRequestParameters) -> list[chat_types.chat_pb2.Tool]:
        """Convert pydantic_ai built-in tools to xAI SDK server-side tools."""
        tools: list[chat_types.chat_pb2.Tool] = []
        for builtin_tool in model_request_parameters.builtin_tools:
            if isinstance(builtin_tool, WebSearchTool):
                # xAI web_search supports:
                # - excluded_domains (from blocked_domains)
                # - allowed_domains
                # Note: user_location and search_context_size are not supported by xAI SDK
                tools.append(
                    web_search(
                        excluded_domains=builtin_tool.blocked_domains,
                        allowed_domains=builtin_tool.allowed_domains,
                        enable_image_understanding=False,  # Not supported by PydanticAI
                    )
                )
            elif isinstance(builtin_tool, CodeExecutionTool):
                # xAI code_execution takes no parameters
                tools.append(code_execution())
            elif isinstance(builtin_tool, MCPServerTool):
                # xAI mcp supports:
                # - server_url, server_label, server_description
                # - allowed_tool_names, authorization, extra_headers
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


@dataclass
class XaiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for xAI SDK."""

    _model_name: str
    _response: _utils.PeekableAsyncStream[tuple[chat_types.Response, Any]]
    _timestamp: datetime
    _provider: Provider[AsyncClient]

    @property
    def system(self) -> str:
        """The model provider system name."""
        return self._provider.name

    def _update_response_state(self, response: chat_types.Response) -> None:
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

    def _handle_reasoning_content(
        self, response: chat_types.Response, reasoning_handled: bool
    ) -> Iterator[ModelResponseStreamEvent]:
        """Handle reasoning content (both readable and encrypted)."""
        if reasoning_handled:
            return

        if response.reasoning_content:
            # reasoning_content is the human-readable summary
            thinking_part = ThinkingPart(
                content=response.reasoning_content,
                signature=None,
                provider_name=None,
            )
            yield self._parts_manager.handle_part(vendor_part_id='reasoning', part=thinking_part)
        elif response.encrypted_content:
            # encrypted_content is a signature that can be sent back for reasoning continuity
            thinking_part = ThinkingPart(
                content='',  # No readable content for encrypted-only reasoning
                signature=response.encrypted_content,
                provider_name=self.system,
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

    def _handle_single_tool_call(self, tool_call: chat_types.chat_pb2.ToolCall) -> Iterator[ModelResponseStreamEvent]:
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
            builtin_tool_name = XaiModel.get_builtin_tool_name(tool_call)
            call_part = BuiltinToolCallPart(
                tool_name=builtin_tool_name,
                args=tool_call.function.arguments,
                tool_call_id=tool_call.id,
                provider_name=self.system,
            )
            yield self._parts_manager.handle_part(vendor_part_id=tool_call.id, part=call_part)

            # Immediately yield the return part since the tool was already executed
            return_part = BuiltinToolReturnPart(
                tool_name=builtin_tool_name,
                content={'status': 'completed'},
                tool_call_id=tool_call.id,
                provider_name=self.system,
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

    def _handle_tool_calls(self, response: chat_types.Response) -> Iterator[ModelResponseStreamEvent]:
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
        """The model provider."""
        return self.system

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp
