"""xAI model implementation using [xAI SDK](https://github.com/xai-org/xai-sdk-python)."""

import json
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

from typing_extensions import assert_never

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
from ..settings import ModelSettings
from ..usage import RequestUsage

try:
    import xai_sdk.chat as chat_types
    from xai_sdk import AsyncClient
    from xai_sdk.chat import assistant, file, image, system, tool, tool_result, user
    from xai_sdk.proto import chat_pb2, sample_pb2, usage_pb2
    from xai_sdk.tools import code_execution, get_tool_call_type, mcp, web_search  # x_search not yet supported
    from xai_sdk.types.model import ChatModel
except ImportError as _import_error:
    raise ImportError(
        'Please install `xai-sdk` to use the xAI model, '
        'you can use the `xai` optional group — `pip install "pydantic-ai-slim[xai]"`'
    ) from _import_error

XaiModelName = str | ChatModel
"""Possible xAI model names."""

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    'stop': 'stop',
    'length': 'length',
    'content_filter': 'content_filter',
    'max_output_tokens': 'length',
    'cancelled': 'error',
    'failed': 'error',
}

# `GetChatCompletionResponse.outputs[*].finish_reason` uses the proto enum (ints), not the string values returned by
# `Response.finish_reason`.
_FINISH_REASON_PROTO_MAP: dict[int, FinishReason] = {
    sample_pb2.FinishReason.REASON_STOP: 'stop',
    sample_pb2.FinishReason.REASON_MAX_LEN: 'length',
    sample_pb2.FinishReason.REASON_TOOL_CALLS: 'tool_call',
}


class XaiModelSettings(ModelSettings, total=False):
    """Settings specific to xAI models.

    See [xAI SDK documentation](https://docs.x.ai/docs) for more details on these parameters.
    """

    xai_logprobs: bool
    """Whether to return log probabilities of the output tokens or not."""

    xai_top_logprobs: int
    """An integer between 0 and 20 specifying the number of most likely tokens to return at each position."""

    xai_user: str
    """A unique identifier representing your end-user, which can help xAI to monitor and detect abuse."""

    xai_store_messages: bool
    """Whether to store messages on xAI's servers for conversation continuity."""

    xai_previous_response_id: str
    """The ID of the previous response to continue the conversation."""

    xai_include_encrypted_content: bool
    """Whether to include the encrypted content in the response.

    Corresponds to the `use_encrypted_content` value of the model settings in the Responses API.
    """

    xai_include_code_execution_output: bool
    """Whether to include the code execution results in the response.

    Corresponds to the `code_interpreter_call.outputs` value of the `include` parameter in the Responses API.
    """

    xai_include_web_search_output: bool
    """Whether to include the web search results in the response.

    Corresponds to the `web_search_call.action.sources` value of the `include` parameter in the Responses API.
    """

    xai_include_inline_citations: bool
    """Whether to include inline citations in the response.

    Corresponds to the `inline_citations` option in the xAI `include` parameter.
    """

    xai_include_mcp_output: bool
    """Whether to include the MCP results in the response.

    Corresponds to the `mcp_call.outputs` value of the `include` parameter in the Responses API.
    """


# Mapping of XaiModelSettings keys to xAI SDK parameter names.
# Most keys are the same, but some differ (e.g., 'stop_sequences' -> 'stop').
_XAI_MODEL_SETTINGS_MAPPING: dict[str, str] = {
    'temperature': 'temperature',
    'top_p': 'top_p',
    'max_tokens': 'max_tokens',
    'stop_sequences': 'stop',
    'parallel_tool_calls': 'parallel_tool_calls',
    'presence_penalty': 'presence_penalty',
    'frequency_penalty': 'frequency_penalty',
    'xai_logprobs': 'logprobs',
    'xai_top_logprobs': 'top_logprobs',
    'xai_user': 'user',
    'xai_store_messages': 'store_messages',
    'xai_previous_response_id': 'previous_response_id',
}


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

        super().__init__(settings=settings, profile=profile or provider.model_profile(model_name))

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return 'xai'

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type]:
        """Return the set of builtin tool types this model can handle."""
        return frozenset({WebSearchTool, CodeExecutionTool, MCPServerTool})

    async def _map_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> list[chat_types.chat_pb2.Message]:
        """Convert pydantic_ai messages to xAI SDK messages."""
        xai_messages: list[chat_types.chat_pb2.Message] = []
        # xAI expects tool results in the same order as tool calls.
        #
        # Pydantic AI doesn't guarantee tool-result part ordering, so we track
        # tool call order as we walk message history and reorder tool results.
        pending_tool_call_ids: list[str] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                mapped_request_parts = await self._map_request_parts(
                    message.parts,
                    pending_tool_call_ids,
                )
                xai_messages.extend(mapped_request_parts)
            elif isinstance(message, ModelResponse):
                xai_messages.extend(self._map_response_parts(message.parts))
                pending_tool_call_ids.extend(
                    part.tool_call_id for part in message.parts if isinstance(part, ToolCallPart) and part.tool_call_id
                )
            else:
                assert_never(message)

        # Insert instructions as a system message after existing system messages if present
        if instructions := self._get_instructions(messages, model_request_parameters):
            system_prompt_count = sum(1 for m in xai_messages if m.role == chat_types.chat_pb2.MessageRole.ROLE_SYSTEM)
            xai_messages.insert(system_prompt_count, system(instructions))

        return xai_messages

    async def _map_request_parts(
        self,
        parts: Sequence[ModelRequestPart],
        pending_tool_call_ids: list[str],
    ) -> list[chat_types.chat_pb2.Message]:
        """Map ModelRequest parts to xAI messages."""
        xai_messages: list[chat_types.chat_pb2.Message] = []
        tool_results: list[ToolReturnPart | RetryPromptPart] = []

        for part in parts:
            if isinstance(part, SystemPromptPart):
                xai_messages.append(system(part.content))
            elif isinstance(part, UserPromptPart):
                if user_msg := await self._map_user_prompt(part):
                    xai_messages.append(user_msg)
            elif isinstance(part, ToolReturnPart):
                tool_results.append(part)
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    xai_messages.append(user(part.model_response()))
                else:
                    tool_results.append(part)
            else:
                assert_never(part)

        # Sort tool results by requested order, then emit
        if tool_results:
            order = {id: i for i, id in enumerate(pending_tool_call_ids)}
            tool_results.sort(key=lambda p: order.get(p.tool_call_id, float('inf')))
            for part in tool_results:
                text = part.model_response_str() if isinstance(part, ToolReturnPart) else part.model_response()
                xai_messages.append(tool_result(text))

        return xai_messages

    def _map_response_parts(self, parts: Sequence[ModelResponsePart]) -> list[chat_types.chat_pb2.Message]:
        """Map ModelResponse parts to xAI assistant messages (one message per part)."""
        messages: list[chat_types.chat_pb2.Message] = []

        # Track builtin tool calls by tool_call_id to update their status with return parts
        builtin_calls: dict[str, chat_types.chat_pb2.ToolCall] = {}

        for item in parts:
            if isinstance(item, TextPart):
                messages.append(assistant(item.content))
            elif isinstance(item, ThinkingPart):
                if (thinking_msg := self._map_thinking_part(item)) is not None:
                    messages.append(thinking_msg)
            elif isinstance(item, ToolCallPart):
                client_side_tool_call = self._map_tool_call(item)
                self._append_tool_call(messages, client_side_tool_call)
            elif isinstance(item, BuiltinToolCallPart):
                builtin_call = self._map_builtin_tool_call_part(item)
                if item.provider_name == self.system and builtin_call:
                    self._append_tool_call(messages, builtin_call)
                    # Track specific tool calls for status updates
                    # Note: tool_call_id is always truthy here since _map_builtin_tool_call_part
                    # returns None when tool_call_id is empty
                    if item.tool_call_id:  # pragma: no branch
                        builtin_calls[item.tool_call_id] = builtin_call
            elif isinstance(item, BuiltinToolReturnPart):
                if (
                    item.provider_name == self.system
                    and item.tool_call_id
                    and (details := item.provider_details) is not None
                    and details.get('status') == 'failed'
                    and (call := builtin_calls.get(item.tool_call_id))
                ):
                    call.status = chat_types.chat_pb2.TOOL_CALL_STATUS_FAILED
                    if error_msg := details.get('error'):
                        call.error_message = str(error_msg)
            elif isinstance(item, FilePart):
                # Files generated by models (e.g., from CodeExecutionTool) are not sent back
                pass
            else:
                assert_never(item)

        return messages

    @staticmethod
    def _append_tool_call(messages: list[chat_types.chat_pb2.Message], tool_call: chat_types.chat_pb2.ToolCall) -> None:
        """Append a tool call to the most recent tool-call assistant message, or create a new one.

        We keep tool calls grouped to avoid generating one assistant message per tool call.
        """
        if messages and messages[-1].tool_calls:
            messages[-1].tool_calls.append(tool_call)
        else:
            msg = assistant('')
            msg.tool_calls.append(tool_call)
            messages.append(msg)

    def _map_thinking_part(self, item: ThinkingPart) -> chat_types.chat_pb2.Message | None:
        """Map a `ThinkingPart` into a single xAI assistant message.

        - Native xAI thinking (with optional signature) is sent via `reasoning_content`/`encrypted_content`
        - Non-xAI (or non-native) thinking is preserved by wrapping in the model profile's thinking tags
        """
        if item.provider_name == self.system and (item.content or item.signature):
            msg = assistant('')
            if item.content:
                msg.reasoning_content = item.content
            if item.signature:
                msg.encrypted_content = item.signature
            return msg
        elif item.content:
            start_tag, end_tag = self.profile.thinking_tags
            return assistant('\n'.join([start_tag, item.content, end_tag]))
        else:
            return None

    def _map_tool_call(self, tool_call_part: ToolCallPart) -> chat_types.chat_pb2.ToolCall:
        """Map a ToolCallPart to an xAI SDK ToolCall."""
        return chat_types.chat_pb2.ToolCall(
            id=tool_call_part.tool_call_id,
            type=chat_types.chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
            status=chat_types.chat_pb2.TOOL_CALL_STATUS_COMPLETED,
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
            # Extract server label from tool_name (format: 'mcp_server:server_label')
            server_label = item.tool_name.split(':', 1)[1] if ':' in item.tool_name else item.tool_name
            args_dict = item.args_as_dict() or {}
            # Extract tool_name and tool_args from the structured args (matches OpenAI/Anthropic pattern)
            actual_tool_name = args_dict.get('tool_name', '')
            tool_args = args_dict.get('tool_args', {})
            # Construct the full function name in xAI's format: 'server_label.tool_name'
            function_name = f'{server_label}.{actual_tool_name}' if actual_tool_name else server_label
            return chat_types.chat_pb2.ToolCall(
                id=item.tool_call_id,
                type=chat_types.chat_pb2.TOOL_CALL_TYPE_MCP_TOOL,
                status=chat_types.chat_pb2.TOOL_CALL_STATUS_COMPLETED,
                function=chat_types.chat_pb2.FunctionCall(
                    name=function_name,
                    arguments=json.dumps(tool_args),
                ),
            )
        return None

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
                image_url = item.url
                if item.force_download:
                    downloaded = await download_item(item, data_format='base64_uri', type_format='extension')
                    image_url = downloaded['data']
                content_items.append(image(image_url, detail=detail))
            elif isinstance(item, BinaryContent):
                if item.is_image:
                    # Convert binary content to data URI and use image()
                    image_detail: chat_types.ImageDetail = 'auto'
                    if item.vendor_metadata and 'detail' in item.vendor_metadata:
                        image_detail = item.vendor_metadata['detail']
                    content_items.append(image(item.data_uri, detail=image_detail))
                elif item.is_audio:
                    raise NotImplementedError('AudioUrl/BinaryContent with audio is not supported by xAI SDK')
                elif item.is_document:
                    # Upload document to xAI files API and reference it
                    filename = item.identifier or f'document.{item.format}'
                    file_id = await self._upload_file_to_xai(item.data, filename)
                    content_items.append(file(file_id))
                else:
                    raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
            elif isinstance(item, AudioUrl):
                raise NotImplementedError('AudioUrl is not supported by xAI SDK')
            elif isinstance(item, DocumentUrl):
                # Download and upload to xAI files API
                downloaded = await download_item(item, data_format='bytes')
                filename = item.identifier or 'document'
                # Add extension if data_type is available from download
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
        model_settings: XaiModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        """Create an xAI chat instance with common setup for both request and stream.

        Returns:
            The xAI SDK chat object, ready to call .sample() or .stream() on.
        """
        # Convert messages to xAI format
        xai_messages = await self._map_messages(messages, model_request_parameters)

        # Convert tools: combine built-in (server-side) tools and custom (client-side) tools
        tools: list[chat_types.chat_pb2.Tool] = []
        if model_request_parameters.builtin_tools:
            tools.extend(_get_builtin_tools(model_request_parameters))
        if model_request_parameters.tool_defs:
            tools.extend(_map_tools(model_request_parameters))
        tools_param = tools if tools else None

        # Set tool_choice based on whether tools are available and text output is allowed
        profile = GrokModelProfile.from_profile(self.profile)
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output and profile.grok_supports_tool_choice_required:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        # Set response_format based on the output_mode
        response_format: chat_pb2.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = _map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted' and not tools and profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = _map_json_object()

        # Map model settings to xAI SDK parameters
        xai_settings = _map_model_settings(model_settings)

        # Populate use_encrypted_content and include based on model settings
        include: list[chat_pb2.IncludeOption] = []
        use_encrypted_content = model_settings.get('xai_include_encrypted_content') or False
        if model_settings.get('xai_include_code_execution_output'):
            include.append(chat_pb2.IncludeOption.INCLUDE_OPTION_CODE_EXECUTION_CALL_OUTPUT)
        if model_settings.get('xai_include_web_search_output'):
            include.append(chat_pb2.IncludeOption.INCLUDE_OPTION_WEB_SEARCH_CALL_OUTPUT)
        if model_settings.get('xai_include_inline_citations'):
            include.append(chat_pb2.IncludeOption.INCLUDE_OPTION_INLINE_CITATIONS)
        # x_search not yet supported
        # collections_search not yet supported (could be mapped to file search)
        if model_settings.get('xai_include_mcp_output'):
            include.append(chat_pb2.IncludeOption.INCLUDE_OPTION_MCP_CALL_OUTPUT)

        # Create and return chat instance
        return self._provider.client.chat.create(
            model=self._model_name,
            messages=xai_messages,
            tools=tools_param,
            tool_choice=tool_choice,
            response_format=response_format,
            use_encrypted_content=use_encrypted_content,
            include=include,
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

        chat = await self._create_chat(messages, cast(XaiModelSettings, model_settings or {}), model_request_parameters)
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

        chat = await self._create_chat(messages, cast(XaiModelSettings, model_settings or {}), model_request_parameters)
        response_stream = chat.stream()
        yield await self._process_streamed_response(response_stream, model_request_parameters)

    def _process_response(self, response: chat_types.Response) -> ModelResponse:
        """Convert xAI SDK response to pydantic_ai ModelResponse.

        Processes response.proto.outputs to extract (in order):
        - ThinkingPart: For reasoning/thinking content
        - TextPart: For text content
        - ToolCallPart: For client-side tool calls
        - BuiltinToolCallPart + BuiltinToolReturnPart: For server-side (builtin) tool calls
        """
        parts: list[ModelResponsePart] = []
        outputs = response.proto.outputs

        for output in outputs:
            message = output.message

            # Add reasoning/thinking content if present
            if message.reasoning_content or message.encrypted_content:
                signature = message.encrypted_content or None
                parts.append(
                    ThinkingPart(
                        content=message.reasoning_content or '',
                        signature=signature,
                        provider_name=self.system if signature else None,
                    )
                )

            # Add text content from assistant messages
            if message.content and message.role == chat_types.chat_pb2.MessageRole.ROLE_ASSISTANT:
                part_provider_details: dict[str, Any] | None = None
                if output.logprobs and output.logprobs.content:
                    part_provider_details = {'logprobs': _map_logprobs(output.logprobs)}
                parts.append(TextPart(content=message.content, provider_details=part_provider_details))

            # Process tool calls in this output
            for tool_call in message.tool_calls:
                tool_result_content = _get_tool_result_content(message.content)
                _, part = _create_tool_call_part(
                    tool_call,
                    tool_result_content,
                    self.system,
                    message_role=message.role,
                )
                parts.append(part)

        # Convert usage with detailed token information
        usage = _extract_usage(response, self._model_name, self._provider.name, self._provider.base_url)

        # Map finish reason.
        #
        # The xAI SDK exposes `response.finish_reason` as a *string* for the overall response, but in
        # multi-output responses (e.g. server-side tools) it can reflect an intermediate TOOL_CALLS
        # output rather than the final STOP output. We derive the finish reason from the final output
        # when available.
        if outputs:
            last_reason = outputs[-1].finish_reason
            finish_reason = _FINISH_REASON_PROTO_MAP.get(last_reason, 'stop')
        else:  # pragma: no cover
            finish_reason = _FINISH_REASON_MAP.get(response.finish_reason, 'stop')

        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=self._model_name,
            timestamp=response.created,
            provider_name=self.system,
            provider_url=self._provider.base_url,
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
        if isinstance(first_item, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        first_response, _ = first_item

        return XaiStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=first_response.created,
            _provider=self._provider,
        )


@dataclass
class XaiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for xAI SDK."""

    _model_name: str
    _response: _utils.PeekableAsyncStream[tuple[chat_types.Response, chat_types.Chunk]]
    _timestamp: datetime
    _provider: Provider[AsyncClient]

    @property
    def system(self) -> str:
        """The model provider system name."""
        return self._provider.name

    @property
    def provider_url(self) -> str:
        """Get the provider base URL."""
        return self._provider.base_url

    def _update_response_state(self, response: chat_types.Response) -> None:
        """Update response state including usage, response ID, and finish reason."""
        # Update usage (SDK Response always provides a usage object)
        self._usage = _extract_usage(response, self._model_name, self._provider.name, self._provider.base_url)

        # Set provider response ID (only set once)
        if response.id and self.provider_response_id is None:
            self.provider_response_id = response.id

        # Handle finish reason (SDK Response always provides a finish_reason)
        self.finish_reason = _FINISH_REASON_MAP.get(response.finish_reason, 'stop')

    def _collect_reasoning_events(
        self,
        *,
        response: chat_types.Response,
        prev_reasoning_content: str,
        prev_encrypted_content: str,
    ) -> tuple[str, str, list[ModelResponseStreamEvent]]:
        """Collect thinking/reasoning events and return updated previous values.

        Note: xAI exposes reasoning via the accumulated Response object (not the per-chunk delta), so we compute
        deltas ourselves to avoid re-emitting the entire accumulated content on every chunk.
        """
        events: list[ModelResponseStreamEvent] = []

        if response.reasoning_content and response.reasoning_content != prev_reasoning_content:
            if response.reasoning_content.startswith(prev_reasoning_content):
                reasoning_delta = response.reasoning_content[len(prev_reasoning_content) :]
            else:
                reasoning_delta = response.reasoning_content
            prev_reasoning_content = response.reasoning_content
            if reasoning_delta:  # pragma: no branch
                events.extend(
                    self._parts_manager.handle_thinking_delta(
                        vendor_part_id='reasoning',
                        content=reasoning_delta,
                        # Only set provider_name when we have an encrypted signature to send back.
                        provider_name=self.system if response.encrypted_content else None,
                    )
                )

        if response.encrypted_content and response.encrypted_content != prev_encrypted_content:
            prev_encrypted_content = response.encrypted_content
            events.extend(
                self._parts_manager.handle_thinking_delta(
                    vendor_part_id='reasoning',
                    signature=response.encrypted_content,
                    provider_name=self.system,
                )
            )

        return prev_reasoning_content, prev_encrypted_content, events

    def _handle_server_side_tool_call(
        self,
        *,
        tool_call: chat_pb2.ToolCall,
        delta: chat_pb2.Delta,
        seen_tool_call_ids: set[str],
        seen_tool_return_ids: set[str],
        last_tool_return_content: dict[str, dict[str, Any] | str | None],
    ) -> Iterator[ModelResponseStreamEvent]:
        """Handle a single server-side tool call delta, yielding stream events."""
        builtin_tool_name = _get_builtin_tool_name(tool_call)

        if delta.role == chat_pb2.MessageRole.ROLE_ASSISTANT:
            # Emit the call part once per tool_call_id.
            if tool_call.id in seen_tool_call_ids:
                return
            seen_tool_call_ids.add(tool_call.id)

            if builtin_tool_name.startswith(MCPServerTool.kind):
                parsed_args = _build_mcp_tool_call_args(tool_call)
            else:
                parsed_args = _parse_tool_args(tool_call.function.arguments)
            call_part = BuiltinToolCallPart(
                tool_name=builtin_tool_name, args=parsed_args, tool_call_id=tool_call.id, provider_name=self.system
            )
            yield self._parts_manager.handle_part(vendor_part_id=tool_call.id, part=call_part)
            return

        if delta.role == chat_pb2.MessageRole.ROLE_TOOL:
            # Emit the return part once per tool_call_id.
            return_vendor_id = f'{tool_call.id}_return'
            tool_result_content = _get_tool_result_content(delta.content)
            if return_vendor_id in seen_tool_return_ids and tool_result_content == last_tool_return_content.get(
                return_vendor_id
            ):
                return
            seen_tool_return_ids.add(return_vendor_id)
            last_tool_return_content[return_vendor_id] = tool_result_content
            return_part = BuiltinToolReturnPart(
                tool_name=builtin_tool_name,
                content=tool_result_content,
                tool_call_id=tool_call.id,
                provider_name=self.system,
            )
            yield self._parts_manager.handle_part(vendor_part_id=return_vendor_id, part=return_part)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Iterate over streaming events from xAI SDK."""
        # Local state to avoid re-emmiting duplicate events.
        prev_reasoning_content = ''
        prev_encrypted_content = ''
        seen_tool_call_ids: set[str] = set()
        seen_tool_return_ids: set[str] = set()
        last_tool_return_content: dict[str, dict[str, Any] | str | None] = {}
        # Track previous tool call args to compute deltas (like we do for reasoning content).
        prev_tool_call_args: dict[str, str] = {}

        async for response, chunk in self._response:
            self._update_response_state(response)

            prev_reasoning_content, prev_encrypted_content, reasoning_events = self._collect_reasoning_events(
                response=response,
                prev_reasoning_content=prev_reasoning_content,
                prev_encrypted_content=prev_encrypted_content,
            )
            for event in reasoning_events:
                yield event

            # Handle text content (property filters for ROLE_ASSISTANT)
            if chunk.content:
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id='content',
                    content=chunk.content,
                ):
                    yield event

            # Handle tool calls/tool results from *this chunk*.
            #
            # Important: xAI SDK `Response` is an accumulated view; `response.tool_calls` includes tool calls from
            # previous chunks. Iterating over it would re-emit tool calls repeatedly. Instead, we read tool calls
            # from the chunk's deltas which represent what changed in this frame.
            for output_chunk in chunk.proto.outputs:
                delta = output_chunk.delta
                if not delta.tool_calls:
                    continue
                for tool_call in delta.tool_calls:
                    if not tool_call.function.name:
                        continue

                    if tool_call.type != chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL:
                        for event in self._handle_server_side_tool_call(
                            tool_call=tool_call,
                            delta=delta,
                            seen_tool_call_ids=seen_tool_call_ids,
                            seen_tool_return_ids=seen_tool_return_ids,
                            last_tool_return_content=last_tool_return_content,
                        ):
                            yield event
                    else:
                        # Client-side tools: emit args as deltas so UI adapters receive PartDeltaEvents
                        # (not repeated PartStartEvents). Use accumulated args from response.tool_calls
                        # and compute the delta like we do for reasoning content.
                        accumulated = next((tc for tc in response.tool_calls if tc.id == tool_call.id), None)
                        accumulated_args = (
                            accumulated.function.arguments
                            if accumulated is not None and accumulated.function.arguments
                            else tool_call.function.arguments
                        )
                        prev_args = prev_tool_call_args.get(tool_call.id, '')
                        is_new_tool_call = tool_call.id not in prev_tool_call_args
                        args_changed = accumulated_args != prev_args

                        if is_new_tool_call or args_changed:
                            # Compute delta: if accumulated starts with prev, extract the new portion.
                            if accumulated_args.startswith(prev_args):
                                args_delta = accumulated_args[len(prev_args) :] or None
                            else:
                                args_delta = accumulated_args or None
                            prev_tool_call_args[tool_call.id] = accumulated_args
                            maybe_event = self._parts_manager.handle_tool_call_delta(
                                vendor_part_id=tool_call.id,
                                # Only pass tool_name on the first call; it would be appended otherwise.
                                tool_name=tool_call.function.name if is_new_tool_call else None,
                                args=args_delta,
                                tool_call_id=tool_call.id,
                            )
                            if maybe_event is not None:  # pragma: no branch
                                yield maybe_event

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


def _map_json_schema(o: OutputObjectDefinition) -> chat_pb2.ResponseFormat:
    """Convert OutputObjectDefinition to xAI ResponseFormat protobuf object."""
    return chat_pb2.ResponseFormat(
        format_type=chat_pb2.FORMAT_TYPE_JSON_SCHEMA,
        schema=json.dumps(o.json_schema),
    )


def _map_json_object() -> chat_pb2.ResponseFormat:
    """Create a ResponseFormat for JSON object mode (prompted output)."""
    return chat_pb2.ResponseFormat(format_type=chat_pb2.FORMAT_TYPE_JSON_OBJECT)


def _map_model_settings(model_settings: XaiModelSettings) -> dict[str, Any]:
    """Map pydantic_ai ModelSettings to xAI SDK parameters."""
    return {
        _XAI_MODEL_SETTINGS_MAPPING[key]: value
        for key, value in model_settings.items()
        if key in _XAI_MODEL_SETTINGS_MAPPING
    }


def _map_tools(model_request_parameters: ModelRequestParameters) -> list[chat_types.chat_pb2.Tool]:
    """Convert pydantic_ai tool definitions to xAI SDK tools."""
    return [
        tool(
            name=tool_def.name,
            description=tool_def.description or '',
            parameters=tool_def.parameters_json_schema,
        )
        for tool_def in model_request_parameters.tool_defs.values()
    ]


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
        else:  # pragma: no cover
            # Defensive fallback - validation in models/__init__.py catches unsupported tools earlier
            raise UserError(
                f'`{builtin_tool.__class__.__name__}` is not supported by `XaiModel`. '
                f'Supported built-in tools: WebSearchTool, CodeExecutionTool, MCPServerTool. '
                f'If XSearchTool should be supported, please file an issue.'
            )
    return tools


def _get_builtin_tool_name(tool_call: chat_types.chat_pb2.ToolCall) -> str:
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
        # Extract server label from "server_label.tool_name" format
        function_name = tool_call.function.name
        server_label = function_name.split('.', 1)[0] if '.' in function_name else function_name
        return f'{MCPServerTool.kind}:{server_label}'
    else:
        # x_search, collections_search, or unknown - use function name
        return tool_call.function.name


def _map_server_side_tools_used_to_name(server_side_tool: usage_pb2.ServerSideTool) -> str:
    """Map xAI SDK ServerSideTool enum from usage.server_side_tools_used to a tool name.

    Args:
        server_side_tool: The ServerSideTool enum value from usage.server_side_tools_used.

    Returns:
        The tool name (e.g., 'web_search', 'code_execution').
    """
    mapping = {
        usage_pb2.SERVER_SIDE_TOOL_WEB_SEARCH: WebSearchTool.kind,
        usage_pb2.SERVER_SIDE_TOOL_CODE_EXECUTION: CodeExecutionTool.kind,
        usage_pb2.SERVER_SIDE_TOOL_MCP: MCPServerTool.kind,
        usage_pb2.SERVER_SIDE_TOOL_X_SEARCH: 'x_search',
        usage_pb2.SERVER_SIDE_TOOL_COLLECTIONS_SEARCH: 'collections_search',
        usage_pb2.SERVER_SIDE_TOOL_VIEW_IMAGE: 'view_image',
        usage_pb2.SERVER_SIDE_TOOL_VIEW_X_VIDEO: 'view_x_video',
    }
    return mapping.get(server_side_tool, 'unknown')


def _extract_usage(
    response: chat_types.Response,
    model: str,
    provider: str,
    provider_url: str,
) -> RequestUsage:
    """Extract usage information from xAI SDK response.

    Extracts token counts and additional usage details including:
    - reasoning_tokens: Tokens used for model reasoning/thinking
    - cache_read_tokens: Tokens read from prompt cache
    - server_side_tools_used: Count of server-side (built-in) tools executed
    """
    usage_obj = response.usage

    # Build usage data dict with all integer fields for genai-prices extraction
    usage_data: dict[str, int] = {
        'prompt_tokens': usage_obj.prompt_tokens or 0,
        'completion_tokens': usage_obj.completion_tokens or 0,
    }

    # Add reasoning tokens if available (optional attribute)
    if usage_obj.reasoning_tokens:
        usage_data['reasoning_tokens'] = usage_obj.reasoning_tokens

    # Add cached prompt tokens if available (optional attribute)
    if usage_obj.cached_prompt_text_tokens:
        usage_data['cache_read_tokens'] = usage_obj.cached_prompt_text_tokens

    # Aggregate server-side tools used by PydanticAI builtin tool name
    if usage_obj.server_side_tools_used:
        tool_counts: dict[str, int] = defaultdict(int)
        for server_side_tool in usage_obj.server_side_tools_used:
            tool_name = _map_server_side_tools_used_to_name(server_side_tool)
            tool_counts[tool_name] += 1
        # Add each tool as a separate details entry (server_side_tools must be flattened to comply with details being dict[str, int])
        for tool_name, count in tool_counts.items():
            usage_data[f'server_side_tools_{tool_name}'] = count

    # Build details from non-standard fields
    details = {k: v for k, v in usage_data.items() if k not in {'prompt_tokens', 'completion_tokens'}}

    extracted = RequestUsage.extract(
        dict(model=model, usage=usage_data),
        provider=provider,
        provider_url=provider_url,
        provider_fallback='x_ai',  # Pricing file is defined as x_ai.yml
        details=details or None,
    )

    # Ensure token counts are set even if genai-prices extraction failed
    if extracted.input_tokens == 0 and usage_data['prompt_tokens']:
        extracted.input_tokens = usage_data['prompt_tokens']
    if extracted.output_tokens == 0 and usage_data['completion_tokens']:
        extracted.output_tokens = usage_data['completion_tokens']

    return extracted


def _get_tool_result_content(content: str) -> dict[str, Any] | str | None:
    """Extract tool result content from a content string.

    Args:
        content: The content string (may be JSON or plain text)

    Returns:
        Tool result content as dict (if JSON), string, or None if no content
    """
    if content:
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content
    return None


def _parse_tool_args(arguments: str) -> dict[str, Any]:
    """Parse tool call arguments from JSON string to dict.

    Args:
        arguments: JSON string of tool arguments

    Returns:
        Parsed arguments as dict, or empty dict if parsing fails
    """
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}


def _build_mcp_tool_call_args(tool_call: chat_pb2.ToolCall) -> dict[str, Any]:
    """Build args dict for MCP server tool calls.

    Follows the same pattern as OpenAI and Anthropic for MCP tool calls,
    including the actual tool name in the args dict.

    Args:
        tool_call: The xAI SDK tool call.

    Returns:
        Args dict with action, tool_name, and tool_args.
    """
    # xAI provides function name in "server_label.tool_name" format
    function_name = tool_call.function.name
    actual_tool_name = function_name.split('.', 1)[1] if '.' in function_name else function_name
    return {
        'action': 'call_tool',
        'tool_name': actual_tool_name,
        'tool_args': _parse_tool_args(tool_call.function.arguments),
    }


def _create_tool_call_part(
    tool_call: chat_pb2.ToolCall,
    tool_result_content: dict[str, Any] | str | None,
    provider_name: str,
    message_role: chat_pb2.MessageRole | None = None,
) -> tuple[str, ModelResponsePart]:
    """Create a part for a tool call, returning (vendor_part_id, part).

    Handles both server-side (builtin) and client-side tool calls.

    Args:
        tool_call: The tool call from the xAI response.
        tool_result_content: The content for the tool result (extracted by caller).
        provider_name: The provider name for builtin tools.
        message_role: The role of the message containing the tool call, if available. For server-side tools in
            non-streamed responses, xAI emits tool *calls* on `ROLE_ASSISTANT` messages and tool *results* on
            `ROLE_TOOL` messages; in those cases, role should take precedence over tool status.

    Returns:
        Tuple of (vendor_part_id, part).
    """
    is_server_side_tool = tool_call.type != chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL

    if is_server_side_tool:
        builtin_tool_name = _get_builtin_tool_name(tool_call)
        status = tool_call.status
        provider_details: dict[str, Any] | None = None

        if status == chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_FAILED:
            provider_details = {
                'status': 'failed',
                'error': tool_call.error_message,
            }

        # If we know the message role (non-streamed responses), use it to decide call vs return.
        # Note: FAILED is always a return part (it may be emitted on an assistant message without a separate ROLE_TOOL
        # message).
        if status == chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_FAILED or message_role == chat_pb2.MessageRole.ROLE_TOOL:
            return (
                f'{tool_call.id}_return',
                BuiltinToolReturnPart(
                    tool_name=builtin_tool_name,
                    content=tool_result_content,
                    tool_call_id=tool_call.id,
                    provider_name=provider_name,
                    provider_details=provider_details,
                ),
            )
        else:
            if builtin_tool_name.startswith(MCPServerTool.kind):
                args = _build_mcp_tool_call_args(tool_call)
            else:
                args = _parse_tool_args(tool_call.function.arguments)
            return (
                tool_call.id,
                BuiltinToolCallPart(
                    tool_name=builtin_tool_name,
                    args=args,
                    tool_call_id=tool_call.id,
                    provider_name=provider_name,
                ),
            )
    else:
        # Client-side tool call
        return (
            tool_call.id,
            ToolCallPart(
                tool_name=tool_call.function.name,
                args=tool_call.function.arguments,
                tool_call_id=tool_call.id,
            ),
        )


def _map_logprobs(logprobs: chat_types.chat_pb2.LogProbs) -> dict[str, Any]:
    """Map xAI logprobs to a dictionary format."""
    return {
        'content': [
            {
                'token': lp.token,
                'bytes': list(lp.bytes) if lp.bytes else None,
                'logprob': lp.logprob,
                'top_logprobs': [
                    {
                        'token': tlp.token,
                        'bytes': list(tlp.bytes) if tlp.bytes else None,
                        'logprob': tlp.logprob,
                    }
                    for tlp in lp.top_logprobs
                ],
            }
            for lp in logprobs.content
        ]
    }
