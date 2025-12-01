from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

from pydantic_ai.messages import FinishReason

from ..conftest import raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    import xai_sdk.chat as chat_types
    from xai_sdk import AsyncClient
    from xai_sdk.proto.v6 import chat_pb2, sample_pb2


def _get_proto_finish_reason(finish_reason: FinishReason) -> sample_pb2.FinishReason:
    """Map pydantic-ai FinishReason to xAI proto FinishReason enum value."""
    finish_reason_to_proto: dict[FinishReason, sample_pb2.FinishReason] = {
        'stop': sample_pb2.FinishReason.REASON_STOP,
        'length': sample_pb2.FinishReason.REASON_MAX_LEN,
        'tool_call': sample_pb2.FinishReason.REASON_TOOL_CALLS,
        'content_filter': sample_pb2.FinishReason.REASON_STOP,  # xAI doesn't have content_filter
    }
    return finish_reason_to_proto.get(finish_reason, sample_pb2.FinishReason.REASON_STOP)


@dataclass
class MockXai:
    """Mock for xAI SDK AsyncClient to simulate xAI API responses."""

    responses: chat_types.Response | Sequence[chat_types.Response] | Exception | Sequence[Exception] | None = None
    stream_data: (
        Sequence[tuple[chat_types.Response, Any]] | Sequence[Sequence[tuple[chat_types.Response, Any]]] | None
    ) = None
    index: int = 0
    chat_create_kwargs: list[dict[str, Any]] = field(default_factory=list)
    api_key: str = 'test-api-key'

    @cached_property
    def chat(self) -> Any:
        """Create mock chat interface."""
        return type('Chat', (), {'create': self.chat_create})

    @cached_property
    def files(self) -> Any:
        """Create mock files interface."""
        return type('Files', (), {'upload': self.files_upload})

    @classmethod
    def create_mock(
        cls,
        responses: chat_types.Response | Sequence[chat_types.Response] | Exception | Sequence[Exception],
        api_key: str = 'test-api-key',
    ) -> AsyncClient:
        """Create a mock AsyncClient for non-streaming responses."""
        return cast(AsyncClient, cls(responses=responses, api_key=api_key))

    @classmethod
    def create_mock_stream(
        cls,
        stream: (Sequence[tuple[chat_types.Response, Any]] | Sequence[Sequence[tuple[chat_types.Response, Any]]]),
        api_key: str = 'test-api-key',
    ) -> AsyncClient:
        """Create a mock AsyncClient for streaming responses."""
        return cast(AsyncClient, cls(stream_data=stream, api_key=api_key))

    def chat_create(self, *_args: Any, **kwargs: Any) -> MockChatInstance:
        """Mock the chat.create method."""
        self.chat_create_kwargs.append(kwargs)
        return MockChatInstance(
            responses=self.responses,
            stream_data=self.stream_data,
            index=self.index,
            parent=self,
        )

    async def files_upload(self, data: bytes, filename: str) -> Any:
        """Mock the files.upload method."""
        # Return a mock uploaded file object with an id
        return type('UploadedFile', (), {'id': f'file-{filename}'})()


@dataclass
class MockChatInstance:
    """Mock for the chat instance returned by client.chat.create()."""

    responses: chat_types.Response | Sequence[chat_types.Response] | Exception | Sequence[Exception] | None = None
    stream_data: (
        Sequence[tuple[chat_types.Response, Any]] | Sequence[Sequence[tuple[chat_types.Response, Any]]] | None
    ) = None
    index: int = 0
    parent: MockXai | None = None

    async def sample(self) -> chat_types.Response:
        """Mock the sample() method for non-streaming responses."""
        assert self.responses is not None, 'you can only use sample() if responses are provided'

        if isinstance(self.responses, Sequence):
            if self.index >= len(self.responses):
                raise IndexError(f'Mock response index {self.index} out of range (length: {len(self.responses)})')
            raise_if_exception(self.responses[self.index])
            response = cast(chat_types.Response, self.responses[self.index])
            # Increment index for next call
            self.index += 1
            if self.parent:
                self.parent.index = self.index
        else:
            raise_if_exception(self.responses)
            response = cast(chat_types.Response, self.responses)

        return response

    def stream(self) -> MockAsyncStream[tuple[chat_types.Response, Any]]:
        """Mock the stream() method for streaming responses."""
        assert self.stream_data is not None, 'you can only use stream() if stream_data is provided'

        # Check if we have nested sequences (multiple streams) vs single stream
        # We need to check if it's a list of tuples (single stream) vs list of lists (multiple streams)
        if isinstance(self.stream_data, list) and len(self.stream_data) > 0:
            first_item = self.stream_data[0]
            # If first item is a list (not a tuple), we have multiple streams
            if isinstance(first_item, list):
                data = cast(list[tuple[chat_types.Response, Any]], self.stream_data[self.index])
            else:
                # Single stream - use the data as is
                data = cast(list[tuple[chat_types.Response, Any]], self.stream_data)
        else:
            data = cast(list[tuple[chat_types.Response, Any]], self.stream_data)

        if self.parent:
            self.parent.index += 1

        return MockAsyncStream(iter(data))


def get_mock_chat_create_kwargs(async_client: AsyncClient) -> list[dict[str, Any]]:
    """Extract the kwargs passed to chat.create from a mock client."""
    if isinstance(async_client, MockXai):
        return async_client.chat_create_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockXai instance')


# =============================================================================
# Proto Object Helpers
# =============================================================================
# These functions create actual xAI SDK proto objects (chat_pb2.*) and wrap them
# in chat_types.Response. This ensures our mocks behave exactly like the real SDK.


def _build_response_with_outputs(
    response_id: str,
    outputs: list[chat_pb2.CompletionOutput],
    usage: Any | None = None,
) -> chat_types.Response:
    """Build a Response from a list of CompletionOutput protos."""
    proto = chat_pb2.GetChatCompletionResponse(id=response_id)
    proto.outputs.extend(outputs)
    if usage:
        proto.usage.CopyFrom(usage)
    proto.created.GetCurrentTime()
    return chat_types.Response(proto, index=None)


def create_response(
    content: str = '',
    tool_calls: list[chat_pb2.ToolCall] | None = None,
    finish_reason: FinishReason = 'stop',
    usage: Any | None = None,
    reasoning_content: str = '',
    encrypted_content: str = '',
) -> chat_types.Response:
    """Create a Response object for testing using actual xAI SDK proto objects.

    Args:
        content: Text content of the response
        tool_calls: List of chat_pb2.ToolCall objects (use create_tool_call() to create)
        finish_reason: pydantic-ai FinishReason ('stop', 'length', 'tool_call', 'content_filter')
        usage: Usage proto object (use create_usage() to create)
        reasoning_content: Reasoning/thinking content
        encrypted_content: Encrypted content signature

    Returns:
        A real chat_types.Response wrapping actual proto objects
    """
    # Create the message proto
    message = chat_pb2.CompletionMessage(
        content=content,
        role=chat_pb2.MessageRole.ROLE_ASSISTANT,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
    )
    if tool_calls:
        message.tool_calls.extend(tool_calls)

    # Create the output proto
    output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=_get_proto_finish_reason(finish_reason),
        message=message,
    )

    return _build_response_with_outputs('grok-123', [output], usage)


def create_tool_call(
    id: str,
    name: str,
    arguments: dict[str, Any],
) -> chat_pb2.ToolCall:
    """Create a ToolCall proto object for testing (client-side tool).

    Args:
        id: Tool call ID
        name: Function name
        arguments: Function arguments (will be JSON-serialized)

    Returns:
        A chat_pb2.ToolCall proto object
    """
    return chat_pb2.ToolCall(
        id=id,
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
        function=chat_pb2.FunctionCall(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def create_server_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    tool_call_id: str = 'server_tool_001',
    tool_type: chat_pb2.ToolCallType = chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
    status: chat_pb2.ToolCallStatus = chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    error_message: str = '',
) -> chat_pb2.ToolCall:
    """Create a server-side tool call proto object.

    Args:
        tool_name: Name of the builtin tool ('code_execution', 'web_search', etc.)
        arguments: Arguments for the tool call (will be JSON-serialized)
        tool_call_id: Unique ID for this tool call
        tool_type: Protobuf enum value for tool type. Use chat_pb2.ToolCallType constants:
            - TOOL_CALL_TYPE_WEB_SEARCH_TOOL
            - TOOL_CALL_TYPE_CODE_EXECUTION_TOOL
            - TOOL_CALL_TYPE_MCP_TOOL
        status: Status of the tool call. Defaults to TOOL_CALL_STATUS_COMPLETED.
        error_message: Error message if the tool call failed.

    Returns:
        A chat_pb2.ToolCall proto object configured as a server-side tool

    Example:
        >>> tool_call = create_server_tool_call(
        ...     tool_name='code_execution',
        ...     arguments={'code': 'print(2+2)'},
        ...     tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL
        ... )
    """
    return chat_pb2.ToolCall(
        id=tool_call_id,
        type=tool_type,
        status=status,
        error_message=error_message,
        function=chat_pb2.FunctionCall(
            name=tool_name,
            arguments=json.dumps(arguments),
        ),
    )


def create_stream_chunk(
    content: str = '',
    tool_calls: list[chat_pb2.ToolCall] | None = None,
    reasoning_content: str = '',
    encrypted_content: str = '',
    finish_reason: FinishReason | None = None,
    index: int = 0,
) -> chat_types.Chunk:
    """Create a streaming chunk using the actual xAI SDK Chunk wrapper.

    This creates a real xAI SDK Chunk object that wraps the proto.
    The Chunk class provides convenience properties like .content, .tool_calls.

    Args:
        content: Text content delta
        tool_calls: List of tool calls in this chunk
        reasoning_content: Reasoning content delta
        encrypted_content: Encrypted content delta
        finish_reason: Finish reason (only set on final chunk)
        index: Output index (usually 0)

    Returns:
        A chat_types.Chunk object wrapping the proto
    """
    # Create the delta with incremental content
    delta = chat_pb2.Delta(
        content=content,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
        role=chat_pb2.MessageRole.ROLE_ASSISTANT,
    )
    if tool_calls:
        delta.tool_calls.extend(tool_calls)

    # Create the output chunk
    output_chunk = chat_pb2.CompletionOutputChunk(
        index=index,
        delta=delta,
    )
    if finish_reason:
        output_chunk.finish_reason = _get_proto_finish_reason(finish_reason)

    # Create the chunk proto
    proto = chat_pb2.GetChatCompletionChunk(id='grok-123')
    proto.outputs.append(output_chunk)
    proto.created.GetCurrentTime()

    # Wrap in Chunk class (like the real SDK does) for convenience properties
    return chat_types.Chunk(proto, index=None)


# =============================================================================
# Builtin Tool Support
# =============================================================================
# The following functions support testing xAI's built-in (server-side) tools:
# - code_execution: Execute Python code server-side
# - web_search: Search the web for information
# - mcp_server: Connect to Model Context Protocol servers
#
# These tools are executed by xAI's infrastructure and return results that need to be
# included in the mock response. Since xAI uses gRPC (not HTTP), we cannot use VCR
# for recording. Instead, we hand-craft mocks similar to how Anthropic and Google
# handle their builtin tools.


def _create_builtin_tool_return_content(
    tool_type: chat_pb2.ToolCallType,
    tool_name: str,
    tool_arguments: dict[str, Any],
    *,
    output: str | None = None,
    return_code: int = 0,
    stderr: str = '',
    results: list[dict[str, str]] | None = None,
    result_content: dict[str, Any] | None = None,
) -> dict[str, Any] | str:
    """Create mock content for BuiltinToolReturnPart based on tool type.

    Args:
        tool_type: The type of builtin tool
        tool_name: Name of the tool
        tool_arguments: Arguments passed to the tool
        output: For code_execution - stdout output
        return_code: For code_execution - exit code
        stderr: For code_execution - stderr output
        results: For web_search - search results
        result_content: For MCP - result content dict

    Returns:
        Content dict or string for the BuiltinToolReturnPart
    """
    if tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL:
        return {
            'output': output or f'Code execution result for: {tool_arguments.get("code", "")}',
            'return_code': return_code,
            'stderr': stderr,
        }
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL:
        return {
            'status': 'completed',
            'results': results or [],
        }
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL:
        return result_content or {
            'result': {
                'content': [{'type': 'text', 'text': f'MCP tool {tool_name} executed successfully'}],
            },
        }
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_DOCUMENT_SEARCH_TOOL:
        return {
            'documents': [
                {
                    'id': 'doc_1',
                    'title': 'Sample Document',
                    'content': 'Sample document content',
                    'url': 'https://example.com/doc1',
                }
            ],
        }
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_COLLECTIONS_SEARCH_TOOL:
        return {
            'results': [
                {
                    'id': 'collection_1',
                    'name': 'Sample Collection',
                    'description': 'Sample collection description',
                }
            ],
        }
    else:
        return {'status': 'completed'}


def create_code_execution_responses(
    code: str,
    output: str,
    *,
    return_code: int = 0,
    stderr: str = '',
    text_content: str = '',
    tool_call_id: str = 'code_exec_001',
) -> chat_types.Response:
    """Create a Response with code execution tool outputs.

    Args:
        code: The Python code that was executed
        output: The stdout output from code execution
        return_code: Exit code (0 = success)
        stderr: Any stderr output
        text_content: Text response from the model after code execution
        tool_call_id: Unique ID for this tool call

    Returns:
        Response configured with outputs for code execution

    Example:
        >>> response = create_code_execution_responses(
        ...     code='print(2 + 2)',
        ...     output='4',
        ...     text_content='The answer is 4.',
        ... )
    """
    tool_call = create_server_tool_call(
        tool_name='code_execution',
        arguments={'code': code},
        tool_call_id=tool_call_id,
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    )

    return_content = _create_builtin_tool_return_content(
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
        tool_name='code_execution',
        tool_arguments={'code': code},
        output=output,
        return_code=return_code,
        stderr=stderr,
    )

    # Output 0: Tool call
    tool_call_message = chat_pb2.CompletionMessage(role=chat_pb2.MessageRole.ROLE_ASSISTANT)
    tool_call_message.tool_calls.append(tool_call)
    tool_call_output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=sample_pb2.FinishReason.REASON_TOOL_CALLS,
        message=tool_call_message,
    )

    # Output 1: Tool result + text content
    if text_content:
        response_content = text_content
    else:
        response_content = json.dumps(return_content) if isinstance(return_content, dict) else return_content

    tool_result_output = chat_pb2.CompletionOutput(
        index=1,
        finish_reason=sample_pb2.FinishReason.REASON_STOP,
        message=chat_pb2.CompletionMessage(
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            content=response_content,
        ),
    )

    return _build_response_with_outputs(
        response_id=f'grok-{tool_call_id}',
        outputs=[tool_call_output, tool_result_output],
    )


def create_code_execution_response(
    code: str,
    output: str,
    *,
    return_code: int = 0,
    stderr: str = '',
    text_content: str = '',
    tool_call_id: str = 'code_exec_001',
) -> chat_types.Response:
    """Create a Response with code execution builtin tool.

    This mimics xAI's server-side code execution tool that runs Python code
    in a sandboxed environment.

    Args:
        code: The Python code that was executed
        output: The stdout output from code execution
        return_code: Exit code (0 = success)
        stderr: Any stderr output
        text_content: Text response from the model after code execution
        tool_call_id: Unique ID for this tool call

    Returns:
        Response configured as a code execution response

    Example:
        >>> response = create_code_execution_response(
        ...     code='print(2 + 2)',
        ...     output='4',
        ...     text_content='The answer is 4.',
        ... )
    """
    tool_call = create_server_tool_call(
        tool_name='code_execution',
        arguments={'code': code},
        tool_call_id=tool_call_id,
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    )

    if text_content:
        response_content = text_content
    else:
        return_content = _create_builtin_tool_return_content(
            tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
            tool_name='code_execution',
            tool_arguments={'code': code},
            output=output,
            return_code=return_code,
            stderr=stderr,
        )
        response_content = json.dumps(return_content) if isinstance(return_content, dict) else return_content

    message = chat_pb2.CompletionMessage(
        role=chat_pb2.MessageRole.ROLE_ASSISTANT,
        content=response_content,
    )
    message.tool_calls.append(tool_call)

    output_proto = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=sample_pb2.FinishReason.REASON_STOP,
        message=message,
    )

    return _build_response_with_outputs(
        response_id=f'grok-{tool_call_id}',
        outputs=[output_proto],
    )


def create_web_search_responses(
    query: str,
    results: list[dict[str, str]],
    *,
    text_content: str = '',
    tool_call_id: str = 'web_search_001',
) -> chat_types.Response:
    """Create a Response with web search tool outputs.

    Args:
        query: The search query
        results: List of search results, each with 'title', 'url', 'snippet'
        text_content: Text response from the model after search
        tool_call_id: Unique ID for this tool call

    Returns:
        Response configured with outputs for web search

    Example:
        >>> response = create_web_search_responses(
        ...     query='current weather',
        ...     results=[{'title': 'Weather.com', 'url': 'https://...', 'snippet': 'Sunny, 75°F'}],
        ...     text_content='The weather is sunny and 75°F.',
        ... )
    """
    tool_call = create_server_tool_call(
        tool_name='web_search',
        arguments={'query': query},
        tool_call_id=tool_call_id,
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    )

    return_content = _create_builtin_tool_return_content(
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
        tool_name='web_search',
        tool_arguments={'query': query},
        results=results,
    )

    # Output 0: Tool call
    tool_call_message = chat_pb2.CompletionMessage(role=chat_pb2.MessageRole.ROLE_ASSISTANT)
    tool_call_message.tool_calls.append(tool_call)
    tool_call_output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=sample_pb2.FinishReason.REASON_TOOL_CALLS,
        message=tool_call_message,
    )

    # Output 1: Tool result + text content
    if text_content:
        response_content = text_content
    else:
        response_content = json.dumps(return_content) if isinstance(return_content, dict) else return_content

    tool_result_output = chat_pb2.CompletionOutput(
        index=1,
        finish_reason=sample_pb2.FinishReason.REASON_STOP,
        message=chat_pb2.CompletionMessage(
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            content=response_content,
        ),
    )

    return _build_response_with_outputs(
        response_id=f'grok-{tool_call_id}',
        outputs=[tool_call_output, tool_result_output],
    )


def create_mcp_server_responses(
    server_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    text_content: str = '',
    tool_call_id: str = 'mcp_001',
    result_content: dict[str, Any] | None = None,
) -> chat_types.Response:
    """Create a Response with MCP server tool outputs.

    Args:
        server_id: ID of the MCP server (e.g., 'linear', 'github')
        tool_name: Name of the tool on the MCP server
        tool_input: Input parameters for the tool
        text_content: Text response from the model after tool execution
        tool_call_id: Unique ID for this tool call
        result_content: Optional custom result content dict

    Returns:
        Response configured with outputs for MCP server tool

    Example:
        >>> response = create_mcp_server_responses(
        ...     server_id='linear',
        ...     tool_name='list_issues',
        ...     tool_input={'status': 'open'},
        ...     text_content='Here are your open issues.',
        ... )
    """
    # For MCP, the tool name is prefixed with the server_id (e.g., "linear.list_issues")
    full_tool_name = f'{server_id}.{tool_name}'

    tool_call = create_server_tool_call(
        tool_name=full_tool_name,
        arguments=tool_input,
        tool_call_id=tool_call_id,
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    )

    return_content = _create_builtin_tool_return_content(
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL,
        tool_name=full_tool_name,
        tool_arguments=tool_input,
        result_content=result_content,
    )

    # Output 0: Tool call
    tool_call_message = chat_pb2.CompletionMessage(role=chat_pb2.MessageRole.ROLE_ASSISTANT)
    tool_call_message.tool_calls.append(tool_call)
    tool_call_output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=sample_pb2.FinishReason.REASON_TOOL_CALLS,
        message=tool_call_message,
    )

    # Output 1: Tool result + text content
    if text_content:
        response_content = text_content
    else:
        response_content = json.dumps(return_content) if isinstance(return_content, dict) else return_content

    tool_result_output = chat_pb2.CompletionOutput(
        index=1,
        finish_reason=sample_pb2.FinishReason.REASON_STOP,
        message=chat_pb2.CompletionMessage(
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            content=response_content,
        ),
    )

    return _build_response_with_outputs(
        response_id=f'grok-{tool_call_id}',
        outputs=[tool_call_output, tool_result_output],
    )


def create_mixed_tools_response(
    server_tools: list[chat_pb2.ToolCall],
    text_content: str = '',
) -> chat_types.Response:
    """Create a response with multiple server-side tool calls.

    Useful for testing scenarios where multiple builtin tools are called in sequence.
    Creates two outputs: one for tool calls (REASON_TOOL_CALLS) and one for results (REASON_STOP).

    Args:
        server_tools: List of chat_pb2.ToolCall proto objects
        text_content: Text response after all tools are executed

    Returns:
        Response with multiple builtin tool calls

    Example:
        >>> tools = [
        ...     create_server_tool_call('web_search', {'query': 'bitcoin price'},
        ...                            tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL),
        ...     create_server_tool_call('code_execution', {'code': 'x=50000; y=65000; (y-x)/x*100'},
        ...                            tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL),
        ... ]
        >>> response = create_mixed_tools_response(tools, 'Bitcoin increased by 30%')
    """
    # Output 0: Tool calls
    tool_call_message = chat_pb2.CompletionMessage(role=chat_pb2.MessageRole.ROLE_ASSISTANT)
    tool_call_message.tool_calls.extend(server_tools)
    tool_call_output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=sample_pb2.FinishReason.REASON_TOOL_CALLS,
        message=tool_call_message,
    )

    # Output 1: Tool results + text content
    tool_result_output = chat_pb2.CompletionOutput(
        index=1,
        finish_reason=sample_pb2.FinishReason.REASON_STOP,
        message=chat_pb2.CompletionMessage(
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            content=text_content,
        ),
    )

    return _build_response_with_outputs(
        response_id='grok-multi-tool',
        outputs=[tool_call_output, tool_result_output],
    )


def create_multi_turn_builtin_sequence(
    *responses: chat_types.Response,
) -> Sequence[chat_types.Response]:
    """Create a sequence of responses for multi-turn builtin tool interactions.

    This is useful for testing conversations where multiple builtin tools are called
    across different turns or where the same tool is called multiple times.

    Args:
        *responses: Variable number of Response objects

    Returns:
        Sequence of responses suitable for MockXai.create_mock()

    Example:
        >>> response1 = create_code_execution_responses(code='x = 2+2', output='4')
        >>> responses = create_multi_turn_builtin_sequence(
        ...     response1,
        ...     create_response(content='The calculation is complete.'),
        ... )
        >>> mock_client = MockXai.create_mock(responses)
    """
    return list(responses)
