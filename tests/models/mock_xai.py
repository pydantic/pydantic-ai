"""Mock xAI SDK for testing without live API calls.

Since xAI uses gRPC, we cannot use VCR for recording. This module provides
mock objects that create real xAI SDK proto objects for accurate testing.
"""

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

# Type aliases
ToolCallArgumentsType = dict[str, Any]
ToolCallOutputType = dict[str, Any] | list[dict[str, Any]] | str


def _serialize_content(content: ToolCallOutputType) -> str:
    """Serialize content to JSON string if not already a string."""
    return content if isinstance(content, str) else json.dumps(content)


def _get_proto_finish_reason(finish_reason: FinishReason) -> sample_pb2.FinishReason:
    """Map pydantic-ai FinishReason to xAI proto FinishReason."""
    return {
        'stop': sample_pb2.FinishReason.REASON_STOP,
        'length': sample_pb2.FinishReason.REASON_MAX_LEN,
        'tool_call': sample_pb2.FinishReason.REASON_TOOL_CALLS,
        'content_filter': sample_pb2.FinishReason.REASON_STOP,
    }.get(finish_reason, sample_pb2.FinishReason.REASON_STOP)


def _build_response_with_outputs(
    response_id: str,
    outputs: list[chat_pb2.CompletionOutput],
    usage: Any | None = None,
) -> chat_types.Response:
    """Build a Response from outputs."""
    proto = chat_pb2.GetChatCompletionResponse(id=response_id, outputs=outputs, usage=usage)
    proto.created.GetCurrentTime()
    return chat_types.Response(proto, index=None)


@dataclass
class MockXai:
    """Mock xAI SDK AsyncClient."""

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
# Response Builders
# =============================================================================


def create_response(
    content: str = '',
    tool_calls: list[chat_pb2.ToolCall] | None = None,
    finish_reason: FinishReason = 'stop',
    usage: Any | None = None,
    reasoning_content: str = '',
    encrypted_content: str = '',
) -> chat_types.Response:
    """Create a Response with a single output."""
    output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=_get_proto_finish_reason(finish_reason),
        message=chat_pb2.CompletionMessage(
            content=content,
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            reasoning_content=reasoning_content,
            encrypted_content=encrypted_content,
            tool_calls=tool_calls or [],
        ),
    )

    return _build_response_with_outputs('grok-123', [output], usage)


def create_tool_call(
    id: str,
    name: str,
    arguments: ToolCallArgumentsType,
) -> chat_pb2.ToolCall:
    """Create a client-side ToolCall proto."""
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
    arguments: ToolCallArgumentsType,
    *,
    tool_call_id: str = 'server_tool_001',
    tool_type: chat_pb2.ToolCallType = chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
    status: chat_pb2.ToolCallStatus = chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    error_message: str = '',
) -> chat_pb2.ToolCall:
    """Create a server-side (builtin) ToolCall proto."""
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
    """Create a streaming Chunk."""
    output_chunk = chat_pb2.CompletionOutputChunk(
        index=index,
        delta=chat_pb2.Delta(
            content=content,
            reasoning_content=reasoning_content,
            encrypted_content=encrypted_content,
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            tool_calls=tool_calls or [],
        ),
    )
    if finish_reason:
        output_chunk.finish_reason = _get_proto_finish_reason(finish_reason)

    proto = chat_pb2.GetChatCompletionChunk(id='grok-123')
    proto.outputs.append(output_chunk)
    proto.created.GetCurrentTime()
    return chat_types.Chunk(proto, index=None)


# =============================================================================
# Builtin Tool Helpers
# =============================================================================


def _get_tool_content(
    tool_type: chat_pb2.ToolCallType,
    content: ToolCallOutputType | None = None,
) -> ToolCallOutputType:
    """Return content if provided, otherwise a realistic default for the tool type."""
    if content is not None:
        return content
    if tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL:
        return {'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''}
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL:
        return ''  # Web search has no content currently
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL:
        return [
            {
                'id': 'issue_001',
                'identifier': 'PROJ-123',
                'title': 'example-issue',
                'description': 'example-issue description',
                'status': 'Todo',
                'priority': {'value': 3, 'name': 'Medium'},
                'url': 'https://linear.app/team/issue/PROJ-123/example-issue',
            }
        ]
    else:
        return {'status': 'completed'}


def _create_builtin_tool_response(
    tool_name: str,
    arguments: ToolCallArgumentsType,
    content: ToolCallOutputType,
    tool_call_id: str,
    tool_type: chat_pb2.ToolCallType,
) -> chat_types.Response:
    """Create a Response with builtin tool outputs (shared helper)."""
    in_progress_output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=sample_pb2.FinishReason.REASON_TOOL_CALLS,
        message=chat_pb2.CompletionMessage(
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            tool_calls=[
                create_server_tool_call(
                    tool_name,
                    arguments,
                    tool_call_id=tool_call_id,
                    tool_type=tool_type,
                    status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_IN_PROGRESS,
                )
            ],
        ),
    )
    completed_output = chat_pb2.CompletionOutput(
        index=1,
        finish_reason=sample_pb2.FinishReason.REASON_STOP,
        message=chat_pb2.CompletionMessage(
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            content=_serialize_content(content),
            tool_calls=[
                create_server_tool_call(
                    tool_name,
                    arguments,
                    tool_call_id=tool_call_id,
                    tool_type=tool_type,
                    status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
                )
            ],
        ),
    )

    return _build_response_with_outputs(
        response_id=f'grok-{tool_call_id}',
        outputs=[in_progress_output, completed_output],
    )


def create_code_execution_responses(
    code: str,
    content: ToolCallOutputType | None = None,
    *,
    tool_call_id: str = 'code_exec_001',
) -> chat_types.Response:
    """Create a Response with code execution tool outputs.

    Example:
        >>> response = create_code_execution_responses(
        ...     code='2 + 2',
        ...     content={'stdout': '4\\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
        ... )
    """
    tool_type = chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL
    actual_content = _get_tool_content(tool_type, content)
    return _create_builtin_tool_response(
        tool_name='code_execution',
        arguments={'code': code},
        content=actual_content,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
    )


def create_web_search_responses(
    query: str,
    content: ToolCallOutputType | None = None,
    *,
    tool_call_id: str = 'web_search_001',
) -> chat_types.Response:
    """Create a Response with web search tool outputs.

    Example:
        >>> response = create_web_search_responses(query='date of Jan 1 in 2026', content='Thursday')
    """
    tool_type = chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL
    actual_content = _get_tool_content(tool_type, content)
    return _create_builtin_tool_response(
        tool_name='web_search',
        arguments={'query': query},
        content=actual_content,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
    )


def create_mcp_server_responses(
    server_id: str,
    tool_name: str,
    content: ToolCallOutputType | None = None,
    *,
    tool_input: ToolCallArgumentsType | None = None,
    tool_call_id: str = 'mcp_001',
) -> chat_types.Response:
    """Create a Response with MCP server tool outputs.

    Example:
        >>> response = create_mcp_server_responses(
        ...     server_id='linear', tool_name='list_issues', content=[{'id': 'issue_001'}]
        ... )
    """
    full_tool_name = f'{server_id}.{tool_name}'
    tool_type = chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL
    actual_content = _get_tool_content(tool_type, content)
    return _create_builtin_tool_response(
        tool_name=full_tool_name,
        arguments=tool_input or {},
        content=actual_content,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
    )


def create_mixed_tools_response(
    server_tools: list[chat_pb2.ToolCall],
    text_content: str = '',
) -> chat_types.Response:
    """Create a response with multiple server-side tool calls."""
    tool_call_output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=sample_pb2.FinishReason.REASON_TOOL_CALLS,
        message=chat_pb2.CompletionMessage(role=chat_pb2.MessageRole.ROLE_ASSISTANT, tool_calls=server_tools),
    )
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
