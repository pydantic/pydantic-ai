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
    from google.protobuf.json_format import MessageToDict
    from xai_sdk import AsyncClient
    from xai_sdk.proto import chat_pb2, sample_pb2, usage_pb2

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

    responses: Sequence[chat_types.Response | Exception] | None = None
    stream_data: Sequence[Sequence[tuple[chat_types.Response, Any]]] | None = None
    index: int = 0
    chat_create_kwargs: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
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
        responses: Sequence[chat_types.Response | Exception],
        api_key: str = 'test-api-key',
    ) -> AsyncClient:
        """Create a mock AsyncClient for non-streaming responses."""
        return cast(AsyncClient, cls(responses=responses, api_key=api_key))

    @classmethod
    def create_mock_stream(
        cls,
        stream: Sequence[Sequence[tuple[chat_types.Response, Any]]],
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

    responses: Sequence[chat_types.Response | Exception] | None = None
    stream_data: Sequence[Sequence[tuple[chat_types.Response, Any]]] | None = None
    index: int = 0
    parent: MockXai = field(default_factory=lambda: MockXai())

    async def sample(self) -> chat_types.Response:
        """Mock the sample() method for non-streaming responses."""
        assert self.responses is not None, 'you can only use sample() if responses are provided'

        if self.index >= len(self.responses):
            raise IndexError(f'Mock response index {self.index} out of range (length: {len(self.responses)})')
        raise_if_exception(self.responses[self.index])
        response = cast(chat_types.Response, self.responses[self.index])
        # Increment index for next call
        self.index += 1
        self.parent.index = self.index

        return response

    def stream(self) -> MockAsyncStream[tuple[chat_types.Response, Any]]:
        """Mock the stream() method for streaming responses."""
        assert self.stream_data is not None, 'you can only use stream() if stream_data is provided'

        data = list(self.stream_data[self.index])
        self.parent.index += 1

        return MockAsyncStream(iter(data))


def get_mock_chat_create_kwargs(async_client: AsyncClient) -> list[dict[str, Any]]:
    """Extract the kwargs passed to chat.create from a mock client.

    Messages, tools, and response_format are automatically converted from protobuf to dicts for easier testing.
    """
    if isinstance(async_client, MockXai):
        result: list[dict[str, Any]] = []
        for kwargs in async_client.chat_create_kwargs:
            kwargs_copy: dict[str, Any] = dict(kwargs)
            if 'messages' in kwargs_copy:  # pragma: no branch
                kwargs_copy['messages'] = [
                    MessageToDict(msg, preserving_proto_field_name=True) for msg in kwargs_copy['messages']
                ]
            if 'tools' in kwargs_copy and kwargs_copy['tools'] is not None:
                kwargs_copy['tools'] = [
                    MessageToDict(tool, preserving_proto_field_name=True) for tool in kwargs_copy['tools']
                ]
            if 'response_format' in kwargs_copy and kwargs_copy['response_format'] is not None:
                kwargs_copy['response_format'] = MessageToDict(
                    kwargs_copy['response_format'], preserving_proto_field_name=True
                )
            result.append(kwargs_copy)
        return result
    else:  # pragma: no cover
        raise RuntimeError('Not a MockXai instance')


# =============================================================================
# Response Builders
# =============================================================================


def create_logprob(
    token: str,
    logprob: float,
    top_logprobs: list[chat_pb2.TopLogProb] | None = None,
) -> chat_pb2.LogProb:
    """Create a LogProb proto.

    Args:
        token: The token string.
        logprob: The log probability value.
        top_logprobs: Optional list of top log probabilities.

    Example:
        >>> logprob = create_logprob('Hello', -0.5)
    """
    return chat_pb2.LogProb(
        token=token,
        logprob=logprob,
        bytes=token.encode('utf-8'),
        top_logprobs=top_logprobs or [],
    )


def create_response(
    content: str = '',
    tool_calls: list[chat_pb2.ToolCall] | None = None,
    finish_reason: FinishReason = 'stop',
    usage: Any | None = None,
    reasoning_content: str = '',
    encrypted_content: str = '',
    logprobs: list[chat_pb2.LogProb] | None = None,
    index: int = 0,
) -> chat_types.Response:
    """Create a Response with a single output."""
    output = chat_pb2.CompletionOutput(
        index=index,
        finish_reason=_get_proto_finish_reason(finish_reason),
        message=chat_pb2.CompletionMessage(
            content=content,
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            reasoning_content=reasoning_content,
            encrypted_content=encrypted_content,
            tool_calls=tool_calls or [],
        ),
    )

    if logprobs is not None:
        output.logprobs.CopyFrom(chat_pb2.LogProbs(content=logprobs))

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
    tool_type: chat_pb2.ToolCallType | None = None,
    status: chat_pb2.ToolCallStatus | None = None,
    error_message: str = '',
) -> chat_pb2.ToolCall:
    """Create a server-side (builtin) ToolCall proto."""
    if tool_type is None:
        tool_type = chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL
    if status is None:
        status = chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED
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
    role: chat_pb2.MessageRole | None = None,
    finish_reason: FinishReason | None = None,
    index: int = 0,
) -> chat_types.Chunk:
    """Create a streaming Chunk."""
    if role is None:
        role = chat_pb2.MessageRole.ROLE_ASSISTANT
    output_chunk = chat_pb2.CompletionOutputChunk(
        index=index,
        delta=chat_pb2.Delta(
            content=content,
            reasoning_content=reasoning_content,
            encrypted_content=encrypted_content,
            role=role,
            tool_calls=tool_calls or [],
        ),
    )
    if finish_reason:
        output_chunk.finish_reason = _get_proto_finish_reason(finish_reason)

    proto = chat_pb2.GetChatCompletionChunk(id='grok-123')
    proto.outputs.append(output_chunk)
    proto.created.GetCurrentTime()
    return chat_types.Chunk(proto, index=None)


def get_grok_tool_chunk(
    tool_name: str | None,
    tool_arguments: str | None,
    finish_reason: str = '',
    accumulated_args: str = '',
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a client-side tool-call streaming chunk for Grok.

    This is used by xAI model streaming tests where:
    - `Chunk` contains the per-frame delta
    - `Response` contains the accumulated view (including accumulated tool args when available)

    Note: Unlike the real xAI SDK (which may only send the tool name in the first chunk),
    this helper includes the effective tool name in every chunk to simplify test tracking.
    """
    # Infer tool name from accumulated state if not provided
    effective_tool_name = tool_name or ('final_result' if accumulated_args else None)

    # Create the chunk tool call (delta)
    chunk_tool_calls: list[chat_pb2.ToolCall] = []
    if effective_tool_name is not None or tool_arguments is not None:
        chunk_tool_calls = [
            chat_pb2.ToolCall(
                id='tool-123',
                type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                function=chat_pb2.FunctionCall(
                    name=effective_tool_name or '',
                    arguments=tool_arguments if tool_arguments is not None else '',
                ),
            )
        ]

    chunk = create_stream_chunk(
        content='',
        tool_calls=chunk_tool_calls,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Create response tool calls (accumulated view)
    response_tool_calls: list[chat_pb2.ToolCall] = []
    if effective_tool_name is not None or accumulated_args:
        response_tool_calls = [
            chat_pb2.ToolCall(
                id='tool-123',
                type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                function=chat_pb2.FunctionCall(
                    name=effective_tool_name or '',
                    arguments=accumulated_args,
                ),
            )
        ]

    usage = usage_pb2.SamplingUsage(prompt_tokens=20, completion_tokens=1) if finish_reason else None
    response = create_response(
        content='',
        tool_calls=response_tool_calls,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=usage,
    )

    return (response, chunk)


def get_grok_text_chunk(text: str, finish_reason: str = 'stop') -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a text streaming chunk for Grok.

    Note: For streaming, `Response` is the accumulated view and `Chunk` is the delta. For simplicity in mocks,
    we set `response.content` to the same value as the chunk delta.
    """
    chunk = create_stream_chunk(
        content=text,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    usage = usage_pb2.SamplingUsage(prompt_tokens=2, completion_tokens=1) if finish_reason else None
    response = create_response(
        content=text,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=usage,
    )

    return (response, chunk)


def get_grok_reasoning_text_chunk(
    text: str,
    reasoning_content: str = '',
    encrypted_content: str = '',
    finish_reason: str = 'stop',
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a text streaming chunk for Grok with reasoning content/signature."""
    chunk = create_stream_chunk(
        content=text,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    usage = usage_pb2.SamplingUsage(prompt_tokens=2, completion_tokens=1) if finish_reason else None
    response = create_response(
        content=text,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=usage,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
    )

    return (response, chunk)


# =============================================================================
# Builtin Tool Helpers
# =============================================================================


def _get_example_tool_output(
    tool_type: chat_pb2.ToolCallType,
    content: ToolCallOutputType | None = None,
) -> ToolCallOutputType:
    """Return content if provided, otherwise a realistic default for the tool type."""
    if content is not None:
        return content
    if tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL:
        return {'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''}
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL:
        return {}  # Web search has no content currently, in future will return inline citations
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
    else:  # pragma: no cover
        # Unknown tool type - return empty dict as fallback
        return {}


def _create_builtin_tool_outputs(
    tool_name: str,
    arguments: ToolCallArgumentsType,
    content: ToolCallOutputType,
    tool_call_id: str,
    tool_type: chat_pb2.ToolCallType,
    initial_status: chat_pb2.ToolCallStatus,
) -> list[chat_pb2.CompletionOutput]:
    """Create CompletionOutputs for builtin tool call and result (shared helper).

    Returns a list of outputs representing the tool call in progress and completed.
    Callers should add a final assistant message output to match real API behavior.
    """
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
                    status=initial_status,
                )
            ],
        ),
    )
    tool_result_output = chat_pb2.CompletionOutput(
        index=1,
        finish_reason=sample_pb2.FinishReason.REASON_TOOL_CALLS,
        message=chat_pb2.CompletionMessage(
            role=chat_pb2.MessageRole.ROLE_TOOL,
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
    return [in_progress_output, tool_result_output]


def create_failed_builtin_tool_response(
    tool_name: str,
    tool_type: chat_pb2.ToolCallType,
    *,
    tool_call_id: str = 'failed_tool_001',
    error_message: str = 'tool failed',
    content: ToolCallOutputType | None = None,
) -> chat_types.Response:
    """Create a Response representing a failed builtin tool call."""
    output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=sample_pb2.FinishReason.REASON_STOP,
        message=chat_pb2.CompletionMessage(
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            content=_serialize_content(content or ''),
            tool_calls=[
                create_server_tool_call(
                    tool_name,
                    {},
                    tool_call_id=tool_call_id,
                    tool_type=tool_type,
                    status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_FAILED,
                    error_message=error_message,
                )
            ],
        ),
    )

    return _build_response_with_outputs(
        response_id=f'grok-{tool_call_id}',
        outputs=[output],
    )


def create_code_execution_response(
    code: str,
    content: ToolCallOutputType | None = None,
    *,
    tool_call_id: str = 'code_exec_001',
    assistant_text: str,
) -> chat_types.Response:
    """Create a Response with code execution tool outputs.

    Args:
        assistant_text: Text for the final assistant message (required to match real API).

    Example:
        >>> response = create_code_execution_response(
        ...     code='2 + 2',
        ...     content={'stdout': '4\\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
        ...     assistant_text='The result is 4.',
        ... )
    """
    tool_type = chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL
    actual_content = _get_example_tool_output(tool_type, content)
    outputs = _create_builtin_tool_outputs(
        tool_name='code_execution',
        arguments={'code': code},
        content=actual_content,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
        initial_status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    )
    # Add final assistant message (matching real API behavior)
    outputs.append(
        chat_pb2.CompletionOutput(
            index=len(outputs),
            finish_reason=sample_pb2.FinishReason.REASON_STOP,
            message=chat_pb2.CompletionMessage(
                role=chat_pb2.MessageRole.ROLE_ASSISTANT,
                content=assistant_text,
            ),
        )
    )
    return _build_response_with_outputs(response_id=f'grok-{tool_call_id}', outputs=outputs)


def create_web_search_response(
    query: str,
    content: ToolCallOutputType | None = None,
    *,
    tool_call_id: str = 'web_search_001',
    assistant_text: str,
) -> chat_types.Response:
    """Create a Response with web search tool outputs.

    Args:
        assistant_text: Text for the final assistant message (required to match real API).

    Example:
        >>> response = create_web_search_response(
        ...     query='date of Jan 1 in 2026',
        ...     content='Thursday',
        ...     assistant_text='January 1, 2026 is a Thursday.',
        ... )
    """
    tool_type = chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL
    actual_content = _get_example_tool_output(tool_type, content)
    outputs = _create_builtin_tool_outputs(
        tool_name='web_search',
        arguments={'query': query},
        content=actual_content,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
        initial_status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    )
    # Add final assistant message (matching real API behavior)
    outputs.append(
        chat_pb2.CompletionOutput(
            index=len(outputs),
            finish_reason=sample_pb2.FinishReason.REASON_STOP,
            message=chat_pb2.CompletionMessage(
                role=chat_pb2.MessageRole.ROLE_ASSISTANT,
                content=assistant_text,
            ),
        )
    )
    return _build_response_with_outputs(response_id=f'grok-{tool_call_id}', outputs=outputs)


def create_mcp_server_response(
    server_id: str,
    tool_name: str,
    tool_input: ToolCallArgumentsType | None = None,
    content: ToolCallOutputType | None = None,
    *,
    tool_call_id: str = 'mcp_001',
    assistant_text: str,
) -> chat_types.Response:
    """Create a Response with MCP server tool outputs.

    Args:
        assistant_text: Text for the final assistant message (required to match real API).

    Example:
        >>> response = create_mcp_server_response(
        ...     server_id='linear',
        ...     tool_name='list_issues',
        ...     content=[{'id': 'issue_001'}],
        ...     assistant_text='Found 1 issue.',
        ... )
    """
    full_tool_name = f'{server_id}.{tool_name}'
    tool_type = chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL
    actual_content = _get_example_tool_output(tool_type, content)
    outputs = _create_builtin_tool_outputs(
        tool_name=full_tool_name,
        arguments=tool_input or {},
        content=actual_content,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
        initial_status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    )
    # Add final assistant message (matching real API behavior)
    outputs.append(
        chat_pb2.CompletionOutput(
            index=len(outputs),
            finish_reason=sample_pb2.FinishReason.REASON_STOP,
            message=chat_pb2.CompletionMessage(
                role=chat_pb2.MessageRole.ROLE_ASSISTANT,
                content=assistant_text,
            ),
        )
    )
    return _build_response_with_outputs(response_id=f'grok-{tool_call_id}', outputs=outputs)


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


def create_response_with_tool_calls(
    content: str = '',
    tool_calls: list[chat_pb2.ToolCall] | None = None,
    finish_reason: FinishReason = 'stop',
    usage: Any | None = None,
) -> chat_types.Response:
    """Create a Response with specific tool calls for testing edge cases."""
    output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=_get_proto_finish_reason(finish_reason),
        message=chat_pb2.CompletionMessage(
            content=content,
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            tool_calls=tool_calls or [],
        ),
    )
    return _build_response_with_outputs('grok-123', [output], usage)


def create_response_without_usage(
    content: str = '',
    finish_reason: FinishReason | None = 'stop',
) -> chat_types.Response:
    """Create a Response without usage data for testing edge cases."""
    output = chat_pb2.CompletionOutput(
        index=0,
        finish_reason=_get_proto_finish_reason(finish_reason)
        if finish_reason
        else sample_pb2.FinishReason.REASON_INVALID,
        message=chat_pb2.CompletionMessage(
            content=content,
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
        ),
    )
    # Pass None for usage explicitly to get response without usage
    return _build_response_with_outputs('grok-123', [output], None)


def create_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_prompt_text_tokens: int = 0,
    server_side_tools_used: list[usage_pb2.ServerSideTool] | None = None,
) -> usage_pb2.SamplingUsage:
    """Helper to create xAI SamplingUsage protobuf objects for tests with all required fields."""
    return usage_pb2.SamplingUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_prompt_text_tokens=cached_prompt_text_tokens,
        server_side_tools_used=server_side_tools_used or [],
    )
