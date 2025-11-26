from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, cast

from ..conftest import raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    import xai_sdk.chat as chat_types
    from xai_sdk import AsyncClient
    from xai_sdk.proto.v6 import chat_pb2

    MockResponse = chat_types.Response | Exception
    # xai_sdk streaming returns tuples of (Response, chunk) where chunk type is not explicitly defined
    MockResponseChunk = tuple[chat_types.Response, Any] | Exception


@dataclass
class MockXai:
    """Mock for xAI SDK AsyncClient to simulate xAI API responses."""

    responses: MockResponse | Sequence[MockResponse] | None = None
    stream_data: Sequence[MockResponseChunk] | Sequence[Sequence[MockResponseChunk]] | None = None
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
        cls, responses: MockResponse | Sequence[MockResponse], api_key: str = 'test-api-key'
    ) -> AsyncClient:
        """Create a mock AsyncClient for non-streaming responses."""
        return cast(AsyncClient, cls(responses=responses, api_key=api_key))

    @classmethod
    def create_mock_stream(
        cls,
        stream: Sequence[MockResponseChunk] | Sequence[Sequence[MockResponseChunk]],
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

    responses: MockResponse | Sequence[MockResponse] | None = None
    stream_data: Sequence[MockResponseChunk] | Sequence[Sequence[MockResponseChunk]] | None = None
    index: int = 0
    parent: MockXai | None = None

    async def sample(self) -> chat_types.Response:
        """Mock the sample() method for non-streaming responses."""
        assert self.responses is not None, 'you can only use sample() if responses are provided'

        if isinstance(self.responses, Sequence):
            raise_if_exception(self.responses[self.index])
            response = cast(chat_types.Response, self.responses[self.index])
        else:
            raise_if_exception(self.responses)
            response = cast(chat_types.Response, self.responses)

        if self.parent:
            self.parent.index += 1

        return response

    def stream(self) -> MockAsyncStream[MockResponseChunk]:
        """Mock the stream() method for streaming responses."""
        assert self.stream_data is not None, 'you can only use stream() if stream_data is provided'

        # Check if we have nested sequences (multiple streams) vs single stream
        # We need to check if it's a list of tuples (single stream) vs list of lists (multiple streams)
        if isinstance(self.stream_data, list) and len(self.stream_data) > 0:
            first_item = self.stream_data[0]
            # If first item is a list (not a tuple), we have multiple streams
            if isinstance(first_item, list):
                data = cast(list[MockResponseChunk], self.stream_data[self.index])
            else:
                # Single stream - use the data as is
                data = cast(list[MockResponseChunk], self.stream_data)
        else:
            data = cast(list[MockResponseChunk], self.stream_data)

        if self.parent:
            self.parent.index += 1

        return MockAsyncStream(iter(data))


def get_mock_chat_create_kwargs(async_client: AsyncClient) -> list[dict[str, Any]]:
    """Extract the kwargs passed to chat.create from a mock client."""
    if isinstance(async_client, MockXai):
        return async_client.chat_create_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockXai instance')


@dataclass
class MockXaiResponse:
    """Mock Response object that mimics xai_sdk.chat.Response interface."""

    id: str = 'grok-123'
    content: str = ''
    tool_calls: list[Any] = field(default_factory=list)
    finish_reason: str = 'stop'
    usage: Any | None = None  # Would be usage_pb2.SamplingUsage in real xai_sdk
    reasoning_content: str = ''  # Human-readable reasoning trace
    encrypted_content: str = ''  # Encrypted reasoning signature
    created: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Note: The real xAI SDK usage object uses protobuf fields:
    # - prompt_tokens (not input_tokens)
    # - completion_tokens (not output_tokens)
    # - reasoning_tokens
    # - cached_prompt_text_tokens


@dataclass
class MockXaiToolCall:
    """Mock ToolCall object that mimics chat_pb2.ToolCall interface."""

    id: str
    function: Any  # Would be chat_pb2.Function with name and arguments


@dataclass
class MockXaiFunction:
    """Mock Function object for tool calls."""

    name: str
    arguments: dict[str, Any]


def create_response(
    content: str = '',
    tool_calls: list[Any] | None = None,
    finish_reason: str = 'stop',
    usage: Any | None = None,
    reasoning_content: str = '',
    encrypted_content: str = '',
) -> chat_types.Response:
    """Create a mock Response object for testing.

    Returns a MockXaiResponse that mimics the xai_sdk.chat.Response interface.
    """
    return cast(
        chat_types.Response,
        MockXaiResponse(
            id='grok-123',
            content=content,
            tool_calls=tool_calls or [],
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=reasoning_content,
            encrypted_content=encrypted_content,
        ),
    )


def create_tool_call(
    id: str,
    name: str,
    arguments: dict[str, Any],
) -> MockXaiToolCall:
    """Create a mock ToolCall object for testing.

    Returns a MockXaiToolCall that mimics the chat_pb2.ToolCall interface.
    """
    return MockXaiToolCall(
        id=id,
        function=MockXaiFunction(name=name, arguments=arguments),
    )


@dataclass
class MockXaiResponseChunk:
    """Mock response chunk for streaming."""

    content: str = ''
    tool_calls: list[Any] = field(default_factory=list)


def create_response_chunk(
    content: str = '',
    tool_calls: list[Any] | None = None,
) -> MockXaiResponseChunk:
    """Create a mock response chunk object for testing.

    Returns a MockXaiResponseChunk for streaming responses.
    """
    return MockXaiResponseChunk(
        content=content,
        tool_calls=tool_calls or [],
    )


# Builtin Tool Support
# ---------------------
# The following classes and functions support testing xAI's built-in (server-side) tools:
# - code_execution: Execute Python code server-side
# - web_search: Search the web for information
# - mcp_server: Connect to Model Context Protocol servers
#
# These tools are executed by xAI's infrastructure and return results that need to be
# included in the mock response. Since xAI uses gRPC (not HTTP), we cannot use VCR
# for recording. Instead, we hand-craft mocks similar to how Anthropic and Google
# handle their builtin tools.
#
# See XAI_MOCKING_DESIGN.md for detailed rationale and comparison with other providers.


@dataclass
class MockXaiServerToolCall:
    """Mock server-side tool call that mimics chat_pb2.ToolCall for builtin tools.

    This represents a tool call for xAI's server-side tools (code_execution, web_search, etc.).
    The key difference from client-side tools is the `type` field which determines how
    the tool is handled by the xAI model integration.
    """

    id: str
    # Mimics the protobuf enum value - needs to be non-zero for server-side tools
    # From xai_sdk chat_pb2.ToolCallType:
    #   0=INVALID, 1=CLIENT_SIDE_TOOL, 2=WEB_SEARCH_TOOL, 3=X_SEARCH_TOOL,
    #   4=CODE_EXECUTION_TOOL, 5=COLLECTIONS_SEARCH_TOOL, 6=MCP_TOOL,
    #   7=DOCUMENT_SEARCH_TOOL, 9=LOCATIONS_SEARCH_TOOL
    type: chat_pb2.ToolCallType
    function: Any  # MockXaiFunction with name and arguments

    def WhichOneof(self, field: str) -> str | None:
        """Mimic protobuf's WhichOneof method for compatibility with get_tool_call_type()."""
        # The xAI SDK's get_tool_call_type checks which oneof field is set
        # Return the type name based on the type value
        type_names = {
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_INVALID: 'invalid',
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL: 'client_side_tool',
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL: 'web_search_tool',
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_X_SEARCH_TOOL: 'x_search_tool',
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL: 'code_execution_tool',
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_COLLECTIONS_SEARCH_TOOL: 'collections_search_tool',
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL: 'mcp_tool',
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_DOCUMENT_SEARCH_TOOL: 'document_search_tool',
            chat_pb2.ToolCallType.TOOL_CALL_TYPE_LOCATIONS_SEARCH_TOOL: 'locations_search_tool',
        }
        return type_names.get(self.type, 'client_side_tool')


def create_server_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    tool_call_id: str = 'server_tool_001',
    tool_type: chat_pb2.ToolCallType = chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
) -> MockXaiServerToolCall:
    """Create a mock server-side tool call.

    Args:
        tool_name: Name of the builtin tool ('code_execution', 'web_search', etc.)
        arguments: Arguments for the tool call
        tool_call_id: Unique ID for this tool call
        tool_type: Protobuf enum value for tool type. Use chat_pb2.ToolCallType constants:
            - TOOL_CALL_TYPE_WEB_SEARCH_TOOL
            - TOOL_CALL_TYPE_CODE_EXECUTION_TOOL
            - TOOL_CALL_TYPE_MCP_TOOL

    Returns:
        MockXaiServerToolCall that will be recognized as a server-side tool

    Example:
        >>> tool_call = create_server_tool_call(
        ...     tool_name='code_execution',
        ...     arguments={'code': 'print(2+2)'},
        ...     tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL
        ... )
    """
    return MockXaiServerToolCall(
        id=tool_call_id,
        type=tool_type,
        function=MockXaiFunction(name=tool_name, arguments=arguments),
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
    """Create a mock response with code execution builtin tool.

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
        MockXaiResponse configured as a code execution response

    Example:
        >>> response = create_code_execution_response(
        ...     code='print(2 + 2)',
        ...     output='4',
        ...     text_content='The answer is 4.',
        ... )
    """
    # Create server-side tool call (type=TOOL_CALL_TYPE_CODE_EXECUTION_TOOL)
    tool_call = create_server_tool_call(
        tool_name='code_execution',
        arguments={'code': code},
        tool_call_id=tool_call_id,
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
    )

    return cast(
        chat_types.Response,
        MockXaiResponse(
            id=f'grok-{tool_call_id}',
            content=text_content,
            tool_calls=[tool_call],
            finish_reason='stop',
        ),
    )


def create_web_search_response(
    query: str,
    results: list[dict[str, str]],
    *,
    text_content: str = '',
    tool_call_id: str = 'web_search_001',
) -> chat_types.Response:
    """Create a mock response with web search builtin tool.

    This mimics xAI's server-side web search tool.

    Args:
        query: The search query
        results: List of search results, each with 'title', 'url', 'snippet'
        text_content: Text response from the model after search
        tool_call_id: Unique ID for this tool call

    Returns:
        MockXaiResponse configured as a web search response

    Example:
        >>> response = create_web_search_response(
        ...     query='current weather',
        ...     results=[
        ...         {'title': 'Weather.com', 'url': 'https://...', 'snippet': 'Sunny, 75°F'},
        ...     ],
        ...     text_content='The weather is sunny and 75°F.',
        ... )
    """
    tool_call = create_server_tool_call(
        tool_name='web_search',
        arguments={'query': query},
        tool_call_id=tool_call_id,
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
    )

    return cast(
        chat_types.Response,
        MockXaiResponse(
            id=f'grok-{tool_call_id}',
            content=text_content,
            tool_calls=[tool_call],
            finish_reason='stop',
        ),
    )


def create_mcp_server_response(
    server_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    text_content: str = '',
    tool_call_id: str = 'mcp_001',
) -> chat_types.Response:
    """Create a mock response with MCP server builtin tool.

    This mimics xAI's MCP (Model Context Protocol) server integration.

    Args:
        server_id: ID of the MCP server (e.g., 'linear', 'github')
        tool_name: Name of the tool on the MCP server
        tool_input: Input parameters for the tool
        text_content: Text response from the model after tool execution
        tool_call_id: Unique ID for this tool call

    Returns:
        MockXaiResponse configured as an MCP server response

    Example:
        >>> response = create_mcp_server_response(
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
    )

    return cast(
        chat_types.Response,
        MockXaiResponse(
            id=f'grok-{tool_call_id}',
            content=text_content,
            tool_calls=[tool_call],
            finish_reason='stop',
        ),
    )


def create_mixed_tools_response(
    server_tools: list[MockXaiServerToolCall],
    text_content: str = '',
) -> chat_types.Response:
    """Create a response with multiple server-side tool calls.

    Useful for testing scenarios where multiple builtin tools are called in sequence.

    Args:
        server_tools: List of server-side tool calls
        text_content: Text response after all tools are executed

    Returns:
        MockXaiResponse with multiple builtin tool calls

    Example:
        >>> tools = [
        ...     create_server_tool_call('web_search', {'query': 'bitcoin price'},
        ...                            tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL),
        ...     create_server_tool_call('code_execution', {'code': 'x=50000; y=65000; (y-x)/x*100'},
        ...                            tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL),
        ... ]
        >>> response = create_mixed_tools_response(tools, 'Bitcoin increased by 30%')
    """
    return cast(
        chat_types.Response,
        MockXaiResponse(
            id='grok-multi-tool',
            content=text_content,
            tool_calls=server_tools,
            finish_reason='stop',
        ),
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
        >>> responses = create_multi_turn_builtin_sequence(
        ...     create_code_execution_response(code='x = 2+2', output='4'),
        ...     create_response(content='The calculation is complete.'),
        ... )
        >>> mock_client = MockXai.create_mock(responses)
    """
    return list(responses)
