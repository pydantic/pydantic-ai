from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

from ..conftest import raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    import xai_sdk.chat as chat_types
    from xai_sdk import AsyncClient

    MockResponse = chat_types.Response | Exception
    # xai_sdk streaming returns tuples of (Response, chunk) where chunk type is not explicitly defined
    MockResponseChunk = tuple[chat_types.Response, Any] | Exception


@dataclass
class MockGrok:
    """Mock for xAI SDK AsyncClient to simulate Grok API responses."""

    responses: MockResponse | Sequence[MockResponse] | None = None
    stream_data: Sequence[MockResponseChunk] | Sequence[Sequence[MockResponseChunk]] | None = None
    index: int = 0
    chat_create_kwargs: list[dict[str, Any]] = field(default_factory=list)
    api_key: str = 'test-api-key'

    @cached_property
    def chat(self) -> Any:
        """Create mock chat interface."""
        return type('Chat', (), {'create': self.chat_create})

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


@dataclass
class MockChatInstance:
    """Mock for the chat instance returned by client.chat.create()."""

    responses: MockResponse | Sequence[MockResponse] | None = None
    stream_data: Sequence[MockResponseChunk] | Sequence[Sequence[MockResponseChunk]] | None = None
    index: int = 0
    parent: MockGrok | None = None

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
    if isinstance(async_client, MockGrok):
        return async_client.chat_create_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockGrok instance')


@dataclass
class MockGrokResponse:
    """Mock Response object that mimics xai_sdk.chat.Response interface."""

    id: str = 'grok-123'
    content: str = ''
    tool_calls: list[Any] = field(default_factory=list)
    finish_reason: str = 'stop'
    usage: Any | None = None  # Would be usage_pb2.SamplingUsage in real xai_sdk


@dataclass
class MockGrokToolCall:
    """Mock ToolCall object that mimics chat_pb2.ToolCall interface."""

    id: str
    function: Any  # Would be chat_pb2.Function with name and arguments


@dataclass
class MockGrokFunction:
    """Mock Function object for tool calls."""

    name: str
    arguments: dict[str, Any]


def create_response(
    content: str = '',
    tool_calls: list[Any] | None = None,
    finish_reason: str = 'stop',
    usage: Any | None = None,
) -> MockGrokResponse:
    """Create a mock Response object for testing.

    Returns a MockGrokResponse that mimics the xai_sdk.chat.Response interface.
    """
    return MockGrokResponse(
        id='grok-123',
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=usage,
    )


def create_tool_call(
    id: str,
    name: str,
    arguments: dict[str, Any],
) -> MockGrokToolCall:
    """Create a mock ToolCall object for testing.

    Returns a MockGrokToolCall that mimics the chat_pb2.ToolCall interface.
    """
    return MockGrokToolCall(
        id=id,
        function=MockGrokFunction(name=name, arguments=arguments),
    )


@dataclass
class MockGrokResponseChunk:
    """Mock response chunk for streaming."""

    content: str = ''
    tool_calls: list[Any] = field(default_factory=list)


def create_response_chunk(
    content: str = '',
    tool_calls: list[Any] | None = None,
) -> MockGrokResponseChunk:
    """Create a mock response chunk object for testing.

    Returns a MockGrokResponseChunk for streaming responses.
    """
    return MockGrokResponseChunk(
        content=content,
        tool_calls=tool_calls or [],
    )
