"""Tests for the synchronous streaming functionality in the direct module."""

import asyncio
import re
import threading
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pydantic_ai.direct import model_request_stream_sync
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import StreamedResponseSync


def test_model_request_stream_sync_basic():
    """Test basic synchronous streaming functionality."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello, world!')]

    with model_request_stream_sync('test', messages) as stream:
        # Collect stream events
        events = list(stream)

        # Should have at least one event
        assert len(events) > 0

        # First event should be a PartStartEvent with TextPart
        assert isinstance(events[0], PartStartEvent)
        assert isinstance(events[0].part, TextPart)

        # Subsequent events should be PartDeltaEvents with TextPartDelta
        for event in events[1:]:
            assert isinstance(event, PartDeltaEvent)
            assert isinstance(event.delta, TextPartDelta)

        # Test that we can get the final response
        response = stream.get()
        assert response.parts
        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content

        # Test usage information
        usage = stream.usage()
        assert usage.total_tokens is not None and usage.total_tokens > 0

        # Test model name and timestamp
        assert stream.model_name == 'test'
        assert stream.timestamp is not None


def test_model_request_stream_sync_context_manager():
    """Test that the context manager works properly."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Test message')]

    # Test that context manager can be entered and exited
    with model_request_stream_sync('test', messages) as stream:
        # Should be able to access stream properties
        assert hasattr(stream, 'get')
        assert hasattr(stream, 'usage')
        assert hasattr(stream, 'model_name')
        assert hasattr(stream, 'timestamp')

        # Should be able to iterate
        events: list[PartStartEvent | PartDeltaEvent] = []
        for i, event in enumerate(stream):
            events.append(event)
            if i >= 2:  # Limit to avoid long runs
                break

        assert len(events) > 0


def test_model_request_stream_sync_early_get():
    """Test getting response before consuming all events."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Short message')]

    with model_request_stream_sync('test', messages) as stream:
        # Consume just the first event
        events: list[PartStartEvent | PartDeltaEvent] = []
        for i, event in enumerate(stream):
            events.append(event)
            if i >= 0:  # Just the first event
                break

        # Should still be able to get partial response
        response = stream.get()
        assert response is not None

        # Usage should be available
        usage = stream.usage()
        assert usage is not None


def test_model_request_stream_sync_error_handling():
    """Test error handling with invalid model."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Test')]

    # Should raise an error for unknown model
    with pytest.raises(Exception):
        with model_request_stream_sync('nonexistent-model', messages) as stream:
            list(stream)  # Try to consume the stream


def test_model_request_stream_sync_requires_context_manager():
    """Test that StreamedResponseSync enforces context manager usage."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Test')]

    # Test that when used properly with context manager, it works
    with model_request_stream_sync('test', messages) as stream:
        # These should work fine within the context manager
        assert hasattr(stream, 'get')
        assert hasattr(stream, 'usage')
        assert hasattr(stream, 'model_name')
        assert hasattr(stream, 'timestamp')

        # Try to consume just one event to ensure stream works
        events: list[PartStartEvent | PartDeltaEvent] = []
        for i, event in enumerate(stream):
            events.append(event)
            if i >= 0:  # Just first event
                break


def test_model_request_stream_sync_without_context_manager():
    """Test that accessing properties or iterating without context manager raises RuntimeError."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Test')]

    expected_error_msg = re.escape(
        'StreamedResponseSync must be used as a context manager. Use: `with model_request_stream_sync(...) as stream:`'
    )

    # Create stream object without entering context manager
    stream_cm = model_request_stream_sync('test', messages)

    # Test that accessing model_name raises RuntimeError
    with pytest.raises(RuntimeError, match=expected_error_msg):
        _ = stream_cm.model_name

    # Test that accessing timestamp raises RuntimeError
    with pytest.raises(RuntimeError, match=expected_error_msg):
        _ = stream_cm.timestamp

    # Test that calling get() raises RuntimeError
    with pytest.raises(RuntimeError, match=expected_error_msg):
        stream_cm.get()

    # Test that calling usage() raises RuntimeError
    with pytest.raises(RuntimeError, match=expected_error_msg):
        stream_cm.usage()

    # Test that iterating raises RuntimeError
    with pytest.raises(RuntimeError, match=expected_error_msg):
        list(stream_cm)

    # Test that manual iteration raises RuntimeError
    with pytest.raises(RuntimeError, match=expected_error_msg):
        for _ in stream_cm:
            break


def test_model_request_stream_sync_exception_in_stream():
    """Test handling of exceptions raised during streaming."""
    # Create a mock async stream that raises an exception
    async_stream_mock = AsyncMock()
    async_stream_mock.__aenter__ = AsyncMock(side_effect=ValueError('Stream error'))

    stream_sync = StreamedResponseSync(_async_stream_cm=async_stream_mock)

    # Test that exception is propagated when iterating
    with stream_sync:
        with pytest.raises(ValueError, match='Stream error'):
            list(stream_sync)


def test_model_request_stream_sync_timeout():
    """Test timeout when stream fails to initialize."""
    async_stream_mock = AsyncMock()

    # Mock a scenario where stream never becomes ready
    async def slow_init():
        await asyncio.sleep(0.1)

    async_stream_mock.__aenter__ = AsyncMock(side_effect=slow_init)

    stream_sync = StreamedResponseSync(_async_stream_cm=async_stream_mock)

    with patch('pydantic_ai.models.STREAM_INITIALIZATION_TIMEOUT', 0.01):
        with stream_sync:
            with pytest.raises(RuntimeError, match='Stream failed to initialize within timeout'):
                stream_sync.get()


def test_model_request_stream_sync_repr_no_stream():
    """Test __repr__ method for StreamedResponseSync when no stream is ready."""
    stream_sync = StreamedResponseSync(_async_stream_cm=Mock())
    repr_str = repr(stream_sync)
    assert 'StreamedResponseSync' in repr_str
    assert 'context_entered=False' in repr_str


def test_streamed_response_sync_start_producer_and_repr():
    mock_cm = Mock()
    srs = StreamedResponseSync(_async_stream_cm=mock_cm)
    srs._start_producer()  # pyright: ignore[reportPrivateUsage]
    t = srs._thread  # pyright: ignore[reportPrivateUsage]
    assert isinstance(t, threading.Thread)
    srs._start_producer()  # pyright: ignore[reportPrivateUsage]
    assert srs._thread is t  # pyright: ignore[reportPrivateUsage]
    r = repr(srs)
    assert r.startswith('StreamedResponseSync') and 'context_entered' in r


def test_model_request_stream_sync_stream_ready_failure():
    """Test stream initialization failure after timeout."""
    async_stream_mock = AsyncMock()

    # Mock a scenario where stream initialization sets ready but stream is still None
    async def mock_init():
        pass

    async_stream_mock.__aenter__ = AsyncMock(side_effect=mock_init)

    stream_sync = StreamedResponseSync(_async_stream_cm=async_stream_mock)

    with patch('pydantic_ai.models.STREAM_INITIALIZATION_TIMEOUT', 0.01):
        with stream_sync:
            # Wait a bit to let the thread start
            import time

            time.sleep(0.02)
            with pytest.raises(RuntimeError, match='Stream failed to initialize'):
                stream_sync.get()
