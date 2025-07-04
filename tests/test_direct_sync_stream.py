"""Tests for the synchronous streaming functionality in the direct module."""

import asyncio
import re
from unittest.mock import AsyncMock, patch

import pytest

from pydantic_ai.direct import StreamedResponseSync, model_request_stream_sync
from pydantic_ai.messages import ModelMessage, ModelRequest


def test_model_request_stream_sync_without_context_manager():
    """Test that accessing properties or iterating without context manager raises RuntimeError."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('x')]

    expected_error_msg = re.escape(
        'StreamedResponseSync must be used as a context manager. Use: `with model_request_stream_sync(...) as stream:`'
    )

    stream_cm = model_request_stream_sync('test', messages)

    stream_repr = repr(stream_cm)
    assert 'StreamedResponseSync' in stream_repr
    assert 'context_entered=False' in stream_repr

    with pytest.raises(RuntimeError, match=expected_error_msg):
        _ = stream_cm.model_name

    with pytest.raises(RuntimeError, match=expected_error_msg):
        _ = stream_cm.timestamp

    with pytest.raises(RuntimeError, match=expected_error_msg):
        stream_cm.get()

    with pytest.raises(RuntimeError, match=expected_error_msg):
        stream_cm.usage()

    with pytest.raises(RuntimeError, match=expected_error_msg):
        list(stream_cm)

    with pytest.raises(RuntimeError, match=expected_error_msg):
        for _ in stream_cm:
            break


def test_model_request_stream_sync_exception_in_stream():
    """Test handling of exceptions raised during streaming."""
    async_stream_mock = AsyncMock()
    async_stream_mock.__aenter__ = AsyncMock(side_effect=ValueError('Stream error'))

    stream_sync = StreamedResponseSync(_async_stream_cm=async_stream_mock)

    with stream_sync:
        with pytest.raises(ValueError, match='Stream error'):
            list(stream_sync)


def test_model_request_stream_sync_timeout():
    """Test timeout when stream fails to initialize."""
    async_stream_mock = AsyncMock()

    async def slow_init():
        await asyncio.sleep(0.1)

    async_stream_mock.__aenter__ = AsyncMock(side_effect=slow_init)

    stream_sync = StreamedResponseSync(_async_stream_cm=async_stream_mock)

    with patch('pydantic_ai.models.STREAM_INITIALIZATION_TIMEOUT', 0.01):
        with stream_sync:
            with pytest.raises(RuntimeError, match='Stream failed to initialize within timeout'):
                stream_sync.get()


def test_model_request_stream_sync_intermediate_get():
    """Test getting properties of StreamedResponse before consuming all events."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('x')]

    with model_request_stream_sync('test', messages) as stream:
        response = stream.get()
        assert response is not None

        usage = stream.usage()
        assert usage is not None
