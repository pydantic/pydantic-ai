"""Cancelling a Bedrock request whose botocore call or stream read is blocked in a worker thread.

botocore blocks its thread until the endpoint answers or its `read_timeout` passes. These offline tests stand in for a
stalled endpoint with a client whose call, or its stream's first read, blocks until the test releases it, so they can
check that a `request_timeout` or a cancellation ends the request while the thread is still blocked, and that what the
thread gets afterwards is closed. A real endpoint can't be made to stall on cue.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, cast

import anyio
import anyio.from_thread
import anyio.to_thread
import pytest

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.exceptions import ModelRequestTimeout
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

with try_import() as imports_successful:
    from botocore.hooks import HierarchicalEmitter

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')

SHORT_TIMEOUT = 0.05
"""For a request that never finishes: how soon it times out only affects how long the test takes."""

RELEASE_TIMEOUT = 30
"""How long a blocked call waits to be released before it gives up, so a regression fails rather than hangs."""

Blocks = Literal['converse', 'converse_stream', 'stream_read']


@dataclass
class _BlockingEventStream:
    """A `ConverseStream` event stream whose first read blocks until released, as one from a stalled endpoint does.

    Closing it waits for a blocked read too, as closing a botocore event stream waits on the read holding its
    response's lock, unless its connection's socket is shut down first, which wakes the read.
    """

    client: _BlockingBedrockClient
    closed: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        def shutdown(how: int) -> None:
            self.client.release.set()

        self._raw_stream = SimpleNamespace(connection=SimpleNamespace(sock=SimpleNamespace(shutdown=shutdown)))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.client.block()
        yield from ()

    def close(self) -> None:
        self.client.release.wait(RELEASE_TIMEOUT)
        self.closed.set()


class _BlockingBedrockClient:
    """A Bedrock client whose call, or its stream's first read, blocks until released."""

    def __init__(self, blocks: Blocks) -> None:
        self.meta = SimpleNamespace(endpoint_url='https://bedrock.stub', events=HierarchicalEmitter())
        self.blocks = blocks
        self.on_blocked: Callable[[], None] | None = None
        """Called on the event loop once the call or read is blocked, to cancel the request from there."""
        self.release = threading.Event()
        self.unblocked = threading.Event()
        self.stream = _BlockingEventStream(self)

    def block(self) -> None:
        if self.on_blocked is not None:
            anyio.from_thread.run_sync(self.on_blocked)
        self.release.wait(RELEASE_TIMEOUT)
        self.unblocked.set()

    def converse(self, **_: Any) -> dict[str, Any]:
        self.block()
        return {'output': {'message': {'role': 'assistant', 'content': [{'text': 'late'}]}}, 'stopReason': 'end_turn'}

    def converse_stream(self, **_: Any) -> dict[str, Any]:
        if self.blocks == 'converse_stream':
            self.block()
        return {'stream': self.stream, 'ResponseMetadata': {}}


def _model(client: _BlockingBedrockClient) -> BedrockConverseModel:
    return BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=BedrockProvider(bedrock_client=cast(Any, client)))


async def _request(model: BedrockConverseModel, blocks: Blocks, *, request_timeout: float | None = None) -> None:
    messages = [ModelRequest.user_text_prompt('hi')]
    settings = ModelSettings(request_timeout=request_timeout) if request_timeout is not None else None
    if blocks == 'converse':
        await model_request(model, messages, model_settings=settings)
    else:
        async with model_request_stream(model, messages, model_settings=settings) as stream:
            async for _ in stream:
                pass  # pragma: no cover


def _assert_ended_while_blocked(client: _BlockingBedrockClient, blocks: Blocks) -> None:
    """Check the request ended without waiting for the blocked thread to be released by the test."""
    if blocks == 'stream_read':
        # Closing the stream on the way out shut its connection down, which woke the read, and closed it.
        assert client.stream.closed.is_set()
    else:
        assert not client.unblocked.is_set()


async def _assert_cleaned_up(client: _BlockingBedrockClient, blocks: Blocks) -> None:
    """Release the blocked thread, and check that what it got is closed."""
    client.release.set()
    if blocks != 'converse':
        assert await anyio.to_thread.run_sync(client.stream.closed.wait, RELEASE_TIMEOUT)


@pytest.mark.parametrize('blocks', ['converse', 'converse_stream', 'stream_read'])
async def test_request_timeout_ends_a_blocked_request(allow_model_requests: None, blocks: Blocks):
    """A `request_timeout` raises `ModelRequestTimeout` while the blocked thread is still waiting."""
    client = _BlockingBedrockClient(blocks)
    model = _model(client)
    with pytest.raises(ModelRequestTimeout):
        await _request(model, blocks, request_timeout=SHORT_TIMEOUT)
    _assert_ended_while_blocked(client, blocks)
    await _assert_cleaned_up(client, blocks)


@pytest.mark.parametrize('blocks', ['converse', 'converse_stream', 'stream_read'])
async def test_cancellation_ends_a_blocked_request(allow_model_requests: None, blocks: Blocks):
    """Cancelling a request ends it while the blocked thread is still waiting."""
    client = _BlockingBedrockClient(blocks)
    model = _model(client)
    with anyio.CancelScope() as scope:
        client.on_blocked = scope.cancel
        await _request(model, blocks)
    assert scope.cancelled_caught
    _assert_ended_while_blocked(client, blocks)
    await _assert_cleaned_up(client, blocks)
