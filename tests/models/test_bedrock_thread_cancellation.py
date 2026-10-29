"""Cancelling a Bedrock request whose botocore call or stream read is blocked in a worker thread.

botocore blocks its thread until the endpoint answers or its `read_timeout` passes. These offline tests stand in for a
stalled endpoint with a client whose call, or its stream's first read, blocks until the test releases it, so they can
check that a `request_timeout` or a cancellation ends the request while the thread is still blocked, and that what the
thread gets afterwards is closed. A real endpoint can't be made to stall on cue.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, cast

import anyio
import anyio.from_thread
import anyio.to_thread
import pytest

from pydantic_ai import ModelMessage, ModelRequest
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.exceptions import ModelRequestTimeout
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

with try_import() as imports_successful:
    import boto3
    from botocore.config import Config
    from botocore.hooks import HierarchicalEmitter

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')

SHORT_TIMEOUT = 0.05
"""For a request that never finishes: how soon it times out only affects how long the test takes."""

RELEASE_TIMEOUT = 30
"""How long a blocked call waits to be released before it gives up, so a regression fails rather than hangs."""

Blocks = Literal['converse', 'converse_stream', 'stream_read', 'count_tokens']


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
        self.called = threading.Event()
        """Set once the call is made: a thread the request gave up on before it started doesn't make it at all."""
        self.stream = _BlockingEventStream(self)

    def block(self) -> None:
        if self.on_blocked is not None:
            anyio.from_thread.run_sync(self.on_blocked)
        self.release.wait(RELEASE_TIMEOUT)
        self.unblocked.set()

    def converse(self, **_: Any) -> dict[str, Any]:
        self.called.set()
        self.block()
        return {'output': {'message': {'role': 'assistant', 'content': [{'text': 'late'}]}}, 'stopReason': 'end_turn'}

    def count_tokens(self, **_: Any) -> dict[str, Any]:
        self.called.set()
        self.block()
        return {'inputTokens': 1}

    def converse_stream(self, **_: Any) -> dict[str, Any]:
        self.called.set()
        if self.blocks == 'converse_stream':
            self.block()
        return {'stream': self.stream, 'ResponseMetadata': {}}


def _model(client: _BlockingBedrockClient) -> BedrockConverseModel:
    return BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=BedrockProvider(bedrock_client=cast(Any, client)))


async def _request(model: BedrockConverseModel, blocks: Blocks, *, request_timeout: float | None = None) -> None:
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('hi')]
    settings = ModelSettings(request_timeout=request_timeout) if request_timeout is not None else None
    if blocks == 'converse':
        await model_request(model, messages, model_settings=settings)
    elif blocks == 'count_tokens':
        await model.count_tokens(messages, settings, ModelRequestParameters())
    else:
        async with model_request_stream(model, messages, model_settings=settings) as stream:
            async for _ in stream:
                pass  # pragma: no cover


def _assert_ended_while_blocked(client: _BlockingBedrockClient) -> None:
    """Check the request ended without waiting for the blocked call or read to give up on its own.

    Only the test releases a blocked call, after this check, while a blocked read is also released by shutting down
    the stream's connection, which closing the stream does on the way out.
    """
    assert client.release.is_set() or not client.unblocked.is_set()


async def _assert_cleaned_up(client: _BlockingBedrockClient, blocks: Blocks) -> None:
    """Release the blocked thread, and check that what it got is closed."""
    client.release.set()
    if blocks in ('converse_stream', 'stream_read') and client.called.is_set():
        assert await anyio.to_thread.run_sync(client.stream.closed.wait, RELEASE_TIMEOUT)


@pytest.mark.parametrize('blocks', ['converse', 'converse_stream', 'stream_read'])
async def test_request_timeout_ends_a_blocked_request(allow_model_requests: None, blocks: Blocks):
    """A `request_timeout` raises `ModelRequestTimeout` while the blocked thread is still waiting."""
    client = _BlockingBedrockClient(blocks)
    model = _model(client)
    with pytest.raises(ModelRequestTimeout):
        await _request(model, blocks, request_timeout=SHORT_TIMEOUT)
    _assert_ended_while_blocked(client)
    await _assert_cleaned_up(client, blocks)


@pytest.mark.parametrize('blocks', ['converse', 'converse_stream', 'stream_read', 'count_tokens'])
async def test_cancellation_ends_a_blocked_request(allow_model_requests: None, blocks: Blocks):
    """Cancelling a request ends it while the blocked thread is still waiting."""
    client = _BlockingBedrockClient(blocks)
    model = _model(client)
    with anyio.CancelScope() as scope:
        client.on_blocked = scope.cancel
        await _request(model, blocks)
    assert scope.cancelled_caught
    _assert_ended_while_blocked(client)
    await _assert_cleaned_up(client, blocks)


async def test_request_timeout_ends_a_stream_read_blocked_on_a_real_connection(allow_model_requests: None):
    """A real botocore event stream, reading from a real socket that stalls after the headers, ends on time.

    Checks what the stubs above stand in for: that closing a real stream shuts down the connection its blocked
    read is waiting on, rather than waiting for the read until botocore's `read_timeout` passes.
    """
    done = threading.Event()
    server = socket.create_server(('127.0.0.1', 0))

    def serve() -> None:
        connection, _ = server.accept()
        with connection:
            connection.recv(65536)
            connection.sendall(
                b'HTTP/1.1 200 OK\r\nContent-Type: application/vnd.amazon.eventstream\r\n'
                b'Transfer-Encoding: chunked\r\n\r\n'
            )
            done.wait(RELEASE_TIMEOUT)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        endpoint_url=f'http://127.0.0.1:{server.getsockname()[1]}',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        config=Config(read_timeout=RELEASE_TIMEOUT, retries={'max_attempts': 1}),
    )
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=BedrockProvider(bedrock_client=client))
    start = time.monotonic()
    try:
        with pytest.raises(ModelRequestTimeout):
            await _request(model, 'stream_read', request_timeout=SHORT_TIMEOUT)
        # Well under botocore's `read_timeout`, which the read would otherwise have waited out.
        assert time.monotonic() - start < RELEASE_TIMEOUT / 3
    finally:
        done.set()
        server.close()
