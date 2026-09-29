"""Tests for requests `AnthropicModel` streams and accumulates into a non-streaming response.

A request whose `max_tokens` is above the Anthropic SDK's non-streaming limit is streamed, and the SDK's accumulator
builds the final message. These tests don't use cassettes: cassetter reads a whole response body before handing it
to the client, which a response that breaks mid-stream can't survive.
"""

from __future__ import annotations as _annotations

from typing import cast

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncStream
    from anthropic.lib.streaming import BetaAsyncMessageStream
    from anthropic.types.beta import BetaRawMessageStartEvent, BetaRawMessageStreamEvent

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        _WithoutUntypedEvents,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


_STREAM_START = (
    b'event: message_start\n'
    b'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":'
    b'"claude-sonnet-4-5","content":[],"stop_reason":null,"stop_sequence":null,'
    b'"usage":{"input_tokens":5,"output_tokens":1}}}\n\n'
)


class _BrokenStream(httpx2.AsyncByteStream):
    async def __aiter__(self):
        yield _STREAM_START
        raise httpx2.ReadError('connection reset')


async def test_stream_that_breaks_raises_model_api_error(allow_model_requests: None) -> None:
    """A transport failure after the streamed response started raises a `ModelAPIError`, like one before it.

    Mocked because a connection can't be broken mid-response on demand.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=_BrokenStream())

    client = AsyncAnthropic(api_key='test', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    agent = Agent(AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client)))
    with pytest.raises(ModelAPIError, match='connection reset'):
        await agent.run('hello')


async def test_empty_stream_raises_unexpected_model_behavior(allow_model_requests: None) -> None:
    """A 200 response whose stream carries no events raises the same error a streamed run raises.

    Mocked because the API doesn't send an empty stream on demand.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=b'')

    client = AsyncAnthropic(api_key='test', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    agent = Agent(AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client)))
    with pytest.raises(UnexpectedModelBehavior, match='Streamed response ended without content or tool calls'):
        await agent.run('hello')


async def test_accumulated_stream_skips_untyped_bedrock_events() -> None:
    """Bedrock's untyped chunks, like a leading `amazon-bedrock-invocationMetrics`, don't break accumulation.

    The SDK's Bedrock stream decoder constructs them as `BetaRawMessageStartEvent(message=None)`, which its message
    accumulator rejects. Tested on the event level because Bedrock's binary event stream can't be produced on demand.
    """
    start = BetaRawMessageStartEvent.model_validate(
        {
            'type': 'message_start',
            'message': {
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'model': 'claude-sonnet-4-5',
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 5, 'output_tokens': 1},
            },
        }
    )
    untyped = BetaRawMessageStartEvent.model_construct(message=None, type=None)
    events: list[BetaRawMessageStreamEvent] = [untyped, start]

    class _Stream:
        response = httpx2.Response(200, request=httpx2.Request('POST', 'https://api.anthropic.com/v1/messages'))

        async def __aiter__(self):
            for event in events:
                yield event

        async def close(self) -> None:
            pass

    stream = _WithoutUntypedEvents(cast(AsyncStream[BetaRawMessageStreamEvent], _Stream()))
    message = await BetaAsyncMessageStream(
        cast(AsyncStream[BetaRawMessageStreamEvent], stream), output_format=NOT_GIVEN
    ).get_final_message()
    assert message.id == 'msg_1'
