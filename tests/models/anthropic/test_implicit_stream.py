"""Tests for the requests `run()` streams behind the scenes.

A request whose `max_tokens` is above the Anthropic SDK's non-streaming limit is streamed, and its response is built
like a streamed run's. These tests don't use cassettes: cassetter reads a whole response body before handing it to
the client, which a response that breaks mid-stream can't survive.
"""

from __future__ import annotations as _annotations

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


_STREAM_START = (
    b'event: message_start\n'
    b'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":'
    b'"claude-sonnet-4-5","content":[],"stop_reason":null,"stop_sequence":null,'
    b'"usage":{"input_tokens":5,"output_tokens":1}}}\n\n'
)


class _BrokenStream(httpx2.AsyncByteStream):
    def __init__(self, sent: bytes) -> None:
        self._sent = sent

    async def __aiter__(self):
        if self._sent:
            yield self._sent
        raise httpx2.ReadError('connection reset')


@pytest.mark.parametrize(
    'sent', [pytest.param(b'', id='before-the-first-event'), pytest.param(_STREAM_START, id='mid-response')]
)
async def test_stream_that_breaks_raises_model_api_error(allow_model_requests: None, sent: bytes) -> None:
    """A transport failure in the streamed response raises a `ModelAPIError`, like one that stops the request.

    Mocked because a connection can't be broken mid-response on demand.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=_BrokenStream(sent))

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
