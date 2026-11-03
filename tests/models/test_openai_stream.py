from __future__ import annotations as _annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
import pytest

from pydantic_ai import Agent, ModelAPIError, ModelConnectionError, ModelMessage, ModelRequest, ModelTimeoutError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types import chat
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings, OpenAIResponsesModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


def _evroc_model(evroc_api_key: str, *, require_finish_reason: bool = False) -> OpenAIChatModel:
    profile = OpenAIModelProfile(openai_chat_streaming_requires_finish_reason=True) if require_finish_reason else None
    return OpenAIChatModel(
        'moonshotai/Kimi-K2.6',
        provider=OpenAIProvider(base_url='https://models.think.evroc.com/v1', api_key=evroc_api_key),
        profile=profile,
    )


def _evroc_request() -> tuple[list[ModelMessage], OpenAIChatModelSettings]:
    settings = OpenAIChatModelSettings(extra_headers={'X-Think-Timeout': '5'})
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt(
            'Output every integer from 1 through 100000, one per line. Do not abbreviate or stop early.'
        )
    ]
    return messages, settings


@pytest.fixture(scope='session')
def evroc_api_key() -> str:
    return os.getenv('EVROC_API_KEY', 'mock-api-key')


@pytest.mark.vcr
async def test_clean_eof_without_finish_reason_is_accepted_by_default(allow_model_requests: None, evroc_api_key: str):
    """Keep missing finish reasons non-fatal unless the model profile opts into strict handling.

    A clean end of stream counts as a `'stop'`, as a complete response without a finish reason does.
    """
    model = _evroc_model(evroc_api_key)
    messages, settings = _evroc_request()

    async with model.request_stream(messages, settings, ModelRequestParameters()) as stream:
        async for _ in stream:
            pass

    response = stream.get()
    assert response.state == 'complete'
    assert response.finish_reason == 'stop'
    assert response.provider_details is not None
    assert 'finish_reason' not in response.provider_details
    assert response.text is not None
    assert response.text.strip().splitlines() == [str(number) for number in range(1, 215)]


@pytest.mark.vcr('test_clean_eof_without_finish_reason_is_accepted_by_default.yaml')
async def test_clean_eof_without_finish_reason_is_rejected_when_required(
    allow_model_requests: None, evroc_api_key: str
):
    """Reject the recorded partial evroc response when its profile requires a terminal finish reason."""
    model = _evroc_model(evroc_api_key, require_finish_reason=True)
    messages, settings = _evroc_request()
    stream = None

    with pytest.raises(ModelAPIError, match='Streamed response ended without a `finish_reason`') as exc_info:
        async with model.request_stream(messages, settings, ModelRequestParameters()) as streamed_response:
            stream = streamed_response
            async for _ in streamed_response:
                pass

    assert exc_info.value.model_name == 'moonshotai/Kimi-K2.6'
    assert stream is not None
    response = stream.get()
    assert response.state == 'incomplete'
    assert response.finish_reason is None
    assert response.text is not None
    assert response.text.strip().splitlines() == [str(number) for number in range(1, 215)]


@pytest.mark.vcr
async def test_complete_stream_is_accepted_when_finish_reason_is_required(
    allow_model_requests: None, openai_api_key: str
):
    """Accept a finish reason followed by a no-choices moderation chunk, which must not reset it."""
    model = OpenAIChatModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        profile=OpenAIModelProfile(openai_chat_streaming_requires_finish_reason=True),
    )
    settings = OpenAIChatModelSettings(openai_moderation={'model': 'omni-moderation-latest'})
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('What is the capital of France?')]

    async with model.request_stream(messages, settings, ModelRequestParameters()) as stream:
        async for _ in stream:
            pass

    response = stream.get()
    assert response.state == 'complete'
    assert response.finish_reason == 'stop'
    assert response.text == 'Paris.'


async def test_cancelled_stream_does_not_require_finish_reason(allow_model_requests: None):
    """A natural EOF after client-side cancellation cannot be recorded reliably from a live provider."""
    chunk = chat.ChatCompletionChunk(
        id='123',
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content='partial', role='assistant'), finish_reason=None)],
        created=1704067200,
        model='gpt-5',
        object='chat.completion.chunk',
    )
    mock_client = MockOpenAI.create_mock_stream([chunk])
    model = OpenAIChatModel(
        'gpt-5',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(openai_chat_streaming_requires_finish_reason=True),
    )
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]

    async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
        iterator = aiter(stream)
        await anext(iterator)
        await stream.cancel()
        async for _ in iterator:
            pass

    response = stream.get()
    assert response.state == 'interrupted'
    assert response.finish_reason is None
    assert response.text == 'partial'


@asynccontextmanager
async def _serve_incomplete_503(*, hang_before_body: bool = False) -> AsyncIterator[str]:
    """Serve a 503 whose error body never completes.

    The response advertises a `Content-Length` longer than the body it sends, so the openai SDK fails while it reads
    the body it needs to build its status error. With `hang_before_body`, the server goes silent after the headers and
    waits for the client read timeout instead.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Read past the request head; its content does not matter for the failure.
        while await reader.readline() not in (b'\r\n', b'\n', b''):
            pass
        body = b'{"error": {"message": "server overloaded", "type": "server_error"}}'
        writer.write(
            b'HTTP/1.1 503 Service Unavailable\r\n'
            b'Content-Type: application/json\r\n'
            b'Content-Length: ' + str(len(body) + 64).encode() + b'\r\n'
            b'Connection: close\r\n'
            b'\r\n'
        )
        await writer.drain()
        if hang_before_body:
            # Stay silent until the client read timeout fires while httpx waits for the body.
            await asyncio.sleep(5)
        else:
            writer.write(body)
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, host='127.0.0.1', port=0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f'http://127.0.0.1:{port}/v1'
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize('model_class', ['OpenAIChatModel', 'OpenAIResponsesModel'])
async def test_run_stream_transport_error_during_error_body_read_raises_model_connection_error(
    allow_model_requests: None, model_class: str
) -> None:
    """A transport error the SDK hits while reading a streamed error body surfaces as `ModelConnectionError`."""
    model_classes = {
        'OpenAIChatModel': OpenAIChatModel,
        'OpenAIResponsesModel': OpenAIResponsesModel,
    }

    async with _serve_incomplete_503() as base_url:
        model = model_classes[model_class]('gpt-4o', provider=OpenAIProvider(base_url=base_url, api_key='test-key'))
        agent = Agent(model)
        with pytest.raises(ModelConnectionError) as exc_info:
            async with agent.run_stream('Hello'):
                pass

    assert type(exc_info.value) is ModelConnectionError
    assert exc_info.value.model_name == 'gpt-4o'
    assert exc_info.value.phase == 'read'
    assert isinstance(exc_info.value.__cause__, httpx2.TransportError)
    assert not isinstance(exc_info.value.__cause__, httpx2.TimeoutException)
    assert str(exc_info.value.__cause__) in exc_info.value.message


async def test_run_stream_timeout_during_error_body_read_raises_model_timeout_error(
    allow_model_requests: None,
) -> None:
    """A read timeout the SDK hits while reading a streamed error body surfaces as `ModelTimeoutError`."""
    async with _serve_incomplete_503(hang_before_body=True) as base_url:
        async with httpx2.AsyncClient(timeout=0.5) as http_client:
            provider = OpenAIProvider(base_url=base_url, api_key='test-key', http_client=http_client)
            agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))
            with pytest.raises(ModelTimeoutError) as exc_info:
                async with agent.run_stream('Hello'):
                    pass

    assert type(exc_info.value) is ModelTimeoutError
    assert exc_info.value.model_name == 'gpt-4o'
    assert exc_info.value.phase == 'read'
    assert isinstance(exc_info.value.__cause__, httpx2.TimeoutException)
    assert str(exc_info.value.__cause__) in exc_info.value.message


async def test_fallback_model_default_config_falls_back_when_stream_open_hits_transport_error(
    allow_model_requests: None,
) -> None:
    """`FallbackModel`'s default configuration falls back when opening a stream raises a mapped connection error."""
    chunk = chat.ChatCompletionChunk(
        id='123',
        choices=[
            ChunkChoice(
                delta=ChoiceDelta(content='fallback output', role='assistant'),
                finish_reason=None,
                index=0,
            )
        ],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion.chunk',
    )

    async with _serve_incomplete_503() as base_url:
        broken = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(base_url=base_url, api_key='test-key'))
        healthy = OpenAIChatModel(
            'gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream([chunk]))
        )
        agent = Agent(FallbackModel(broken, healthy))
        async with agent.run_stream('Hello') as result:
            output = await result.get_output()

    assert output == 'fallback output'
