from __future__ import annotations as _annotations

import os
from collections.abc import AsyncIterator
from typing import Literal

import httpx2
import pytest

from pydantic_ai import Agent, ModelAPIError, ModelConnectionError, ModelMessage, ModelRequest, ModelTimeoutError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI
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


class _ErrorBodyBreakingOff(httpx2.AsyncByteStream):
    """A 503 error body that breaks off after its first chunk, as when a proxy cuts the connection."""

    def __init__(self, error: httpx2.TransportError) -> None:
        self.error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"error": {"mess'
        raise self.error


def _client_with_error_body_breaking_off(error: httpx2.TransportError) -> AsyncOpenAI:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503, headers={'content-type': 'application/json'}, stream=_ErrorBodyBreakingOff(error))

    return AsyncOpenAI(
        api_key='test-key',
        base_url='http://localhost/v1',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )


@pytest.mark.parametrize('model_type', ['chat', 'responses'])
@pytest.mark.parametrize(
    'error_class,expected_error_class',
    [
        pytest.param(httpx2.RemoteProtocolError, ModelConnectionError, id='protocol-error'),
        pytest.param(httpx2.ReadTimeout, ModelTimeoutError, id='read-timeout'),
    ],
)
async def test_run_stream_transport_error_during_error_body_read(
    allow_model_requests: None,
    model_type: Literal['chat', 'responses'],
    error_class: type[httpx2.TransportError],
    expected_error_class: type[ModelConnectionError],
) -> None:
    """A transport error while `openai` reads a streamed error body surfaces as `ModelConnectionError`.

    A timeout surfaces as its `ModelTimeoutError` subclass. For a streamed request the SDK reads the body of an error
    status outside the code that wraps transport errors in `APIConnectionError`. A cassette can't replay a broken-off
    connection, so a mock transport raises it.
    """
    error = error_class('connection broke off while reading the error body')
    model_class = OpenAIChatModel if model_type == 'chat' else OpenAIResponsesModel

    async with _client_with_error_body_breaking_off(error) as client:
        agent = Agent(model_class('gpt-5.4', provider=OpenAIProvider(openai_client=client)))
        with pytest.raises(ModelAPIError) as exc_info:
            async with agent.run_stream('Hello') as result:
                await result.get_output()

    assert type(exc_info.value) is expected_error_class
    assert exc_info.value.model_name == 'gpt-5.4'
    assert exc_info.value.phase == 'read'
    assert exc_info.value.message == 'connection broke off while reading the error body'
    assert exc_info.value.__cause__ is error


async def test_fallback_model_falls_back_on_transport_error_during_error_body_read(allow_model_requests: None) -> None:
    """`FallbackModel`'s default `fallback_on` catches the mapped error, so a run stream moves on to the next model."""

    async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'fallback output'

    error = httpx2.RemoteProtocolError('peer closed connection without sending complete message body')
    async with _client_with_error_body_breaking_off(error) as client:
        broken = OpenAIChatModel('gpt-5.4', provider=OpenAIProvider(openai_client=client))
        agent = Agent(FallbackModel(broken, FunctionModel(stream_function=stream_function)))
        async with agent.run_stream('Hello') as result:
            assert await result.get_output() == 'fallback output'
