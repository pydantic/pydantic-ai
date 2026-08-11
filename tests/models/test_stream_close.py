from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime, timezone
from typing import Literal, Protocol, TypeVar

import anyio
import pytest

from pydantic_ai._utils import PeekableAsyncStream
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from ..conftest import try_import

with try_import() as imports_successful:
    import xai_sdk.chat as xai_chat
    from google.genai.types import GenerateContentResponse
    from huggingface_hub import ChatCompletionStreamOutput
    from xai_sdk.proto import chat_pb2

    from pydantic_ai.models.google import GeminiStreamedResponse
    from pydantic_ai.models.huggingface import HuggingFaceStreamedResponse
    from pydantic_ai.models.xai import XaiStreamedResponse
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='provider SDKs not installed'),
    pytest.mark.anyio,
]

T = TypeVar('T')


class _ClosableStream(Protocol):
    async def peek(self) -> object: ...

    async def __anext__(self) -> object: ...


async def _active_stream(first: T, pull_started: anyio.Event, finalization_started: anyio.Event) -> AsyncIterator[T]:
    try:
        yield first
        pull_started.set()
        await anyio.sleep_forever()
    finally:
        finalization_started.set()
        await anyio.sleep(0)


async def _assert_close_cancels_active_pull(
    response: StreamedResponse,
    stream: _ClosableStream,
    first: object,
    pull_started: anyio.Event,
    finalization_started: anyio.Event,
) -> None:
    assert await stream.peek() is first
    assert await anext(stream) is first

    pull_finished = anyio.Event()

    async def pull() -> None:
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        pull_finished.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(pull)
        await pull_started.wait()
        with anyio.fail_after(1):
            await response.close_stream()
            await pull_finished.wait()

    assert finalization_started.is_set()


@pytest.mark.parametrize('provider', ['google', 'huggingface', 'xai'])
async def test_provider_close_stream_cancels_active_pull(provider: Literal['google', 'huggingface', 'xai']):
    """Provider shutdown must synchronize with an active pull through `PeekableAsyncStream`."""
    pull_started = anyio.Event()
    finalization_started = anyio.Event()

    if provider == 'google':
        first = GenerateContentResponse()
        google_stream: PeekableAsyncStream[GenerateContentResponse, AsyncIterator[GenerateContentResponse]] = (
            PeekableAsyncStream(_active_stream(first, pull_started, finalization_started))
        )
        response: StreamedResponse = GeminiStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            _model_name='gemini-2.0-flash',
            _response=google_stream,
            _provider_name='google',
            _provider_url='https://generativelanguage.googleapis.com',
        )
        stream: _ClosableStream = google_stream
    elif provider == 'huggingface':
        first = ChatCompletionStreamOutput(
            choices=[], created=0, id='response-id', model='model', system_fingerprint='fingerprint'
        )
        huggingface_source: AsyncIterable[ChatCompletionStreamOutput] = _active_stream(
            first, pull_started, finalization_started
        )
        huggingface_stream: PeekableAsyncStream[
            ChatCompletionStreamOutput, AsyncIterable[ChatCompletionStreamOutput]
        ] = PeekableAsyncStream(huggingface_source)
        response = HuggingFaceStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            _model_name='model',
            _model_profile={},
            _response=huggingface_stream,
            _provider_name='huggingface',
            _provider_url='https://api-inference.huggingface.co',
        )
        stream = huggingface_stream
    else:
        first = (
            xai_chat.Response(chat_pb2.GetChatCompletionResponse(), index=None),
            xai_chat.Chunk(chat_pb2.GetChatCompletionChunk(), index=None),
        )
        xai_source: AsyncIterator[tuple[xai_chat.Response, object]] = _active_stream(
            first, pull_started, finalization_started
        )
        xai_stream: PeekableAsyncStream[
            tuple[xai_chat.Response, xai_chat.Chunk], AsyncIterator[tuple[xai_chat.Response, object]]
        ] = PeekableAsyncStream(xai_source)
        response = XaiStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            _model_name='grok-4-fast-non-reasoning',
            _response=xai_stream,
            _timestamp=datetime.now(timezone.utc),
            _provider=XaiProvider(api_key='xai-api-key'),
        )
        stream = xai_stream

    await _assert_close_cancels_active_pull(response, stream, first, pull_started, finalization_started)
