from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal, Protocol, TypeVar

import anyio
import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RequestUsage, TextPart, UserPromptPart
from pydantic_ai._utils import PeekableAsyncStream
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    import xai_sdk.chat as xai_chat
    from google.genai.types import GenerateContentResponse
    from huggingface_hub import AsyncInferenceClient, ChatCompletionStreamOutput
    from xai_sdk.proto import chat_pb2

    from pydantic_ai.models.google import GeminiStreamedResponse, GoogleModel
    from pydantic_ai.models.huggingface import HuggingFaceModel, HuggingFaceStreamedResponse
    from pydantic_ai.models.xai import XaiModel, XaiStreamedResponse
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.huggingface import HuggingFaceProvider
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='provider SDKs not installed'),
    pytest.mark.anyio,
]

T = TypeVar('T')


@dataclass(frozen=True)
class Case:
    id: Literal['google', 'huggingface', 'xai']
    model_name: str
    expected_messages: list[ModelMessage]


CASES = [
    Case(
        id='google',
        model_name='gemini-2.5-flash',
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Reply with exactly: Paris', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Paris')],
                    usage=RequestUsage(
                        details={'thoughts_tokens': 35, 'text_prompt_tokens': 6},
                        input_tokens=6,
                        input_text_tokens=6,
                        output_tokens=36,
                        output_reasoning_tokens=35,
                        cost=Decimal('0.0000918'),
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                    state='interrupted',
                ),
            ]
        ),
    ),
    Case(
        id='huggingface',
        model_name='meta-llama/Llama-3.1-8B-Instruct',
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Reply with exactly: Paris', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Paris')],
                    usage=RequestUsage(cost=Decimal('0.00')),
                    model_name='meta-llama/llama-3.1-8b-instruct',
                    timestamp=IsDatetime(),
                    provider_name='huggingface',
                    provider_url='https://router.huggingface.co/novita',
                    provider_details={'timestamp': IsDatetime()},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                    state='interrupted',
                ),
            ]
        ),
    ),
    Case(
        id='xai',
        model_name='grok-4-fast-non-reasoning',
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Reply with exactly: Paris', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Paris')],
                    usage=RequestUsage(
                        input_tokens=189, cache_read_tokens=128, output_tokens=1, cost=Decimal('0.0000191')
                    ),
                    model_name='grok-4-fast-non-reasoning',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                    provider_url='https://api.x.ai/v1',
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                    state='interrupted',
                ),
            ]
        ),
    ),
]


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
            _model_id_namespace='google',
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


@pytest.mark.vcr
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_cancel_recorded_provider_stream(
    case: Case,
    allow_model_requests: None,
    gemini_api_key: str,
    huggingface_api_key: str,
    xai_provider: XaiProvider | None,
):
    """Recorded SDK streams preserve public early-cancellation behavior for each affected provider."""
    huggingface_client: AsyncInferenceClient | None = None
    if case.id == 'google':
        model = GoogleModel(case.model_name, provider=GoogleProvider(api_key=gemini_api_key))
    elif case.id == 'huggingface':
        provider = HuggingFaceProvider(provider_name='novita', api_key=huggingface_api_key)
        huggingface_client = provider.client
        model = HuggingFaceModel(case.model_name, provider=provider)
    else:
        assert xai_provider is not None
        model = XaiModel(case.model_name, provider=xai_provider)

    try:
        agent = Agent(model, model_settings={'temperature': 0.0})
        async with agent.run_stream('Reply with exactly: Paris') as result:
            async for text in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                assert text
                break
            await result.cancel()
            await result.cancel()
            assert result.cancelled
        assert result.all_messages() == case.expected_messages
    finally:
        if huggingface_client is not None:
            await huggingface_client.close()
