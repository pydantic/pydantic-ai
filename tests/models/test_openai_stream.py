from __future__ import annotations as _annotations

import json
import os
from collections.abc import Iterator
from decimal import Decimal

import httpx2
import pytest

from pydantic_ai import Agent, ModelAPIError, ModelConnectionError, ModelMessage, ModelRequest
from pydantic_ai.capabilities import ModelFallback, RetryOnModelErrors
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types import chat
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI
    from .test_openai import chunk_with_usage

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
    """Reject the recorded partial evroc response as cut off when its profile requires a terminal finish reason."""
    model = _evroc_model(evroc_api_key, require_finish_reason=True)
    messages, settings = _evroc_request()
    stream = None

    with pytest.raises(ModelConnectionError, match='Streamed response ended without a `finish_reason`') as exc_info:
        async with model.request_stream(messages, settings, ModelRequestParameters()) as streamed_response:
            stream = streamed_response
            async for _ in streamed_response:
                pass

    # A cut-off stream, like a dropped connection, and still a `ModelAPIError` for code that caught that before.
    assert isinstance(exc_info.value, ModelAPIError)
    assert exc_info.value.phase == 'read'
    assert exc_info.value.model_name == 'moonshotai/Kimi-K2.6'
    assert stream is not None
    response = stream.get()
    assert response.state == 'incomplete'
    assert response.finish_reason is None
    assert response.text is not None
    assert response.text.strip().splitlines() == [str(number) for number in range(1, 215)]


def _sse_model(*streams: list[dict[str, object]]) -> OpenAIChatModel:
    """A model that requires a finish reason, answering each request with the next of `streams`' chunk choices."""
    envelope = {'id': 'chatcmpl-1', 'created': 0, 'model': 'test-model', 'object': 'chat.completion.chunk'}
    responses: Iterator[list[dict[str, object]]] = iter(streams)

    def handler(request: httpx2.Request) -> httpx2.Response:
        events = [f'data: {json.dumps({**envelope, "choices": [choice]})}\n\n' for choice in next(responses)]
        return httpx2.Response(
            200, content=''.join([*events, 'data: [DONE]\n\n']), headers={'content-type': 'text/event-stream'}
        )

    client = AsyncOpenAI(
        api_key='test',
        base_url='https://example.com/v1',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAIChatModel(
        'test-model',
        provider=OpenAIProvider(openai_client=client),
        profile=OpenAIModelProfile(openai_chat_streaming_requires_finish_reason=True),
    )


# A stream cut off after its role chunk, before any output: nothing has reached the consumer yet.
_CUT_OFF_STREAM: list[dict[str, object]] = [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
_COMPLETE_STREAM: list[dict[str, object]] = [
    {'index': 0, 'delta': {'role': 'assistant', 'content': 'Paris.'}, 'finish_reason': 'stop'}
]


async def test_stream_without_required_finish_reason_is_retried(allow_model_requests: None):
    """`RetryOnModelErrors` retries a stream cut off before its finish reason, like a dropped connection.

    A mock transport stands in for a cassette because no provider cuts a stream off on demand.
    """
    agent = Agent(_sse_model(_CUT_OFF_STREAM, _COMPLETE_STREAM), capabilities=[RetryOnModelErrors(initial_delay=0)])

    async with agent.run_stream('What is the capital of France?') as result:
        assert await result.get_output() == 'Paris.'

    assert [attempt.kind for attempt in result.response.failed_attempts or []] == snapshot(['retry'])


async def test_stream_without_required_finish_reason_falls_back(allow_model_requests: None):
    """`ModelFallback` falls back by default on a stream cut off before its finish reason.

    A mock transport stands in for a cassette because no provider cuts a stream off on demand.
    """
    agent = Agent(_sse_model(_CUT_OFF_STREAM), capabilities=[ModelFallback(TestModel(custom_output_text='Paris.'))])

    async with agent.run_stream('What is the capital of France?') as result:
        assert await result.get_output() == 'Paris.'

    assert result.response.model_name == 'test'


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


@pytest.mark.vcr
async def test_gemini_stream_usage_replaces_running_totals(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture
):
    """Gemini's OpenAI-compatible endpoint repeats the running usage on every streamed chunk, so each replaces the last.

    The recording's 7 chunks each report 14 prompt tokens, with completion tokens growing from 2 to 110. Summing them
    would report 98/455; the input tokens staying at 14 across steps shows each total replaces the last. Usage is
    requested with `include_usage` alone: the replacement comes from the provider's profile, not
    `continuous_usage_stats`.
    """
    provider = OpenAIProvider(
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        api_key=gemini_api_key,
        http_client=request_capture.client,
    )
    agent = Agent(OpenAIChatModel('gemini-3.1-flash-lite', provider=provider))

    async with agent.run_stream('Count from 1 to 40, separated by spaces.') as result:
        usage_at_each_step = [response.usage async for response in result.stream_response(debounce_by=None)]

    assert request_capture.body('/chat/completions')['stream_options'] == snapshot({'include_usage': True})
    assert usage_at_each_step == snapshot(
        [
            RequestUsage(input_tokens=14, output_tokens=2),
            RequestUsage(input_tokens=14, output_tokens=20),
            RequestUsage(input_tokens=14, output_tokens=47),
            RequestUsage(input_tokens=14, output_tokens=71),
            RequestUsage(input_tokens=14, output_tokens=95),
            RequestUsage(input_tokens=14, output_tokens=110),
            RequestUsage(input_tokens=14, output_tokens=110, cost=Decimal('0.0001685')),
            RequestUsage(input_tokens=14, output_tokens=110, cost=Decimal('0.0001685')),
        ]
    )
    assert result.usage == snapshot(RunUsage(requests=1, input_tokens=14, output_tokens=110, cost=Decimal('0.0001685')))


async def test_stream_cumulative_usage_profile_opt_out_sums_chunks(allow_model_requests: None):
    """A `gemini*` name on a server that reports per-chunk deltas keeps correct usage by opting out in its profile.

    A unit test because no recorded server sends deltas. The flag-on side is pinned against a real recording by
    `test_gemini_stream_usage_replaces_running_totals`.
    """
    stream: list[chat.ChatCompletionChunk] = [
        chunk_with_usage(
            [ChoiceDelta(content='hello ', role='assistant')], completion_tokens=5, prompt_tokens=10, total_tokens=15
        ),
        chunk_with_usage([ChoiceDelta(content='world')], completion_tokens=5, prompt_tokens=0, total_tokens=5),
        chunk_with_usage([], finish_reason='stop', completion_tokens=5, prompt_tokens=0, total_tokens=5),
    ]
    model = OpenAIChatModel(
        'gemini-3.1-flash-lite',
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(stream)),
        profile=OpenAIModelProfile(openai_chat_streaming_repeats_usage_totals=False),
    )

    async with Agent(model).run_stream('') as result:
        await result.get_output()

    assert result.usage == snapshot(RunUsage(requests=1, input_tokens=10, output_tokens=15))


async def test_stream_cumulative_usage_profile_replaces_at_each_step(allow_model_requests: None):
    """A `gemini*` name takes each chunk's running total as the usage so far, across a chunk that reports none.

    A unit test because no recording has a usage-less chunk between two that report usage.
    """
    usage_less_chunk = chunk_with_usage([ChoiceDelta(content='world')])
    usage_less_chunk.usage = None
    stream: list[chat.ChatCompletionChunk] = [
        chunk_with_usage(
            [ChoiceDelta(content='hello ', role='assistant')], completion_tokens=5, prompt_tokens=10, total_tokens=15
        ),
        usage_less_chunk,
        chunk_with_usage([ChoiceDelta(content='!')], completion_tokens=15, prompt_tokens=10, total_tokens=25),
        chunk_with_usage([], finish_reason='stop', completion_tokens=15, prompt_tokens=10, total_tokens=25),
    ]
    model = OpenAIChatModel(
        'gemini-3.1-flash-lite', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(stream))
    )

    async with Agent(model).run_stream('') as result:
        usage_at_each_step = [response.usage async for response in result.stream_response(debounce_by=None)]

    assert usage_at_each_step == snapshot(
        [
            RequestUsage(input_tokens=10, output_tokens=5),
            RequestUsage(input_tokens=10, output_tokens=5),
            RequestUsage(input_tokens=10, output_tokens=15),
            RequestUsage(input_tokens=10, output_tokens=15),
            RequestUsage(input_tokens=10, output_tokens=15),
        ]
    )
