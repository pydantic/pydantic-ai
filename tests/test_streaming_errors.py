"""Tests for streaming error wrapping across all providers.

Mid-stream errors from provider SDKs should be wrapped in ModelHTTPError/ModelAPIError
to enable FallbackModel and consistent error handling. See #4729.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.fallback import FallbackModel

from .conftest import try_import

with try_import() as anthropic_imports:
    from anthropic import APIConnectionError as AnthropicConnectionError, APIStatusError as AnthropicStatusError
    from anthropic.types.beta import BetaMessage, BetaRawMessageStartEvent, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic

with try_import() as openai_imports:
    from openai import APIConnectionError as OpenAIConnectionError, APIStatusError as OpenAIStatusError
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAI

with try_import() as groq_imports:
    from groq import APIStatusError as GroqStatusError
    from groq.types.chat import ChatCompletionChunk as GroqChunk
    from groq.types.chat.chat_completion_chunk import Choice as GroqChoice, ChoiceDelta as GroqChoiceDelta

    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    from .models.test_groq import MockGroq

with try_import() as bedrock_imports:
    from botocore.exceptions import ClientError

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.profiles import DEFAULT_PROFILE
    from pydantic_ai.providers import Provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _httpx_response(status_code: int, url: str = 'https://test.example.com') -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request('POST', url))


def _anthropic_start_event() -> BetaRawMessageStartEvent:
    return BetaRawMessageStartEvent(
        type='message_start',
        message=BetaMessage(
            id='msg_1',
            content=[],
            model='claude-haiku-4-5',
            role='assistant',
            stop_reason=None,
            type='message',
            usage=BetaUsage(input_tokens=1, output_tokens=0),
        ),
    )


def _openai_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id='chatcmpl-1',
        choices=[Choice(delta=ChoiceDelta(content='hello'), index=0, finish_reason=None)],
        created=1234567890,
        model='gpt-4o',
        object='chat.completion.chunk',
    )


def _groq_chunk() -> GroqChunk:
    return GroqChunk(
        id='chatcmpl-1',
        choices=[GroqChoice(delta=GroqChoiceDelta(content='hello', role='assistant'), index=0, finish_reason=None)],
        created=1234567890,
        model='llama-3.3-70b-versatile',
        object='chat.completion.chunk',
        x_groq=None,
    )


# ---------------------------------------------------------------------------
# Bedrock helpers (stub provider for error injection)
# ---------------------------------------------------------------------------


class _StubBedrockClient:
    def __init__(self, error: ClientError):
        self._error = error
        self.meta = SimpleNamespace(endpoint_url='https://bedrock.stub')

    def converse(self, **_: Any) -> None:
        raise self._error

    def converse_stream(self, **_: Any) -> None:
        raise self._error


class _StubBedrockProvider(Provider[Any]):
    def __init__(self, client: _StubBedrockClient):
        self._client = client

    @property
    def name(self) -> str:
        return 'bedrock-stub'

    @property
    def base_url(self) -> str:
        return 'https://bedrock.stub'

    @property
    def client(self) -> _StubBedrockClient:
        return self._client

    @staticmethod
    def model_profile(model_name: str):
        return DEFAULT_PROFILE


def _bedrock_model_with_error(error: ClientError) -> BedrockConverseModel:
    return BedrockConverseModel(
        'us.amazon.nova-micro-v1:0',
        provider=_StubBedrockProvider(_StubBedrockClient(error)),
    )


# ---------------------------------------------------------------------------
# Anthropic Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
async def test_anthropic_midstream_status_error(allow_model_requests: None):
    """APIStatusError during stream iteration is wrapped as ModelHTTPError."""
    error = AnthropicStatusError(
        message='Overloaded',
        response=_httpx_response(529),
        body={'type': 'error', 'error': {'type': 'overloaded_error'}},
    )
    stream = [_anthropic_start_event(), error]
    mock_client = MockAnthropic.create_stream_mock(stream)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelHTTPError) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    assert exc_info.value.status_code == 529
    assert exc_info.value.model_name == 'claude-haiku-4-5'


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
async def test_anthropic_midstream_connection_error(allow_model_requests: None):
    """APIConnectionError during stream iteration is wrapped as ModelAPIError."""
    error = AnthropicConnectionError(request=httpx.Request('POST', 'https://api.anthropic.com'))
    stream = [_anthropic_start_event(), error]
    mock_client = MockAnthropic.create_stream_mock(stream)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelAPIError) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    assert exc_info.value.model_name == 'claude-haiku-4-5'


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
async def test_anthropic_peek_error(allow_model_requests: None):
    """APIStatusError during peek is wrapped as ModelHTTPError."""
    error = AnthropicStatusError(
        message='Rate limited',
        response=_httpx_response(429),
        body={'type': 'error', 'error': {'type': 'rate_limit_error'}},
    )
    stream = [error]
    mock_client = MockAnthropic.create_stream_mock(stream)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelHTTPError) as exc_info:
        async with agent.run_stream('hello'):
            pass

    assert exc_info.value.status_code == 429


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
async def test_anthropic_midstream_sse_error_status_200(allow_model_requests: None):
    """Anthropic SSE error event arrives as APIStatusError with status_code=200 and is wrapped as ModelAPIError.

    This is the specific bug from #4729: mid-stream overloaded_error comes as HTTP 200 + SSE error event.
    """
    error = AnthropicStatusError(
        message='Overloaded',
        response=_httpx_response(200),
        body={'type': 'error', 'error': {'type': 'overloaded_error'}},
    )
    stream = [_anthropic_start_event(), error]
    mock_client = MockAnthropic.create_stream_mock(stream)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelAPIError) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    assert exc_info.value.model_name == 'claude-haiku-4-5'
    assert 'Overloaded' in exc_info.value.message


# ---------------------------------------------------------------------------
# OpenAI Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_midstream_status_error(allow_model_requests: None):
    """APIStatusError during stream iteration is wrapped as ModelHTTPError."""
    error = OpenAIStatusError(
        message='Server error',
        response=_httpx_response(500),
        body={'error': {'message': 'Internal server error'}},
    )
    stream = [_openai_chunk(), error]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelHTTPError) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    assert exc_info.value.status_code == 500
    assert exc_info.value.model_name == 'gpt-4o'


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_midstream_connection_error(allow_model_requests: None):
    """APIConnectionError during stream iteration is wrapped as ModelAPIError."""
    error = OpenAIConnectionError(request=httpx.Request('POST', 'https://api.openai.com'))
    stream = [_openai_chunk(), error]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelAPIError) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    assert exc_info.value.model_name == 'gpt-4o'


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_peek_error(allow_model_requests: None):
    """APIStatusError during peek is wrapped as ModelHTTPError."""
    error = OpenAIStatusError(
        message='Rate limited',
        response=_httpx_response(429),
        body={'error': {'message': 'Rate limit exceeded'}},
    )
    stream = [error]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelHTTPError) as exc_info:
        async with agent.run_stream('hello'):
            pass

    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# Groq Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
async def test_groq_midstream_status_error(allow_model_requests: None):
    """APIStatusError during stream iteration is wrapped as ModelHTTPError."""
    error = GroqStatusError(
        message='Service unavailable',
        response=_httpx_response(503),
        body={'error': {'message': 'Service unavailable'}},
    )
    stream = [_groq_chunk(), error]
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelHTTPError) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    assert exc_info.value.status_code == 503
    assert exc_info.value.model_name == 'llama-3.3-70b-versatile'


# ---------------------------------------------------------------------------
# Bedrock Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not bedrock_imports(), reason='botocore not installed')
async def test_bedrock_stream_creation_error(allow_model_requests: None):
    """ClientError during stream creation is wrapped as ModelHTTPError."""
    error = ClientError(
        {
            'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'},
            'ResponseMetadata': {
                'RequestId': '',
                'HostId': '',
                'HTTPStatusCode': 429,
                'HTTPHeaders': {},
                'RetryAttempts': 0,
            },
        },
        'converse_stream',
    )
    model = _bedrock_model_with_error(error)
    agent = Agent(model)

    with pytest.raises(ModelHTTPError) as exc_info:
        async with agent.run_stream('hello'):
            pass

    assert exc_info.value.status_code == 429
    assert exc_info.value.model_name == 'us.amazon.nova-micro-v1:0'


@pytest.mark.skipif(not bedrock_imports(), reason='botocore not installed')
async def test_bedrock_stream_non_http_error(allow_model_requests: None):
    """ClientError without HTTP status code is wrapped as ModelAPIError."""
    error = ClientError(
        {'Error': {'Code': 'TestException', 'Message': 'broken connection'}},
        'converse_stream',
    )
    model = _bedrock_model_with_error(error)
    agent = Agent(model)

    with pytest.raises(ModelAPIError) as exc_info:
        async with agent.run_stream('hello'):
            pass

    assert 'broken connection' in exc_info.value.message


# ---------------------------------------------------------------------------
# FallbackModel Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not anthropic_imports() or not openai_imports(), reason='anthropic+openai not installed')
async def test_fallback_model_streaming_error_triggers_fallback(allow_model_requests: None):
    """FallbackModel falls back to the next model when the first model errors during peek."""
    # First model: Anthropic that errors on peek (first event is the error)
    anthropic_error = AnthropicStatusError(
        message='Overloaded',
        response=_httpx_response(529),
        body={'type': 'error', 'error': {'type': 'overloaded_error'}},
    )
    anthropic_stream = [anthropic_error]
    anthropic_mock = MockAnthropic.create_stream_mock(anthropic_stream)
    anthropic_model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=anthropic_mock))

    # Second model: OpenAI that succeeds
    openai_finish_chunk = ChatCompletionChunk(
        id='chatcmpl-2',
        choices=[Choice(delta=ChoiceDelta(content=None), index=0, finish_reason='stop')],
        created=1234567890,
        model='gpt-4o',
        object='chat.completion.chunk',
    )
    openai_stream = [_openai_chunk(), openai_finish_chunk]
    openai_mock = MockOpenAI.create_mock_stream(openai_stream)
    openai_model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=openai_mock))

    fallback = FallbackModel(anthropic_model, openai_model)
    agent = Agent(fallback)

    async with agent.run_stream('hello') as result:
        text = await result.get_output()

    assert text == 'hello'
