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
    from openai.types import responses
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAI, MockOpenAIResponses, response_message

with try_import() as groq_imports:
    from groq import APIConnectionError as GroqConnectionError, APIStatusError as GroqStatusError
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

    class _StubBedrockClient:
        def __init__(self, error: ClientError):
            self._error = error
            self.meta = SimpleNamespace(endpoint_url='https://bedrock.stub')

        def converse(self, **_: Any) -> None:  # pragma: no cover
            raise self._error

        def converse_stream(self, **_: Any) -> None:
            raise self._error

    class _StubBedrockProvider(Provider[Any]):
        def __init__(self, client: Any):
            self._client = client

        @property
        def name(self) -> str:
            return 'bedrock-stub'

        @property
        def base_url(self) -> str:  # pragma: no cover
            return 'https://bedrock.stub'

        @property
        def client(self) -> Any:
            return self._client

        @staticmethod
        def model_profile(model_name: str):
            return DEFAULT_PROFILE

    def _bedrock_model_with_error(error: ClientError) -> BedrockConverseModel:
        return BedrockConverseModel(
            'us.amazon.nova-micro-v1:0',
            provider=_StubBedrockProvider(_StubBedrockClient(error)),
        )

    class _StubBedrockStreamClient:
        """Bedrock client that returns a stream yielding one event then raising."""

        def __init__(self, error: ClientError):
            self._error = error
            self.meta = SimpleNamespace(endpoint_url='https://bedrock.stub')

        def converse_stream(self, **_: Any) -> dict[str, Any]:
            def _stream():
                yield {'messageStart': {'role': 'assistant'}}
                raise self._error

            return {'stream': _stream(), 'ResponseMetadata': {'RequestId': 'stub'}}

    def _bedrock_model_with_midstream_error(error: ClientError) -> BedrockConverseModel:
        return BedrockConverseModel(
            'us.amazon.nova-micro-v1:0',
            provider=_StubBedrockProvider(_StubBedrockStreamClient(error)),
        )


with try_import() as huggingface_imports:
    from huggingface_hub import (
        ChatCompletionStreamOutput,
        ChatCompletionStreamOutputChoice,
        ChatCompletionStreamOutputDelta,
    )
    from huggingface_hub.errors import HfHubHTTPError

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    from .models.test_huggingface import MockHuggingFace

with try_import() as mistral_imports:
    from mistralai.models import (
        CompletionChunk as MistralCompletionChunk,
        CompletionEvent as MistralCompletionEvent,
        CompletionResponseStreamChoice as MistralCompletionResponseStreamChoice,
        DeltaMessage as MistralDeltaMessage,
        SDKError,
        UsageInfo as MistralUsageInfo,
    )

    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

    from .models.test_mistral import MockMistralAI

with try_import() as xai_imports:
    import grpc

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

    from .models.mock_xai import MockXai, get_grok_text_chunk

    class _StubRpcError(grpc.RpcError):
        """Stub gRPC error with configurable code and details."""

        def __init__(self, code: grpc.StatusCode, details: str):
            self._code = code
            self._details = details

        def code(self) -> grpc.StatusCode:
            return self._code

        def details(self) -> str:
            return self._details


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
@pytest.mark.parametrize(
    'error_factory,expected_exc',
    [
        pytest.param(
            lambda: AnthropicStatusError(message='SSE error', response=_httpx_response(200), body={'type': 'error'}),
            ModelAPIError,
            id='status_lt_400',
        ),
        pytest.param(
            lambda: AnthropicConnectionError(request=httpx.Request('POST', 'https://api.anthropic.com')),
            ModelAPIError,
            id='connection',
        ),
    ],
)
async def test_anthropic_peek_non_http_error(
    allow_model_requests: None, error_factory: Any, expected_exc: type[Exception]
):
    """APIStatusError with status<400 or APIConnectionError during peek is wrapped as ModelAPIError."""
    stream = [error_factory()]
    mock_client = MockAnthropic.create_stream_mock(stream)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc):
        async with agent.run_stream('hello'):
            pass


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


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize(
    'error_factory,expected_exc',
    [
        pytest.param(
            lambda: OpenAIStatusError(message='SSE error', response=_httpx_response(200), body={}),
            ModelAPIError,
            id='status_lt_400',
        ),
        pytest.param(
            lambda: OpenAIConnectionError(request=httpx.Request('POST', 'https://api.openai.com')),
            ModelAPIError,
            id='connection',
        ),
    ],
)
async def test_openai_peek_non_http_error(
    allow_model_requests: None, error_factory: Any, expected_exc: type[Exception]
):
    """APIStatusError with status<400 or APIConnectionError during peek is wrapped as ModelAPIError."""
    stream = [error_factory()]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc):
        async with agent.run_stream('hello'):
            pass


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_midstream_non_http_error(allow_model_requests: None):
    """APIStatusError with status<400 during stream iteration is wrapped as ModelAPIError."""
    error = OpenAIStatusError(message='SSE error', response=_httpx_response(200), body={})
    stream = [_openai_chunk(), error]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(ModelAPIError):
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass


# ---------------------------------------------------------------------------
# OpenAI Responses Tests
# ---------------------------------------------------------------------------


def _openai_responses_created_event() -> responses.ResponseCreatedEvent:
    resp = response_message([])
    return responses.ResponseCreatedEvent(response=resp, type='response.created', sequence_number=0)


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize(
    'error_factory,expected_exc,expected_status',
    [
        pytest.param(
            lambda: OpenAIStatusError(message='Server error', response=_httpx_response(500), body={}),
            ModelHTTPError,
            500,
            id='http',
        ),
        pytest.param(
            lambda: OpenAIStatusError(message='SSE error', response=_httpx_response(200), body={}),
            ModelAPIError,
            None,
            id='status_lt_400',
        ),
        pytest.param(
            lambda: OpenAIConnectionError(request=httpx.Request('POST', 'https://api.openai.com')),
            ModelAPIError,
            None,
            id='connection',
        ),
    ],
)
async def test_openai_responses_peek_error(
    allow_model_requests: None, error_factory: Any, expected_exc: type[Exception], expected_status: int | None
):
    """Errors during peek on OpenAI Responses model are wrapped correctly."""
    stream = [error_factory()]
    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc) as exc_info:
        async with agent.run_stream('hello'):
            pass

    if expected_status is not None:
        assert isinstance(exc_info.value, ModelHTTPError)
        assert exc_info.value.status_code == expected_status


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize(
    'error_factory,expected_exc',
    [
        pytest.param(
            lambda: OpenAIStatusError(message='Server error', response=_httpx_response(500), body={}),
            ModelHTTPError,
            id='http',
        ),
        pytest.param(
            lambda: OpenAIStatusError(message='SSE error', response=_httpx_response(200), body={}),
            ModelAPIError,
            id='status_lt_400',
        ),
        pytest.param(
            lambda: OpenAIConnectionError(request=httpx.Request('POST', 'https://api.openai.com')),
            ModelAPIError,
            id='connection',
        ),
    ],
)
async def test_openai_responses_midstream_error(
    allow_model_requests: None, error_factory: Any, expected_exc: type[Exception]
):
    """Errors during stream iteration on OpenAI Responses model are wrapped correctly."""
    stream = [_openai_responses_created_event(), error_factory()]
    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc):
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass


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


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
@pytest.mark.parametrize(
    'error_factory,expected_exc',
    [
        pytest.param(
            lambda: GroqStatusError(message='SSE error', response=_httpx_response(200), body={}),
            ModelAPIError,
            id='status_lt_400',
        ),
        pytest.param(
            lambda: GroqConnectionError(request=httpx.Request('POST', 'https://api.groq.com')),
            ModelAPIError,
            id='connection',
        ),
    ],
)
async def test_groq_midstream_non_http_error(
    allow_model_requests: None, error_factory: Any, expected_exc: type[Exception]
):
    """APIStatusError with status<400 or APIConnectionError during iteration is wrapped as ModelAPIError."""
    stream = [_groq_chunk(), error_factory()]
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc):
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass


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


@pytest.mark.skipif(not bedrock_imports(), reason='botocore not installed')
@pytest.mark.parametrize(
    'error_factory,expected_exc,check_status',
    [
        pytest.param(
            lambda: ClientError(
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
            ),
            ModelHTTPError,
            429,
            id='http',
        ),
        pytest.param(
            lambda: ClientError({'Error': {'Code': 'TestException', 'Message': 'broken'}}, 'converse_stream'),
            ModelAPIError,
            None,
            id='non_http',
        ),
    ],
)
async def test_bedrock_midstream_error(
    allow_model_requests: None, error_factory: Any, expected_exc: type[Exception], check_status: int | None
):
    """ClientError during stream iteration is wrapped correctly."""
    model = _bedrock_model_with_midstream_error(error_factory())
    agent = Agent(model)

    with pytest.raises(expected_exc) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    if check_status is not None:
        assert isinstance(exc_info.value, ModelHTTPError)
        assert exc_info.value.status_code == check_status


# ---------------------------------------------------------------------------
# HuggingFace Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not huggingface_imports(), reason='huggingface_hub not installed')
async def test_huggingface_midstream_error(allow_model_requests: None):
    """HfHubHTTPError during stream iteration is wrapped as ModelHTTPError."""
    hf_chunk = ChatCompletionStreamOutput(
        id='x',
        choices=[
            ChatCompletionStreamOutputChoice(
                index=0, delta=ChatCompletionStreamOutputDelta(content='hi', role='assistant'), finish_reason=None
            )
        ],
        created=1704067200,
        model='hf-model',
        system_fingerprint='',
    )
    error = HfHubHTTPError(
        'Server error',
        response=httpx.Response(500, request=httpx.Request('POST', 'https://api.hf.co'), content=b'error'),
    )
    stream = [hf_chunk, error]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('test-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='test'))
    agent = Agent(m)

    with pytest.raises(ModelHTTPError) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# Mistral Tests
# ---------------------------------------------------------------------------


def _mistral_chunk() -> MistralCompletionEvent:
    return MistralCompletionEvent(
        data=MistralCompletionChunk(
            id='x',
            choices=[
                MistralCompletionResponseStreamChoice(
                    index=0, delta=MistralDeltaMessage(content='hi', role='assistant'), finish_reason=None
                )
            ],
            created=1704067200,
            model='mistral-large',
            object='chat.completion.chunk',
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )


def _mistral_client_raising(error: Any) -> Any:
    """Create a minimal Mistral client mock where stream_async raises the given error."""

    async def _raise(**_: Any) -> None:
        raise error

    return SimpleNamespace(
        chat=SimpleNamespace(stream_async=_raise),
        sdk_configuration=SimpleNamespace(get_server_details=lambda: ('https://api.mistral.ai',)),
    )


def _mistral_sdk_error(status_code: int, message: str) -> SDKError:
    return SDKError(message, httpx.Response(status_code, request=httpx.Request('POST', 'https://api.mistral.ai')))


@pytest.mark.skipif(not mistral_imports(), reason='mistral not installed')
@pytest.mark.parametrize(
    'error_factory,expected_exc,check_status',
    [
        pytest.param(lambda: _mistral_sdk_error(500, 'Server error'), ModelHTTPError, 500, id='http'),
        pytest.param(lambda: _mistral_sdk_error(200, 'SSE error'), ModelAPIError, None, id='non_http'),
    ],
)
async def test_mistral_stream_creation_error(
    allow_model_requests: None, error_factory: Any, expected_exc: type[Exception], check_status: int | None
):
    """SDKError during stream creation is wrapped correctly."""
    mock_client = _mistral_client_raising(error_factory())
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc) as exc_info:
        async with agent.run_stream('hello'):
            pass

    if check_status is not None:
        assert isinstance(exc_info.value, ModelHTTPError)
        assert exc_info.value.status_code == check_status


@pytest.mark.skipif(not mistral_imports(), reason='mistral not installed')
@pytest.mark.parametrize(
    'error_factory,expected_exc,check_status',
    [
        pytest.param(lambda: _mistral_sdk_error(500, 'Server error'), ModelHTTPError, 500, id='http'),
        pytest.param(lambda: _mistral_sdk_error(200, 'SSE error'), ModelAPIError, None, id='non_http'),
    ],
)
async def test_mistral_midstream_error(
    allow_model_requests: None, error_factory: Any, expected_exc: type[Exception], check_status: int | None
):
    """SDKError during stream iteration is wrapped correctly."""
    mock_client = MockMistralAI.create_stream_mock([_mistral_chunk(), error_factory()])
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    if check_status is not None:
        assert isinstance(exc_info.value, ModelHTTPError)
        assert exc_info.value.status_code == check_status


# ---------------------------------------------------------------------------
# xAI Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not xai_imports(), reason='xai-sdk not installed')
@pytest.mark.parametrize(
    'grpc_code_name,expected_exc,expected_status',
    [
        pytest.param('UNAVAILABLE', ModelHTTPError, 503, id='http_mappable'),
        pytest.param('CANCELLED', ModelAPIError, None, id='unmapped'),
    ],
)
async def test_xai_request_error(
    allow_model_requests: None, grpc_code_name: str, expected_exc: type[Exception], expected_status: int | None
):
    """gRPC errors during non-streaming request are wrapped correctly."""
    error = _StubRpcError(getattr(grpc.StatusCode, grpc_code_name), 'gRPC error')
    mock_client = MockXai.create_mock([error])
    m = XaiModel('grok-3-mini', provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc) as exc_info:
        await agent.run('hello')

    if expected_status is not None:
        assert isinstance(exc_info.value, ModelHTTPError)
        assert exc_info.value.status_code == expected_status


@pytest.mark.skipif(not xai_imports(), reason='xai-sdk not installed')
@pytest.mark.parametrize(
    'grpc_code_name,expected_exc,expected_status',
    [
        pytest.param('UNAVAILABLE', ModelHTTPError, 503, id='http_mappable'),
        pytest.param('CANCELLED', ModelAPIError, None, id='unmapped'),
    ],
)
async def test_xai_midstream_error(
    allow_model_requests: None, grpc_code_name: str, expected_exc: type[Exception], expected_status: int | None
):
    """gRPC errors during stream iteration are wrapped correctly."""
    error = _StubRpcError(getattr(grpc.StatusCode, grpc_code_name), 'gRPC error')
    stream_data = [[get_grok_text_chunk('hello'), error]]
    mock_client = MockXai.create_mock_stream(stream_data)  # pyright: ignore[reportArgumentType]
    m = XaiModel('grok-3-mini', provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(expected_exc) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text():
                pass

    if expected_status is not None:
        assert isinstance(exc_info.value, ModelHTTPError)
        assert exc_info.value.status_code == expected_status


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
