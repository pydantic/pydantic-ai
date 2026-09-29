"""Provider errors are raised as the provider-neutral error categories from `pydantic_ai.exceptions`.

Not VCR tests: providers don't return rate limits, overloads, dropped connections, or context window overflows on
demand, so each case replays the error's wire shape (taken from a real response where one was recorded) through the
provider SDK, using a mock transport where the SDK has an HTTP one and a stub where it doesn't (Bedrock, xAI).
Mock transports use `localhost` base URLs, which VCR ignores.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, cast

import httpx2
import pytest

from pydantic_ai import (
    Agent,
    ContextWindowExceeded,
    ModelAPIError,
    ModelConnectionError,
    ModelHTTPError,
    ModelOverloadedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from pydantic_ai.models import Model

from ..conftest import try_import

with try_import() as imports_successful:
    import grpc
    import httpx
    from anthropic import AsyncAnthropic
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        EndpointConnectionError,
        EventStreamError,
        ReadTimeoutError,
    )
    from cohere import AsyncClientV2
    from groq import AsyncGroq
    from huggingface_hub import AsyncInferenceClient
    from huggingface_hub.errors import HfHubHTTPError
    from openai import AsyncOpenAI

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.cohere import CohereProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.huggingface import HuggingFaceProvider
    from pydantic_ai.providers.mistral import MistralProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.xai import XaiProvider

    from .mock_xai import MockXai
    from .test_bedrock import _bedrock_model_with_error  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.skipif(not imports_successful(), reason='provider SDKs not installed')

Handler = Callable[[Any], Any]
"""An `httpx` or `httpx2` mock transport handler: returns a response, or raises a transport error."""

_CATEGORIES: tuple[type[ModelAPIError], ...] = (
    ModelHTTPError,
    ModelRateLimitError,
    ModelOverloadedError,
    ModelConnectionError,
    ModelTimeoutError,
    ContextWindowExceeded,
)


def _openai(handler: Handler) -> Model:
    client = AsyncOpenAI(
        api_key='test',
        base_url='http://localhost/v1',
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(openai_client=client))


def _openrouter(handler: Handler) -> Model:
    client = AsyncOpenAI(
        api_key='test',
        base_url='http://localhost/v1',
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenRouterModel('openai/gpt-5.2', provider=OpenRouterProvider(openai_client=client))


def _anthropic(handler: Handler) -> Model:
    client = AsyncAnthropic(
        api_key='test',
        base_url='http://localhost',
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))


def _groq(handler: Handler) -> Model:
    client = AsyncGroq(
        api_key='test',
        base_url='http://localhost',
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=client))


def _google(handler: Handler) -> Model:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return GoogleModel(
        'gemini-2.5-flash',
        provider=GoogleProvider(api_key='test', http_client=http_client, base_url='http://localhost'),
    )


def _mistral(handler: Handler) -> Model:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return MistralModel(
        'mistral-large-latest',
        provider=MistralProvider(api_key='test', base_url='http://localhost', http_client=http_client),
    )


def _cohere(handler: Handler) -> Model:
    client = AsyncClientV2(
        api_key='test',
        base_url='http://localhost',
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return CohereModel('command-r7b-12-2024', provider=CohereProvider(cohere_client=client))


def _json(status_code: int, body: Any, headers: dict[str, str] | None = None) -> Handler:
    def handler(request: Any) -> Any:
        response_class = httpx.Response if isinstance(request, httpx.Request) else httpx2.Response
        return response_class(status_code, content=json.dumps(body).encode(), headers=headers)

    return handler


def _sse(*events: tuple[str, dict[str, Any]]) -> Handler:
    content = b''.join(f'event: {event}\ndata: {json.dumps(data)}\n\n'.encode() for event, data in events)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=content, headers={'content-type': 'text/event-stream'})

    return handler


def _sse_data(*data: dict[str, Any]) -> Handler:
    """An OpenAI-style SSE stream of `data:` events."""
    content = b''.join(f'data: {json.dumps(d)}\n\n'.encode() for d in data)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=content, headers={'content-type': 'text/event-stream'})

    return handler


def _raise(kind: Literal['connect', 'timeout']) -> Handler:
    def handler(request: Any) -> Any:
        if isinstance(request, httpx.Request):
            raise (httpx.ConnectError if kind == 'connect' else httpx.ReadTimeout)('failed', request=request)
        raise (httpx2.ConnectError if kind == 'connect' else httpx2.ReadTimeout)('failed', request=request)

    return handler


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[Handler], Model]
    handler: Handler
    categories: set[type[ModelAPIError]]
    """The exception's classes among `_CATEGORIES`: its error categories, plus `ModelHTTPError` if it has a status."""
    attrs: dict[str, Any] = field(default_factory=dict[str, Any])
    stream: bool = False
    has_cause: bool = True
    """Whether the error was raised from a provider SDK exception, which it is unless we parsed it from a 200 body."""


_ANTHROPIC_MESSAGE_START: tuple[str, dict[str, Any]] = (
    'message_start',
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
            'usage': {'input_tokens': 1, 'output_tokens': 1},
        },
    },
)

CASES = [
    Case(
        id='openai-rate-limit',
        model=_openai,
        handler=_json(
            429,
            {'error': {'message': 'Rate limit reached', 'type': 'requests', 'code': 'rate_limit_exceeded'}},
            headers={'retry-after': '3'},
        ),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={
            'status_code': 429,
            'provider_error_code': 'rate_limit_exceeded',
            'provider_error_type': 'requests',
            'retry_after': 3.0,
        },
    ),
    Case(
        id='openai-insufficient-quota',
        model=_openai,
        handler=_json(
            429,
            {'error': {'message': 'Quota exceeded', 'type': 'insufficient_quota', 'code': 'insufficient_quota'}},
        ),
        categories={ModelHTTPError},
        attrs={'status_code': 429, 'provider_error_code': 'insufficient_quota'},
    ),
    Case(
        id='openai-overloaded',
        model=_openai,
        handler=_json(503, {'error': {'message': 'The engine is currently overloaded', 'type': 'server_error'}}),
        categories={ModelHTTPError, ModelOverloadedError},
        attrs={'status_code': 503, 'provider_error_code': None, 'provider_error_type': 'server_error'},
    ),
    Case(
        id='openai-context-window',
        model=_openai,
        handler=_json(
            400,
            {
                'error': {
                    'message': "This model's maximum context length is 128000 tokens. However, your messages "
                    'resulted in 150008 tokens. Please reduce the length of the messages.',
                    'type': 'invalid_request_error',
                    'param': 'messages',
                    'code': 'context_length_exceeded',
                }
            },
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={'status_code': 400, 'provider_error_code': 'context_length_exceeded'},
    ),
    Case(
        id='openai-connection',
        model=_openai,
        handler=_raise('connect'),
        categories={ModelConnectionError},
    ),
    Case(
        id='openai-timeout',
        model=_openai,
        handler=_raise('timeout'),
        categories={ModelConnectionError, ModelTimeoutError},
    ),
    Case(
        id='openai-stream-rate-limit',
        model=_openai,
        handler=_sse_data(
            {'error': {'message': 'Rate limit reached', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
        ),
        categories={ModelRateLimitError},
        attrs={'provider_error_code': 'rate_limit_exceeded', 'provider_error_type': 'tokens', 'retry_after': None},
        stream=True,
    ),
    Case(
        id='openai-stream-context-window',
        model=_openai,
        handler=_sse_data(
            {
                'error': {
                    'message': "This model's maximum context length is 128000 tokens.",
                    'type': 'invalid_request_error',
                    'code': 'context_length_exceeded',
                }
            }
        ),
        categories={ContextWindowExceeded},
        attrs={
            'provider_error_code': 'context_length_exceeded',
            'body': {
                'message': "This model's maximum context length is 128000 tokens.",
                'type': 'invalid_request_error',
                'code': 'context_length_exceeded',
            },
        },
        stream=True,
    ),
    Case(
        id='openai-stream-overloaded-type',
        model=_openai,
        handler=_sse_data({'error': {'message': 'Overloaded', 'type': 'overloaded'}}),
        categories={ModelOverloadedError},
        attrs={'provider_error_code': None, 'provider_error_type': 'overloaded'},
        stream=True,
    ),
    Case(
        id='openrouter-http-context-window',
        model=_openrouter,
        handler=_json(
            400,
            {
                'error': {
                    'code': 400,
                    'message': "This endpoint's maximum context length is 128000 tokens. However, you requested "
                    'about 150008 tokens. Please reduce the length of either one.',
                }
            },
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={'status_code': 400, 'provider_error_code': '400'},
    ),
    Case(
        id='openrouter-body-rate-limit',
        model=_openrouter,
        handler=_json(200, {'error': {'code': 429, 'message': 'Rate limit exceeded'}}),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'provider_error_code': '429', 'body': 'Rate limit exceeded'},
        has_cause=False,
    ),
    Case(
        id='openrouter-body-context-window',
        model=_openrouter,
        handler=_json(
            200,
            {'error': {'code': 400, 'message': "This endpoint's maximum context length is 128000 tokens."}},
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={'status_code': 400, 'provider_error_code': '400'},
        has_cause=False,
    ),
    Case(
        id='anthropic-overloaded',
        model=_anthropic,
        handler=_json(529, {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}),
        categories={ModelHTTPError, ModelOverloadedError},
        attrs={'status_code': 529, 'provider_error_code': None, 'provider_error_type': 'overloaded_error'},
    ),
    Case(
        id='anthropic-rate-limit',
        model=_anthropic,
        handler=_json(
            429,
            {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'Rate limited'}},
            headers={'retry-after': '12'},
        ),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'provider_error_type': 'rate_limit_error', 'retry_after': 12.0},
    ),
    Case(
        id='anthropic-context-window',
        model=_anthropic,
        handler=_json(
            400,
            {
                'type': 'error',
                'error': {
                    'type': 'invalid_request_error',
                    'message': 'prompt is too long: 200027 tokens > 200000 maximum',
                },
                'request_id': 'req_011CXsbVC34PujYNC6P8wAbP',
            },
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={'status_code': 400, 'provider_error_type': 'invalid_request_error'},
    ),
    Case(
        id='anthropic-stream-overloaded',
        model=_anthropic,
        handler=_sse(
            _ANTHROPIC_MESSAGE_START,
            ('error', {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}),
        ),
        categories={ModelOverloadedError},
        attrs={
            'provider_error_type': 'overloaded_error',
            'body': {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}},
            'retry_after': None,
        },
        stream=True,
    ),
    Case(
        id='anthropic-stream-api-error',
        model=_anthropic,
        handler=_sse(
            _ANTHROPIC_MESSAGE_START,
            ('error', {'type': 'error', 'error': {'type': 'api_error', 'message': 'Internal server error'}}),
        ),
        categories=set(),
        attrs={'provider_error_type': 'api_error'},
        stream=True,
    ),
    Case(
        id='anthropic-timeout',
        model=_anthropic,
        handler=_raise('timeout'),
        categories={ModelConnectionError, ModelTimeoutError},
    ),
    Case(
        id='groq-context-window-without-code',
        model=_groq,
        handler=_json(
            400,
            {
                'error': {
                    'message': 'Please reduce the length of the messages or completion.',
                    'type': 'invalid_request_error',
                    'param': 'messages',
                }
            },
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={'status_code': 400, 'provider_error_code': None, 'provider_error_type': 'invalid_request_error'},
    ),
    Case(
        id='groq-context-window',
        model=_groq,
        handler=_json(
            400,
            {
                'error': {
                    'message': 'Please reduce the length of the messages or completion.',
                    'type': 'invalid_request_error',
                    'code': 'context_length_exceeded',
                }
            },
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={'status_code': 400, 'provider_error_code': 'context_length_exceeded'},
    ),
    Case(
        id='groq-rate-limit',
        model=_groq,
        handler=_json(
            429, {'error': {'message': 'Rate limit reached', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
        ),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'provider_error_code': 'rate_limit_exceeded', 'provider_error_type': 'tokens'},
    ),
    Case(
        id='groq-connection',
        model=_groq,
        handler=_raise('connect'),
        categories={ModelConnectionError},
    ),
    Case(
        id='groq-timeout',
        model=_groq,
        handler=_raise('timeout'),
        categories={ModelConnectionError, ModelTimeoutError},
    ),
    Case(
        id='google-rate-limit',
        model=_google,
        handler=_json(
            429, {'error': {'code': 429, 'message': 'Resource has been exhausted', 'status': 'RESOURCE_EXHAUSTED'}}
        ),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'provider_error_code': 'RESOURCE_EXHAUSTED'},
    ),
    Case(
        id='google-overloaded',
        model=_google,
        handler=_json(503, {'error': {'code': 503, 'message': 'The model is overloaded.', 'status': 'UNAVAILABLE'}}),
        categories={ModelHTTPError, ModelOverloadedError},
        attrs={'status_code': 503, 'provider_error_code': 'UNAVAILABLE'},
    ),
    Case(
        id='google-context-window',
        model=_google,
        handler=_json(
            400,
            {
                'error': {
                    'code': 400,
                    'message': 'The input token count (1100010) exceeds the maximum number of tokens allowed (1048575).',
                    'status': 'INVALID_ARGUMENT',
                }
            },
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={'status_code': 400, 'provider_error_code': 'INVALID_ARGUMENT'},
    ),
    Case(
        id='google-invalid-argument',
        model=_google,
        handler=_json(
            400, {'error': {'code': 400, 'message': 'Invalid value at `temperature`', 'status': 'INVALID_ARGUMENT'}}
        ),
        categories={ModelHTTPError},
        attrs={'status_code': 400, 'provider_error_code': 'INVALID_ARGUMENT'},
    ),
    Case(
        id='mistral-context-window',
        model=_mistral,
        handler=_json(
            400,
            {
                'object': 'error',
                'message': 'Prompt contains 150004 tokens and 0 draft tokens, too large for model with 131072 '
                'maximum context length',
                'type': 'invalid_request_invalid_args',
                'param': None,
                'code': '3051',
            },
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={
            'status_code': 400,
            'provider_error_code': '3051',
            'provider_error_type': 'invalid_request_invalid_args',
        },
    ),
    Case(
        id='mistral-rate-limit',
        model=_mistral,
        handler=_json(429, {'object': 'error', 'message': 'Requests rate limit exceeded', 'type': 'rate_limited'}),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'provider_error_code': None, 'provider_error_type': 'rate_limited'},
    ),
    Case(
        id='mistral-non-json-error',
        model=_mistral,
        handler=lambda request: httpx2.Response(502, content=b'Bad Gateway', headers={'content-type': 'text/plain'}),
        categories={ModelHTTPError},
        attrs={'status_code': 502, 'provider_error_code': None},
    ),
    Case(
        id='cohere-context-window',
        model=_cohere,
        handler=_json(
            400,
            {
                'id': 'bda42bf8-db3f-4c32-bb2c-f76466e42a1e',
                'message': 'too many tokens: size limit exceeded by 19296 tokens. Try using shorter or fewer inputs. '
                'The limit for this model is 132000 tokens.',
            },
        ),
        categories={ModelHTTPError, ContextWindowExceeded},
        attrs={'status_code': 400},
    ),
]


@pytest.mark.parametrize('case', CASES, ids=[case.id for case in CASES])
async def test_http_provider_error_category(allow_model_requests: None, case: Case):
    agent = Agent(case.model(case.handler))
    with pytest.raises(ModelAPIError) as exc_info:
        if case.stream:
            # The error arrives before the first output part, while the stream is being opened.
            await agent.run_stream('hello').__aenter__()
        else:
            await agent.run('hello')

    exc = exc_info.value
    assert {cls for cls in _CATEGORIES if isinstance(exc, cls)} == case.categories
    for attr, expected in case.attrs.items():
        assert getattr(exc, attr) == expected, attr
    assert (exc.__cause__ is not None) == case.has_cause


def _bedrock_error(code: str, message: str, status_code: int | None) -> ClientError:
    response: dict[str, Any] = {'Error': {'Code': code, 'Message': message}}
    if status_code is None:
        # An exception event inside a `ConverseStream` response: botocore gives it no response metadata.
        return EventStreamError(cast(Any, response), 'ConverseStream')
    response['ResponseMetadata'] = {'HTTPStatusCode': status_code, 'HTTPHeaders': {}}
    return ClientError(cast(Any, response), 'Converse')


@pytest.mark.parametrize(
    ('make_error', 'categories', 'attrs'),
    [
        pytest.param(
            lambda: _bedrock_error('ThrottlingException', 'Too many requests, please wait before trying again.', 429),
            {ModelHTTPError, ModelRateLimitError},
            {'status_code': 429, 'provider_error_code': 'ThrottlingException'},
            id='rate-limit',
        ),
        pytest.param(
            lambda: _bedrock_error('ServiceUnavailableException', 'Service unavailable.', 503),
            {ModelHTTPError, ModelOverloadedError},
            {'status_code': 503, 'provider_error_code': 'ServiceUnavailableException'},
            id='overloaded',
        ),
        pytest.param(
            lambda: _bedrock_error(
                'ValidationException',
                'The model returned the following errors: Input is too long for requested model.',
                400,
            ),
            {ModelHTTPError, ContextWindowExceeded},
            {'status_code': 400, 'provider_error_code': 'ValidationException'},
            id='context-window',
        ),
        pytest.param(
            lambda: _bedrock_error('ValidationException', 'Malformed input request.', 400),
            {ModelHTTPError},
            {'status_code': 400, 'provider_error_code': 'ValidationException'},
            id='validation',
        ),
        pytest.param(
            lambda: _bedrock_error('throttlingException', 'Too many tokens, please wait before trying again.', None),
            {ModelRateLimitError},
            {'provider_error_code': 'throttlingException', 'retry_after': None},
            id='stream-rate-limit',
        ),
        pytest.param(
            lambda: _bedrock_error('validationException', 'Input is too long for requested model.', None),
            {ContextWindowExceeded},
            {'provider_error_code': 'validationException'},
            id='stream-context-window',
        ),
        pytest.param(
            lambda: ReadTimeoutError(endpoint_url='https://bedrock.stub'),
            {ModelConnectionError, ModelTimeoutError},
            {},
            id='timeout',
        ),
        pytest.param(
            lambda: EndpointConnectionError(endpoint_url='https://bedrock.stub'),
            {ModelConnectionError},
            {},
            id='connection',
        ),
    ],
)
async def test_bedrock_error_category(
    allow_model_requests: None,
    make_error: Callable[[], ClientError | BotoCoreError],
    categories: set[type[ModelAPIError]],
    attrs: dict[str, Any],
):
    """The errors are built in the test, not at collection, so the module can be collected without `boto3`."""
    error = make_error()
    model = _bedrock_model_with_error(error)
    with pytest.raises(ModelAPIError) as exc_info:
        await Agent(model).run('hello')

    exc = exc_info.value
    assert {cls for cls in _CATEGORIES if isinstance(exc, cls)} == categories
    for attr, expected in attrs.items():
        assert getattr(exc, attr) == expected, attr
    assert exc.__cause__ is error


def _rpc_error(status: str, details: str) -> grpc.RpcError:
    """An `RpcError` with the named `grpc.StatusCode`, looked up in the test so collection doesn't need `grpc`."""

    class RpcError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode[status]

        def details(self) -> str:
            return details

    return RpcError()


@pytest.mark.parametrize(
    ('status', 'details', 'categories', 'attrs'),
    [
        pytest.param(
            'RESOURCE_EXHAUSTED',
            'Rate limit exceeded',
            {ModelHTTPError, ModelRateLimitError},
            {'status_code': 429, 'provider_error_code': 'RESOURCE_EXHAUSTED'},
            id='rate-limit',
        ),
        pytest.param(
            'UNAVAILABLE',
            'Service unavailable',
            {ModelHTTPError, ModelOverloadedError},
            {'status_code': 503, 'provider_error_code': 'UNAVAILABLE'},
            id='overloaded',
        ),
        pytest.param(
            'INVALID_ARGUMENT',
            "This model's maximum prompt length is 131072 but the request contains 150004 tokens.",
            {ContextWindowExceeded},
            {'provider_error_code': 'INVALID_ARGUMENT'},
            id='context-window',
        ),
        pytest.param(
            'DEADLINE_EXCEEDED',
            'Deadline Exceeded',
            {ModelHTTPError},
            {'status_code': 504, 'provider_error_code': 'DEADLINE_EXCEEDED'},
            id='deadline-exceeded',
        ),
    ],
)
async def test_xai_error_category(
    allow_model_requests: None,
    status: str,
    details: str,
    categories: set[type[ModelAPIError]],
    attrs: dict[str, Any],
):
    model = XaiModel(
        'grok-4-1-fast-non-reasoning',
        provider=XaiProvider(xai_client=MockXai.create_mock([_rpc_error(status, details)])),
    )
    with pytest.raises(ModelAPIError) as exc_info:
        await Agent(model).run('hello')

    exc = exc_info.value
    assert {cls for cls in _CATEGORIES if isinstance(exc, cls)} == categories
    for attr, expected in attrs.items():
        assert getattr(exc, attr) == expected, attr


async def test_huggingface_error_category(allow_model_requests: None):
    response = httpx.Response(
        503,
        content=b'{"error":"Model is overloaded","error_type":"overloaded"}',
        request=httpx.Request('POST', 'http://localhost/v1/chat/completions'),
    )
    error = HfHubHTTPError('Model is overloaded', response=response)

    async def create(*args: Any, **kwargs: Any) -> Any:
        raise error

    hf_client = cast(
        AsyncInferenceClient, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )
    model = HuggingFaceModel('m', provider=HuggingFaceProvider(hf_client=hf_client, api_key='test'))
    with pytest.raises(ModelOverloadedError) as exc_info:
        await Agent(model).run('hello')

    assert isinstance(exc_info.value, ModelHTTPError)
    assert exc_info.value.status_code == 503
