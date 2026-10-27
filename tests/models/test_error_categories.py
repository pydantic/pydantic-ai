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
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
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

    from .mock_xai import MockXai, get_grok_text_chunk
    from .test_bedrock import _bedrock_model_with_error  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.skipif(not imports_successful(), reason='provider SDKs not installed')

Handler = Callable[[Any], Any]
"""An `httpx` or `httpx2` mock transport handler: returns a response, or raises a transport error."""

_CATEGORIES: tuple[type[ModelAPIError], ...] = (
    ModelHTTPError,
    ModelRateLimitError,
    ModelQuotaExceededError,
    ModelUnavailableError,
    ModelOverloadedError,
    ModelServerError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelContextWindowExceededError,
    ModelRequestRejectedError,
    ModelAuthenticationError,
    ModelNotFoundError,
)

_OVERLOADED: set[type[ModelAPIError]] = {ModelHTTPError, ModelUnavailableError, ModelOverloadedError}
"""An overloaded provider is also unavailable."""

_REJECTED: set[type[ModelAPIError]] = {ModelHTTPError, ModelRequestRejectedError}
_AUTHENTICATION: set[type[ModelAPIError]] = {ModelHTTPError, ModelRequestRejectedError, ModelAuthenticationError}
"""A rejected authentication is also a rejected request."""
_NOT_FOUND: set[type[ModelAPIError]] = {ModelHTTPError, ModelRequestRejectedError, ModelNotFoundError}
_CONTEXT_WINDOW: set[type[ModelAPIError]] = {ModelHTTPError, ModelRequestRejectedError, ModelContextWindowExceededError}
"""A context window overflow is also a rejected request."""


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


def _raise(kind: Literal['connect', 'timeout', 'pool']) -> Handler:
    def handler(request: Any) -> Any:
        if isinstance(request, httpx.Request):
            error_class = {'connect': httpx.ConnectError, 'timeout': httpx.ReadTimeout, 'pool': httpx.PoolTimeout}[kind]
            raise error_class('failed', request=request)
        error_class2 = {'connect': httpx2.ConnectError, 'timeout': httpx2.ReadTimeout, 'pool': httpx2.PoolTimeout}[kind]
        raise error_class2('failed', request=request)

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
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 429, 'provider_error_code': 'insufficient_quota'},
    ),
    Case(
        id='openai-overloaded',
        model=_openai,
        handler=_json(503, {'error': {'message': 'The engine is currently overloaded', 'type': 'server_error'}}),
        categories=_OVERLOADED,
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
        categories=_CONTEXT_WINDOW,
        attrs={'status_code': 400, 'provider_error_code': 'context_length_exceeded'},
    ),
    Case(
        id='openai-connection',
        model=_openai,
        handler=_raise('connect'),
        categories={ModelConnectionError},
        attrs={'phase': 'connect'},
    ),
    Case(
        id='openai-timeout',
        model=_openai,
        handler=_raise('timeout'),
        categories={ModelConnectionError, ModelTimeoutError},
        attrs={'phase': 'read'},
    ),
    Case(
        id='openai-pool-timeout',
        model=_openai,
        handler=_raise('pool'),
        categories={ModelConnectionError, ModelTimeoutError},
        attrs={'phase': 'pool'},
    ),
    Case(
        id='openai-stream-rate-limit',
        model=_openai,
        handler=_sse_data(
            {'error': {'message': 'Rate limit reached', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
        ),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={
            'status_code': 429,
            'in_stream': True,
            'provider_error_code': 'rate_limit_exceeded',
            'provider_error_type': 'tokens',
            'retry_after': None,
        },
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
        categories=_CONTEXT_WINDOW,
        attrs={
            'status_code': 400,
            'in_stream': True,
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
        id='openai-stream-numeric-code',
        model=_openai,
        # Gateways like OpenRouter send the HTTP status as the `code`.
        handler=_sse_data({'error': {'message': 'Rate limited', 'code': 429}}),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'in_stream': True, 'provider_error_code': '429'},
        stream=True,
    ),
    Case(
        id='openai-stream-unclassified',
        model=_openai,
        handler=_sse_data({'error': {'message': 'Something went wrong'}}),
        categories=set(),
        attrs={'in_stream': True, 'message': 'Something went wrong'},
        stream=True,
    ),
    Case(
        id='openai-stream-overloaded-type',
        model=_openai,
        handler=_sse_data({'error': {'message': 'Overloaded', 'type': 'overloaded'}}),
        categories=_OVERLOADED,
        attrs={'status_code': 503, 'in_stream': True, 'provider_error_code': None, 'provider_error_type': 'overloaded'},
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
        categories=_CONTEXT_WINDOW,
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
        categories=_CONTEXT_WINDOW,
        attrs={'status_code': 400, 'provider_error_code': '400'},
        has_cause=False,
    ),
    Case(
        id='anthropic-overloaded',
        model=_anthropic,
        handler=_json(529, {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}),
        categories=_OVERLOADED,
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
        categories=_CONTEXT_WINDOW,
        attrs={'status_code': 400, 'provider_error_type': 'invalid_request_error'},
    ),
    Case(
        id='anthropic-stream-overloaded',
        model=_anthropic,
        handler=_sse(
            _ANTHROPIC_MESSAGE_START,
            ('error', {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}),
        ),
        # The error event follows a 200, but `AnthropicModel` reports the status the same error has on a
        # non-streaming request, since it may stream a `run()` behind the scenes.
        categories=_OVERLOADED,
        attrs={
            'status_code': 529,
            'in_stream': True,
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
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500, 'in_stream': True, 'provider_error_type': 'api_error'},
        stream=True,
    ),
    Case(
        id='anthropic-timeout',
        model=_anthropic,
        handler=_raise('timeout'),
        categories={ModelConnectionError, ModelTimeoutError},
        attrs={'phase': 'read'},
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
        categories=_CONTEXT_WINDOW,
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
        categories=_CONTEXT_WINDOW,
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
        attrs={'phase': 'connect'},
    ),
    Case(
        id='groq-timeout',
        model=_groq,
        handler=_raise('timeout'),
        categories={ModelConnectionError, ModelTimeoutError},
        attrs={'phase': 'read'},
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
        categories=_OVERLOADED,
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
        categories=_CONTEXT_WINDOW,
        attrs={'status_code': 400, 'provider_error_code': 'INVALID_ARGUMENT'},
    ),
    Case(
        id='google-invalid-argument',
        model=_google,
        handler=_json(
            400, {'error': {'code': 400, 'message': 'Invalid value at `temperature`', 'status': 'INVALID_ARGUMENT'}}
        ),
        categories=_REJECTED,
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
        categories=_CONTEXT_WINDOW,
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
        categories={ModelHTTPError, ModelServerError},
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
        categories=_CONTEXT_WINDOW,
        attrs={'status_code': 400},
    ),
    Case(
        id='openai-unavailable',
        model=_openai,
        handler=_json(503, {'error': {'message': 'Service Unavailable', 'type': 'server_error'}}),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503},
    ),
    Case(
        id='openai-server-error',
        model=_openai,
        handler=_json(500, {'error': {'message': 'The server had an error', 'type': 'server_error'}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500, 'provider_error_type': 'server_error'},
    ),
    Case(
        id='openai-bad-gateway',
        model=_openai,
        handler=_json(502, {'error': {'message': 'Bad gateway', 'type': 'server_error'}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 502},
    ),
    Case(
        id='openai-billing-hard-limit',
        model=_openai,
        handler=_json(
            400,
            {
                'error': {
                    'message': 'Billing hard limit has been reached',
                    'type': 'invalid_request_error',
                    'code': 'billing_hard_limit_reached',
                }
            },
        ),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 400, 'provider_error_code': 'billing_hard_limit_reached'},
    ),
    Case(
        id='openai-stream-insufficient-quota',
        model=_openai,
        handler=_sse_data({'error': {'message': 'Quota exceeded', 'code': 'insufficient_quota'}}),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 429, 'in_stream': True, 'provider_error_code': 'insufficient_quota'},
        stream=True,
    ),
    Case(
        id='openai-credit-balance-exhausted',
        model=_openai,
        handler=_json(
            429,
            {'error': {'message': 'Limit reached', 'type': 'rate_limit_error', 'code': 'credit_balance_exhausted'}},
        ),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 429, 'provider_error_code': 'credit_balance_exhausted'},
    ),
    Case(
        id='openai-organization-spend-limit-exceeded',
        model=_openai,
        handler=_json(
            429,
            {
                'error': {
                    'message': 'Limit reached',
                    'type': 'rate_limit_error',
                    'code': 'organization_spend_limit_exceeded',
                }
            },
        ),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 429, 'provider_error_code': 'organization_spend_limit_exceeded'},
    ),
    Case(
        id='openai-project-spend-limit-exceeded',
        model=_openai,
        handler=_json(
            429,
            {
                'error': {
                    'message': 'Limit reached',
                    'type': 'rate_limit_error',
                    'code': 'project_spend_limit_exceeded',
                }
            },
        ),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 429, 'provider_error_code': 'project_spend_limit_exceeded'},
    ),
    Case(
        id='openai-organization-usage-limit-exceeded',
        model=_openai,
        handler=_json(
            429,
            {
                'error': {
                    'message': 'Limit reached',
                    'type': 'rate_limit_error',
                    'code': 'organization_usage_limit_exceeded',
                }
            },
        ),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 429, 'provider_error_code': 'organization_usage_limit_exceeded'},
    ),
    Case(
        id='openai-too-many-requests',
        model=_openai,
        handler=_json(
            429,
            {
                'error': {
                    'message': 'No capacity',
                    'type': 'too_many_requests',
                    'code': 'no_capacity',
                }
            },
        ),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'provider_error_code': 'no_capacity'},
    ),
    Case(
        id='openai-stream-credit-balance-exhausted',
        model=_openai,
        handler=_sse_data({'error': {'message': 'No credits left', 'code': 'credit_balance_exhausted'}}),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 429, 'in_stream': True, 'provider_error_code': 'credit_balance_exhausted'},
        stream=True,
    ),
    Case(
        id='openai-stream-server-is-overloaded',
        model=_openai,
        handler=_sse_data(
            {
                'error': {
                    'message': 'Slow down',
                    'code': 'server_is_overloaded',
                    'type': 'service_unavailable_error',
                }
            }
        ),
        categories=_OVERLOADED,
        attrs={'status_code': 503, 'in_stream': True, 'provider_error_code': 'server_is_overloaded'},
        stream=True,
    ),
    Case(
        id='openai-stream-slow-down',
        model=_openai,
        handler=_sse_data({'error': {'message': 'Slow down', 'code': 'slow_down', 'type': 'rate_limit_error'}}),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'in_stream': True, 'provider_error_type': 'rate_limit_error'},
        stream=True,
    ),
    Case(
        id='openai-stream-too-many-requests',
        model=_openai,
        handler=_sse_data({'error': {'message': 'No capacity', 'type': 'too_many_requests', 'code': 'no_capacity'}}),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={
            'status_code': 429,
            'in_stream': True,
            'provider_error_code': 'no_capacity',
            'provider_error_type': 'too_many_requests',
        },
        stream=True,
    ),
    Case(
        id='openai-stream-server-error',
        model=_openai,
        handler=_sse_data({'error': {'message': 'The server had an error', 'code': 'server_error'}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500, 'in_stream': True, 'provider_error_code': 'server_error'},
        stream=True,
    ),
    Case(
        id='openai-stream-unavailable',
        model=_openai,
        handler=_sse_data({'error': {'message': 'Service unavailable', 'code': 'service_unavailable'}}),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503, 'in_stream': True, 'provider_error_code': 'service_unavailable'},
        stream=True,
    ),
    Case(
        id='openrouter-body-insufficient-credits',
        model=_openrouter,
        handler=_json(200, {'error': {'code': 402, 'message': 'Insufficient credits'}}),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 402, 'provider_error_code': '402'},
        has_cause=False,
    ),
    Case(
        id='openrouter-body-bad-gateway',
        model=_openrouter,
        handler=_json(200, {'error': {'code': 502, 'message': 'Provider returned error'}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 502, 'provider_error_code': '502'},
        has_cause=False,
    ),
    Case(
        id='anthropic-billing-error',
        model=_anthropic,
        handler=_json(402, {'type': 'error', 'error': {'type': 'billing_error', 'message': 'Billing issue'}}),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 402, 'provider_error_type': 'billing_error'},
    ),
    Case(
        id='anthropic-credit-balance',
        model=_anthropic,
        handler=_json(
            400,
            {
                'type': 'error',
                'error': {
                    'type': 'invalid_request_error',
                    'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & '
                    'Billing to upgrade or purchase credits.',
                },
            },
        ),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 400, 'provider_error_type': 'invalid_request_error'},
    ),
    Case(
        id='anthropic-api-error',
        model=_anthropic,
        handler=_json(500, {'type': 'error', 'error': {'type': 'api_error', 'message': 'Internal server error'}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500, 'provider_error_type': 'api_error'},
    ),
    Case(
        id='anthropic-unavailable',
        model=_anthropic,
        handler=_json(503, {'type': 'error', 'error': {'type': 'api_error', 'message': 'Service unavailable'}}),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503},
    ),
    Case(
        id='anthropic-stream-rate-limit',
        model=_anthropic,
        handler=_sse(
            _ANTHROPIC_MESSAGE_START,
            ('error', {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'Rate limited'}}),
        ),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429, 'in_stream': True, 'provider_error_type': 'rate_limit_error'},
        stream=True,
    ),
    Case(
        id='groq-unavailable',
        model=_groq,
        handler=_json(503, {'error': {'message': 'Service Unavailable', 'type': 'internal_server_error'}}),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503},
    ),
    Case(
        id='groq-server-error',
        model=_groq,
        handler=_json(500, {'error': {'message': 'Internal Server Error', 'type': 'internal_server_error'}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500},
    ),
    Case(
        id='google-unavailable',
        model=_google,
        handler=_json(
            503, {'error': {'code': 503, 'message': 'The service is currently unavailable.', 'status': 'UNAVAILABLE'}}
        ),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503, 'provider_error_code': 'UNAVAILABLE'},
    ),
    Case(
        id='google-internal',
        model=_google,
        handler=_json(
            500, {'error': {'code': 500, 'message': 'An internal error has occurred.', 'status': 'INTERNAL'}}
        ),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500, 'provider_error_code': 'INTERNAL'},
    ),
    Case(
        id='google-deadline-exceeded',
        model=_google,
        handler=_json(
            504,
            {
                'error': {
                    'code': 504,
                    'message': 'Deadline expired before operation could complete.',
                    'status': 'DEADLINE_EXCEEDED',
                }
            },
        ),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 504, 'provider_error_code': 'DEADLINE_EXCEEDED'},
    ),
    Case(
        id='google-stream-unavailable',
        model=_google,
        handler=_sse_data(
            {'error': {'code': 503, 'message': 'The service is currently unavailable.', 'status': 'UNAVAILABLE'}}
        ),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503, 'in_stream': True, 'provider_error_code': 'UNAVAILABLE'},
        stream=True,
    ),
    Case(
        id='google-stream-internal',
        model=_google,
        handler=_sse_data({'error': {'code': 500, 'message': 'An internal error has occurred.', 'status': 'INTERNAL'}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500, 'in_stream': True, 'provider_error_code': 'INTERNAL'},
        stream=True,
    ),
    Case(
        id='mistral-unavailable',
        model=_mistral,
        handler=_json(503, {'object': 'error', 'message': 'Service unavailable', 'type': 'service_unavailable'}),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503},
    ),
    Case(
        # A 529 is overloaded by its status alone, even when neither the body's type nor its message says so.
        id='mistral-status-529',
        model=_mistral,
        handler=_json(529, {'object': 'error', 'message': 'Please try again later', 'type': 'unknown_error'}),
        categories=_OVERLOADED,
        attrs={'status_code': 529},
    ),
    Case(
        id='mistral-payment-required',
        model=_mistral,
        handler=_json(402, {'object': 'error', 'message': 'Payment required', 'type': 'payment_required'}),
        categories={ModelHTTPError, ModelQuotaExceededError},
        attrs={'status_code': 402},
    ),
    Case(
        id='cohere-server-error',
        model=_cohere,
        handler=_json(500, {'id': 'x', 'message': 'internal server error'}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500},
    ),
    Case(
        id='cohere-unavailable',
        model=_cohere,
        handler=_json(503, {'id': 'x', 'message': 'service unavailable'}),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503},
    ),
    Case(
        id='openai-invalid-api-key',
        model=_openai,
        handler=_json(
            401,
            {
                'error': {
                    'message': 'Incorrect API key provided: sk-test. You can find your API key at '
                    'https://platform.openai.com/account/api-keys.',
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': 'invalid_api_key',
                }
            },
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401, 'provider_error_code': 'invalid_api_key'},
    ),
    Case(
        id='openai-unsupported-country',
        model=_openai,
        handler=_json(
            403,
            {
                'error': {
                    'message': 'Country, region, or territory not supported',
                    'type': 'request_forbidden',
                    'param': None,
                    'code': 'unsupported_country_region_territory',
                }
            },
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 403, 'provider_error_code': 'unsupported_country_region_territory'},
    ),
    Case(
        id='openai-model-not-found',
        model=_openai,
        handler=_json(
            404,
            {
                'error': {
                    'message': 'The model `gpt-5.2` does not exist or you do not have access to it.',
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': 'model_not_found',
                }
            },
        ),
        categories=_NOT_FOUND,
        attrs={'status_code': 404, 'provider_error_code': 'model_not_found'},
    ),
    Case(
        # The Responses API sends an unknown model as a 400; its code says what it is.
        id='openai-model-not-found-400',
        model=_openai,
        handler=_json(
            400,
            {
                'error': {
                    'message': "The requested model 'gpt-5.2' does not exist.",
                    'type': 'invalid_request_error',
                    'param': 'model',
                    'code': 'model_not_found',
                }
            },
        ),
        categories=_NOT_FOUND,
        attrs={'status_code': 400, 'provider_error_code': 'model_not_found'},
    ),
    Case(
        id='openai-bad-request',
        model=_openai,
        handler=_json(
            400,
            {
                'error': {
                    'message': "Invalid value for 'temperature': expected a number.",
                    'type': 'invalid_request_error',
                    'param': 'temperature',
                    'code': None,
                }
            },
        ),
        categories=_REJECTED,
        attrs={'status_code': 400, 'provider_error_type': 'invalid_request_error'},
    ),
    Case(
        id='openai-unprocessable',
        model=_openai,
        handler=_json(422, {'error': {'message': 'Unprocessable entity', 'type': 'invalid_request_error'}}),
        categories=_REJECTED,
        attrs={'status_code': 422},
    ),
    Case(
        id='openai-conflict',
        model=_openai,
        handler=_json(409, {'error': {'message': 'Conflict', 'type': 'invalid_request_error'}}),
        categories={ModelHTTPError},
        attrs={'status_code': 409},
    ),
    Case(
        # Its status before a stream opens depends on the API, so an in-stream one gets none, only the category.
        id='openai-stream-model-not-found',
        model=_openai,
        handler=_sse_data({'error': {'message': 'The model does not exist.', 'code': 'model_not_found'}}),
        categories={ModelRequestRejectedError, ModelNotFoundError},
        attrs={'in_stream': True, 'provider_error_code': 'model_not_found'},
        stream=True,
    ),
    Case(
        id='openrouter-body-unauthorized',
        model=_openrouter,
        handler=_json(200, {'error': {'code': 401, 'message': 'No auth credentials found'}}),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401, 'provider_error_code': '401'},
        has_cause=False,
    ),
    Case(
        # OpenRouter sends flagged input as a 403 with the moderation result; it's not a credentials problem.
        id='openrouter-http-moderation',
        model=_openrouter,
        handler=_json(
            403,
            {
                'error': {
                    'code': 403,
                    'message': 'openai/gpt-5.2 requires moderation',
                    'metadata': {
                        'reasons': ['violence'],
                        'flagged_input': 'How do I...',
                        'provider_name': 'OpenAI',
                        'model_slug': 'openai/gpt-5.2',
                    },
                }
            },
        ),
        categories=_REJECTED,
        attrs={'status_code': 403},
    ),
    Case(
        id='openrouter-body-moderation',
        model=_openrouter,
        handler=_json(
            200,
            {
                'error': {
                    'code': 403,
                    'message': 'openai/gpt-5.2 requires moderation',
                    'metadata': {
                        'reasons': ['violence'],
                        'flagged_input': 'How do I...',
                        'provider_name': 'OpenAI',
                        'model_slug': 'openai/gpt-5.2',
                    },
                }
            },
        ),
        categories=_REJECTED,
        attrs={'status_code': 403, 'provider_error_code': '403'},
        has_cause=False,
    ),
    Case(
        id='openrouter-http-forbidden',
        model=_openrouter,
        handler=_json(403, {'error': {'code': 403, 'message': 'Forbidden'}}),
        categories=_AUTHENTICATION,
        attrs={'status_code': 403},
    ),
    Case(
        id='openrouter-http-bad-request',
        model=_openrouter,
        handler=_json(400, {'error': {'code': 400, 'message': 'Provider returned error'}}),
        categories=_REJECTED,
        attrs={'status_code': 400},
    ),
    Case(
        id='anthropic-authentication',
        model=_anthropic,
        handler=_json(
            401, {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401, 'provider_error_type': 'authentication_error'},
    ),
    Case(
        id='anthropic-permission',
        model=_anthropic,
        handler=_json(
            403,
            {
                'type': 'error',
                'error': {
                    'type': 'permission_error',
                    'message': 'Your API key does not have permission to use the specified resource.',
                },
            },
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 403, 'provider_error_type': 'permission_error'},
    ),
    Case(
        id='anthropic-not-found',
        model=_anthropic,
        handler=_json(
            404, {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-sonnet-4-5'}}
        ),
        categories=_NOT_FOUND,
        attrs={'status_code': 404, 'provider_error_type': 'not_found_error'},
    ),
    Case(
        id='anthropic-invalid-request',
        model=_anthropic,
        handler=_json(
            400,
            {
                'type': 'error',
                'error': {
                    'type': 'invalid_request_error',
                    'message': 'messages: text content blocks must be non-empty',
                },
            },
        ),
        categories=_REJECTED,
        attrs={'status_code': 400, 'provider_error_type': 'invalid_request_error'},
    ),
    Case(
        id='anthropic-request-too-large',
        model=_anthropic,
        handler=_json(
            413,
            {'type': 'error', 'error': {'type': 'request_too_large', 'message': 'Request exceeds the maximum size'}},
        ),
        categories={ModelHTTPError},
        attrs={'status_code': 413, 'provider_error_type': 'request_too_large'},
    ),
    Case(
        id='anthropic-stream-invalid-request',
        model=_anthropic,
        handler=_sse(
            _ANTHROPIC_MESSAGE_START,
            ('error', {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Invalid request'}}),
        ),
        categories=_REJECTED,
        attrs={'status_code': 400, 'in_stream': True, 'provider_error_type': 'invalid_request_error'},
        stream=True,
    ),
    Case(
        id='groq-model-not-found',
        model=_groq,
        handler=_json(
            404,
            {
                'error': {
                    'message': 'The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.',
                    'type': 'invalid_request_error',
                    'code': 'model_not_found',
                }
            },
        ),
        categories=_NOT_FOUND,
        attrs={'status_code': 404, 'provider_error_code': 'model_not_found'},
    ),
    Case(
        id='groq-invalid-api-key',
        model=_groq,
        handler=_json(
            401,
            {'error': {'message': 'Invalid API Key', 'type': 'invalid_request_error', 'code': 'invalid_api_key'}},
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401, 'provider_error_code': 'invalid_api_key'},
    ),
    Case(
        # Gemini sends an invalid API key as `INVALID_ARGUMENT`; the error's `reason` says what it is.
        id='google-api-key-invalid',
        model=_google,
        handler=_json(
            400,
            {
                'error': {
                    'code': 400,
                    'message': 'API key not valid. Please pass a valid API key.',
                    'status': 'INVALID_ARGUMENT',
                    'details': [
                        {
                            '@type': 'type.googleapis.com/google.rpc.ErrorInfo',
                            'reason': 'API_KEY_INVALID',
                            'domain': 'googleapis.com',
                            'metadata': {'service': 'generativelanguage.googleapis.com'},
                        }
                    ],
                }
            },
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 400, 'provider_error_code': 'INVALID_ARGUMENT'},
    ),
    Case(
        id='google-permission-denied',
        model=_google,
        handler=_json(
            403,
            {
                'error': {
                    'code': 403,
                    'message': "Method doesn't allow unregistered callers.",
                    'status': 'PERMISSION_DENIED',
                    'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': []}],
                }
            },
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 403, 'provider_error_code': 'PERMISSION_DENIED'},
    ),
    Case(
        id='google-unauthenticated',
        model=_google,
        handler=_json(
            401,
            {
                'error': {
                    'code': 401,
                    'message': 'Request had invalid authentication credentials.',
                    'status': 'UNAUTHENTICATED',
                }
            },
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401, 'provider_error_code': 'UNAUTHENTICATED'},
    ),
    Case(
        id='google-not-found',
        model=_google,
        handler=_json(
            404,
            {
                'error': {
                    'code': 404,
                    'message': 'models/gemini-2.5-flash is not found for API version v1beta, or is not supported for '
                    'generateContent.',
                    'status': 'NOT_FOUND',
                }
            },
        ),
        categories=_NOT_FOUND,
        attrs={'status_code': 404, 'provider_error_code': 'NOT_FOUND'},
    ),
    Case(
        id='google-failed-precondition',
        model=_google,
        handler=_json(
            400,
            {
                'error': {
                    'code': 400,
                    'message': 'User location is not supported for the API use.',
                    'status': 'FAILED_PRECONDITION',
                }
            },
        ),
        categories=_REJECTED,
        attrs={'status_code': 400, 'provider_error_code': 'FAILED_PRECONDITION'},
    ),
    Case(
        id='mistral-unauthorized',
        model=_mistral,
        handler=_json(401, {'detail': 'Unauthorized'}),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401},
    ),
    Case(
        id='mistral-unprocessable',
        model=_mistral,
        handler=_json(
            422,
            {
                'object': 'error',
                'message': {'detail': [{'type': 'missing', 'loc': ['body', 'messages'], 'msg': 'Field required'}]},
                'type': 'invalid_request_message_order',
                'code': '3230',
            },
        ),
        categories=_REJECTED,
        attrs={'status_code': 422, 'provider_error_code': '3230'},
    ),
    Case(
        id='cohere-unauthorized',
        model=_cohere,
        handler=_json(401, {'id': 'x', 'message': 'invalid api token'}),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401},
    ),
    Case(
        id='cohere-not-found',
        model=_cohere,
        handler=_json(404, {'id': 'x', 'message': "model 'command-r7b-12-2024' not found"}),
        categories=_NOT_FOUND,
        attrs={'status_code': 404},
    ),
    Case(
        id='cohere-bad-request',
        model=_cohere,
        handler=_json(400, {'id': 'x', 'message': 'invalid request: message must be at least 1 token long'}),
        categories=_REJECTED,
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
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 503, 'provider_error_code': 'ServiceUnavailableException'},
            id='unavailable',
        ),
        pytest.param(
            lambda: _bedrock_error('ServiceUnavailableException', 'The model is overloaded. Try again later.', 503),
            _OVERLOADED,
            {'status_code': 503, 'provider_error_code': 'ServiceUnavailableException'},
            id='overloaded',
        ),
        pytest.param(
            lambda: _bedrock_error('ModelNotReadyException', 'The model is not ready to serve requests.', 429),
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 429, 'provider_error_code': 'ModelNotReadyException'},
            id='model-not-ready',
        ),
        pytest.param(
            lambda: _bedrock_error(
                'ServiceQuotaExceededException', 'Your request exceeds the service quota for your account.', 400
            ),
            {ModelHTTPError, ModelQuotaExceededError},
            {'status_code': 400, 'provider_error_code': 'ServiceQuotaExceededException'},
            id='quota',
        ),
        pytest.param(
            lambda: _bedrock_error('InternalServerException', 'An internal server error occurred.', 500),
            {ModelHTTPError, ModelServerError},
            {'status_code': 500, 'provider_error_code': 'InternalServerException'},
            id='server-error',
        ),
        pytest.param(
            lambda: _bedrock_error('serviceUnavailableException', 'Service unavailable.', None),
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 503, 'in_stream': True, 'provider_error_code': 'serviceUnavailableException'},
            id='stream-unavailable',
        ),
        pytest.param(
            lambda: _bedrock_error('internalServerException', 'An internal server error occurred.', None),
            {ModelHTTPError, ModelServerError},
            {'status_code': 500, 'in_stream': True, 'provider_error_code': 'internalServerException'},
            id='stream-server-error',
        ),
        pytest.param(
            lambda: _bedrock_error(
                'ValidationException',
                'The model returned the following errors: Input is too long for requested model.',
                400,
            ),
            _CONTEXT_WINDOW,
            {'status_code': 400, 'provider_error_code': 'ValidationException'},
            id='context-window',
        ),
        pytest.param(
            lambda: _bedrock_error('ValidationException', 'Malformed input request.', 400),
            _REJECTED,
            {'status_code': 400, 'provider_error_code': 'ValidationException'},
            id='validation',
        ),
        pytest.param(
            lambda: _bedrock_error('ValidationException', 'The provided model identifier is invalid.', 400),
            _NOT_FOUND,
            {'status_code': 400, 'provider_error_code': 'ValidationException'},
            id='invalid-model-identifier',
        ),
        pytest.param(
            lambda: _bedrock_error(
                'AccessDeniedException', "You don't have access to the model with the specified model ID.", 403
            ),
            _AUTHENTICATION,
            {'status_code': 403, 'provider_error_code': 'AccessDeniedException'},
            id='access-denied',
        ),
        pytest.param(
            lambda: _bedrock_error(
                'UnrecognizedClientException', 'The security token included in the request is invalid.', 403
            ),
            _AUTHENTICATION,
            {'status_code': 403, 'provider_error_code': 'UnrecognizedClientException'},
            id='unrecognized-client',
        ),
        pytest.param(
            # The code says what it is whatever the status.
            lambda: _bedrock_error(
                'ExpiredTokenException', 'The security token included in the request is expired', 400
            ),
            _AUTHENTICATION,
            {'status_code': 400, 'provider_error_code': 'ExpiredTokenException'},
            id='expired-token',
        ),
        pytest.param(
            # Transient despite its 403: AWS says to try again once the Marketplace subscription is processed.
            lambda: _bedrock_error(
                'MPAgreementBeingCreated',
                'Your AWS Marketplace subscription for this model is still being processed. Try again after 15 minutes.',
                403,
            ),
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 403, 'provider_error_code': 'MPAgreementBeingCreated'},
            id='marketplace-agreement-being-created',
        ),
        pytest.param(
            lambda: _bedrock_error('ResourceNotFoundException', 'Model not found.', 404),
            _NOT_FOUND,
            {'status_code': 404, 'provider_error_code': 'ResourceNotFoundException'},
            id='resource-not-found',
        ),
        pytest.param(
            lambda: _bedrock_error('validationException', 'Malformed input request.', None),
            _REJECTED,
            {'status_code': 400, 'in_stream': True, 'provider_error_code': 'validationException'},
            id='stream-validation',
        ),
        pytest.param(
            lambda: _bedrock_error('throttlingException', 'Too many tokens, please wait before trying again.', None),
            {ModelHTTPError, ModelRateLimitError},
            {'status_code': 429, 'in_stream': True, 'provider_error_code': 'throttlingException', 'retry_after': None},
            id='stream-rate-limit',
        ),
        pytest.param(
            lambda: _bedrock_error('validationException', 'Input is too long for requested model.', None),
            _CONTEXT_WINDOW,
            {'status_code': 400, 'in_stream': True, 'provider_error_code': 'validationException'},
            id='stream-context-window',
        ),
        pytest.param(
            lambda: _bedrock_error('modelStreamErrorException', 'The model stream failed.', None),
            {ModelHTTPError, ModelServerError},
            {'status_code': 424, 'in_stream': True, 'provider_error_code': 'modelStreamErrorException'},
            id='stream-model-stream-error',
        ),
        pytest.param(
            lambda: _bedrock_error('modelTimeoutException', 'The model took too long.', None),
            {ModelHTTPError},
            {'status_code': 408, 'in_stream': True, 'provider_error_code': 'modelTimeoutException'},
            id='stream-model-timeout',
        ),
        pytest.param(
            lambda: _bedrock_error('unknownStreamException', 'Something else failed.', None),
            set[type[ModelAPIError]](),
            {'in_stream': True, 'provider_error_code': 'unknownStreamException'},
            id='stream-unmapped',
        ),
        pytest.param(
            lambda: ReadTimeoutError(endpoint_url='https://bedrock.stub'),
            {ModelConnectionError, ModelTimeoutError},
            {'phase': 'read'},
            id='timeout',
        ),
        pytest.param(
            lambda: EndpointConnectionError(endpoint_url='https://bedrock.stub'),
            {ModelConnectionError},
            {'phase': 'connect'},
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
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 503, 'provider_error_code': 'UNAVAILABLE'},
            id='unavailable',
        ),
        pytest.param(
            'UNAVAILABLE',
            'The model is overloaded, please try again',
            _OVERLOADED,
            {'status_code': 503, 'provider_error_code': 'UNAVAILABLE'},
            id='overloaded',
        ),
        pytest.param(
            'INTERNAL',
            'Internal error',
            {ModelHTTPError, ModelServerError},
            {'status_code': 500, 'provider_error_code': 'INTERNAL'},
            id='internal',
        ),
        pytest.param(
            'INVALID_ARGUMENT',
            "This model's maximum prompt length is 131072 but the request contains 150004 tokens.",
            _CONTEXT_WINDOW,
            {'status_code': 400, 'provider_error_code': 'INVALID_ARGUMENT'},
            id='context-window',
        ),
        pytest.param(
            'UNAVAILABLE',
            # The server stopping mid-call; grpc-core's own transport failures below are worded differently.
            'Cancelling all calls',
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 503, 'provider_error_code': 'UNAVAILABLE'},
            id='server-shutdown',
        ),
        pytest.param(
            'DEADLINE_EXCEEDED',
            'Upstream request timed out',
            {ModelHTTPError, ModelServerError},
            {'status_code': 504, 'provider_error_code': 'DEADLINE_EXCEEDED'},
            id='server-deadline-exceeded',
        ),
        # The rest are the gRPC client's own transport failures, worded as grpcio 1.84 and 1.72 (`xai-sdk`'s floor)
        # report them.
        pytest.param(
            'UNAVAILABLE',
            'failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:59585: '
            'Failed to connect to remote host: Connection refused',
            {ModelConnectionError},
            {'phase': 'connect', 'provider_error_code': 'UNAVAILABLE', 'in_stream': False},
            id='connection-refused',
        ),
        pytest.param(
            'UNAVAILABLE',
            'failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:38243: Tls handshake failed '
            '(TSI_PROTOCOL_FAILURE): SSL_ERROR_SSL: error:100000f7:SSL routines:OPENSSL_internal:WRONG_VERSION_NUMBER',
            {ModelConnectionError},
            {'phase': 'connect'},
            id='tls-handshake-failed',
        ),
        pytest.param(
            'UNAVAILABLE',
            'errors resolving api.x.invalid: [field:hostname lookup error:address lookup failed for api.x.invalid: '
            'Domain name not found]',
            {ModelConnectionError},
            {'phase': 'connect'},
            id='dns-failed',
        ),
        pytest.param(
            'UNAVAILABLE',
            'DNS resolution failed for api.x.invalid: C-ares status is not ARES_SUCCESS qtype=A name=api.x.invalid '
            'is_balancer=0: Domain name not found',
            {ModelConnectionError},
            {'phase': 'connect'},
            id='dns-failed-c-ares',
        ),
        pytest.param(
            'UNAVAILABLE',
            'Stream removed (recvmsg:Connection reset by peer (104))',
            {ModelConnectionError},
            {
                'phase': None,
                'provider_error_code': 'UNAVAILABLE',
                'body': 'Stream removed (recvmsg:Connection reset by peer (104))',
            },
            id='connection-reset',
        ),
        pytest.param(
            'UNAVAILABLE',
            'recvmsg:Connection reset by peer',
            {ModelConnectionError},
            {'phase': None},
            id='connection-reset-grpcio-1.72',
        ),
        pytest.param(
            'UNAVAILABLE',
            'Socket closed',
            {ModelConnectionError},
            {'phase': None},
            id='connection-closed-grpcio-1.72',
        ),
        pytest.param(
            'DEADLINE_EXCEEDED',
            'Deadline Exceeded',
            {ModelConnectionError, ModelTimeoutError},
            {'phase': None, 'provider_error_code': 'DEADLINE_EXCEEDED', 'in_stream': False},
            id='deadline-exceeded',
        ),
        pytest.param(
            'INVALID_ARGUMENT',
            'Invalid request content',
            _REJECTED,
            {'status_code': 400, 'provider_error_code': 'INVALID_ARGUMENT'},
            id='invalid-argument',
        ),
        pytest.param(
            'UNAUTHENTICATED',
            'Incorrect API key provided',
            _AUTHENTICATION,
            {'status_code': 401, 'provider_error_code': 'UNAUTHENTICATED'},
            id='unauthenticated',
        ),
        pytest.param(
            'PERMISSION_DENIED',
            'The API key does not have permission to use this model',
            _AUTHENTICATION,
            {'status_code': 403, 'provider_error_code': 'PERMISSION_DENIED'},
            id='permission-denied',
        ),
        pytest.param(
            'NOT_FOUND',
            'The model grok-4-1-fast-non-reasoning does not exist',
            _NOT_FOUND,
            {'status_code': 404, 'provider_error_code': 'NOT_FOUND'},
            id='not-found',
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


async def test_xai_error_category_mid_stream(allow_model_requests: None):
    """A gRPC error that ends a stream after it started is marked `in_stream`; one before the first response isn't."""
    error = _rpc_error('UNAVAILABLE', 'Service unavailable')
    stream = cast(list[Any], [get_grok_text_chunk('Hello', ''), error])
    model = XaiModel(
        'grok-4-1-fast-non-reasoning', provider=XaiProvider(xai_client=MockXai.create_mock_stream([stream]))
    )
    with pytest.raises(ModelUnavailableError) as exc_info:
        async with Agent(model).run_stream('hello') as result:
            await result.get_output()

    exc = exc_info.value
    assert not isinstance(exc, ModelOverloadedError)
    assert isinstance(exc, ModelHTTPError)
    assert exc.status_code == 503
    assert exc.in_stream is True
    assert exc.__cause__ is error


@pytest.mark.parametrize(
    ('status', 'details', 'categories'),
    [
        pytest.param('UNAVAILABLE', 'Stream removed (Socket closed)', {ModelConnectionError}, id='connection-lost'),
        pytest.param(
            'DEADLINE_EXCEEDED',
            'Stream removed (Deadline Exceeded)',
            {ModelConnectionError, ModelTimeoutError},
            id='deadline',
        ),
    ],
)
async def test_xai_transport_error_mid_stream(
    allow_model_requests: None, status: str, details: str, categories: set[type[ModelAPIError]]
):
    """A transport failure after the stream started sending responses happened while reading the response."""
    error = _rpc_error(status, details)
    stream = cast(list[Any], [get_grok_text_chunk('Hello', ''), error])
    model = XaiModel(
        'grok-4-1-fast-non-reasoning', provider=XaiProvider(xai_client=MockXai.create_mock_stream([stream]))
    )
    with pytest.raises(ModelConnectionError) as exc_info:
        async with Agent(model).run_stream('hello') as result:
            await result.get_output()

    exc = exc_info.value
    assert {cls for cls in _CATEGORIES if isinstance(exc, cls)} == categories
    assert exc.phase == 'read'
    assert exc.in_stream is True
    assert exc.message == details
    assert exc.__cause__ is error


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


@pytest.mark.parametrize(
    ('status_code', 'content', 'categories'),
    [
        pytest.param(
            503, b'{"error":"Model is currently loading"}', {ModelHTTPError, ModelUnavailableError}, id='unavailable'
        ),
        pytest.param(
            402,
            b'{"error":"You have exceeded your monthly included credits for Inference Providers."}',
            {ModelHTTPError, ModelQuotaExceededError},
            id='quota',
        ),
        pytest.param(500, b'{"error":"Internal Server Error"}', {ModelHTTPError, ModelServerError}, id='server-error'),
        pytest.param(
            401, b'{"error":"Invalid credentials in Authorization header"}', _AUTHENTICATION, id='unauthorized'
        ),
        pytest.param(404, b'{"error":"Model m does not exist"}', _NOT_FOUND, id='not-found'),
        pytest.param(422, b'{"error":"Input validation error"}', _REJECTED, id='unprocessable'),
    ],
)
async def test_huggingface_status_category(
    allow_model_requests: None, status_code: int, content: bytes, categories: set[type[ModelAPIError]]
):
    response = httpx.Response(
        status_code, content=content, request=httpx.Request('POST', 'http://localhost/v1/chat/completions')
    )
    error = HfHubHTTPError('failed', response=response)

    async def create(*args: Any, **kwargs: Any) -> Any:
        raise error

    hf_client = cast(
        AsyncInferenceClient, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )
    model = HuggingFaceModel('m', provider=HuggingFaceProvider(hf_client=hf_client, api_key='test'))
    with pytest.raises(ModelAPIError) as exc_info:
        await Agent(model).run('hello')

    assert {cls for cls in _CATEGORIES if isinstance(exc_info.value, cls)} == categories
    assert isinstance(exc_info.value, ModelHTTPError)
    assert exc_info.value.status_code == status_code
