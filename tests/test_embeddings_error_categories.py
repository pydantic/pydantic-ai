"""Embedding model errors are raised as the same provider-neutral error categories as chat model errors.

Not VCR tests, for the reason given in `tests/models/test_error_categories.py`: providers don't return rate limits,
overloads, or dropped connections on demand. Each case replays the error's wire shape through the provider SDK, using
a mock transport where the SDK has an HTTP one and a stub where it doesn't (Bedrock, VoyageAI). The input overflow
bodies were recorded from the real APIs. Mock transports use `localhost` base URLs, which VCR ignores.
"""

from __future__ import annotations as _annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch

import httpx2
import pytest

from pydantic_ai import (
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
from pydantic_ai.embeddings import Embedder, EmbeddingModel

from .conftest import try_import

with try_import() as imports_successful:
    import httpx
    from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError, ReadTimeoutError
    from cohere import AsyncClientV2
    from openai import AsyncOpenAI

    from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel
    from pydantic_ai.embeddings.cohere import CohereEmbeddingModel
    from pydantic_ai.embeddings.google import GoogleEmbeddingModel
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.cohere import CohereProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='provider SDKs not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]

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

_REJECTED: set[type[ModelAPIError]] = {ModelHTTPError, ModelRequestRejectedError}
_AUTHENTICATION: set[type[ModelAPIError]] = {ModelHTTPError, ModelRequestRejectedError, ModelAuthenticationError}
_NOT_FOUND: set[type[ModelAPIError]] = {ModelHTTPError, ModelRequestRejectedError, ModelNotFoundError}


def _categories(exc: ModelAPIError) -> set[type[ModelAPIError]]:
    return {cls for cls in _CATEGORIES if isinstance(exc, cls)}


def _openai(handler: Handler) -> EmbeddingModel:
    client = AsyncOpenAI(
        api_key='test',
        base_url='http://localhost/v1',
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(openai_client=client))


def _cohere(handler: Handler) -> EmbeddingModel:
    client = AsyncClientV2(
        api_key='test',
        base_url='http://localhost',
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(cohere_client=client))


def _google(handler: Handler) -> EmbeddingModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return GoogleEmbeddingModel(
        'gemini-embedding-001',
        provider=GoogleProvider(api_key='test', http_client=http_client, base_url='http://localhost'),
    )


def _json(status_code: int, body: Any, headers: dict[str, str] | None = None) -> Handler:
    def handler(request: Any) -> Any:
        response_class = httpx.Response if isinstance(request, httpx.Request) else httpx2.Response
        return response_class(status_code, content=json.dumps(body).encode(), headers=headers)

    return handler


def _raise(kind: Literal['connect', 'timeout']) -> Handler:
    def handler(request: Any) -> Any:
        if isinstance(request, httpx.Request):
            raise {'connect': httpx.ConnectError, 'timeout': httpx.ReadTimeout}[kind]('failed', request=request)
        raise {'connect': httpx2.ConnectError, 'timeout': httpx2.ReadTimeout}[kind]('failed', request=request)

    return handler


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[Handler], EmbeddingModel]
    handler: Handler
    categories: set[type[ModelAPIError]]
    """The exception's classes among `_CATEGORIES`: its error categories, plus `ModelHTTPError` if it has a status."""
    attrs: dict[str, Any] = field(default_factory=dict[str, Any])


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
        id='openai-unavailable',
        model=_openai,
        handler=_json(503, {'error': {'message': 'Service unavailable', 'type': 'server_error', 'code': None}}),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503},
    ),
    Case(
        id='openai-server-error',
        model=_openai,
        handler=_json(500, {'error': {'message': 'Internal error', 'type': 'server_error', 'code': None}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500, 'provider_error_type': 'server_error'},
    ),
    Case(
        # Recorded from the OpenAI API; it doesn't use the chat completions overflow code or wording.
        id='openai-input-too-long',
        model=_openai,
        handler=_json(
            400,
            {
                'error': {
                    'message': "Invalid 'input[0]': maximum input length is 8192 tokens.",
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': None,
                }
            },
        ),
        categories=_REJECTED,
        attrs={'status_code': 400, 'provider_error_type': 'invalid_request_error'},
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
        id='cohere-rate-limit',
        model=_cohere,
        handler=_json(429, {'id': 'x', 'message': 'You are using a Trial key, which is limited to 100 API calls'}),
        categories={ModelHTTPError, ModelRateLimitError},
        attrs={'status_code': 429},
    ),
    Case(
        id='cohere-unavailable',
        model=_cohere,
        handler=_json(503, {'id': 'x', 'message': 'service unavailable'}),
        categories={ModelHTTPError, ModelUnavailableError},
        attrs={'status_code': 503},
    ),
    Case(
        id='cohere-server-error',
        model=_cohere,
        handler=_json(500, {'id': 'x', 'message': 'internal server error'}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500},
    ),
    Case(
        # Recorded from the Cohere API; unlike a chat overflow, it doesn't start with `too many tokens`.
        id='cohere-input-too-long',
        model=_cohere,
        handler=_json(
            400,
            {
                'id': '0307c969-32e2-4895-bc9a-827cfa04abab',
                'message': 'invalid request: texts[0] is too long. Max tokens per text must be at most 508 - received '
                '3334',
            },
        ),
        categories=_REJECTED,
        attrs={'status_code': 400},
    ),
    Case(
        id='cohere-timeout',
        model=_cohere,
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
        categories={ModelHTTPError, ModelUnavailableError, ModelOverloadedError},
        attrs={'status_code': 503, 'provider_error_code': 'UNAVAILABLE'},
    ),
    Case(
        id='google-internal',
        model=_google,
        handler=_json(500, {'error': {'code': 500, 'message': 'Internal error encountered.', 'status': 'INTERNAL'}}),
        categories={ModelHTTPError, ModelServerError},
        attrs={'status_code': 500, 'provider_error_code': 'INTERNAL'},
    ),
    Case(
        id='google-invalid-argument',
        model=_google,
        handler=_json(400, {'error': {'code': 400, 'message': 'Invalid value', 'status': 'INVALID_ARGUMENT'}}),
        categories=_REJECTED,
        attrs={'status_code': 400, 'provider_error_code': 'INVALID_ARGUMENT'},
    ),
    Case(
        id='openai-invalid-api-key',
        model=_openai,
        handler=_json(
            401,
            {
                'error': {
                    'message': 'Incorrect API key provided',
                    'type': 'invalid_request_error',
                    'code': 'invalid_api_key',
                }
            },
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401, 'provider_error_code': 'invalid_api_key'},
    ),
    Case(
        id='openai-model-not-found',
        model=_openai,
        handler=_json(
            404,
            {
                'error': {
                    'message': 'The model `text-embedding-3-small` does not exist or you do not have access to it.',
                    'type': 'invalid_request_error',
                    'code': 'model_not_found',
                }
            },
        ),
        categories=_NOT_FOUND,
        attrs={'status_code': 404, 'provider_error_code': 'model_not_found'},
    ),
    Case(
        id='cohere-unauthorized',
        model=_cohere,
        handler=_json(401, {'id': 'x', 'message': 'invalid api token'}),
        categories=_AUTHENTICATION,
        attrs={'status_code': 401},
    ),
    Case(
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
                        {'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': 'API_KEY_INVALID'},
                    ],
                }
            },
        ),
        categories=_AUTHENTICATION,
        attrs={'status_code': 400, 'provider_error_code': 'INVALID_ARGUMENT'},
    ),
    Case(
        id='google-not-found',
        model=_google,
        handler=_json(404, {'error': {'code': 404, 'message': 'Model not found.', 'status': 'NOT_FOUND'}}),
        categories=_NOT_FOUND,
        attrs={'status_code': 404, 'provider_error_code': 'NOT_FOUND'},
    ),
    Case(
        id='google-timeout',
        model=_google,
        handler=_raise('timeout'),
        categories={ModelConnectionError, ModelTimeoutError},
        attrs={'phase': 'read'},
    ),
]


@pytest.mark.parametrize('case', CASES, ids=[case.id for case in CASES])
async def test_http_provider_error_category(case: Case):
    with pytest.raises(ModelAPIError) as exc_info:
        await Embedder(case.model(case.handler)).embed_query('hello')

    exc = exc_info.value
    assert _categories(exc) == case.categories
    for attr, expected in case.attrs.items():
        assert getattr(exc, attr) == expected, attr
    assert exc.__cause__ is not None


def _bedrock_error(code: str, message: str, status_code: int) -> ClientError:
    response: dict[str, Any] = {
        'Error': {'Code': code, 'Message': message},
        'ResponseMetadata': {'HTTPStatusCode': status_code, 'HTTPHeaders': {}},
    }
    return ClientError(cast(Any, response), 'InvokeModel')


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
            lambda: _bedrock_error(
                'ServiceQuotaExceededException', 'Your request exceeds the service quota for your account.', 400
            ),
            {ModelHTTPError, ModelQuotaExceededError},
            {'status_code': 400, 'provider_error_code': 'ServiceQuotaExceededException'},
            id='quota',
        ),
        pytest.param(
            lambda: _bedrock_error('ServiceUnavailableException', 'Service unavailable.', 503),
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 503, 'provider_error_code': 'ServiceUnavailableException'},
            id='unavailable',
        ),
        pytest.param(
            lambda: _bedrock_error('ModelNotReadyException', 'The model is not ready to serve requests.', 429),
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 429, 'provider_error_code': 'ModelNotReadyException'},
            id='model-not-ready',
        ),
        pytest.param(
            lambda: _bedrock_error('InternalServerException', 'An internal server error occurred.', 500),
            {ModelHTTPError, ModelServerError},
            {'status_code': 500, 'provider_error_code': 'InternalServerException'},
            id='server-error',
        ),
        pytest.param(
            # Recorded from Titan Text Embeddings V2; unlike a chat overflow, it doesn't say the input is too long.
            lambda: _bedrock_error(
                'ValidationException',
                '400 Bad Request: Too many input tokens. Max input tokens: 8192, request input token count: 30524 ',
                400,
            ),
            _REJECTED,
            {'status_code': 400, 'provider_error_code': 'ValidationException'},
            id='input-too-long',
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
    make_error: Callable[[], ClientError | BotoCoreError],
    categories: set[type[ModelAPIError]],
    attrs: dict[str, Any],
):
    """The errors are built in the test, not at collection, so the module can be collected without `boto3`."""
    error = make_error()
    provider = BedrockProvider(region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test')
    model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=provider)
    with patch.object(model.client, 'invoke_model', side_effect=error):
        with pytest.raises(ModelAPIError) as exc_info:
            await Embedder(model).embed_query('hello')

    exc = exc_info.value
    assert _categories(exc) == categories
    for attr, expected in attrs.items():
        assert getattr(exc, attr) == expected, attr
    assert exc.__cause__ is error


@pytest.fixture(scope='module')
def voyageai_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """`voyageai` imports `torch` when `sentence-transformers` is installed, so it's imported once, not at collection.

    See "Test cost" in `tests/AGENTS.md`.
    """
    return (
        importlib.import_module('voyageai.error'),
        importlib.import_module('pydantic_ai.embeddings.voyageai'),
        importlib.import_module('pydantic_ai.providers.voyageai'),
    )


@pytest.mark.skipif(find_spec('voyageai') is None, reason='voyageai not installed')
@pytest.mark.parametrize(
    ('error_class', 'args', 'categories', 'attrs'),
    [
        pytest.param(
            # How the SDK builds the error for a 429: the response's `detail`, its body, status, JSON and headers.
            'RateLimitError',
            (
                'Rate limit exceeded.',
                '{"detail": "Rate limit exceeded."}',
                429,
                {'detail': 'Rate limit exceeded.'},
                {'Retry-After': '2'},
            ),
            {ModelHTTPError, ModelRateLimitError},
            {'status_code': 429, 'body': {'detail': 'Rate limit exceeded.'}, 'retry_after': 2.0},
            id='rate-limit',
        ),
        pytest.param(
            # The SDK's message for any 502, 503 or 504; only the body is the server's own.
            'ServiceUnavailableError',
            ('The server is overloaded or not ready yet.', 'Service Unavailable', 503),
            {ModelHTTPError, ModelUnavailableError},
            {'status_code': 503, 'body': 'Service Unavailable'},
            id='unavailable',
        ),
        pytest.param(
            'ServerError',
            ('The server failed to process the request.', 'Internal Server Error', 500),
            {ModelHTTPError, ModelServerError},
            {'status_code': 500},
            id='server-error',
        ),
        pytest.param(
            'InvalidRequestError',
            (
                'Model nonexistent is not supported.',
                '{"detail": "Model nonexistent is not supported."}',
                400,
                {'detail': 'Model nonexistent is not supported.'},
            ),
            _REJECTED,
            {'status_code': 400, 'body': {'detail': 'Model nonexistent is not supported.'}},
            id='invalid-request',
        ),
        pytest.param(
            'AuthenticationError',
            (
                'Provided API key is invalid.',
                '{"detail": "Provided API key is invalid."}',
                401,
                {'detail': 'Provided API key is invalid.'},
            ),
            _AUTHENTICATION,
            {'status_code': 401},
            id='authentication',
        ),
        pytest.param(
            # A 4xx without a category stays a plain `ModelAPIError`, as before.
            'InvalidRequestError',
            ('Conflict.', '{"detail": "Conflict."}', 409, {'detail': 'Conflict.'}),
            set[type[ModelAPIError]](),
            {'message': 'Conflict.'},
            id='uncategorized',
        ),
        pytest.param(
            'Timeout',
            ('Request timed out',),
            {ModelConnectionError, ModelTimeoutError},
            {'message': 'Request timed out'},
            id='timeout',
        ),
        pytest.param(
            'APIConnectionError',
            ('Error communicating with VoyageAI',),
            {ModelConnectionError},
            {'message': 'Error communicating with VoyageAI'},
            id='connection',
        ),
    ],
)
async def test_voyageai_error_category(
    voyageai_modules: tuple[ModuleType, ModuleType, ModuleType],
    error_class: str,
    args: tuple[Any, ...],
    categories: set[type[ModelAPIError]],
    attrs: dict[str, Any],
):
    """The errors are `voyageai.error` classes, built with the arguments the SDK passes them."""
    error_module, embeddings_module, provider_module = voyageai_modules
    error: Exception = getattr(error_module, error_class)(*args)

    async def embed(**kwargs: Any) -> Any:
        raise error

    provider = provider_module.VoyageAIProvider(voyageai_client=SimpleNamespace(embed=embed))
    model = embeddings_module.VoyageAIEmbeddingModel('voyage-3.5', provider=provider)
    with pytest.raises(ModelAPIError) as exc_info:
        await Embedder(model).embed_query('hello')

    exc = exc_info.value
    assert _categories(exc) == categories
    for attr, expected in attrs.items():
        assert getattr(exc, attr) == expected, attr
    assert exc.__cause__ is error
