"""Helpers for model adapters to raise the error category a provider error belongs to."""

from __future__ import annotations as _annotations

from collections.abc import Mapping
from typing import Protocol, cast

import pydantic_core

from ._utils import is_str_dict
from .exceptions import (
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
    TransportPhase,
)

HTTPErrorCategory = type[
    ModelRateLimitError | ModelQuotaExceededError | ModelUnavailableError | ModelServerError | ModelRequestRejectedError
]
"""An error category an HTTP error can belong to; see `ModelHTTPError.for_category`."""

_OVERLOADED_PHRASES = ('overloaded', 'over capacity')


def says_overloaded(*texts: object) -> bool:
    """Whether any of a provider's error code, type or message says it's overloaded, e.g. `'overloaded_error'`."""
    return any(
        isinstance(text, str) and any(phrase in text.lower() for phrase in _OVERLOADED_PHRASES) for text in texts
    )


def http_status_category(status_code: int, message: object = None) -> HTTPErrorCategory | None:
    """The error category an HTTP status code implies on its own, without the provider's error code.

    A 503 whose `message` says the provider is overloaded is a `ModelOverloadedError`; any other 503 is only
    unavailable. A 400 or 422 is a `ModelRequestRejectedError`, unless the caller found a more specific category in
    the provider's own error code or message first.
    """
    if status_code in (401, 403):
        return ModelAuthenticationError
    if status_code == 404:
        return ModelNotFoundError
    if status_code in (400, 422):
        return ModelRequestRejectedError
    if status_code == 402:
        return ModelQuotaExceededError
    if status_code == 429:
        return ModelRateLimitError
    if status_code == 529:
        return ModelOverloadedError
    if status_code == 503:
        return ModelOverloadedError if says_overloaded(message) else ModelUnavailableError
    if 500 <= status_code < 600:
        return ModelServerError
    return None


_QUOTA_CODES = (
    'insufficient_quota',
    'billing_hard_limit_reached',
    'credit_balance_exhausted',
    'organization_spend_limit_exceeded',
    'project_spend_limit_exceeded',
    'organization_usage_limit_exceeded',
)
"""OpenAI error codes that say the account's quota, credits or spend limit is exhausted, rather than a rate limit.

OpenAI sends them with a 429, like a rate limit; see https://developers.openai.com/api/docs/guides/error-codes.
"""

_OPENAI_COMPATIBLE_STATUSES: dict[str, int] = {
    'rate_limit_exceeded': 429,
    'rate_limit_error': 429,
    'too_many_requests': 429,
    **dict.fromkeys(_QUOTA_CODES, 429),
    'context_length_exceeded': 400,
    'service_unavailable': 503,
    'service_unavailable_error': 503,
    'server_is_overloaded': 503,
    'overloaded': 503,
    'server_error': 500,
}
"""The HTTP status OpenAI-compatible APIs use for an error, by its `code` or `type`, for errors that come without one."""

_CONTEXT_WINDOW_MESSAGES = ('maximum context length', 'reduce the length of the messages')
"""How OpenAI-compatible APIs that send no `context_length_exceeded` code (OpenRouter, vLLM, Groq) word an overflow."""


def openai_compatible_status(code: str | None, error_type: str | None) -> int | None:
    """The HTTP status an OpenAI-compatible API uses for an error it reported without one, e.g. inside a stream."""
    if (
        status := _OPENAI_COMPATIBLE_STATUSES.get(code or '') or _OPENAI_COMPATIBLE_STATUSES.get(error_type or '')
    ) is not None:
        return status
    if code is not None and code.isdigit() and 400 <= int(code) < 600:
        # Gateways like OpenRouter send the HTTP status itself as the `code`.
        return int(code)
    return None


def openai_compatible_category(
    status_code: int | None, code: str | None, error_type: str | None, message: object
) -> HTTPErrorCategory | None:
    """The error category of an error from an OpenAI-compatible API (OpenAI, Groq, OpenRouter), wherever it was sent."""
    if code in _QUOTA_CODES or error_type in _QUOTA_CODES:
        # Exhausted quota is also a 429, but waiting won't help.
        return ModelQuotaExceededError
    if code == 'context_length_exceeded' or (
        isinstance(message, str) and any(m in message.lower() for m in _CONTEXT_WINDOW_MESSAGES)
    ):
        return ModelContextWindowExceededError
    if says_overloaded(code, error_type):
        return ModelOverloadedError
    if code == 'model_not_found':
        # The Responses API sends it as a 400, the Chat Completions API as a 404.
        return ModelNotFoundError
    if code == 'invalid_api_key':
        return ModelAuthenticationError
    return http_status_category(status_code, message) if status_code is not None else None


_GOOGLE_CONTEXT_WINDOW_MESSAGES = ('input token count', 'number of tokens allowed')
"""Gemini reports a context window overflow as `INVALID_ARGUMENT`, identifiable only by its message.

The full message is `The input token count (N) exceeds the maximum number of tokens allowed (M)`; the file-upload
variant drops the leading clause.
"""


def google_error_reason(details: object) -> str | None:
    """The `reason` of the `google.rpc.ErrorInfo` in a Google API error body, like `'API_KEY_INVALID'`, if any."""
    error = details.get('error') if is_str_dict(details) else None
    error_details = error.get('details') if is_str_dict(error) else None
    if isinstance(error_details, list):
        for detail in cast(list[object], error_details):
            if is_str_dict(detail) and isinstance(reason := detail.get('reason'), str):
                return reason
    return None


def google_status_category(status: str | None, message: object, reason: str | None = None) -> HTTPErrorCategory | None:
    """The error category for a Google API error status, like `RESOURCE_EXHAUSTED`.

    Gemini sends `RESOURCE_EXHAUSTED` for both rate limits and exhausted quota, with the same message, so it's
    always a rate limit. It sends an invalid API key as `INVALID_ARGUMENT`, with the `reason` `API_KEY_INVALID`.
    """
    if reason == 'API_KEY_INVALID':
        return ModelAuthenticationError
    match status:
        case 'RESOURCE_EXHAUSTED':
            return ModelRateLimitError
        case 'UNAVAILABLE':
            # Like `The model is overloaded. Please try again later.`
            return ModelOverloadedError if says_overloaded(message) else ModelUnavailableError
        case 'INTERNAL':
            return ModelServerError
        case 'INVALID_ARGUMENT' if isinstance(message, str) and any(
            m in message.lower() for m in _GOOGLE_CONTEXT_WINDOW_MESSAGES
        ):
            return ModelContextWindowExceededError
        case 'INVALID_ARGUMENT':
            return ModelRequestRejectedError
        case 'UNAUTHENTICATED' | 'PERMISSION_DENIED':
            return ModelAuthenticationError
        case 'NOT_FOUND':
            return ModelNotFoundError
        case _:
            return None


_BEDROCK_CONTEXT_WINDOW_MESSAGES = ('input is too long', 'input tokens exceeded', 'prompt is too long')
"""Bedrock reports a context window overflow as a `ValidationException`, identifiable only by its message."""


def bedrock_code_category(code: str | None, message: object) -> HTTPErrorCategory | None:
    """The error category for a Bedrock error code, which is capitalized in an HTTP error and not in a stream."""
    match code and code.lower():
        case 'throttlingexception':
            return ModelRateLimitError
        case 'servicequotaexceededexception':
            return ModelQuotaExceededError
        case 'modelstreamerrorexception':
            # The model failed while streaming its response; Bedrock says to retry the request.
            return ModelServerError
        case 'serviceunavailableexception' | 'modelnotreadyexception':
            return ModelOverloadedError if says_overloaded(message) else ModelUnavailableError
        case 'mpagreementbeingcreated':
            # A 403 while the account's AWS Marketplace subscription to the model is still being processed; AWS says
            # to try again after 15 minutes, so it's not a permanent authorization failure.
            # https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html#ts-mp-agreement-created
            return ModelUnavailableError
        case 'validationexception' if isinstance(message, str) and any(
            m in message.lower() for m in _BEDROCK_CONTEXT_WINDOW_MESSAGES
        ):
            return ModelContextWindowExceededError
        case (
            'accessdeniedexception'
            | 'unrecognizedclientexception'
            | 'invalidsignatureexception'
            | 'expiredtokenexception'
            | 'expiredtoken'
        ):
            return ModelAuthenticationError
        case 'resourcenotfoundexception':
            return ModelNotFoundError
        case _:
            return None


def stream_error(model_name: str, message: str, body: object) -> ModelAPIError:
    """Map an error object sent inside a 200 stream by an OpenAI-compatible API, classified by its `code` or `type`.

    It gets the status and category the same error has before a stream opens, so it's handled the same way whether
    or not the request was streamed, with `in_stream` set. An error with no clear status isn't a `ModelHTTPError`.
    """
    code = body.get('code') if is_str_dict(body) else None
    error_type = body.get('type') if is_str_dict(body) else None
    code = str(code) if isinstance(code, str | int) else None
    error_type = error_type if isinstance(error_type, str) else None
    status_code = openai_compatible_status(code, error_type)
    category = openai_compatible_category(status_code, code, error_type, message)
    if status_code is None:
        return (category or ModelAPIError)(
            model_name=model_name,
            message=message,
            body=body,
            provider_error_code=code,
            provider_error_type=error_type,
            in_stream=True,
        )
    return ModelHTTPError.for_category(
        category,
        status_code=status_code,
        model_name=model_name,
        body=body,
        provider_error_code=code,
        provider_error_type=error_type,
        in_stream=True,
    )


_HTTPX_PHASES: dict[str, TransportPhase] = {
    'PoolTimeout': 'pool',
    'ConnectTimeout': 'connect',
    'ConnectError': 'connect',
    'WriteTimeout': 'write',
    'WriteError': 'write',
    'ReadTimeout': 'read',
    'ReadError': 'read',
    'RemoteProtocolError': 'read',
}

_TRANSPORT_PHASES: dict[tuple[str, str], TransportPhase] = {
    # `httpx` and `httpx2` share their exception names.
    **{(package, name): phase for package in ('httpx', 'httpx2') for name, phase in _HTTPX_PHASES.items()},
    ('botocore', 'ConnectTimeoutError'): 'connect',
    ('botocore', 'EndpointConnectionError'): 'connect',
    ('botocore', 'ReadTimeoutError'): 'read',
    ('botocore', 'ResponseStreamingError'): 'read',
    # botocore reads an event stream straight from urllib3, so these only reach us while reading a response.
    ('urllib3', 'ReadTimeoutError'): 'read',
    ('urllib3', 'ProtocolError'): 'read',
}
"""Transport exceptions whose stage of the request is known, by top-level package and class name.

Matched by name so the HTTP libraries an adapter doesn't use needn't be importable.
"""


def transport_phase(error: BaseException) -> TransportPhase | None:
    """The stage of the request at which a transport error happened, from it or the exceptions it was raised from.

    SDKs like `openai` and `anthropic` wrap the HTTP library's exception, keeping it as `__cause__`.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for cls in type(current).__mro__:
            if (phase := _TRANSPORT_PHASES.get((cls.__module__.partition('.')[0], cls.__name__))) is not None:
                return phase
        current = current.__cause__
    return None


def connection_error(model_name: str, message: str, error: BaseException, *, timeout: bool) -> ModelConnectionError:
    """A `ModelConnectionError` (or `ModelTimeoutError`) for a transport `error`, with its phase if known."""
    error_class = ModelTimeoutError if timeout else ModelConnectionError
    return error_class(model_name=model_name, message=message, phase=transport_phase(error))


class HTTPStatusResponse(Protocol):
    """The parts of an `httpx` or `httpx2` response that `transport_status_error` reads."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    @property
    def content(self) -> bytes: ...


def transport_status_error(model_name: str, response: HTTPStatusResponse) -> ModelHTTPError:
    """The `ModelHTTPError` for an error status the HTTP transport raised before the provider SDK saw the response.

    A retrying transport's `validate_response` raises `HTTPStatusError` for a status it retries, and re-raises it
    once it runs out of attempts. Most SDKs let that exception through unchanged, and `anthropic` and `groq` wrap it
    in their connection error, though the provider did respond.
    """
    try:
        content = response.content
    except RuntimeError:  # `ResponseNotRead`: the transport closed the response without reading its body.
        content = b''
    text = content.decode(errors='replace')
    body: object = None
    if content:
        try:
            body = pydantic_core.from_json(content)
        except ValueError:
            body = text
    return ModelHTTPError.for_category(
        http_status_category(response.status_code, text),
        status_code=response.status_code,
        model_name=model_name,
        body=body,
        headers=dict(response.headers),
    )
