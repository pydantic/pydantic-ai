"""Mapping Anthropic API errors to Pydantic AI exceptions."""

from __future__ import annotations as _annotations

from collections.abc import Generator
from contextlib import contextmanager

import httpx2
from anthropic import APIConnectionError, APIStatusError, APITimeoutError

from ... import _model_errors, _utils
from ...exceptions import (
    ModelAPIError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
)
from .. import (
    _suggest_known_model_id_from_provider_error,  # pyright: ignore[reportPrivateUsage]
)
from .._transport_errors import transport_error_message

# https://platform.claude.com/docs/en/api/errors#http-errors
_ERROR_TYPE_STATUS_CODES = {
    'invalid_request_error': 400,
    'authentication_error': 401,
    'billing_error': 402,
    'permission_error': 403,
    'not_found_error': 404,
    'request_too_large': 413,
    'rate_limit_error': 429,
    'api_error': 500,
    'timeout_error': 504,
    'overloaded_error': 529,
}


def error_status_code(error: APIStatusError) -> int:
    """The HTTP status of an Anthropic error, including one reported in a stream after a 200 response.

    An error event in a stream arrives after the response status was already 200, so the SDK raises it with that
    status; the error `type` identifies the status the same error has on a non-streaming request.
    """
    if error.status_code >= 400:
        return error.status_code
    body = error.body
    if _utils.is_str_dict(body) and _utils.is_str_dict(error_body := body.get('error')):
        if isinstance(error_type := error_body.get('type'), str) and error_type in _ERROR_TYPE_STATUS_CODES:
            return _ERROR_TYPE_STATUS_CODES[error_type]
    return error.status_code


def reported_error_body(body: object | None) -> object | None:
    """The object in an error body that reports the failure.

    The direct API wraps it in an `error` object; Bedrock returns it at the top level.
    """
    return body.get('error', body) if _utils.is_str_dict(body) else None


def is_expired_container_error(error: APIStatusError) -> bool:
    """Whether Anthropic rejected a request because the container it reuses has expired.

    Anthropic answers an expired container with a 404 `not_found_error`, and earlier with a 500.
    """
    status_code = error_status_code(error)
    if status_code == 500:
        return True
    error_body = reported_error_body(error.body)
    return (
        status_code == 404
        and _utils.is_str_dict(error_body)
        and str(error_body.get('message', '')).startswith('Container not found')
    )


@contextmanager
def map_api_errors(model_name: str, model_id_namespace: str = 'anthropic') -> Generator[None]:
    try:
        yield
    except APIStatusError as e:
        body: object | None = e.body
        error = nested if _utils.is_str_dict(body) and _utils.is_str_dict(nested := body.get('error')) else {}
        error_type, error_message = error.get('type'), error.get('message')
        category = _ERROR_TYPE_CATEGORIES.get(error_type) if isinstance(error_type, str) else None
        if error_type == 'invalid_request_error' and isinstance(error_message, str):
            if 'prompt is too long' in error_message.lower():
                category = ModelContextWindowExceededError
            elif 'credit balance is too low' in error_message.lower():
                # Anthropic reports an exhausted prepaid balance as a 400, identifiable only by its message.
                category = ModelQuotaExceededError
        provider_error_type = error_type if isinstance(error_type, str) else None
        if (status_code := error_status_code(e)) >= 400:
            suggested_model_id = None
            if error_type == 'not_found_error' and error_message == f'model: {model_name}':
                suggested_model_id = _suggest_known_model_id_from_provider_error(model_id_namespace, model_name)
            raise ModelHTTPError.for_category(
                category or _model_errors.http_status_category(status_code, error_message),
                status_code=status_code,
                model_name=model_name,
                body=body,
                headers=dict(e.response.headers),
                suggested_model_id=suggested_model_id,
                provider_error_type=provider_error_type,
                # An `error` event inside a stream comes with the stream's own 200 status; it got its status above.
                in_stream=e.status_code < 400,
            ) from e
        # An `error` event inside a stream whose type has no known status.
        raise (category or ModelAPIError)(
            model_name=model_name, message=e.message, body=body, provider_error_type=provider_error_type, in_stream=True
        ) from e
    except APIConnectionError as e:
        raise _model_errors.connection_error(model_name, e.message, e, timeout=isinstance(e, APITimeoutError)) from e
    except httpx2.TransportError as e:
        # `anthropic` wraps transport failures in `APIConnectionError` only until the response starts; one that breaks
        # off a stream mid-way surfaces as the raw `httpx2` error.
        timeout = isinstance(e, httpx2.TimeoutException)
        raise _model_errors.connection_error(model_name, transport_error_message(e), e, timeout=timeout) from e


_ERROR_TYPE_CATEGORIES: dict[str, _model_errors.HTTPErrorCategory] = {
    'rate_limit_error': ModelRateLimitError,
    'billing_error': ModelQuotaExceededError,
    'overloaded_error': ModelOverloadedError,
}
"""Error categories for the `error.type` Anthropic sends, both in an HTTP error body and in a stream's `error` event."""
