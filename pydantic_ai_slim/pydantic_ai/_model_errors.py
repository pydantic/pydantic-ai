"""Helpers for model adapters to raise the error category a provider error belongs to."""

from __future__ import annotations as _annotations

from ._utils import is_str_dict
from .exceptions import ContextWindowExceeded, ModelAPIError, ModelHTTPError, ModelOverloadedError, ModelRateLimitError


class HTTPModelRateLimitError(ModelHTTPError, ModelRateLimitError):
    """A rate limit reported with an HTTP status code."""


class HTTPModelOverloadedError(ModelHTTPError, ModelOverloadedError):
    """An overloaded provider reported with an HTTP status code."""


class HTTPContextWindowExceeded(ModelHTTPError, ContextWindowExceeded):
    """A context window overflow reported with an HTTP status code."""


_HTTP_ERROR_CLASSES: dict[type[ModelAPIError], type[ModelHTTPError]] = {
    ModelRateLimitError: HTTPModelRateLimitError,
    ModelOverloadedError: HTTPModelOverloadedError,
    ContextWindowExceeded: HTTPContextWindowExceeded,
}


def http_error_class(category: type[ModelAPIError] | None) -> type[ModelHTTPError]:
    """The `ModelHTTPError` class to raise for an HTTP error in `category`, which is also an instance of the category."""
    return ModelHTTPError if category is None else _HTTP_ERROR_CLASSES[category]


def http_status_category(status_code: int) -> type[ModelAPIError] | None:
    """The error category an HTTP status code implies on its own, without the provider's error code."""
    if status_code == 429:
        return ModelRateLimitError
    if status_code in (503, 529):
        return ModelOverloadedError
    return None


_STREAM_ERROR_CATEGORIES: dict[str, type[ModelAPIError]] = {
    'rate_limit_exceeded': ModelRateLimitError,
    'context_length_exceeded': ContextWindowExceeded,
    'service_unavailable': ModelOverloadedError,
    'overloaded': ModelOverloadedError,
}


def stream_error(model_name: str, message: str, body: object) -> ModelAPIError:
    """Map an error object sent inside a 200 stream by an OpenAI-compatible API, classified by its `code` or `type`.

    The HTTP status was already 200, so none is reported.
    """
    code = body.get('code') if is_str_dict(body) else None
    error_type = body.get('type') if is_str_dict(body) else None
    code = str(code) if isinstance(code, str | int) else None
    error_type = error_type if isinstance(error_type, str) else None
    category = _STREAM_ERROR_CATEGORIES.get(code or '') or _STREAM_ERROR_CATEGORIES.get(error_type or '')
    return (category or ModelAPIError)(
        model_name=model_name, message=message, body=body, provider_error_code=code, provider_error_type=error_type
    )
