"""Helpers for model adapters to raise the error category a provider error belongs to."""

from __future__ import annotations as _annotations

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
