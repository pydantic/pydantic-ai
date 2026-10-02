"""Helpers for model adapters to raise the error category a provider error belongs to."""

from __future__ import annotations as _annotations

from ._utils import is_str_dict
from .exceptions import (
    ContextWindowExceeded,
    ModelAPIError,
    ModelConnectionError,
    ModelHTTPError,
    ModelOverloadedError,
    ModelRateLimitError,
    ModelTimeoutError,
    TransportPhase,
)


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


_OPENAI_COMPATIBLE_STATUSES: dict[str, int] = {
    'rate_limit_exceeded': 429,
    'insufficient_quota': 429,
    'context_length_exceeded': 400,
    'service_unavailable': 503,
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
) -> type[ModelAPIError] | None:
    """The error category of an error from an OpenAI-compatible API (OpenAI, Groq, OpenRouter), wherever it was sent."""
    if 'insufficient_quota' in (code, error_type):
        # Exhausted quota is also a 429, but waiting won't help.
        return None
    if code == 'context_length_exceeded' or (
        isinstance(message, str) and any(m in message.lower() for m in _CONTEXT_WINDOW_MESSAGES)
    ):
        return ContextWindowExceeded
    return http_status_category(status_code) if status_code is not None else None


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
    return http_error_class(category)(
        status_code,
        model_name,
        body,
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
