"""Shared HTTP client types and helpers for the HTTPX2 clients Pydantic AI creates and owns."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypeAlias, TypeVar

# Import httpcore2 eagerly: httpx2 defers it to first client construction, which performs blocking
# I/O if that happens inside the event loop.
import httpcore2  # noqa: F401  # pyright: ignore[reportUnusedImport]
import httpx2

from ._warnings import PydanticAIDeprecationWarning

__all__ = (
    'DEFAULT_HTTP_TIMEOUT',
    'DEFAULT_MAX_CONNECTIONS',
    'DEFAULT_MAX_KEEPALIVE_CONNECTIONS',
    'AsyncHTTPClient',
    'ConnectPoolTimeoutCap',
    'HTTPAuth',
    'HTTPTimeout',
    'create_async_httpx2_client',
    'legacy_httpx',
    'to_httpx2_timeout',
    'warn_if_legacy_httpx_client',
)

DEFAULT_HTTP_TIMEOUT: int = 600
"""Default HTTP timeout in seconds for API requests.

This matches the default timeout used by OpenAI's Python client.
See https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9
"""

try:
    import httpx as legacy_httpx
except ImportError:
    legacy_httpx = None

if TYPE_CHECKING:
    import httpx

    AsyncHTTPClient: TypeAlias = httpx.AsyncClient | httpx2.AsyncClient
    HTTPAuth: TypeAlias = httpx.Auth | httpx2.Auth
    HTTPTimeout: TypeAlias = httpx.Timeout | httpx2.Timeout
    LegacyTimeout: TypeAlias = httpx.Timeout
elif legacy_httpx is not None:
    AsyncHTTPClient = legacy_httpx.AsyncClient | httpx2.AsyncClient
    HTTPAuth = legacy_httpx.Auth | httpx2.Auth
    HTTPTimeout = legacy_httpx.Timeout | httpx2.Timeout
    LegacyTimeout = legacy_httpx.Timeout
else:
    AsyncHTTPClient = httpx2.AsyncClient
    HTTPAuth = httpx2.Auth
    HTTPTimeout = httpx2.Timeout
    # Without legacy HTTPX no `ModelSettings.timeout` can hold one of its `Timeout` objects, so the
    # `isinstance` check in `to_httpx2_timeout` falls back to the type the SDKs already accept and
    # the conversion just rebuilds an equivalent value.
    LegacyTimeout = httpx2.Timeout

_NotGivenT = TypeVar('_NotGivenT')


# The OpenAI and Anthropic SDKs' own connection pool limits, rather than HTTPX's 100 and 20.
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 100


def create_async_httpx2_client(
    *,
    timeout: float | httpx2.Timeout = httpx2.Timeout(DEFAULT_HTTP_TIMEOUT, connect=5),
    limits: httpx2.Limits = httpx2.Limits(
        max_connections=DEFAULT_MAX_CONNECTIONS, max_keepalive_connections=DEFAULT_MAX_KEEPALIVE_CONNECTIONS
    ),
) -> httpx2.AsyncClient:
    """Create an `httpx2.AsyncClient` with Pydantic AI's default timeouts, connection limits and user agent.

    This is the client a provider creates when you don't pass your own `http_client`. Call it yourself
    to adjust the timeouts or connection pool limits, and pass the result to the provider as
    `http_client`. A client you create this way is yours to close.

    Args:
        timeout: The client's timeout, in seconds or as an `httpx2.Timeout` that sets each phase
            separately. Defaults to 600 seconds, with a 5-second connect timeout.
        limits: The connection pool limits. Defaults to 1000 connections, of which up to 100 are kept
            alive while idle, matching the OpenAI and Anthropic SDKs' own clients.

    A request timeout whose phases are all equal, such as a number of seconds, can shorten the
    client's connect and pool timeouts but never lengthen them; see
    [`ModelSettings.timeout`][pydantic_ai.settings.ModelSettings.timeout].
    """
    from .models import get_user_agent

    client_timeout = httpx2.Timeout(timeout)
    return httpx2.AsyncClient(
        timeout=client_timeout,
        limits=limits,
        headers={'User-Agent': get_user_agent()},
        event_hooks={'request': [ConnectPoolTimeoutCap(connect=client_timeout.connect, pool=client_timeout.pool)]},
    )


class ConnectPoolTimeoutCap:
    """Request hook that keeps a per-request timeout from lengthening a client's connect and pool timeouts.

    Provider SDKs send a timeout with every request: their own default, a number of seconds from
    [`ModelSettings.timeout`][pydantic_ai.settings.ModelSettings.timeout], or (for google-genai) the
    scalar the provider pins on `HttpOptions`. A scalar sets every phase, so a 600-second read
    timeout would also allow 600 seconds to connect or to wait for a pooled connection. Installed on
    the clients Pydantic AI creates, this hook takes the shorter of the requested and the client's own
    connect and pool timeouts when all four phases of the requested timeout are equal, as a scalar or
    `None` makes them. A timeout whose phases differ was set phase by phase and is left as it is, and
    a client phase without a timeout (`None`) caps nothing.

    It is a request event hook rather than a custom transport because passing `transport=` to an HTTPX
    client turns off the proxies it would otherwise pick up from the environment. Event hooks run for
    every redirect hop, after the client has applied its default timeout to the request.
    """

    def __init__(self, *, connect: float | None, pool: float | None) -> None:
        self._connect = connect
        self._pool = pool

    async def __call__(self, request: httpx2.Request | httpx.Request) -> None:
        requested: dict[str, float | None] | None = request.extensions.get('timeout')
        if requested is None:
            # `AsyncClient.send` sets one before hooks run; only a request handed to the hook directly lacks it.
            return
        # A hand-built request may carry a partial mapping; HTTPX treats a missing phase as no limit.
        phases = {requested.get(phase) for phase in ('connect', 'read', 'write', 'pool')}
        if len(phases) > 1:
            # Phases set separately, e.g. `httpx.Timeout(60, connect=30)`, are what the caller asked for.
            return
        (scalar,) = phases
        request.extensions = {
            **request.extensions,
            'timeout': {
                **requested,
                'connect': _shorter_timeout(scalar, self._connect),
                'pool': _shorter_timeout(scalar, self._pool),
            },
        }


def _shorter_timeout(requested: float | None, cap: float | None) -> float | None:
    # `None` disables a timeout, so any other value is shorter.
    if requested is None:
        return cap
    if cap is None:
        return requested
    return min(requested, cap)


def to_httpx2_timeout(timeout: float | LegacyTimeout | _NotGivenT) -> float | httpx2.Timeout | _NotGivenT:
    """Rebuild a legacy `httpx.Timeout` as the `httpx2.Timeout` that migrated SDKs accept.

    Anything else — a plain number, or the SDK's own not-given sentinel — passes through unchanged,
    so callers can hand [`ModelSettings.timeout`][pydantic_ai.settings.ModelSettings.timeout] straight
    to a client whose HTTPX family no longer matches the one the setting is typed against.
    """
    if isinstance(timeout, LegacyTimeout):
        return httpx2.Timeout(connect=timeout.connect, read=timeout.read, write=timeout.write, pool=timeout.pool)
    return timeout


# TODO(v3): remove, along with the legacy `httpx.AsyncClient` support it warns about.
def warn_if_legacy_httpx_client(http_client: object, *, consumer: str, stacklevel: int) -> None:
    """Warn when a caller-owned HTTP client is a legacy `httpx.AsyncClient` rather than an `httpx2.AsyncClient`.

    Does nothing when legacy `httpx` isn't installed, since no client can then be an instance of it.

    Args:
        http_client: The client the caller was handed; only legacy `httpx.AsyncClient` instances warn.
        consumer: Name of the surface accepting the client, interpolated into the warning message.
        stacklevel: The stacklevel the caller would pass to its own `warnings.warn` call — this helper
            adds 1 to account for its own frame. Callers pick the value that lands the warning on the
            user's provider-constructor call site.
    """
    if legacy_httpx is None:
        return

    if isinstance(http_client, legacy_httpx.AsyncClient):
        warnings.warn(
            f'`httpx.AsyncClient` support for {consumer} is deprecated and will be removed in v3; '
            'use `httpx2.AsyncClient` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=stacklevel + 1,
        )
