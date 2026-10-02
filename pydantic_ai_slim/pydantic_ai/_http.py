"""Shared HTTP client types and helpers for the HTTPX2 clients Pydantic AI creates and owns."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

# Import httpcore2 eagerly: httpx2 defers it to first client construction, which performs blocking
# I/O if that happens inside the event loop.
import httpcore2  # noqa: F401  # pyright: ignore[reportUnusedImport]
import httpx2

from ._warnings import PydanticAIDeprecationWarning

__all__ = (
    'DEFAULT_HTTP_TIMEOUT',
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


def create_async_httpx2_client(*, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5) -> httpx2.AsyncClient:
    """Create an `httpx2.AsyncClient` with Pydantic AI's default timeouts and user agent.

    Each call creates a new client instance. When used via a [`Provider`][pydantic_ai.providers.Provider],
    the client's lifecycle is managed automatically — it will be closed when the provider (or agent) exits.

    A number of seconds passed as an individual request's timeout can shorten the client's `connect`
    timeout and its pool timeout (`timeout`), but never lengthen them; see `ConnectPoolTimeoutCap`.
    """
    from .models import get_user_agent

    return httpx2.AsyncClient(
        timeout=httpx2.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
        event_hooks={'request': [ConnectPoolTimeoutCap(connect=connect, pool=timeout)]},
    )


class ConnectPoolTimeoutCap:
    """Request hook that keeps a per-request timeout from lengthening a client's connect and pool timeouts.

    Provider SDKs send a timeout with every request: their own default, a number of seconds from
    [`ModelSettings.timeout`][pydantic_ai.settings.ModelSettings.timeout], or (for google-genai) the
    scalar the provider pins on `HttpOptions`. A scalar sets every phase, so a 600-second read
    timeout would also allow 600 seconds to connect or to wait for a pooled connection. Installed on
    the clients Pydantic AI creates, this hook takes the shorter of the requested and the client's own
    connect and pool timeouts when all four phases of the requested timeout are equal, as a scalar or
    `None` makes them. A timeout whose phases differ was set phase by phase and is left as it is.

    It is a request event hook rather than a custom transport because passing `transport=` to an HTTPX
    client turns off the proxies it would otherwise pick up from the environment. Event hooks run for
    every redirect hop, after the client has applied its default timeout to the request.
    """

    def __init__(self, *, connect: float, pool: float) -> None:
        self._connect = connect
        self._pool = pool

    async def __call__(self, request: httpx2.Request | httpx.Request) -> None:
        requested: dict[str, Any] | None = request.extensions.get('timeout')
        if requested is None:
            # `AsyncClient.send` sets one before hooks run; only a request handed to the hook directly lacks it.
            return
        if len({requested['connect'], requested['read'], requested['write'], requested['pool']}) > 1:
            # Phases set separately, e.g. `httpx.Timeout(60, connect=30)`, are what the caller asked for.
            return
        request.extensions = {
            **request.extensions,
            'timeout': {
                **requested,
                'connect': _shorter_timeout(requested['connect'], self._connect),
                'pool': _shorter_timeout(requested['pool'], self._pool),
            },
        }


def _shorter_timeout(requested: float | None, cap: float) -> float:
    # `None` disables a timeout, so any cap is shorter.
    return cap if requested is None else min(requested, cap)


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
