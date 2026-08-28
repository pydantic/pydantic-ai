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
    'AsyncHTTPClient',
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
    LegacyTimeout: TypeAlias = httpx.Timeout
elif legacy_httpx is not None:
    AsyncHTTPClient = legacy_httpx.AsyncClient | httpx2.AsyncClient
    LegacyTimeout = legacy_httpx.Timeout
else:
    AsyncHTTPClient = httpx2.AsyncClient
    # Without legacy HTTPX no `ModelSettings.timeout` can hold one of its `Timeout` objects, so the
    # `isinstance` check in `to_httpx2_timeout` falls back to the type the SDKs already accept and
    # the conversion just rebuilds an equivalent value.
    LegacyTimeout = httpx2.Timeout

_NotGivenT = TypeVar('_NotGivenT')


def create_async_httpx2_client(*, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5) -> httpx2.AsyncClient:
    """Create an `httpx2.AsyncClient` with Pydantic AI's default timeouts and user agent.

    Each call creates a new client instance. When used via a [`Provider`][pydantic_ai.providers.Provider],
    the client's lifecycle is managed automatically — it will be closed when the provider (or agent) exits.
    """
    from .models import get_user_agent

    return httpx2.AsyncClient(
        timeout=httpx2.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


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
