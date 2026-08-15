"""Shared HTTP client helpers for the HTTPX2 clients Pydantic AI creates and owns."""

from __future__ import annotations

import warnings

# Import httpcore2 eagerly: httpx2 defers it to first client construction, which performs blocking
# I/O if that happens inside the event loop.
import httpcore2  # noqa: F401  # pyright: ignore[reportUnusedImport]
import httpx2

from ._warnings import PydanticAIDeprecationWarning
from .models import DEFAULT_HTTP_TIMEOUT, get_user_agent


def create_async_httpx2_client(*, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5) -> httpx2.AsyncClient:
    """Create an `httpx2.AsyncClient` with Pydantic AI's default timeouts and user agent.

    Each call creates a new client instance. When used via a [`Provider`][pydantic_ai.providers.Provider],
    the client's lifecycle is managed automatically — it will be closed when the provider (or agent) exits.
    """
    return httpx2.AsyncClient(
        timeout=httpx2.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


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
    try:
        import httpx
    except ImportError:
        return

    if isinstance(http_client, httpx.AsyncClient):
        warnings.warn(
            f'`httpx.AsyncClient` support for {consumer} is deprecated and will be removed in v3; '
            'use `httpx2.AsyncClient` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=stacklevel + 1,
        )
