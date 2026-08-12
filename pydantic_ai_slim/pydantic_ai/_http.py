from __future__ import annotations

import warnings

import httpx2

from ._warnings import PydanticAIDeprecationWarning
from .models import DEFAULT_HTTP_TIMEOUT, get_user_agent


def create_httpx2_client(*, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        timeout=httpx2.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


def warn_if_legacy_httpx_client(http_client: object, *, consumer: str, stacklevel: int) -> None:
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
