from __future__ import annotations

import httpx2

from .models import DEFAULT_HTTP_TIMEOUT, get_user_agent


def create_httpx2_client(*, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        timeout=httpx2.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )
