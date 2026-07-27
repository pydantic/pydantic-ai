"""Canary for when provider SDKs start accepting `httpx2` clients.

Pydantic AI uses `httpx2` for the HTTP it owns end to end, but the provider SDK boundary is
still `httpx`: every SDK below runs `isinstance(http_client, httpx.AsyncClient)` and raises
`TypeError: Invalid 'http_client' argument`. That assert is what keeps `httpx` in
`pydantic-ai-slim`'s required dependencies.

These tests assert the rejection still happens, so they **fail as soon as an SDK ships httpx2
support** — CI resolves the locked versions, so the signal arrives on the monthly Dependabot
bump that brings the new SDK in. A failure here is good news: drop that provider from the list
and move its `http_client` parameter to `httpx2`.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from .conftest import try_import

with try_import() as openai_imports_successful:
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports_successful:
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as groq_imports_successful:
    from pydantic_ai.providers.groq import GroqProvider


def _openai_provider(http_client: httpx2.AsyncClient) -> Any:
    return OpenAIProvider(api_key='api-key', http_client=http_client)  # pyright: ignore[reportArgumentType]


def _anthropic_provider(http_client: httpx2.AsyncClient) -> Any:
    return AnthropicProvider(api_key='api-key', http_client=http_client)  # pyright: ignore[reportArgumentType]


def _groq_provider(http_client: httpx2.AsyncClient) -> Any:
    return GroqProvider(api_key='api-key', http_client=http_client)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    'build_provider',
    [
        pytest.param(
            _openai_provider,
            id='openai',
            marks=pytest.mark.skipif(not openai_imports_successful(), reason='need to install openai'),
        ),
        pytest.param(
            _anthropic_provider,
            id='anthropic',
            marks=pytest.mark.skipif(not anthropic_imports_successful(), reason='need to install anthropic'),
        ),
        pytest.param(
            _groq_provider,
            id='groq',
            marks=pytest.mark.skipif(not groq_imports_successful(), reason='need to install groq'),
        ),
    ],
)
def test_sdk_still_rejects_httpx2_client(build_provider: Callable[[httpx2.AsyncClient], Any]) -> None:
    """When this fails, the SDK accepts `httpx2` — see the module docstring."""
    with pytest.raises(TypeError, match='http_client'):
        build_provider(httpx2.AsyncClient())
