"""Shared fixtures for Google model tests."""

from __future__ import annotations as _annotations

from collections.abc import Callable

import httpx
import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    GoogleModelFactory = Callable[..., GoogleModel]


@pytest.fixture
def google_model(gemini_api_key: str) -> GoogleModelFactory:
    """Factory to create Google models. Used by VCR-recorded integration tests."""

    def _create_model(
        model_name: str,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> GoogleModel:
        provider = (
            GoogleProvider(api_key=api_key or gemini_api_key, http_client=http_client)
            if http_client is not None
            else GoogleProvider(api_key=api_key or gemini_api_key)
        )
        return GoogleModel(model_name, provider=provider)

    return _create_model
