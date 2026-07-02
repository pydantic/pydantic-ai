"""Shared fixtures for OpenRouter model tests."""

from __future__ import annotations as _annotations

from collections.abc import Callable
from functools import cache

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    OpenRouterModelFactory = Callable[..., OpenRouterModel]


@pytest.fixture
def openrouter_model(openrouter_api_key: str) -> OpenRouterModelFactory:
    """Factory to create OpenRouter models. Used by VCR-recorded integration tests."""

    @cache
    def _create_model(model_name: str) -> OpenRouterModel:
        return OpenRouterModel(model_name, provider=OpenRouterProvider(api_key=openrouter_api_key))

    return _create_model
