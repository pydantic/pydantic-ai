"""Shared fixtures for Anthropic model tests."""

from __future__ import annotations as _annotations

from collections.abc import Callable

import pytest

from ...conftest import try_import
from ..test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaUsage
    from pydantic import BaseModel

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

AnthropicModelFactory = Callable[..., AnthropicModel]


# Model factory fixture for live API tests
@pytest.fixture
def anthropic_model(anthropic_api_key: str) -> AnthropicModelFactory:
    """Factory to create Anthropic models with custom configuration."""

    def _create_model(
        model_name: str,
        api_key: str | None = None,
    ) -> AnthropicModel:
        """Create an AnthropicModel with the specified configuration.

        Args:
            model_name: Model name like 'claude-sonnet-4-5'
            api_key: Optional API key, defaults to the fixture's anthropic_api_key
        """
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=api_key or anthropic_api_key),
        )

    return _create_model


# Mock model fixtures for unit tests
@pytest.fixture
def mock_sonnet_4_5(allow_model_requests: None) -> tuple[AnthropicModel, AsyncAnthropic]:
    """Mock claude-sonnet-4-5 model for unit tests."""
    c = completion_message(
        [BetaTextBlock(text='{"city": "Mexico City", "country": "Mexico"}', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    return model, mock_client


# Schema fixtures
@pytest.fixture
def city_location_schema() -> type[BaseModel]:
    """Standard CityLocation schema for testing."""

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    return CityLocation


@pytest.fixture
def country_language_schema() -> type[BaseModel]:
    """Standard CountryLanguage schema for testing."""

    class CountryLanguage(BaseModel):
        country: str
        language: str

    return CountryLanguage


# Mock response fixtures
@pytest.fixture
def weather_tool_responses() -> list[BetaMessage]:
    """Standard mock responses for weather tool tests."""
    return [
        completion_message(
            [
                BetaToolUseBlock(
                    id='tool_123',
                    name='get_weather',
                    input={'location': 'Paris'},
                    type='tool_use',
                )
            ],
            BetaUsage(input_tokens=5, output_tokens=10),
        ),
        completion_message(
            [BetaTextBlock(text='The weather in Paris is sunny.', type='text')],
            BetaUsage(input_tokens=3, output_tokens=5),
        ),
    ]
