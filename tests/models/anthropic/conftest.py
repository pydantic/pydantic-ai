"""Shared fixtures for Anthropic model tests."""

from __future__ import annotations as _annotations

import os

import pytest

from ...conftest import RequestCapture, try_import
from ..test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaUsage
    from pydantic import BaseModel

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider


# The `anthropic_model` factory lives in the parent `tests/models/conftest.py` so that the tests in
# that directory can reach it too, and so `capture=True` can hand out the shared request-capture
# client from one place.


def _bedrock_api_key() -> str | None:
    """The `api_key` for a Bedrock client that records with any AWS credentials and replays with none.

    The SDK signs with SigV4 whenever botocore can resolve credentials, and refuses a bearer token
    alongside them — so recording with static keys leaves `api_key` unset, recording with a Bedrock API
    key passes `AWS_BEARER_TOKEN_BEDROCK`, and replay falls back to a placeholder bearer token. Without
    that fallback botocore reaches for the instance metadata endpoint on a runner that has no
    credentials at all, and the request never gets as far as the cassette. `botocore` itself only ships
    under the `bedrock` extra.
    """
    pytest.importorskip('botocore')
    return (
        None
        if os.environ.get('AWS_ACCESS_KEY_ID')
        else os.environ.get('AWS_BEARER_TOKEN_BEDROCK', 'test-bedrock-token')
    )


@pytest.fixture
def anthropic_bedrock_client(request_capture: RequestCapture) -> AsyncAnthropicBedrock:
    """An `AsyncAnthropicBedrock` (InvokeModel) client routed through `request_capture`."""
    return AsyncAnthropicBedrock(
        api_key=_bedrock_api_key(),
        aws_region=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
        http_client=request_capture.client,
    )


@pytest.fixture
def anthropic_bedrock_mantle_client(request_capture: RequestCapture) -> AsyncAnthropicBedrockMantle:
    """An `AsyncAnthropicBedrockMantle` (Bedrock Messages API) client routed through `request_capture`."""
    return AsyncAnthropicBedrockMantle(
        api_key=_bedrock_api_key(),
        aws_region=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
        http_client=request_capture.client,
    )


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
