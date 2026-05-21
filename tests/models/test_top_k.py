from __future__ import annotations as _annotations

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.anyio


async def test_google_top_k_propagation(allow_model_requests: None, mocker: MockerFixture):
    provider = GoogleProvider(api_key='test')
    model = GoogleModel('gemini-1.5-flash', provider=provider)

    # Mock the response
    from google.genai.types import Candidate, Content, GenerateContentResponse, Part

    response = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text='Paris')], role='model'))],
        response_id='1',
        model_version='gemini-1.5-flash',
    )

    mock_generate = mocker.patch.object(model.client.aio.models, 'generate_content', return_value=response)

    agent = Agent(model=model, model_settings={'top_k': 40})
    await agent.run('test')

    # Verify top_k was passed in the config
    assert mock_generate.call_count == 1
    _, kwargs = mock_generate.call_args
    assert kwargs['config']['top_k'] == 40


async def test_anthropic_top_k_propagation(allow_model_requests: None, mocker: MockerFixture):
    provider = AnthropicProvider(api_key='test')
    model = AnthropicModel('claude-3-5-sonnet-latest', provider=provider)

    # Mock the response
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage

    response = BetaMessage(
        id='1',
        content=[BetaTextBlock(text='Paris', type='text')],
        model='claude-3-5-sonnet-latest',
        role='assistant',
        type='message',
        usage=BetaUsage(input_tokens=1, output_tokens=1),
    )

    from unittest.mock import AsyncMock

    mock_create = mocker.patch.object(model.client.beta.messages, 'create', new_callable=AsyncMock)
    mock_create.return_value = response

    agent = Agent(model=model, model_settings={'top_k': 40})
    await agent.run('test')

    # Verify top_k was passed
    assert mock_create.call_count == 1
    _, kwargs = mock_create.call_args
    assert kwargs['top_k'] == 40
