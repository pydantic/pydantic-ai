from __future__ import annotations as _annotations

import os

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, NativeOutput, PromptedOutput, ThinkingPart

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.fireworks import FireworksProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


class City(BaseModel):
    city: str
    country: str


@pytest.mark.parametrize('model_name', ['kimi-k3', 'llama-v3p3-70b-instruct'])
async def test_fireworks_native_output(allow_model_requests: None, request_capture: RequestCapture, model_name: str):
    """Fireworks enforces native schemas for unprofiled and family-profiled models."""
    provider = FireworksProvider(
        api_key=os.getenv('FIREWORKS_API_KEY', 'mock-api-key'), http_client=request_capture.client
    )
    agent = Agent(
        OpenAIChatModel(f'accounts/fireworks/models/{model_name}', provider=provider), output_type=NativeOutput(City)
    )
    result = await agent.run('Where is the Eiffel Tower? Reply in JSON.')
    assert result.output == City(city='Paris', country='France')
    assert not any(isinstance(part, ThinkingPart) for part in result.response.parts)
    response_format = request_capture.body('/chat/completions')['response_format']
    if model_name == 'kimi-k3':
        assert response_format == snapshot()
    else:
        assert response_format == snapshot()


@pytest.mark.parametrize('model_name', ['kimi-k3', 'gemma-3-27b-it'])
async def test_fireworks_prompted_output(allow_model_requests: None, request_capture: RequestCapture, model_name: str):
    """Native schema support preserves prompted output's family-specific JSON-mode behavior."""
    provider = FireworksProvider(
        api_key=os.getenv('FIREWORKS_API_KEY', 'mock-api-key'), http_client=request_capture.client
    )
    agent = Agent(
        OpenAIChatModel(f'accounts/fireworks/models/{model_name}', provider=provider), output_type=PromptedOutput(City)
    )
    result = await agent.run('Where is the Eiffel Tower?')
    assert result.output == City(city='Paris', country='France')
    response_format = request_capture.body('/chat/completions').get('response_format')
    if model_name == 'gemma-3-27b-it':
        assert response_format == {'type': 'json_object'}
    else:
        assert response_format is None
