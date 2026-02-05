"""Tests for ContextWindowExceeded exception detection across providers."""

from __future__ import annotations

import os
import shutil

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ContextWindowExceeded
from pydantic_ai.providers.gateway import gateway_provider

from .conftest import try_import

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel

with try_import() as groq_imports_successful:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as google_imports_successful:
    from pydantic_ai.models.google import GoogleModel  # pyright: ignore[reportUnusedImport] # noqa: F401

with try_import() as bedrock_imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel

with try_import() as mistral_imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

HUGE_PROMPT = 'word ' * 150_000
ANTHROPIC_HUGE_PROMPT = 'word ' * 250_000
GOOGLE_HUGE_PROMPT = 'word ' * 1_100_000


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_context_window_exceeded(allow_model_requests: None, openai_api_key: str):
    """Test that OpenAI context length exceeded errors raise ContextWindowExceeded."""
    model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'gpt-4o-mini'


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_context_window_exceeded(allow_model_requests: None, gateway_api_key: str):
    """Test that Anthropic context length exceeded errors raise ContextWindowExceeded."""
    model = AnthropicModel('claude-3-5-haiku-latest', provider=gateway_provider('anthropic', api_key=gateway_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(ANTHROPIC_HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'claude-3-5-haiku-latest'


@pytest.mark.skipif(not groq_imports_successful(), reason='groq not installed')
async def test_groq_context_window_exceeded(allow_model_requests: None, groq_api_key: str):
    """Test that Groq context length exceeded errors raise ContextWindowExceeded."""
    model = GroqModel('llama-3.1-8b-instant', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'llama-3.1-8b-instant'


@pytest.mark.skipif(not google_imports_successful(), reason='google-genai not installed')
@pytest.mark.skipif(
    not os.getenv('GOOGLE_APPLICATION_CREDENTIALS') and not shutil.which('gcloud'),
    reason='Google credentials not available',
)
async def test_google_context_window_exceeded(allow_model_requests: None):
    """Test that Google context length exceeded errors raise ContextWindowExceeded."""
    agent = Agent('google-vertex:gemini-2.0-flash')

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(GOOGLE_HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'gemini-2.0-flash'


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
async def test_bedrock_context_window_exceeded(allow_model_requests: None, gateway_api_key: str):
    """Test that Bedrock context length exceeded errors raise ContextWindowExceeded."""
    model = BedrockConverseModel(
        'us.amazon.nova-micro-v1:0', provider=gateway_provider('bedrock', api_key=gateway_api_key)
    )
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'us.amazon.nova-micro-v1:0'


@pytest.mark.skipif(not mistral_imports_successful(), reason='mistral not installed')
async def test_mistral_context_window_exceeded(allow_model_requests: None, mistral_api_key: str):
    """Test that Mistral context length exceeded errors raise ContextWindowExceeded."""
    model = MistralModel('mistral-small-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'mistral-small-latest'
