from __future__ import annotations as _annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider, infer_provider, infer_provider_class

from ..conftest import try_import

with try_import() as imports_successful:
    from google.auth.exceptions import GoogleAuthError
    from openai import OpenAIError

    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.cohere import CohereProvider
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.fireworks import FireworksProvider
    from pydantic_ai.providers.github import GitHubProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.grok import GrokProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.heroku import HerokuProvider
    from pydantic_ai.providers.litellm import LiteLLMProvider
    from pydantic_ai.providers.mistral import MistralProvider
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider
    from pydantic_ai.providers.nebius import NebiusProvider
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.outlines import OutlinesProvider
    from pydantic_ai.providers.ovhcloud import OVHcloudProvider
    from pydantic_ai.providers.together import TogetherProvider
    from pydantic_ai.providers.vercel import VercelProvider

    test_infer_provider_params = [
        ('anthropic', AnthropicProvider, 'ANTHROPIC_API_KEY'),
        ('cohere', CohereProvider, 'CO_API_KEY'),
        ('deepseek', DeepSeekProvider, 'DEEPSEEK_API_KEY'),
        ('openrouter', OpenRouterProvider, 'OPENROUTER_API_KEY'),
        ('vercel', VercelProvider, 'VERCEL_AI_GATEWAY_API_KEY'),
        ('openai', OpenAIProvider, 'OPENAI_API_KEY'),
        ('azure', AzureProvider, 'AZURE_OPENAI'),
        ('google-vertex', GoogleProvider, 'Your default credentials were not found'),
        ('google-gla', GoogleProvider, 'GOOGLE_API_KEY'),
        ('groq', GroqProvider, 'GROQ_API_KEY'),
        ('mistral', MistralProvider, 'MISTRAL_API_KEY'),
        ('grok', GrokProvider, 'GROK_API_KEY'),
        ('moonshotai', MoonshotAIProvider, 'MOONSHOTAI_API_KEY'),
        ('fireworks', FireworksProvider, 'FIREWORKS_API_KEY'),
        ('together', TogetherProvider, 'TOGETHER_API_KEY'),
        ('heroku', HerokuProvider, 'HEROKU_INFERENCE_KEY'),
        ('github', GitHubProvider, 'GITHUB_API_KEY'),
        ('ollama', OllamaProvider, 'OLLAMA_BASE_URL'),
        ('litellm', LiteLLMProvider, None),
        ('nebius', NebiusProvider, 'NEBIUS_API_KEY'),
        ('ovhcloud', OVHcloudProvider, 'OVHCLOUD_API_KEY'),
        ('gateway/chat', OpenAIProvider, 'PYDANTIC_AI_GATEWAY_API_KEY'),
        ('gateway/groq', GroqProvider, 'PYDANTIC_AI_GATEWAY_API_KEY'),
        ('gateway/gemini', GoogleProvider, 'PYDANTIC_AI_GATEWAY_API_KEY'),
        ('gateway/anthropic', AnthropicProvider, 'PYDANTIC_AI_GATEWAY_API_KEY'),
        ('gateway/converse', BedrockProvider, 'PYDANTIC_AI_GATEWAY_API_KEY'),
        ('outlines', OutlinesProvider, None),
    ]

if not imports_successful():
    test_infer_provider_params = []  # pragma: lax no cover

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='need to install all extra packages'),
]


@pytest.fixture(autouse=True)
def empty_env():
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.mark.parametrize(('provider', 'provider_cls', 'exception_has'), test_infer_provider_params)
def test_infer_provider(provider: str, provider_cls: type[Provider[Any]], exception_has: str | None):
    if exception_has is not None:
        with pytest.raises((UserError, OpenAIError, GoogleAuthError), match=rf'.*{exception_has}.*'):
            infer_provider(provider)
    else:
        assert isinstance(infer_provider(provider), provider_cls)


@pytest.mark.parametrize(('provider', 'provider_cls', 'exception_has'), test_infer_provider_params)
def test_infer_provider_class(provider: str, provider_cls: type[Provider[Any]], exception_has: str | None):
    if provider.startswith('gateway/'):
        pytest.skip('Gateway providers are not supported for this test')

    assert infer_provider_class(provider) == provider_cls
