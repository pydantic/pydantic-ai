"""Shared tests for the `max_retries` parameter on OpenAI SDK-based providers.

Every provider that builds its own `AsyncOpenAI`/`AsyncAzureOpenAI` client exposes a
`max_retries` parameter that is forwarded to the SDK client, defaulting to `2` to match
the OpenAI SDK. These tests assert the parameter is plumbed through for each provider.
"""

# pyright: reportDeprecated=false

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.alibaba import AlibabaProvider
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.fireworks import FireworksProvider
    from pydantic_ai.providers.github import GitHubProvider
    from pydantic_ai.providers.grok import GrokProvider
    from pydantic_ai.providers.heroku import HerokuProvider
    from pydantic_ai.providers.litellm import LiteLLMProvider
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider
    from pydantic_ai.providers.nebius import NebiusProvider
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.ovhcloud import OVHcloudProvider
    from pydantic_ai.providers.sambanova import SambaNovaProvider
    from pydantic_ai.providers.together import TogetherProvider
    from pydantic_ai.providers.vercel import VercelProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def _params() -> list[Any]:
    # Each entry: (provider factory, kwargs required to construct it without hitting the network).
    providers: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
        'openai': (OpenAIProvider, {'api_key': 'api-key'}),
        'azure': (
            AzureProvider,
            {
                'azure_endpoint': 'https://example.openai.azure.com/',
                'api_version': '2024-12-01-preview',
                'api_key': 'api-key',
            },
        ),
        'deepseek': (DeepSeekProvider, {'api_key': 'api-key'}),
        'alibaba': (AlibabaProvider, {'api_key': 'api-key'}),
        'cerebras': (CerebrasProvider, {'api_key': 'api-key'}),
        'fireworks': (FireworksProvider, {'api_key': 'api-key'}),
        'github': (GitHubProvider, {'api_key': 'api-key'}),
        'grok': (GrokProvider, {'api_key': 'api-key'}),
        'heroku': (HerokuProvider, {'api_key': 'api-key'}),
        'litellm': (LiteLLMProvider, {'api_key': 'api-key'}),
        'moonshotai': (MoonshotAIProvider, {'api_key': 'api-key'}),
        'nebius': (NebiusProvider, {'api_key': 'api-key'}),
        'ollama': (OllamaProvider, {'base_url': 'http://localhost:11434/v1', 'api_key': 'api-key'}),
        'openrouter': (OpenRouterProvider, {'api_key': 'api-key'}),
        'ovhcloud': (OVHcloudProvider, {'api_key': 'api-key'}),
        'sambanova': (SambaNovaProvider, {'api_key': 'api-key'}),
        'together': (TogetherProvider, {'api_key': 'api-key'}),
        'vercel': (VercelProvider, {'api_key': 'api-key'}),
    }
    # `GrokProvider` is deprecated in favor of `XaiProvider`, but still supports `max_retries`.
    marks = {'grok': pytest.mark.filterwarnings('ignore:`GrokProvider` is deprecated:DeprecationWarning')}
    return [
        pytest.param(factory, kwargs, id=name, marks=marks.get(name, ()))
        for name, (factory, kwargs) in providers.items()
    ]


@pytest.mark.parametrize('factory, kwargs', _params())
def test_provider_max_retries(factory: Callable[..., Any], kwargs: dict[str, Any]) -> None:
    assert factory(**kwargs).client.max_retries == 2  # default matches the OpenAI SDK
    assert factory(**kwargs, max_retries=5).client.max_retries == 5
    assert factory(**kwargs, max_retries=0).client.max_retries == 0
