from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropicVertex

    from pydantic_ai.providers.anthropic_vertex import AnthropicVertexProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='need to install anthropic-vertex')


def test_anthropic_provider_with_project_and_region():
    mock_region = 'us-east5'
    client = AsyncAnthropicVertex(project_id='test-project', region=mock_region)
    provider = AnthropicVertexProvider(anthropic_client=client)
    assert provider.name == 'anthropic-vertex'
    assert provider.base_url == f'https://{mock_region}-aiplatform.googleapis.com/v1/'
    assert isinstance(provider.client, AsyncAnthropicVertex)
    assert provider.client.region == mock_region


def test_anthropic_provider_with_empty_client_and_valid_env(monkeypatch: pytest.MonkeyPatch):
    mock_region = 'europe-west3'
    monkeypatch.setenv('CLOUD_ML_REGION', mock_region)

    client = AsyncAnthropicVertex()
    provider = AnthropicVertexProvider(anthropic_client=client)
    assert provider.name == 'anthropic-vertex'
    assert provider.client.region == mock_region
    assert provider.base_url == f'https://{mock_region}-aiplatform.googleapis.com/v1/'
    assert isinstance(provider.client, AsyncAnthropicVertex)
