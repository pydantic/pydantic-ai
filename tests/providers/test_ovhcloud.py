import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.ovhcloud import OVHcloudAIEndpointsProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_ovhcloud_provider():
    provider = OVHcloudAIEndpointsProvider(api_key='your-api-key')
    assert provider.name == 'ovhcloud'
    assert provider.base_url == 'https://oai.endpoints.kepler.ai.cloud.ovh.net/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'


def test_ovhcloud_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OVHCLOUD_AI_ENDPOINTS_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OVHCLOUD_AI_ENDPOINTS_API_KEY` environment variable or pass it via '
            '`OVHcloudAIEndpointsProvider(api_key=...)` to use OVHcloud AI Endpoints provider.'
        ),
    ):
        OVHcloudAIEndpointsProvider()


def test_ovhcloud_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='your-api-key')
    provider = OVHcloudAIEndpointsProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_ovhcloud_pass_http_client():
    http_client = httpx.AsyncClient()
    provider = OVHcloudAIEndpointsProvider(api_key='your-api-key', http_client=http_client)
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'
