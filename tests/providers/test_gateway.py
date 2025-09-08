import re
from typing import Any, Literal

import httpx
import pytest

from pydantic_ai import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.gateway import GatewayProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
    pytest.mark.anyio,
]


@pytest.mark.parametrize(
    'provider_name, provider_cls',
    [('openai', OpenAIProvider), ('openai-chat', OpenAIProvider), ('openai-responses', OpenAIProvider)],
)
def test_init_with_base_url(
    provider_name: Literal['openai', 'openai-chat', 'openai-responses'], provider_cls: type[Provider[Any]]
):
    provider = GatewayProvider(provider=provider_name, base_url='https://example.com/', api_key='foobar')
    assert isinstance(provider, provider_cls)
    assert provider.base_url == 'https://example.com/openai/'
    assert provider.client.api_key == 'foobar'


def test_init_gateway_without_api_key_raises_error(env: TestEnv):
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `GatewayProvider(api_key=...)` to use the Pydantic AI Gateway provider.'
        ),
    ):
        GatewayProvider(provider='openai')


async def test_init_with_http_client():
    async with httpx.AsyncClient() as http_client:
        provider = GatewayProvider(provider='openai', http_client=http_client, api_key='foobar')
        assert provider.client._client == http_client  # type: ignore
