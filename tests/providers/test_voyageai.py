from __future__ import annotations as _annotations

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from voyageai.client_async import AsyncClient

    from pydantic_ai.providers.voyageai import VoyageAIProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='voyageai not installed')


def test_voyageai_provider() -> None:
    provider = VoyageAIProvider(api_key='api-key')
    assert provider.name == 'voyageai'
    assert provider.base_url == 'https://api.voyageai.com/v1'
    assert isinstance(provider.client, AsyncClient)


def test_voyageai_provider_need_api_key(env: TestEnv) -> None:
    env.remove('VOYAGE_API_KEY')
    with pytest.raises(UserError, match='VOYAGE_API_KEY'):
        VoyageAIProvider()


def test_voyageai_provider_pass_voyageai_client() -> None:
    voyageai_client = AsyncClient(api_key='test-api-key')
    provider = VoyageAIProvider(voyageai_client=voyageai_client)
    assert provider.client == voyageai_client
