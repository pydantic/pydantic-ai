from __future__ import annotations as _annotations

import importlib
from importlib.util import find_spec

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv

pytestmark = [
    pytest.mark.skipif(find_spec('voyageai') is None, reason='voyageai not installed'),
    pytest.mark.usefixtures('voyageai_imported'),
]


@pytest.fixture(scope='module')
def voyageai_imported() -> None:
    """Pay for importing `voyageai` in shared setup, outside each test's time budget.

    With `sentence-transformers` installed, `voyageai` imports it and `torch`, which takes seconds: importing it at
    module level would make every worker collecting this module pay for it. See "Test cost" in `tests/AGENTS.md`.
    """
    importlib.import_module('pydantic_ai.providers.voyageai')


def test_voyageai_provider() -> None:
    from voyageai.client_async import AsyncClient

    from pydantic_ai.providers.voyageai import VoyageAIProvider

    provider = VoyageAIProvider(api_key='api-key')
    assert provider.name == 'voyageai'
    assert provider.base_url == 'https://api.voyageai.com/v1'
    assert isinstance(provider.client, AsyncClient)


def test_voyageai_provider_need_api_key(env: TestEnv) -> None:
    from pydantic_ai.providers.voyageai import VoyageAIProvider

    env.remove('VOYAGE_API_KEY')
    with pytest.raises(UserError, match='VOYAGE_API_KEY'):
        VoyageAIProvider()


def test_voyageai_provider_pass_voyageai_client() -> None:
    from voyageai.client_async import AsyncClient

    from pydantic_ai.providers.voyageai import VoyageAIProvider

    voyageai_client = AsyncClient(api_key='test-api-key')
    provider = VoyageAIProvider(voyageai_client=voyageai_client)
    assert provider.client == voyageai_client
