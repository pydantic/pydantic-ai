import re

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.otari import OtariProvider, OtariRegion


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_otari_provider(env: TestEnv) -> None:
    env.remove('OTARI_BASE_URL')
    provider = OtariProvider(api_key='otari-key')
    assert provider.name == 'otari'
    assert provider.base_url == 'https://api.otari.ai/api/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'otari-key'


def test_otari_provider_reads_the_environment(env: TestEnv) -> None:
    env.set('OTARI_API_KEY', 'env-key')
    env.set('OTARI_BASE_URL', 'http://localhost:8000/api/v1')
    provider = OtariProvider()
    assert provider.client.api_key == 'env-key'
    assert provider.base_url == 'http://localhost:8000/api/v1'


def test_otari_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OTARI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OTARI_API_KEY` environment variable or pass it via '
            '`OtariProvider(api_key=...)` to use the Otari provider.'
        ),
    ):
        OtariProvider()


@pytest.mark.parametrize(
    ('region', 'base_url'),
    [('us', 'https://api.otari.ai/api/v1'), ('eu', 'https://eu.api.otari.ai/api/v1')],
)
def test_otari_region(env: TestEnv, region: OtariRegion, base_url: str) -> None:
    # An explicit region wins over `OTARI_BASE_URL`.
    env.set('OTARI_BASE_URL', 'http://localhost:8000/api/v1')
    provider = OtariProvider(api_key='otari-key', region=region)
    assert provider.base_url == base_url
    assert str(provider.client.base_url) == f'{base_url}/'


def test_otari_region_and_base_url_conflict() -> None:
    with pytest.raises(UserError, match=re.escape('Cannot set `region` together with `base_url` or `openai_client`.')):
        OtariProvider(api_key='otari-key', base_url='http://localhost:8000/api/v1', region='eu')  # pyright: ignore[reportCallIssue]


def test_otari_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='otari-key', base_url='https://otari.internal.example/api/v1')
    provider = OtariProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert provider.base_url == 'https://otari.internal.example/api/v1/'
