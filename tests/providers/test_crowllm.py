import re

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.crowllm import CrowLLMProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_crowllm_provider():
    provider = CrowLLMProvider(api_key='api-key')
    assert provider.name == 'crowllm'
    assert provider.base_url == 'https://crowllm.com/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_crowllm_provider_need_api_key(env: TestEnv) -> None:
    env.remove('CROWLLM_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `CROWLLM_API_KEY` environment variable or pass it via `CrowLLMProvider(api_key=...)`'
            ' to use the CrowLLM provider.'
        ),
    ):
        CrowLLMProvider()


def test_crowllm_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = CrowLLMProvider(openai_client=openai_client)
    assert provider.client == openai_client
