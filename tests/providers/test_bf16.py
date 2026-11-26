import re

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.bf16 import BF16Provider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_bf16_provider(env: TestEnv) -> None:
    env.remove('BF16_BASE_URL')
    provider = BF16Provider(api_key='bf16-key')
    assert provider.name == 'bf16'
    assert provider.base_url == 'https://api.bf16.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'bf16-key'


def test_bf16_provider_reads_the_environment(env: TestEnv) -> None:
    env.set('BF16_API_KEY', 'env-key')
    env.set('BF16_BASE_URL', 'https://proxy.bf16.example/v1')
    provider = BF16Provider()
    assert provider.client.api_key == 'env-key'
    assert provider.base_url == 'https://proxy.bf16.example/v1'


def test_bf16_provider_need_api_key(env: TestEnv) -> None:
    env.remove('BF16_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `BF16_API_KEY` environment variable or pass it via '
            '`BF16Provider(api_key=...)` to use the bf16.ai provider.'
        ),
    ):
        BF16Provider()


def test_bf16_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='bf16-key', base_url='https://bf16.internal.example/v1')
    provider = BF16Provider(openai_client=openai_client)
    assert provider.client is openai_client
    assert provider.base_url == 'https://bf16.internal.example/v1/'


def test_bf16_model_profiles() -> None:
    deepseek_profile = BF16Provider.model_profile('deepseek/deepseek-v4.1-flash-uncensored')
    assert deepseek_profile is not None
    assert deepseek_profile.get('may_tokenize_special_tokens_in_content') is True

    bf16_deepseek_profile = BF16Provider.model_profile('bf16/deepseek-v4.1-flash-1017-exl3-2.5bpw')
    assert bf16_deepseek_profile == deepseek_profile

    glm_profile = BF16Provider.model_profile('zai-org/glm-5.3-uncensored')
    assert glm_profile is not None
    assert glm_profile.get('may_tokenize_special_tokens_in_content') is True

    kimi_profile = BF16Provider.model_profile('moonshotai/kimi-k3.1-pro-uncensored')
    assert kimi_profile is not None
    assert kimi_profile.get('may_tokenize_special_tokens_in_content') is True

    router_profile = BF16Provider.model_profile('bf16/auto')
    assert router_profile is not None
    assert router_profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer
    assert router_profile.get('may_tokenize_special_tokens_in_content') is True
