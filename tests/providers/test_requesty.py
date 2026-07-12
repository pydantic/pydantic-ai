import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.requesty import RequestyProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_requesty_provider():
    provider = RequestyProvider(api_key='api-key')
    assert provider.name == 'requesty'
    assert provider.base_url == 'https://router.requesty.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_requesty_provider_with_app_attribution():
    provider = RequestyProvider(api_key='api-key', app_url='test.com', app_title='test')
    assert provider.name == 'requesty'
    assert provider.base_url == 'https://router.requesty.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'
    assert provider.client.default_headers['X-Title'] == 'test'
    assert provider.client.default_headers['HTTP-Referer'] == 'test.com'


def test_requesty_provider_need_api_key(env: TestEnv) -> None:
    env.remove('REQUESTY_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `REQUESTY_API_KEY` environment variable or pass it via `RequestyProvider(api_key=...)`'
            ' to use the Requesty provider.'
        ),
    ):
        RequestyProvider()


def test_requesty_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = RequestyProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_requesty_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = RequestyProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_requesty_provider_model_profile(mocker: MockerFixture):
    provider = RequestyProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.requesty'
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    openai_model_profile_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)
    anthropic_model_profile_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    grok_model_profile_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    amazon_model_profile_mock = mocker.patch(f'{ns}.amazon_model_profile', wraps=amazon_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    moonshotai_model_profile_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)

    google_profile = provider.model_profile('google/gemini-2.5-pro-preview')
    google_model_profile_mock.assert_called_with('gemini-2.5-pro-preview')
    assert google_profile is not None
    assert google_profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    openai_profile = provider.model_profile('openai/o1-mini')
    openai_model_profile_mock.assert_called_with('o1-mini')
    assert openai_profile is not None
    assert openai_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    # Requesty only accepts the older `max_tokens` field, never `max_completion_tokens` — even for OpenAI
    # models, whose own profile defaults the flag to `True`; the merge must not clobber Requesty's `False`.
    assert openai_profile.get('openai_chat_supports_max_completion_tokens', True) is False

    anthropic_profile = provider.model_profile('anthropic/claude-3.5-sonnet')
    anthropic_model_profile_mock.assert_called_with('claude-3-5-sonnet')
    assert anthropic_profile is not None
    assert anthropic_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    anthropic_profile = provider.model_profile('anthropic/claude-sonnet-4.5')
    anthropic_model_profile_mock.assert_called_with('claude-sonnet-4-5')
    assert anthropic_profile is not None
    assert anthropic_profile.get('supports_json_schema_output', False) is True

    mistral_profile = provider.model_profile('mistral/mistral-large-2407')
    mistral_model_profile_mock.assert_called_with('mistral-large-2407')
    assert mistral_profile is not None
    assert mistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    qwen_profile = provider.model_profile('qwen/qwen-2.5-coder-32b')
    qwen_model_profile_mock.assert_called_with('qwen-2.5-coder-32b')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    grok_profile = provider.model_profile('xai/grok-3')
    grok_model_profile_mock.assert_called_with('grok-3')
    assert grok_profile is not None
    assert grok_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    cohere_profile = provider.model_profile('cohere/command-a')
    cohere_model_profile_mock.assert_called_with('command-a')
    assert cohere_profile is not None
    assert cohere_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    amazon_profile = provider.model_profile('amazon/titan-text-express-v1')
    amazon_model_profile_mock.assert_called_with('titan-text-express-v1')
    assert amazon_profile is not None
    assert amazon_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    deepseek_profile = provider.model_profile('deepseek/deepseek-r1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    meta_profile = provider.model_profile('meta/llama-4-maverick')
    meta_model_profile_mock.assert_called_with('llama-4-maverick')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    moonshotai_profile = provider.model_profile('moonshotai/kimi-k2')
    moonshotai_model_profile_mock.assert_called_with('kimi-k2')
    assert moonshotai_profile is not None
    assert moonshotai_profile.get('ignore_streamed_leading_whitespace', False) is True
    assert moonshotai_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown/model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
