import re

import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models import infer_model
    from pydantic_ai.models.crusoe import CrusoeModel
    from pydantic_ai.providers.crusoe import CrusoeProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_crusoe_provider():
    provider = CrusoeProvider(api_key='api-key')
    assert provider.name == 'crusoe'
    assert provider.base_url == 'https://api.inference.crusoecloud.com/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_crusoe_provider_need_api_key(env: TestEnv) -> None:
    env.remove('CRUSOE_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `CRUSOE_API_KEY` environment variable or pass it via '
            '`CrusoeProvider(api_key=...)` to use the Crusoe provider.'
        ),
    ):
        CrusoeProvider()


def test_crusoe_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = CrusoeProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_crusoe_provider_model_profile(mocker: MockerFixture):
    provider = CrusoeProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.crusoe'

    # Mock all profile functions
    meta_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    zai_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    harmony_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)

    # Test meta provider
    meta_profile = provider.model_profile('meta-llama/Llama-3.3-70B-Instruct')
    meta_mock.assert_called_with('llama-3.3-70b-instruct')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    # Test deepseek provider
    profile = provider.model_profile('deepseek-ai/DeepSeek-V3-0324')
    deepseek_mock.assert_called_with('deepseek-v3-0324')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    # Test qwen provider
    qwen_profile = provider.model_profile('Qwen/Qwen3-235B-A22B-Instruct-2507')
    qwen_mock.assert_called_with('qwen3-235b-a22b-instruct-2507')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    # Test google provider
    google_profile = provider.model_profile('google/gemma-4-31b-it')
    google_mock.assert_called_with('gemma-4-31b-it')
    assert google_profile is not None
    assert google_profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    # Test harmony (for openai gpt-oss) provider
    profile = provider.model_profile('openai/gpt-oss-120b')
    harmony_mock.assert_called_with('gpt-oss-120b')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    # Test moonshotai provider
    moonshotai_profile = provider.model_profile('moonshotai/Kimi-K2.6')
    moonshotai_mock.assert_called_with('kimi-k2.6')
    assert moonshotai_profile is not None
    assert moonshotai_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    # Test zai provider
    zai_profile = provider.model_profile('zai/GLM-5.2')
    zai_mock.assert_called_with('glm-5.2')
    assert zai_profile is not None

    # Test unknown vendor
    unknown_profile = provider.model_profile('unknown-vendor/unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


@pytest.mark.parametrize(
    'model_name',
    [
        'zai/GLM-5.2',  # `zai_model_profile` doesn't claim native structured output support
        'meta-llama/Llama-3.3-70B-Instruct',
        'unknown-vendor/unknown-model',
        'bare-model-name',
    ],
)
def test_crusoe_provider_supports_structured_output(model_name: str):
    """Crusoe serves every model with guided decoding, so `response_format` works regardless of family."""
    profile = CrusoeProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_json_schema_output') is True
    assert profile.get('supports_json_object_output') is True


def test_infer_crusoe_model(env: TestEnv):
    """`crusoe:` resolves to `CrusoeModel`, not the bare `OpenAIChatModel`."""
    env.set('CRUSOE_API_KEY', 'test-api-key')
    model = infer_model('crusoe:zai/GLM-5.2')
    assert isinstance(model, CrusoeModel)
    assert model.model_name == 'zai/GLM-5.2'


def test_crusoe_provider_model_name_without_slash():
    profile = CrusoeProvider.model_profile('bare-model-name')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
