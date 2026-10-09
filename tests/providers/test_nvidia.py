import re

import httpx2
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.nvidia import nvidia_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models import infer_model
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.nvidia import NVIDIAProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_nvidia_provider():
    provider = NVIDIAProvider(api_key='api-key')
    assert provider.name == 'nvidia'
    assert provider.base_url == 'https://integrate.api.nvidia.com/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'
    assert str(provider.client.base_url) == 'https://integrate.api.nvidia.com/v1/'


def test_nvidia_provider_with_env_api_key(env: TestEnv) -> None:
    env.set('NVIDIA_API_KEY', 'env-api-key')
    provider = NVIDIAProvider()
    assert provider.client.api_key == 'env-api-key'


def test_nvidia_provider_need_api_key(env: TestEnv) -> None:
    env.remove('NVIDIA_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `NVIDIA_API_KEY` environment variable or pass it via `NVIDIAProvider(api_key=...)`'
            ' to use the NVIDIA provider.'
        ),
    ):
        NVIDIAProvider()


def test_nvidia_provider_custom_base_url() -> None:
    provider = NVIDIAProvider(api_key='api-key', base_url='http://localhost:8000/v1')
    assert provider.base_url == 'http://localhost:8000/v1'
    assert str(provider.client.base_url) == 'http://localhost:8000/v1/'


def test_nvidia_pass_openai_client(env: TestEnv) -> None:
    env.remove('NVIDIA_API_KEY')
    openai_client = openai.AsyncOpenAI(api_key='api-key', base_url='http://localhost:8000/v1')
    provider = NVIDIAProvider(openai_client=openai_client)
    assert provider.client == openai_client
    assert provider.base_url == 'http://localhost:8000/v1/'


def test_nvidia_provider_pass_http_client() -> None:
    http_client = httpx2.AsyncClient()
    provider = NVIDIAProvider(api_key='api-key', http_client=http_client)
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_infer_nvidia_model(env: TestEnv) -> None:
    env.set('NVIDIA_API_KEY', 'api-key')
    model = infer_model('nvidia:nvidia/nemotron-3-super-120b-a12b')
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == 'nvidia/nemotron-3-super-120b-a12b'
    assert model.system == 'nvidia'
    assert isinstance(model._provider, NVIDIAProvider)  # type: ignore[reportPrivateUsage]


def test_nvidia_provider_model_profile(mocker: MockerFixture):
    provider = NVIDIAProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.nvidia'
    nvidia_mock = mocker.patch(f'{ns}.nvidia_model_profile', wraps=nvidia_model_profile)
    meta_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    mistral_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)
    zai_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)
    harmony_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)

    profile = provider.model_profile('nvidia/llama-3.1-nemotron-70b-instruct')
    nvidia_mock.assert_called_with('llama-3.1-nemotron-70b-instruct')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    profile = provider.model_profile('nvidia/nemotron-3-super-120b-a12b')
    nvidia_mock.assert_called_with('nemotron-3-super-120b-a12b')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('meta/llama-3.2-90b-vision-instruct')
    meta_mock.assert_called_with('llama-3.2-90b-vision-instruct')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    profile = provider.model_profile('google/gemma-3-12b-it')
    google_mock.assert_called_with('gemma-3-12b-it')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    profile = provider.model_profile('mistralai/mistral-large-2-instruct')
    mistral_mock.assert_called_with('mistral-large-2-instruct')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('deepseek-ai/deepseek-v4.1-flash')
    deepseek_mock.assert_called_with('deepseek-v4.1-flash')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('qwen/qwen3-235b-a22b')
    qwen_mock.assert_called_with('qwen3-235b-a22b')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    profile = provider.model_profile('moonshotai/kimi-k3')
    moonshotai_mock.assert_called_with('kimi-k3')
    assert profile is not None
    assert profile.get('supports_thinking') is True

    profile = provider.model_profile('z-ai/glm-5.3')
    zai_mock.assert_called_with('glm-5.3')
    assert profile is not None
    assert profile.get('supports_thinking') is True

    profile = provider.model_profile('openai/gpt-oss-20b')
    harmony_mock.assert_called_with('gpt-oss-20b')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    # Unknown vendor and names without a vendor prefix fall back to the default OpenAI-compatible profile.
    profile = provider.model_profile('unknown/some-model')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('some-model')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
