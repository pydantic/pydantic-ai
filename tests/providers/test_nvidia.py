import os
import re

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.nvidia import NVIDIAProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_nvidia_provider():
    provider = NVIDIAProvider(api_key='api-key')
    assert provider.name == 'nvidia'
    assert provider.base_url == 'https://integrate.api.nvidia.com/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_nvidia_provider_need_api_key(env: TestEnv) -> None:
    env.remove('NVIDIA_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            os.path.normpath(
                'Set the `NVIDIA_API_KEY` environment variable or pass it via `NVIDIAProvider(api_key=...)`'
                ' to use the NVIDIA provider.'
            )
        ),
    ):
        NVIDIAProvider()


def test_nvidia_provider_base_url_from_env(env: TestEnv) -> None:
    env.set('NVIDIA_BASE_URL', 'https://custom.example.com/v1')
    provider = NVIDIAProvider(api_key='api-key')
    assert provider.base_url == 'https://custom.example.com/v1'


def test_nvidia_provider_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = NVIDIAProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_nvidia_provider_model_profile() -> None:
    provider = NVIDIAProvider(api_key='api-key')
    model = OpenAIChatModel('some-model', provider=provider)
    assert provider.model_profile('some-model') == OpenAIModelProfile()
    assert model.profile['supports_tools'] is True