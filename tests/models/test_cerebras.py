import re

import pytest

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.cerebras import CerebrasModel
    from pydantic_ai.providers.cerebras import CerebrasProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_cerebras_model_init():
    model = CerebrasModel('llama-3.3-70b', provider=CerebrasProvider(api_key='test_key'))
    assert model.model_name == 'llama-3.3-70b'
    assert isinstance(model._provider, CerebrasProvider)  # type: ignore[reportPrivateUsage]
    assert model._provider.client.api_key == 'test_key'  # type: ignore[reportPrivateUsage]


def test_cerebras_model_profile():
    provider = CerebrasProvider(api_key='test_key')

    # Test Llama model
    model = CerebrasModel('llama-3.3-70b', provider=provider)
    profile = model.profile
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert OpenAIModelProfile.from_profile(profile).openai_chat_supports_web_search is False

    # Test Qwen model
    model = CerebrasModel('qwen-3-235b-a22b-instruct-2507', provider=provider)
    profile = model.profile
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert OpenAIModelProfile.from_profile(profile).openai_chat_supports_web_search is False

    # Test GPT-OSS model
    model = CerebrasModel('gpt-oss-120b', provider=provider)
    profile = model.profile
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert OpenAIModelProfile.from_profile(profile).openai_chat_supports_web_search is False

    # Test unknown model - use zai-glm which is valid but won't match any prefix
    model = CerebrasModel('zai-glm-4.6', provider=provider)
    profile = model.profile
    assert isinstance(profile, OpenAIModelProfile)
    assert OpenAIModelProfile.from_profile(profile).openai_chat_supports_web_search is False


def test_cerebras_provider_need_api_key(env: TestEnv) -> None:
    env.remove('CEREBRAS_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `CEREBRAS_API_KEY` environment variable or pass it via `CerebrasProvider(api_key=...)` '
            'to use the Cerebras provider.'
        ),
    ):
        CerebrasProvider()


def test_cerebras_provider_pass_openai_client() -> None:
    openai_client = AsyncOpenAI(api_key='test_key', base_url='https://api.cerebras.ai/v1')
    provider = CerebrasProvider(openai_client=openai_client)
    assert provider.client == openai_client
