import pytest

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.cerebras import CerebrasModel
    from pydantic_ai.providers.cerebras import CerebrasProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_cerebras_model_init():
    model = CerebrasModel('llama-3.3-70b', settings={'api_key': 'test_key'})
    assert model.model_name == 'llama-3.3-70b'
    assert isinstance(model._provider, CerebrasProvider)
    assert model._provider.client.api_key == 'test_key'


def test_cerebras_model_profile():
    # Test Llama model
    model = CerebrasModel('llama-3.3-70b', settings={'api_key': 'test_key'})
    profile = model.profile
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert OpenAIModelProfile.from_profile(profile).openai_chat_supports_web_search is False

    # Test Qwen model
    model = CerebrasModel('qwen-3-235b-a22b-instruct-2507', settings={'api_key': 'test_key'})
    profile = model.profile
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert OpenAIModelProfile.from_profile(profile).openai_chat_supports_web_search is False

    # Test GPT-OSS model
    model = CerebrasModel('gpt-oss-120b', settings={'api_key': 'test_key'})
    profile = model.profile
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert OpenAIModelProfile.from_profile(profile).openai_chat_supports_web_search is False

    # Test unknown model - use zai-glm which is valid but won't match any prefix
    model = CerebrasModel('zai-glm-4.6', settings={'api_key': 'test_key'})
    profile = model.profile
    assert isinstance(profile, OpenAIModelProfile)
    assert OpenAIModelProfile.from_profile(profile).openai_chat_supports_web_search is False
