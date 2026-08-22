import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
    pytest.mark.anyio,
]


def test_init_with_base_url():
    provider = OpenAIProvider(base_url='https://example.com/v1', api_key='foobar')
    assert provider.base_url == 'https://example.com/v1/'
    assert provider.client.api_key == 'foobar'


def test_init_with_no_api_key_will_still_setup_client():
    provider = OpenAIProvider(base_url='http://localhost:19434/v1')
    assert provider.base_url == 'http://localhost:19434/v1/'


def test_init_with_non_openai_model():
    provider = OpenAIProvider(base_url='https://example.com/v1/')
    assert provider.base_url == 'https://example.com/v1/'


def test_init_of_openai_without_api_key_raises_error(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    with pytest.raises(
        UserError,
        match=(
            r'Set the `OPENAI_API_KEY` environment variable or pass it via `OpenAIProvider\(api_key=\.\.\.\)`'
            r" to use the OpenAI provider\. To try Pydantic AI without an API key, use the built-in test model: `Agent\('test'\)`\."
        ),
    ):
        OpenAIProvider()


def test_init_of_openai_with_base_url_and_without_api_key(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    provider = OpenAIProvider(base_url='https://example.com/v1')
    assert provider.client.api_key == 'api-key-not-set'


def test_init_of_openai_with_base_url_env_var_and_without_api_key(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    env.set('OPENAI_BASE_URL', 'https://example.com/v1')
    provider = OpenAIProvider()
    assert provider.client.api_key == 'api-key-not-set'


@pytest.mark.parametrize(
    'model_name',
    ['gpt-4o', 'gpt-5.4', 'o3-mini', 'o4-mini', 'chatgpt-4o-latest', 'ft:gpt-4o-2024-08-06:org:name:id'],
)
def test_first_party_openai_models_keep_leading_whitespace_flag_off(model_name: str):
    """Official Chat Completions names must not ignore leading streamed whitespace.

    Applying the flag to first-party models dropped leading whitespace tokens from
    official OpenAI streams.
    """
    profile = OpenAIProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('ignore_streamed_leading_whitespace', False) is False


@pytest.mark.parametrize(
    'model_name',
    ['Ornith-1.5-35B-A3B-MLX-4bit', 'qwen3', 'my-local-model', 'foobar'],
)
def test_unrecognized_openai_compatible_names_ignore_leading_whitespace(model_name: str):
    """Unrecognized names used with OpenAIProvider are custom-endpoint configurations."""
    profile = OpenAIProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('ignore_streamed_leading_whitespace') is True
