import re

import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

from .._inline_snapshot import snapshot
from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import (
        OpenRouterProvider,
        _OpenRouterGoogleJsonSchemaTransformer,  # pyright: ignore[reportPrivateUsage]
    )


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_openrouter_provider():
    provider = OpenRouterProvider(api_key='api-key')
    assert provider.name == 'openrouter'
    assert provider.base_url == 'https://openrouter.ai/api/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_openrouter_provider_with_app_attribution():
    provider = OpenRouterProvider(api_key='api-key', app_url='test.com', app_title='test')
    assert provider.name == 'openrouter'
    assert provider.base_url == 'https://openrouter.ai/api/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'
    assert provider.client.default_headers['X-Title'] == 'test'
    assert provider.client.default_headers['HTTP-Referer'] == 'test.com'


def test_openrouter_provider_app_attribution_from_env(env: TestEnv):
    """`app_url` and `app_title` fall back to the environment when omitted.

    Asserts the constructed client's `default_headers` rather than a recorded request, since the
    attribution headers are set once at client construction and a cassette match is not sensitive
    to them.
    """
    env.set('OPENROUTER_APP_URL', 'env.test.com')
    env.set('OPENROUTER_APP_TITLE', 'env test')

    provider = OpenRouterProvider(api_key='api-key')
    assert provider.client.default_headers['HTTP-Referer'] == 'env.test.com'
    assert provider.client.default_headers['X-Title'] == 'env test'


def test_openrouter_provider_app_attribution_skipped_for_prebuilt_client(env: TestEnv):
    """A prebuilt `openai_client` is reused as-is, so the environment fallbacks do not reach it.

    The overloads already stop a caller from passing `app_url`/`app_title` alongside `openai_client`,
    so the environment variables are the path that can silently go missing: they apply without the
    caller writing any attribution argument at all.
    """
    env.set('OPENROUTER_APP_URL', 'env.test.com')
    env.set('OPENROUTER_APP_TITLE', 'env test')

    client = openai.AsyncOpenAI(api_key='api-key', base_url='https://openrouter.ai/api/v1')
    provider = OpenRouterProvider(openai_client=client)

    assert provider.client is client
    assert 'HTTP-Referer' not in provider.client.default_headers
    assert 'X-Title' not in provider.client.default_headers


def test_openrouter_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OPENROUTER_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OPENROUTER_API_KEY` environment variable or pass it via `OpenRouterProvider(api_key=...)`'
            ' to use the OpenRouter provider.'
        ),
    ):
        OpenRouterProvider()


def test_openrouter_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = OpenRouterProvider(openai_client=openai_client)
    assert provider.client == openai_client


async def test_openrouter_with_google_model(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    agent = Agent(model, instructions='Be helpful.')
    response = await agent.run('Tell me a joke.')
    assert response.output == snapshot("""\
Why don't scientists trust atoms? \n\

Because they make up everything!
""")


def test_openrouter_provider_model_profile(mocker: MockerFixture):
    provider = OpenRouterProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.openrouter'
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
    assert google_profile.get('json_schema_transformer', None) == _OpenRouterGoogleJsonSchemaTransformer

    google_profile = provider.model_profile('google/gemma-3n-e4b-it:free')
    google_model_profile_mock.assert_called_with('gemma-3n-e4b-it')
    assert google_profile is not None
    assert google_profile.get('json_schema_transformer', None) == _OpenRouterGoogleJsonSchemaTransformer

    openai_profile = provider.model_profile('openai/o1-mini')
    openai_model_profile_mock.assert_called_with('o1-mini')
    assert openai_profile is not None
    assert openai_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    # OpenRouter only accepts the older `max_tokens` field, never `max_completion_tokens` — even for OpenAI
    # models, whose own profile defaults the flag to `True`; the merge must not clobber OpenRouter's `False`.
    assert openai_profile.get('openai_chat_supports_max_completion_tokens', True) is False

    anthropic_profile = provider.model_profile('anthropic/claude-3.5-sonnet')
    anthropic_model_profile_mock.assert_called_with('claude-3-5-sonnet')
    assert anthropic_profile is not None
    assert anthropic_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    anthropic_profile = provider.model_profile('anthropic/claude-sonnet-4.5')
    anthropic_model_profile_mock.assert_called_with('claude-sonnet-4-5')
    assert anthropic_profile is not None
    assert anthropic_profile.get('supports_json_schema_output', False) is True

    anthropic_profile = provider.model_profile('anthropic/claude-haiku-4.5:free')
    anthropic_model_profile_mock.assert_called_with('claude-haiku-4-5')
    assert anthropic_profile is not None
    assert anthropic_profile.get('supports_json_schema_output', False) is True

    mistral_profile = provider.model_profile('mistralai/mistral-large-2407')
    mistral_model_profile_mock.assert_called_with('mistral-large-2407')
    assert mistral_profile is not None
    assert mistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    qwen_profile = provider.model_profile('qwen/qwen-2.5-coder-32b')
    qwen_model_profile_mock.assert_called_with('qwen-2.5-coder-32b')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    grok_profile = provider.model_profile('x-ai/grok-3')
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

    meta_profile = provider.model_profile('meta-llama/llama-4-maverick')
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


@pytest.mark.parametrize(
    ('model_name', 'expected_flags'),
    [
        # Anthropic: full cache support, TTL, tool-definition caching, dynamic-instruction split, 4-breakpoint cap.
        (
            'anthropic/claude-sonnet-4.6',
            {
                'openrouter_supports_cache_control': True,
                'openrouter_supports_cache_ttl': True,
                'openrouter_supports_tool_cache': True,
                'openrouter_supports_dynamic_instruction_cache': True,
                'openrouter_max_cache_points': 4,
            },
        ),
        # Google: cache_control only — no TTL, no tool caching, no dynamic-instruction split, no cap.
        (
            'google/gemini-2.5-flash',
            {
                'openrouter_supports_cache_control': True,
                'openrouter_supports_cache_ttl': False,
                'openrouter_supports_tool_cache': False,
                'openrouter_supports_dynamic_instruction_cache': False,
                'openrouter_max_cache_points': None,
            },
        ),
        # Unsupported downstream provider: no cache support at all.
        (
            'openai/gpt-5-mini',
            {
                'openrouter_supports_cache_control': False,
                'openrouter_supports_cache_ttl': False,
                'openrouter_supports_tool_cache': False,
                'openrouter_supports_dynamic_instruction_cache': False,
                'openrouter_max_cache_points': None,
            },
        ),
        # `~provider` latest-alias models resolve to the same downstream cache capabilities.
        (
            '~anthropic/claude-sonnet-latest',
            {
                'openrouter_supports_cache_control': True,
                'openrouter_supports_cache_ttl': True,
                'openrouter_supports_tool_cache': True,
                'openrouter_supports_dynamic_instruction_cache': True,
                'openrouter_max_cache_points': 4,
            },
        ),
        (
            '~google/gemini-pro-latest',
            {
                'openrouter_supports_cache_control': True,
                'openrouter_supports_cache_ttl': False,
                'openrouter_supports_tool_cache': False,
                'openrouter_supports_dynamic_instruction_cache': False,
                'openrouter_max_cache_points': None,
            },
        ),
    ],
)
def test_openrouter_model_profile_cache_capabilities(model_name: str, expected_flags: dict[str, object]) -> None:
    """Cache capability flags are derived from the downstream provider, not model-name matching."""
    provider = OpenRouterProvider(api_key='api-key')
    profile = provider.model_profile(model_name)
    assert profile is not None

    actual = {flag: value for flag, value in profile.items() if flag in expected_flags}
    assert actual == expected_flags


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    [
        # Anthropic rejects a forced `tool_choice` while thinking is enabled; OpenRouter swallows that
        # incompatibility by dropping `reasoning`, so the gateway marks the combination unsupported.
        ('anthropic/claude-sonnet-4.6', False),
        ('~anthropic/claude-sonnet-latest', False),
        # Other downstream providers honor `reasoning` alongside a forced `tool_choice`.
        ('google/gemini-2.5-flash', True),
        ('openai/gpt-5-mini', True),
        ('unknown/model', True),
    ],
)
def test_openrouter_model_profile_forced_tool_choice_with_thinking(model_name: str, expected: bool) -> None:
    """Forced-`tool_choice`-with-thinking support is derived from the downstream provider."""
    provider = OpenRouterProvider(api_key='api-key')
    profile = provider.model_profile(model_name)
    assert profile is not None
    assert profile.get('openrouter_supports_forced_tool_choice_with_thinking') is expected


def test_openrouter_model_profile_requires_provider_prefix() -> None:
    provider = OpenRouterProvider(api_key='api-key')
    with pytest.raises(UserError, match=re.escape("e.g. 'openai/gpt-4o', not 'gpt-4o'")):
        provider.model_profile('gpt-4o')


def test_openrouter_google_json_schema_transformer():
    """Test _OpenRouterGoogleJsonSchemaTransformer covers all transformation cases."""
    schema = {
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'title': 'TestSchema',
        'type': 'object',
        'properties': {
            'status': {'const': 'active'},
            'category': {'oneOf': [{'type': 'string'}, {'type': 'integer'}]},
            'email': {'type': 'string', 'format': 'email', 'description': 'User email'},
            'date': {'type': 'string', 'format': 'date'},
        },
    }

    transformer = _OpenRouterGoogleJsonSchemaTransformer(schema)
    result = transformer.walk()

    # const -> enum conversion
    assert result['properties']['status'] == {'enum': ['active'], 'type': 'string'}

    # oneOf -> anyOf conversion
    assert 'anyOf' in result['properties']['category']
    assert 'oneOf' not in result['properties']['category']

    # format -> description with existing description
    assert result['properties']['email']['description'] == 'User email (format: email)'
    assert 'format' not in result['properties']['email']

    # format -> description without existing description
    assert result['properties']['date']['description'] == 'Format: date'
    assert 'format' not in result['properties']['date']

    # Removed fields
    assert '$schema' not in result
    assert 'title' not in result
