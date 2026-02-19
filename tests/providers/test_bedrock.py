from typing import cast, get_args

import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from pydantic_ai.models.bedrock import LatestBedrockModelNames
    from pydantic_ai.providers.bedrock import (
        BEDROCK_GEO_PREFIXES,
        BedrockModelProfile,
        BedrockProvider,
        remove_bedrock_geo_prefix,
    )

if not imports_successful():
    BEDROCK_GEO_PREFIXES: tuple[str, ...] = ()  # pragma: lax no cover  # type: ignore[no-redef]

pytestmark = pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')


def test_bedrock_provider(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()
    assert isinstance(provider, BedrockProvider)
    assert provider.name == 'bedrock'
    assert provider.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'


def test_bedrock_provider_bearer_token_env_var(env: TestEnv, mocker: MockerFixture):
    """Test that AWS_BEARER_TOKEN_BEDROCK env var is used for bearer token auth."""
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    env.set('AWS_BEARER_TOKEN_BEDROCK', 'test-bearer-token')

    mock_session = mocker.patch('pydantic_ai.providers.bedrock._BearerTokenSession')

    provider = BedrockProvider()

    mock_session.assert_called_once_with('test-bearer-token')
    assert provider.name == 'bedrock'


def test_bedrock_provider_timeout(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    env.set('AWS_READ_TIMEOUT', '1')
    env.set('AWS_CONNECT_TIMEOUT', '1')
    provider = BedrockProvider()
    assert isinstance(provider, BedrockProvider)
    assert provider.name == 'bedrock'

    config = cast(BedrockRuntimeClient, provider.client).meta.config
    assert config.read_timeout == 1  # type: ignore
    assert config.connect_timeout == 1  # type: ignore


def test_bedrock_provider_model_profile(env: TestEnv, mocker: MockerFixture):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()

    ns = 'pydantic_ai.providers.bedrock'
    anthropic_model_profile_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    amazon_model_profile_mock = mocker.patch(f'{ns}.amazon_model_profile', wraps=amazon_model_profile)

    anthropic_profile = provider.model_profile('us.anthropic.claude-3-5-sonnet-20240620-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-3-5-sonnet-20240620')
    assert isinstance(anthropic_profile, BedrockModelProfile)
    assert anthropic_profile.bedrock_supports_tool_choice is True
    # Bedrock does not support native structured output, even for models that support it via direct Anthropic API
    assert anthropic_profile.supports_json_schema_output is False
    assert anthropic_profile.json_schema_transformer is None
    assert anthropic_profile.supported_builtin_tools == frozenset()

    anthropic_profile = provider.model_profile('anthropic.claude-instant-v1')
    anthropic_model_profile_mock.assert_called_with('claude-instant')
    assert isinstance(anthropic_profile, BedrockModelProfile)
    assert anthropic_profile.bedrock_supports_tool_choice is True
    assert anthropic_profile.supports_json_schema_output is False
    assert anthropic_profile.json_schema_transformer is None
    assert anthropic_profile.supported_builtin_tools == frozenset()

    mistral_profile = provider.model_profile('mistral.mistral-large-2407-v1:0')
    mistral_model_profile_mock.assert_called_with('mistral-large-2407')
    assert isinstance(mistral_profile, BedrockModelProfile)
    assert mistral_profile.bedrock_tool_result_format == 'json'
    assert mistral_profile.supported_builtin_tools == frozenset()

    meta_profile = provider.model_profile('meta.llama3-8b-instruct-v1:0')
    meta_model_profile_mock.assert_called_with('llama3-8b-instruct')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert meta_profile.supported_builtin_tools == frozenset()

    cohere_profile = provider.model_profile('cohere.command-text-v14')
    cohere_model_profile_mock.assert_called_with('command-text')
    assert cohere_profile is not None
    assert cohere_profile.supported_builtin_tools == frozenset()

    deepseek_profile = provider.model_profile('deepseek.deepseek-r1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.ignore_streamed_leading_whitespace is True
    assert deepseek_profile.supported_builtin_tools == frozenset()

    amazon_profile = provider.model_profile('us.amazon.nova-pro-v1:0')
    amazon_model_profile_mock.assert_called_with('nova-pro')
    assert isinstance(amazon_profile, BedrockModelProfile)
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert amazon_profile.bedrock_supports_tool_choice is True
    assert amazon_profile.bedrock_supports_prompt_caching is True
    assert amazon_profile.supported_builtin_tools == frozenset()

    amazon_profile = provider.model_profile('us.amazon.nova-2-lite-v1:0')
    amazon_model_profile_mock.assert_called_with('nova-2-lite')
    assert isinstance(amazon_profile, BedrockModelProfile)
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert amazon_profile.bedrock_supports_tool_choice is True
    assert amazon_profile.bedrock_supports_prompt_caching is True
    assert amazon_profile.supported_builtin_tools == frozenset({CodeExecutionTool})

    amazon_profile = provider.model_profile('us.amazon.titan-text-express-v1:0')
    amazon_model_profile_mock.assert_called_with('titan-text-express')
    assert amazon_profile is not None
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert amazon_profile.supported_builtin_tools == frozenset()

    unknown_model = provider.model_profile('unknown-model')
    assert unknown_model is None

    unknown_model = provider.model_profile('unknown.unknown-model')
    assert unknown_model is None


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    [
        ('us.anthropic.claude-sonnet-4-20250514-v1:0', 'anthropic.claude-sonnet-4-20250514-v1:0'),
        ('eu.amazon.nova-micro-v1:0', 'amazon.nova-micro-v1:0'),
        ('apac.meta.llama3-8b-instruct-v1:0', 'meta.llama3-8b-instruct-v1:0'),
        ('anthropic.claude-3-7-sonnet-20250219-v1:0', 'anthropic.claude-3-7-sonnet-20250219-v1:0'),
    ],
)
def test_remove_inference_geo_prefix(model_name: str, expected: str):
    assert remove_bedrock_geo_prefix(model_name) == expected


@pytest.mark.parametrize('prefix', BEDROCK_GEO_PREFIXES)
def test_bedrock_provider_model_profile_all_geo_prefixes(env: TestEnv, prefix: str):
    """Test that all cross-region inference geo prefixes are correctly handled."""
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()

    model_name = f'{prefix}.anthropic.claude-sonnet-4-5-20250929-v1:0'
    profile = provider.model_profile(model_name)

    assert profile is not None, f'model_profile returned None for {model_name}'


def test_bedrock_provider_model_profile_with_unknown_geo_prefix(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()

    model_name = 'narnia.anthropic.claude-sonnet-4-5-20250929-v1:0'
    profile = provider.model_profile(model_name)
    assert profile is None, f'model_profile returned {profile} for {model_name}'


def test_latest_bedrock_model_names_geo_prefixes_are_supported():
    """Ensure all geo prefixes used in LatestBedrockModelNames are in BEDROCK_GEO_PREFIXES.

    This test prevents adding new model names with geo prefixes that aren't handled
    by the provider's model_profile method.
    """
    model_names = get_args(LatestBedrockModelNames)

    missing_prefixes: set[str] = set()

    for model_name in model_names:
        # Model names with geo prefixes have 3+ dot-separated parts:
        # - No prefix: "anthropic.claude-xxx" (2 parts)
        # - With prefix: "us.anthropic.claude-xxx" (3 parts)
        parts = model_name.split('.')
        if len(parts) >= 3:
            geo_prefix = parts[0]
            if geo_prefix not in BEDROCK_GEO_PREFIXES:  # pragma: no cover
                missing_prefixes.add(geo_prefix)

    if missing_prefixes:  # pragma: no cover
        pytest.fail(
            f'Found geo prefixes in LatestBedrockModelNames that are not in BEDROCK_GEO_PREFIXES: {missing_prefixes}. '
            f'Please add them to BEDROCK_GEO_PREFIXES'
        )
