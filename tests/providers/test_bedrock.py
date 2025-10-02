from typing import cast

import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')


def test_bedrock_provider(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()
    assert isinstance(provider, BedrockProvider)
    assert provider.name == 'bedrock'
    assert provider.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'


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

    anthropic_profile = provider.model_profile('anthropic.claude-instant-v1')
    anthropic_model_profile_mock.assert_called_with('claude-instant')
    assert isinstance(anthropic_profile, BedrockModelProfile)
    assert anthropic_profile.bedrock_supports_tool_choice is True

    mistral_profile = provider.model_profile('mistral.mistral-large-2407-v1:0')
    mistral_model_profile_mock.assert_called_with('mistral-large-2407')
    assert isinstance(mistral_profile, BedrockModelProfile)
    assert mistral_profile.bedrock_tool_result_format == 'json'

    meta_profile = provider.model_profile('meta.llama3-8b-instruct-v1:0')
    meta_model_profile_mock.assert_called_with('llama3-8b-instruct')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    cohere_profile = provider.model_profile('cohere.command-text-v14')
    cohere_model_profile_mock.assert_called_with('command-text')
    assert cohere_profile is None

    deepseek_profile = provider.model_profile('deepseek.deepseek-r1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.ignore_streamed_leading_whitespace is True

    amazon_profile = provider.model_profile('us.amazon.nova-pro-v1:0')
    amazon_model_profile_mock.assert_called_with('nova-pro')
    assert isinstance(amazon_profile, BedrockModelProfile)
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert amazon_profile.bedrock_supports_tool_choice is True

    amazon_profile = provider.model_profile('us.amazon.titan-text-express-v1:0')
    amazon_model_profile_mock.assert_called_with('titan-text-express')
    assert amazon_profile is not None
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    unknown_model = provider.model_profile('unknown-model')
    assert unknown_model is None

    unknown_model = provider.model_profile('unknown.unknown-model')
    assert unknown_model is None
