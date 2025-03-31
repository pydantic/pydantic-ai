from typing import cast

import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from pydantic_ai.providers.bedrock import BedrockProvider


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
