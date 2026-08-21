import re

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.spark import SparkProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_spark_provider():
    """Test basic iFLYTEK Spark provider initialization."""
    provider = SparkProvider(api_key='api-key')
    assert provider.name == 'spark'
    assert provider.base_url == 'https://spark-api-open.xf-yun.com/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_spark_provider_need_api_key(env: TestEnv) -> None:
    """Test that the iFLYTEK Spark provider requires an API key."""
    env.remove('SPARK_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `SPARK_API_KEY` environment variable or pass it via `SparkProvider(api_key=...)`'
            ' to use the iFLYTEK Spark provider.'
        ),
    ):
        SparkProvider()


def test_spark_pass_openai_client() -> None:
    """Test passing a custom OpenAI client to the iFLYTEK Spark provider."""
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = SparkProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_spark_model_profile():
    provider = SparkProvider(api_key='api-key')
    model = OpenAIChatModel('4.0Ultra', provider=provider)
    assert isinstance(model.profile, dict)
    assert model.profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert model.profile.get('supports_json_object_output', False) is True
