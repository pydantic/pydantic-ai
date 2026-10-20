import re

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.meta import MetaProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_meta_provider():
    provider = MetaProvider(api_key='api-key')
    assert provider.name == 'meta'
    assert provider.base_url == 'https://api.meta.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_meta_provider_need_api_key(env: TestEnv) -> None:
    env.remove('META_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `META_API_KEY` environment variable or pass it via `MetaProvider(api_key=...)`'
            ' to use the Meta provider.'
        ),
    ):
        MetaProvider()


def test_meta_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = MetaProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_meta_model_profile():
    # Unit (not VCR): these pin the profile flags resolved from the model id. Cassette matchers aren't
    # body-sensitive, so a regression flipping a flag could still replay an existing recording green.
    for model_name in ('muse-spark-1.1', 'muse-spark-1.3-contributor', 'muse-wonder-1.0'):
        profile = MetaProvider.model_profile(model_name)
        assert profile is not None
        assert profile.get('thinking_always_enabled') is True
        assert profile.get('openai_supports_phase') is True
        assert profile.get('openai_supports_encrypted_reasoning_content') is True
        assert profile.get('supports_forced_tool_choice') is False
        # Sampling params stay: `openai_supports_reasoning` would drop them while reasoning.
        assert profile.get('openai_supports_reasoning', False) is False

    # Image and transcription models don't get the text-model reasoning profile.
    profile = MetaProvider.model_profile('muse-image-1.0')
    assert profile is not None
    assert profile.get('thinking_always_enabled', False) is False
    assert profile.get('supports_forced_tool_choice') is False
