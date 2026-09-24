import re

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_deep_seek_provider():
    provider = DeepSeekProvider(api_key='api-key')
    assert provider.name == 'deepseek'
    assert provider.base_url == 'https://api.deepseek.com'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_deep_seek_provider_need_api_key(env: TestEnv) -> None:
    env.remove('DEEPSEEK_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `DEEPSEEK_API_KEY` environment variable or pass it via `DeepSeekProvider(api_key=...)`'
            ' to use the DeepSeek provider.'
        ),
    ):
        DeepSeekProvider()


def test_deep_seek_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = DeepSeekProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_deep_seek_model_profile():
    provider = DeepSeekProvider(api_key='api-key')
    model = OpenAIChatModel('deepseek-r1', provider=provider)
    assert model.profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert model.profile.get('supports_thinking', False) is True
    assert model.profile.get('thinking_always_enabled', False) is True


# 'deepseek-v4-turbo' stands in for an unreleased SKU: the fact is set for every DeepSeek model, so a
# new alias cannot silently miss the grouping fix.
@pytest.mark.parametrize(
    'model_name', ['deepseek-chat', 'deepseek-reasoner', 'deepseek-v4-flash', 'deepseek-v4-pro', 'deepseek-v4-turbo']
)
def test_deep_seek_responses_function_call_grouping_profile(model_name: str) -> None:
    model = OpenAIResponsesModel(model_name, provider=DeepSeekProvider(api_key='api-key'))
    assert model.profile.get('openai_responses_supports_interleaved_function_calls', True) is False


def test_openai_responses_function_call_grouping_profile_defaults_on() -> None:
    openai_model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key='api-key'))
    assert openai_model.profile.get('openai_responses_supports_interleaved_function_calls', True) is True
    default_model = OpenAIResponsesModel(
        'custom-model', provider=OpenAIProvider(api_key='api-key'), profile=OpenAIModelProfile()
    )
    assert default_model.profile.get('openai_responses_supports_interleaved_function_calls', True) is True


@pytest.mark.parametrize('model_name', ['deepseek-v4-flash', 'deepseek-v4-pro'])
def test_deep_seek_v4_model_profile(model_name: str):
    provider = DeepSeekProvider(api_key='api-key')
    profile = provider.model_profile(model_name)
    assert profile is not None
    assert isinstance(profile, dict)
    assert profile.get('supports_thinking', False) is True
    assert profile.get('thinking_always_enabled', False) is False
    # V4 can turn thinking off, so forcing is restricted per request rather than outright.
    assert profile.get('openai_supports_tool_choice_required', True) is True
    assert profile.get('openai_supports_forced_tool_choice_with_thinking', True) is False
    assert profile.get('openai_reasoning_enabled_by_default', False) is True
    assert profile.get('openai_responses_supports_json_schema_output', False) is True


def test_deep_seek_chat_model_profile():
    provider = DeepSeekProvider(api_key='api-key')
    profile = provider.model_profile('deepseek-chat')
    assert profile is not None
    assert isinstance(profile, dict)
    assert profile.get('supports_thinking', False) is False
    # `deepseek-chat` is pinned to non-thinking mode, so forcing is never restricted.
    assert profile.get('openai_supports_tool_choice_required', True) is True
    assert profile.get('openai_supports_forced_tool_choice_with_thinking', True) is True
    assert profile.get('openai_reasoning_enabled_by_default', False) is False


def test_deep_seek_r1_model_profile():
    """Regression anchor: deepseek-r1 must always have thinking enabled."""
    provider = DeepSeekProvider(api_key='api-key')
    profile = provider.model_profile('deepseek-r1')
    assert profile is not None
    assert isinstance(profile, dict)
    assert profile.get('supports_thinking', False) is True
    assert profile.get('thinking_always_enabled', False) is True


def test_deep_seek_reasoner_model_profile():
    provider = DeepSeekProvider(api_key='api-key')
    profile = provider.model_profile('deepseek-reasoner')
    assert profile is not None
    assert isinstance(profile, dict)
    assert profile.get('supports_thinking', False) is True
    assert profile.get('thinking_always_enabled', False) is True
    # `deepseek-reasoner` cannot turn thinking off, so its restriction stays unconditional.
    assert profile.get('openai_supports_tool_choice_required', True) is False
    assert profile.get('openai_reasoning_enabled_by_default', False) is True


def test_deep_seek_v4_future_sku_inherits_tool_choice_restriction():
    """Future deepseek-v4-* SKUs must inherit the thinking-conditional restriction via the startswith predicate."""
    provider = DeepSeekProvider(api_key='api-key')
    profile = provider.model_profile('deepseek-v4-turbo')
    assert profile is not None
    assert isinstance(profile, dict)
    assert profile.get('openai_supports_forced_tool_choice_with_thinking', True) is False
    assert profile.get('openai_reasoning_enabled_by_default', False) is True
