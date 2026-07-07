from __future__ import annotations as _annotations

from typing import Any, cast

import pytest

from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.settings import ModelSettings

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )

    from pydantic_ai.models.snowflake import (
        SnowflakeModel,
        SnowflakeModelSettings,
        SnowflakeReasoning,
        _snowflake_settings_to_openai_settings,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.snowflake import SnowflakeProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


@pytest.fixture
def provider() -> SnowflakeProvider:
    return SnowflakeProvider(account='myorg-myaccount', token='pat')


def test_snowflake_model(provider: SnowflakeProvider):
    model = SnowflakeModel('claude-sonnet-4-6', provider=provider)
    assert model.system == 'snowflake'
    assert model.model_name == 'claude-sonnet-4-6'
    assert model.base_url == 'https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/v1/'


def test_infer_snowflake_model(env: TestEnv):
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-myaccount')
    env.set('SNOWFLAKE_TOKEN', 'pat')
    model = infer_model('snowflake:llama4-maverick')
    assert isinstance(model, SnowflakeModel)
    assert model.system == 'snowflake'
    assert model.model_name == 'llama4-maverick'


async def test_snowflake_settings_transformation():
    """`SnowflakeModelSettings` are transformed to `OpenAIChatModelSettings` with `reasoning` in `extra_body`."""
    params = ModelRequestParameters()

    # An explicit `snowflake_reasoning` is moved into `extra_body['reasoning']`.
    settings = SnowflakeModelSettings(snowflake_reasoning=SnowflakeReasoning(max_tokens=1024))
    transformed = _snowflake_settings_to_openai_settings(settings, params, is_claude=True)
    assert cast(dict[str, Any], transformed.get('extra_body', {})).get('reasoning') == {'max_tokens': 1024}

    # An empty settings object stays empty.
    transformed_empty = _snowflake_settings_to_openai_settings(SnowflakeModelSettings(), params, is_claude=True)
    assert transformed_empty.get('extra_body') is None

    # Unified thinking maps to a reasoning effort for Claude models...
    params_thinking = ModelRequestParameters(thinking='high')
    transformed_thinking = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(), params_thinking, is_claude=True
    )
    assert cast(dict[str, Any], transformed_thinking.get('extra_body', {})).get('reasoning') == {'effort': 'high'}

    # ...but not for other models, which don't take the `reasoning` object.
    transformed_other = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(), params_thinking, is_claude=False
    )
    assert transformed_other.get('extra_body') is None

    # An explicit `snowflake_reasoning` wins over unified thinking.
    transformed_both = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(snowflake_reasoning=SnowflakeReasoning(effort='low')), params_thinking, is_claude=True
    )
    assert cast(dict[str, Any], transformed_both.get('extra_body', {})).get('reasoning') == {'effort': 'low'}

    # `thinking=False` does not enable reasoning.
    params_no_thinking = ModelRequestParameters(thinking=False)
    transformed_disabled = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(), params_no_thinking, is_claude=True
    )
    assert transformed_disabled.get('extra_body') is None


async def test_snowflake_unified_thinking(provider: SnowflakeProvider):
    """The unified `thinking` setting only flows through to the `reasoning` object for Claude models."""
    params = ModelRequestParameters()

    claude = SnowflakeModel('claude-sonnet-4-6', provider=provider)
    claude_settings, _ = claude.prepare_request(ModelSettings(thinking='medium'), params)
    assert claude_settings is not None
    assert cast(dict[str, Any], claude_settings.get('extra_body', {})).get('reasoning') == {'effort': 'medium'}

    # Llama models don't support thinking on Cortex, so the setting is dropped at the profile gate.
    llama = SnowflakeModel('llama4-maverick', provider=provider)
    llama_settings, _ = llama.prepare_request(ModelSettings(thinking='medium'), params)
    assert llama_settings is not None
    assert llama_settings.get('extra_body') is None


def test_snowflake_validate_completion_coerces_empty_finish_reason(provider: SnowflakeProvider):
    """Cortex returns an empty `finish_reason` for Claude models, which must not fail response validation.

    This can't be reached through a recorded request because the OpenAI SDK parses responses
    leniently; the strict validation only happens in our `_validate_completion` hook.
    """
    model = SnowflakeModel('claude-sonnet-4-6', provider=provider)

    def completion(message: chat.ChatCompletionMessage) -> chat.ChatCompletion:
        return chat.ChatCompletion.model_construct(
            id='chatcmpl-123',
            choices=[Choice.model_construct(finish_reason='', index=0, message=message)],
            created=1751234567,
            model='claude-sonnet-4-6',
            object='chat.completion',
        )

    text_response = completion(chat.ChatCompletionMessage(role='assistant', content='4'))
    validated = model._validate_completion(text_response)  # pyright: ignore[reportPrivateUsage]
    assert validated.choices[0].finish_reason == 'stop'

    tool_call_response = completion(
        chat.ChatCompletionMessage(
            role='assistant',
            tool_calls=[
                ChatCompletionMessageFunctionToolCall(
                    id='call_123', type='function', function=Function(name='get_weather', arguments='{}')
                )
            ],
        )
    )
    validated_tool = model._validate_completion(tool_call_response)  # pyright: ignore[reportPrivateUsage]
    assert validated_tool.choices[0].finish_reason == 'tool_calls'
