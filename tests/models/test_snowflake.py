from __future__ import annotations as _annotations

import json
from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.messages import ThinkingPart
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.output import NativeOutput
from pydantic_ai.settings import ModelSettings

from ..conftest import IsStr, TestEnv, try_import

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
    from pydantic_ai.providers.snowflake import SnowflakeModelProfile, SnowflakeProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture
def provider() -> SnowflakeProvider:
    return SnowflakeProvider(account='myorg-myaccount', token='pat')


@pytest.fixture
def live_provider(snowflake_account: str, snowflake_token: str) -> SnowflakeProvider:
    return SnowflakeProvider(account=snowflake_account, token=snowflake_token)


class CityInfo(BaseModel):
    city: str
    country: str


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
    claude_profile = SnowflakeModelProfile(
        snowflake_supports_reasoning=True, snowflake_reasoning_requires_temperature_1=True
    )

    # An explicit `snowflake_reasoning` is moved into `extra_body['reasoning']`, and `temperature`
    # defaults to 1 since Cortex's server-side default temperature is rejected with thinking enabled.
    settings = SnowflakeModelSettings(snowflake_reasoning=SnowflakeReasoning(max_tokens=1024))
    transformed = _snowflake_settings_to_openai_settings(settings, params, profile=claude_profile)
    assert cast(dict[str, Any], transformed.get('extra_body', {})).get('reasoning') == {'max_tokens': 1024}
    assert transformed.get('temperature') == 1

    # An explicit `temperature` wins over the reasoning default.
    settings_temp = SnowflakeModelSettings(snowflake_reasoning=SnowflakeReasoning(max_tokens=1024), temperature=0.5)
    transformed_temp = _snowflake_settings_to_openai_settings(settings_temp, params, profile=claude_profile)
    assert transformed_temp.get('temperature') == 0.5

    # An empty settings object stays empty.
    transformed_empty = _snowflake_settings_to_openai_settings(SnowflakeModelSettings(), params, profile=claude_profile)
    assert transformed_empty.get('extra_body') is None

    # Unified thinking maps to a reasoning effort for Claude models...
    params_thinking = ModelRequestParameters(thinking='high')
    transformed_thinking = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(), params_thinking, profile=claude_profile
    )
    assert cast(dict[str, Any], transformed_thinking.get('extra_body', {})).get('reasoning') == {'effort': 'high'}

    # ...but not for other models, which don't take the `reasoning` object.
    transformed_other = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(), params_thinking, profile=SnowflakeModelProfile()
    )
    assert transformed_other.get('extra_body') is None

    # An explicit `snowflake_reasoning` wins over unified thinking.
    transformed_both = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(snowflake_reasoning=SnowflakeReasoning(effort='low')),
        params_thinking,
        profile=claude_profile,
    )
    assert cast(dict[str, Any], transformed_both.get('extra_body', {})).get('reasoning') == {'effort': 'low'}

    # `thinking=False` does not enable reasoning.
    params_no_thinking = ModelRequestParameters(thinking=False)
    transformed_disabled = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(), params_no_thinking, profile=claude_profile
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


async def test_snowflake_model_simple(allow_model_requests: None, live_provider: SnowflakeProvider):
    model = SnowflakeModel('claude-sonnet-4-6', provider=live_provider)
    agent = Agent(model)
    result = await agent.run('What is 2 + 2? Reply with just the number.')
    assert result.output == snapshot('4')
    # Cortex returns an empty `finish_reason` for Claude models, which we coerce.
    assert result.all_messages()[-1].finish_reason == 'stop'  # type: ignore[union-attr]


async def test_snowflake_model_streaming(allow_model_requests: None, live_provider: SnowflakeProvider):
    model = SnowflakeModel('claude-sonnet-4-6', provider=live_provider)
    agent = Agent(model)
    async with agent.run_stream('What is 2 + 2? Reply with just the number.') as result:
        output = await result.get_output()
    assert output == snapshot('4')
    assert result.usage.total_tokens is not None


async def test_snowflake_tool_calling(allow_model_requests: None, live_provider: SnowflakeProvider):
    model = SnowflakeModel('claude-sonnet-4-6', provider=live_provider)
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return 'Sunny, 25°C'

    result = await agent.run('What is the weather in Mexico City? Reply with a short sentence.')
    assert result.output == snapshot(
        'The weather in Mexico City is currently sunny with a pleasant temperature of 25°C.'
    )
    # The tool-call response's empty `finish_reason` is coerced based on the presence of tool calls.
    tool_call_response = result.all_messages()[1]
    assert tool_call_response.finish_reason == 'tool_call'  # type: ignore[union-attr]


async def test_snowflake_native_output(allow_model_requests: None, live_provider: SnowflakeProvider):
    model = SnowflakeModel('claude-sonnet-4-6', provider=live_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))
    result = await agent.run('The capital of Mexico')
    assert result.output == snapshot(CityInfo(city='Mexico City', country='Mexico'))


async def test_snowflake_thinking(allow_model_requests: None, live_provider: SnowflakeProvider, vcr: Cassette):
    """Unified thinking maps to the `reasoning` object, thinking comes back as `reasoning_details`
    with a signature, and prior thinking is replayed as `reasoning_details` on the next request."""
    model = SnowflakeModel('claude-sonnet-4-6', provider=live_provider)
    agent = Agent(model, model_settings=ModelSettings(thinking='low'))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return 'Sunny, 25°C'

    result = await agent.run('What is the weather in Mexico City? Reply with a short sentence.')
    assert result.output == snapshot("It's currently sunny and 25°C in Mexico City! ☀️")

    thinking_parts = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ThinkingPart)
    ]
    assert thinking_parts
    assert thinking_parts[0].provider_name == 'snowflake'
    assert thinking_parts[0].provider_details == snapshot(
        {'format': 'anthropic-claude-v1', 'index': 0, 'type': 'reasoning.text'}
    )

    first_request = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert first_request['reasoning'] == {'effort': 'low'}
    # Cortex applies a non-1 default temperature server-side, which Claude rejects with thinking enabled.
    assert first_request['temperature'] == 1

    # The second request replays the thinking as `reasoning_details` on the assistant message.
    second_request = json.loads(vcr.requests[1].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assistant_message = next(m for m in second_request['messages'] if m['role'] == 'assistant')
    assert assistant_message['reasoning_details'][0]['type'] == 'reasoning.text'
    assert assistant_message['reasoning_details'][0]['signature']


async def test_snowflake_thinking_streaming(allow_model_requests: None, live_provider: SnowflakeProvider):
    """Streaming delivers thinking via `delta.reasoning_details`."""
    model = SnowflakeModel('claude-sonnet-4-6', provider=live_provider)
    agent = Agent(model, model_settings=ModelSettings(thinking='low'))
    async with agent.run_stream('What is 15 * 27?') as result:
        output = await result.get_output()
    assert output == snapshot("""\
15 × 27 = **405**

Here's the breakdown:
- 15 × 20 = 300
- 15 × 7 = 105
- 300 + 105 = **405**\
""")

    messages = result.all_messages()
    thinking_parts = [part for message in messages for part in message.parts if isinstance(part, ThinkingPart)]
    assert thinking_parts
    assert thinking_parts[0].content == IsStr()


async def test_snowflake_llama(allow_model_requests: None, live_provider: SnowflakeProvider):
    """Non-OpenAI/Claude families don't support tools or `response_format`, so structured output
    falls back to prompted mode."""
    model = SnowflakeModel('llama3.1-8b', provider=live_provider)
    agent = Agent(model, output_type=CityInfo)
    result = await agent.run('The capital of Mexico')
    assert result.output == snapshot(CityInfo(city='Mexico City', country='Mexico'))


async def test_snowflake_openai_model(allow_model_requests: None, live_provider: SnowflakeProvider):
    """OpenAI models return a proper `finish_reason` that passes through the coercion unchanged."""
    model = SnowflakeModel('openai-gpt-4.1', provider=live_provider)
    agent = Agent(model)
    result = await agent.run('What is 2 + 2? Reply with just the number.')
    assert result.output == snapshot('4')
    assert result.all_messages()[-1].finish_reason == 'stop'  # type: ignore[union-attr]


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
