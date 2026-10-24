from __future__ import annotations as _annotations

from typing import Any, cast

import pytest
from cassetter import Cassette
from inline_snapshot import snapshot
from pydantic import BaseModel, ValidationError

from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, ThinkingPart
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.output import NativeOutput
from pydantic_ai.settings import ModelSettings

from ..cassette_utils import request_json
from ..conftest import IsStr, TestEnv, try_import

with try_import() as imports_successful:
    from openai import omit
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
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

    from .mock_openai import MockOpenAI

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
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

    # A reasoning-capable model that doesn't require `temperature=1` doesn't get a default temperature.
    profile_no_temp = SnowflakeModelProfile(snowflake_supports_reasoning=True)
    transformed_no_temp = _snowflake_settings_to_openai_settings(
        SnowflakeModelSettings(snowflake_reasoning=SnowflakeReasoning(effort='low')), params, profile=profile_no_temp
    )
    assert cast(dict[str, Any], transformed_no_temp.get('extra_body', {})).get('reasoning') == {'effort': 'low'}
    assert 'temperature' not in transformed_no_temp


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


def test_snowflake_translate_thinking(provider: SnowflakeProvider):
    """An explicit `openai_reasoning_effort` is passed through for Claude models, but a unified
    thinking level is not (it flows through the `reasoning` object instead)."""
    from pydantic_ai.models.openai import OpenAIChatModelSettings

    claude = SnowflakeModel('claude-sonnet-4-6', provider=provider)
    params = ModelRequestParameters(thinking='high')
    assert claude._translate_thinking(OpenAIChatModelSettings(openai_reasoning_effort='low'), params) == 'low'  # pyright: ignore[reportPrivateUsage]
    assert claude._translate_thinking(OpenAIChatModelSettings(), params) is omit  # pyright: ignore[reportPrivateUsage]

    # OpenAI models fall back to the base behavior, which maps unified thinking to a reasoning effort.
    gpt = SnowflakeModel('openai-gpt-5.2', provider=provider)
    assert gpt._translate_thinking(OpenAIChatModelSettings(), params) == 'high'  # pyright: ignore[reportPrivateUsage]


def test_snowflake_replays_foreign_thinking_as_tags(provider: SnowflakeProvider):
    """A `ThinkingPart` from a different provider is replayed as tags, not as `reasoning_details`."""
    from pydantic_ai import ModelResponse

    model = SnowflakeModel('claude-sonnet-4-6', provider=provider)
    response = ModelResponse(parts=[ThinkingPart(content='foreign reasoning', provider_name='other', id='xyz')])
    message = model._map_model_response(response)  # pyright: ignore[reportPrivateUsage]
    assert message is not None
    assert 'foreign reasoning' in str(message.get('content'))


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
    # The tool-call response's empty `finish_reason` is coerced based on the presence of tool calls, but isn't
    # reported as Cortex's own.
    tool_call_response = result.all_messages()[1]
    assert isinstance(tool_call_response, ModelResponse)
    assert tool_call_response.finish_reason == 'tool_call'
    assert tool_call_response.provider_details is not None
    assert 'finish_reason' not in tool_call_response.provider_details


@pytest.mark.parametrize(
    ('delta', 'finish_reason'),
    [
        (ChoiceDelta(role='assistant', content='4'), 'stop'),
        (
            ChoiceDelta(
                role='assistant',
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id='call_123',
                        type='function',
                        function=ChoiceDeltaToolCallFunction(name='get_weather', arguments='{}'),
                    )
                ],
            ),
            'tool_call',
        ),
    ],
)
async def test_snowflake_stream_without_finish_reason(
    allow_model_requests: None, provider: SnowflakeProvider, delta: ChoiceDelta, finish_reason: str
):
    """Cortex streams from Claude models have no finish reason, which is then picked as for a complete response."""
    stream = [
        chat.ChatCompletionChunk(
            id='chatcmpl-123',
            choices=[ChunkChoice(index=0, delta=delta)],
            created=1751234567,
            model='claude-sonnet-4-6',
            object='chat.completion.chunk',
        )
    ]
    model = SnowflakeModel(
        'claude-sonnet-4-6', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock_stream(stream))
    )

    async with model.request_stream(
        [ModelRequest.user_text_prompt('What is the weather?')], None, ModelRequestParameters()
    ) as response:
        async for _ in response:
            pass

    assert response.get().finish_reason == finish_reason
    assert 'finish_reason' not in (response.get().provider_details or {})


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

    first_request = request_json(vcr.requests[0])
    assert first_request['reasoning'] == {'effort': 'low'}
    # Cortex applies a non-1 default temperature server-side, which Claude rejects with thinking enabled.
    assert first_request['temperature'] == 1

    # The second request replays the thinking as `reasoning_details` on the assistant message.
    second_request = request_json(vcr.requests[1])
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


async def test_snowflake_tool_output_unsupported_family(allow_model_requests: None, provider: SnowflakeProvider):
    """Cortex errors on `tools` for non-OpenAI/Claude families, so the profile disables tool support.
    Tool-based structured output is then rejected client-side (the default falls back to prompted;
    see `test_snowflake_llama`), before any request is made."""
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.output import ToolOutput

    model = SnowflakeModel('llama4-maverick', provider=provider)
    agent = Agent(model, output_type=ToolOutput(CityInfo))

    with pytest.raises(UserError, match='Tool output is not supported by this model'):
        await agent.run('The capital of Mexico')


async def test_snowflake_openai_model(allow_model_requests: None, live_provider: SnowflakeProvider):
    """OpenAI models return a proper `finish_reason` that passes through the coercion unchanged."""
    model = SnowflakeModel('openai-gpt-4.1', provider=live_provider)
    agent = Agent(model)
    result = await agent.run('What is 2 + 2? Reply with just the number.')
    assert result.output == snapshot('4')
    assert result.all_messages()[-1].finish_reason == 'stop'  # type: ignore[union-attr]


@pytest.mark.parametrize('raw_finish_reason', ['', None])
@pytest.mark.parametrize(
    ('message', 'finish_reason'),
    [
        (chat.ChatCompletionMessage(role='assistant', content='4'), 'stop'),
        (
            chat.ChatCompletionMessage(
                role='assistant',
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id='call_123', type='function', function=Function(name='get_weather', arguments='{}')
                    )
                ],
            ),
            'tool_call',
        ),
    ],
)
async def test_snowflake_response_without_finish_reason(
    allow_model_requests: None,
    raw_finish_reason: str | None,
    message: chat.ChatCompletionMessage,
    finish_reason: str,
):
    """Cortex returns an empty `finish_reason` for Claude models, which is then picked as for a stream without one.

    A `null` one is treated the same. Neither is in a recorded response: the OpenAI SDK parses responses leniently,
    so the strict validation only happens in our `_validate_completion` hook, and Cortex hasn't been seen to send `null`.
    """
    completion = chat.ChatCompletion.model_construct(
        id='chatcmpl-123',
        choices=[Choice.model_construct(finish_reason=raw_finish_reason, index=0, message=message)],
        created=1751234567,
        model='claude-sonnet-4-6',
        object='chat.completion',
    )
    model = SnowflakeModel(
        'claude-sonnet-4-6', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock(completion))
    )

    response = await model.request(
        [ModelRequest.user_text_prompt('What is the weather?')], None, ModelRequestParameters()
    )

    assert response.finish_reason == finish_reason
    assert 'finish_reason' not in (response.provider_details or {})


_EMPTY_MODERATION_STUB: dict[str, Any] = {
    'input': {'type': '', 'results': None, 'model': 'openai-gpt-5.2'},
    'output': {'type': '', 'results': None, 'model': 'openai-gpt-5.2'},
}


async def test_snowflake_response_with_empty_moderation_stub(allow_model_requests: None):
    """Cortex's empty moderation placeholder must not fail strict validation.

    Existing recordings omit moderation, so construct the reported stub as a leniently parsed SDK response.
    """
    completion = chat.ChatCompletion.model_construct(
        id='chatcmpl-123',
        choices=[
            Choice.model_construct(
                finish_reason='stop',
                index=0,
                message=chat.ChatCompletionMessage(role='assistant', content='hello'),
            )
        ],
        created=1751234567,
        model='openai-gpt-5.2',
        object='chat.completion',
        moderation=_EMPTY_MODERATION_STUB,
    )
    model = SnowflakeModel(
        'openai-gpt-5.2', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock(completion))
    )

    response = await model.request([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters())

    assert response.parts[0].content == 'hello'  # type: ignore[union-attr]
    assert 'moderation' not in (response.provider_details or {})


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('extra_stub', [False, True])
async def test_snowflake_response_keeps_valid_moderation(allow_model_requests: None, stream: bool, extra_stub: bool):
    """Real moderation results must still reach `provider_details`, including when extra keys contain placeholders.

    Existing recordings omit moderation, so use constructed SDK responses to cover these valid combinations.
    """
    moderation: dict[str, Any] = {
        'input': {'type': 'moderation_results', 'model': 'omni-moderation-latest', 'results': []},
        'output': {'type': 'moderation_results', 'model': 'omni-moderation-latest', 'results': []},
    }
    if extra_stub:
        moderation['extra'] = _EMPTY_MODERATION_STUB['input']

    if stream:
        chunk = chat.ChatCompletionChunk.model_construct(
            id='chatcmpl-123',
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role='assistant', content='hello'), finish_reason='stop')],
            created=1751234567,
            model='openai-gpt-5.2',
            object='chat.completion.chunk',
            moderation=moderation,
        )
        model = SnowflakeModel(
            'openai-gpt-5.2', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock_stream([chunk]))
        )
        async with model.request_stream(
            [ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()
        ) as response:
            async for _ in response:
                pass
        result = response.get()
    else:
        completion = chat.ChatCompletion.model_construct(
            id='chatcmpl-123',
            choices=[
                Choice.model_construct(
                    finish_reason='stop',
                    index=0,
                    message=chat.ChatCompletionMessage(role='assistant', content='hello'),
                )
            ],
            created=1751234567,
            model='openai-gpt-5.2',
            object='chat.completion',
            moderation=moderation,
        )
        model = SnowflakeModel(
            'openai-gpt-5.2', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock(completion))
        )
        result = await model.request([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters())

    assert (result.provider_details or {}).get('moderation') == moderation


async def test_snowflake_response_still_rejects_other_malformed_fields(allow_model_requests: None):
    """Dropping the moderation stub must not hide malformed response fields.

    The bogus `finish_reason` is synthesized, so use a constructed SDK response rather than a recording.
    """

    completion = chat.ChatCompletion.model_construct(
        id='chatcmpl-123',
        choices=[
            Choice.model_construct(
                finish_reason='bogus', index=0, message=chat.ChatCompletionMessage(role='assistant', content='x')
            )
        ],
        created=1751234567,
        model='openai-gpt-5.2',
        object='chat.completion',
        moderation=_EMPTY_MODERATION_STUB,
    )
    model = SnowflakeModel(
        'openai-gpt-5.2', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock(completion))
    )

    with pytest.raises(UnexpectedModelBehavior):
        await model.request([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters())


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(
    'moderation',
    [
        pytest.param(
            {'input': {}, 'output': {'type': 'moderation_results', 'model': 'm', 'results': []}},
            id='missing-fields',
        ),
        pytest.param(
            {
                'input': _EMPTY_MODERATION_STUB['input'],
                'output': {'type': 'moderation_results', 'model': 'm', 'results': []},
            },
            id='stub-input-valid-output',
        ),
        pytest.param(
            {
                'input': {'type': 'moderation_results', 'model': 'm', 'results': []},
                'output': _EMPTY_MODERATION_STUB['output'],
            },
            id='valid-input-stub-output',
        ),
        pytest.param({'input': _EMPTY_MODERATION_STUB['input'], 'output': {}}, id='stub-input-malformed-output'),
        pytest.param({'input': {}, 'output': _EMPTY_MODERATION_STUB['output']}, id='malformed-input-stub-output'),
        pytest.param({'input': _EMPTY_MODERATION_STUB['input'], 'output': 'invalid'}, id='non-dict-output'),
        pytest.param(
            {'input': _EMPTY_MODERATION_STUB['input'], 'output': {'type': '', 'results': []}}, id='non-null-results'
        ),
        pytest.param({'input': _EMPTY_MODERATION_STUB['input']}, id='missing-output'),
    ],
)
async def test_snowflake_response_rejects_malformed_moderation_that_is_not_the_stub(
    allow_model_requests: None, stream: bool, moderation: dict[str, Any]
):
    """Only the complete Cortex placeholder is dropped; other malformed moderation is still rejected.

    Existing recordings omit moderation, so construct synthetic malformed variants to pin strict validation.
    """
    if stream:
        chunk = chat.ChatCompletionChunk.model_construct(
            id='chatcmpl-123',
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role='assistant', content='x'), finish_reason='stop')],
            created=1751234567,
            model='openai-gpt-5.2',
            object='chat.completion.chunk',
            moderation=moderation,
        )
        model = SnowflakeModel(
            'openai-gpt-5.2', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock_stream([chunk]))
        )
        with pytest.raises(ValidationError, match='moderation'):
            async with model.request_stream(
                [ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()
            ) as response:
                async for _ in response:
                    pass
    else:
        completion = chat.ChatCompletion.model_construct(
            id='chatcmpl-123',
            choices=[
                Choice.model_construct(
                    finish_reason='stop', index=0, message=chat.ChatCompletionMessage(role='assistant', content='x')
                )
            ],
            created=1751234567,
            model='openai-gpt-5.2',
            object='chat.completion',
            moderation=moderation,
        )
        model = SnowflakeModel(
            'openai-gpt-5.2', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock(completion))
        )
        with pytest.raises(UnexpectedModelBehavior, match='moderation'):
            await model.request([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters())


async def test_snowflake_stream_with_empty_moderation_stub(allow_model_requests: None):
    """Cortex's empty moderation placeholder must not fail streamed validation.

    Existing recordings omit moderation, so construct a chunk containing the reported stub.
    """

    stream = [
        chat.ChatCompletionChunk.model_construct(
            id='chatcmpl-123',
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role='assistant', content='hello'), finish_reason='stop')],
            created=1751234567,
            model='openai-gpt-5.2',
            object='chat.completion.chunk',
            moderation=_EMPTY_MODERATION_STUB,
        )
    ]
    model = SnowflakeModel(
        'openai-gpt-5.2', provider=SnowflakeProvider(openai_client=MockOpenAI.create_mock_stream(stream))
    )

    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as response:
        async for _ in response:
            pass

    assert response.get().parts[0].content == 'hello'  # type: ignore[union-attr]
    assert 'moderation' not in (response.get().provider_details or {})
