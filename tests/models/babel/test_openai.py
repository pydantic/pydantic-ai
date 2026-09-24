from __future__ import annotations as _annotations

from datetime import datetime, timezone
from typing import Any

import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, AudioUrl, ModelRequest, TextPart, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import FinalResultEvent, PartDeltaEvent, PartEndEvent, PartStartEvent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ...conftest import try_import
from ..mock_openai import MockOpenAI, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from openai import APIStatusError
    from openai.types import chat
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.completion_usage import CompletionUsage

    from pydantic_ai.models.babel.openai import BabelOpenAIChatModel, BabelOpenAIStreamedResponse
    from pydantic_ai.models.openai import OpenAIChatModelSettings
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai or llm-babel not installed'),
    pytest.mark.anyio,
]


def completion(message: dict[str, Any], finish_reason: str) -> chat.ChatCompletion:
    return chat.ChatCompletion.model_validate(
        {
            'id': 'chatcmpl-1',
            'object': 'chat.completion',
            'created': 1704067200,
            'model': 'gpt-4o-123',
            'choices': [{'index': 0, 'finish_reason': finish_reason, 'message': {'role': 'assistant', **message}}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 8, 'total_tokens': 28},
        }
    )


def chunk(
    delta: ChoiceDelta,
    finish_reason: str | None = None,
    usage: bool = True,
    id: str = 'chunk-1',
    model: str = 'gpt-4o-123',
) -> chat.ChatCompletionChunk:
    return chat.ChatCompletionChunk(
        id=id,
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],  # pyright: ignore[reportArgumentType]
        created=1704067200,
        model=model,
        object='chat.completion.chunk',
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3) if usage else None,
    )


def make_model(mock_client: Any, **kwargs: Any) -> BabelOpenAIChatModel:
    return BabelOpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client), **kwargs)


async def test_tool_loop(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(
        [
            completion(
                {
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
                        }
                    ],
                },
                'tool_calls',
            ),
            completion({'content': 'It is sunny in Paris.'}, 'stop'),
        ]
    )
    agent = Agent(make_model(mock_client), system_prompt='You are a weather assistant.', instructions='Be terse.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'{city}: sunny'

    result = await agent.run('What is the weather in Paris?')
    assert result.output == 'It is sunny in Paris.'
    assert (result.usage.requests, result.usage.input_tokens, result.usage.output_tokens) == (2, 40, 16)
    response = result.response
    assert response.parts == [TextPart(content='It is sunny in Paris.')]
    assert response.usage == snapshot(RequestUsage(input_tokens=20, output_tokens=8))
    assert response.model_name == 'gpt-4o-123'
    assert response.provider_name == 'openai'
    assert response.provider_url == 'https://api.openai.com/v1'
    assert response.provider_response_id == 'chatcmpl-1'
    assert response.finish_reason == 'stop'
    assert response.provider_details == snapshot(
        {'finish_reason': 'stop', 'timestamp': datetime(2024, 1, 1, tzinfo=timezone.utc)}
    )
    assert get_mock_chat_completion_kwargs(mock_client)[1]['messages'] == snapshot(
        [
            {'role': 'system', 'content': 'You are a weather assistant.'},
            {'role': 'system', 'content': 'Be terse.'},
            {'role': 'user', 'content': 'What is the weather in Paris?'},
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'get_weather', 'arguments': '{"city":"Paris"}'},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Paris: sunny'},
        ]
    )


async def test_plain_text_response_body(allow_model_requests: None):
    model = make_model(MockOpenAI.create_mock(completion({'content': 'hi'}, 'stop')))
    body = completion({'content': 'from a string body'}, 'stop').model_dump_json()
    response = model._process_response(body)  # pyright: ignore[reportPrivateUsage]
    assert response.parts == [TextPart(content='from a string body')]


async def test_system_prompt_profile(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(completion({'content': 'hi'}, 'stop'))
    profile = OpenAIModelProfile(openai_system_prompt_role='developer')
    agent = Agent(make_model(mock_client, profile=profile), system_prompt='be nice')
    await agent.run('hello')
    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'][0] == {'role': 'developer', 'content': 'be nice'}


async def test_single_system_message_profile(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock([completion({'content': 'hi'}, 'stop')] * 2)
    profile = OpenAIModelProfile(openai_chat_supports_multiple_system_messages=False)
    model = make_model(mock_client, profile=profile)
    agent = Agent(model, system_prompt=['first', 'second'])
    await agent.run('hello')
    messages = get_mock_chat_completion_kwargs(mock_client)[0]['messages']
    assert messages[:2] == snapshot(
        [{'role': 'system', 'content': 'first\n\nsecond'}, {'role': 'user', 'content': 'hello'}]
    )
    # A single leading system message is left alone.
    single = Agent(model, system_prompt='only')
    await single.run('hello')
    assert get_mock_chat_completion_kwargs(mock_client)[1]['messages'][0] == {'role': 'system', 'content': 'only'}


async def test_audio_url_is_downloaded(allow_model_requests: None, mocker: Any):
    mocker.patch(
        'pydantic_ai.models.babel._adapters.download_item',
        return_value={'data': b'bytes', 'data_type': 'audio/mpeg'},
    )
    mock_client = MockOpenAI.create_mock(completion({'content': 'hi'}, 'stop'))
    agent = Agent(make_model(mock_client))
    await agent.run(['listen', AudioUrl(url='https://x/y.mp3')])
    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'][0]['content'] == snapshot(
        [
            {'type': 'text', 'text': 'listen'},
            {'type': 'input_audio', 'input_audio': {'data': 'Ynl0ZXM=', 'format': 'mp3'}},
        ]
    )


async def test_stream_text_and_tool_call(allow_model_requests: None):
    stream = [
        chunk(ChoiceDelta(content='It is ', role='assistant')),
        chunk(ChoiceDelta(content='sunny.'), usage=False),
        chunk(
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id='call_1',
                        type='function',
                        function=ChoiceDeltaToolCallFunction(name='f', arguments=''),
                    )
                ]
            )
        ),
        chunk(
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='{"a": 1}'))]
            )
        ),
        chunk(ChoiceDelta(), finish_reason='tool_calls'),
    ]
    model = make_model(MockOpenAI.create_mock_stream(stream))
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as response:
        assert isinstance(response, BabelOpenAIStreamedResponse)
        events = [event async for event in response]
    assert [type(event) for event in events] == snapshot(
        [
            PartStartEvent,
            FinalResultEvent,
            PartDeltaEvent,
            PartEndEvent,
            PartStartEvent,
            PartDeltaEvent,
            PartEndEvent,
        ]
    )
    assert response.get().parts == snapshot(
        [TextPart(content='It is sunny.'), ToolCallPart(tool_name='f', args='{"a":1}', tool_call_id='call_1')]
    )
    assert response.finish_reason == 'tool_call'
    assert response.provider_response_id == 'chunk-1'
    assert response.model_name == 'gpt-4o-123'
    # Four chunks carried usage, accumulated as the native model does without continuous usage stats.
    assert response.usage == snapshot(RequestUsage(input_tokens=8, output_tokens=4))


async def test_stream_api_error_is_mapped(allow_model_requests: None):
    error = APIStatusError(
        'boom',
        response=httpx2.Response(status_code=500, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'boom'},
    )
    model = make_model(MockOpenAI.create_mock_stream([chunk(ChoiceDelta(content='a', role='assistant')), error]))
    with pytest.raises(ModelHTTPError) as exc_info:
        async with model.request_stream(
            [ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()
        ) as response:
            _ = [event async for event in response]
    assert exc_info.value.status_code == 500
    assert exc_info.value.body == {'error': 'boom'}


async def test_stream_continuous_usage_stats(allow_model_requests: None):
    stream = [
        chunk(ChoiceDelta(content='a', role='assistant')),
        # Azure sends an empty id and model on some chunks; the first values seen stay in place.
        chunk(ChoiceDelta(content='b'), id='', model=''),
        chunk(ChoiceDelta(), finish_reason='stop'),
    ]
    model = make_model(MockOpenAI.create_mock_stream(stream))
    settings = OpenAIChatModelSettings(openai_continuous_usage_stats=True)
    async with model.request_stream(
        [ModelRequest.user_text_prompt('hi')], settings, ModelRequestParameters()
    ) as response:
        _ = [event async for event in response]
    # Each chunk reports the cumulative total, so the last one is the total.
    assert response.usage == snapshot(RequestUsage(input_tokens=2, output_tokens=1))
    assert response.get().parts == [TextPart(content='ab')]
