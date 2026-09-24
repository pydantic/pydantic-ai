from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.messages import FinalResultEvent, PartDeltaEvent, PartEndEvent, PartStartEvent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ...conftest import IsStr, try_import

with try_import() as imports_successful:
    from google.genai.types import GenerateContentResponse

    from pydantic_ai.models.babel.google import BabelGeminiStreamedResponse, BabelGoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai or llm-babel not installed'),
    pytest.mark.anyio,
]


def response(
    parts: list[dict[str, Any]],
    finish_reason: str = 'STOP',
    usage: bool = True,
    response_id: str | None = 'resp_1',
    create_time: datetime | None = None,
) -> GenerateContentResponse:
    data: dict[str, Any] = {
        'candidates': [{'content': {'role': 'model', 'parts': parts}, 'finish_reason': finish_reason, 'index': 0}],
        'response_id': response_id,
        'model_version': 'gemini-2.5-flash-001',
        'create_time': create_time,
    }
    if usage:
        data['usage_metadata'] = {
            'prompt_token_count': 12,
            'candidates_token_count': 5,
            'cached_content_token_count': 4,
            'total_token_count': 17,
        }
    return GenerateContentResponse.model_validate(data)


def make_model() -> BabelGoogleModel:
    return BabelGoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))


async def test_tool_loop(allow_model_requests: None, mocker: Any):
    model = make_model()
    generate = mocker.patch.object(
        model.client.aio.models,
        'generate_content',
        side_effect=[
            response(
                [
                    {'text': 'thinking', 'thought': True, 'thought_signature': b'SIG'},
                    {'function_call': {'name': 'get_weather', 'args': {'city': 'Paris'}}},
                ]
            ),
            response([{'text': 'It is sunny in Paris.'}]),
        ],
    )
    agent = Agent(model, system_prompt='You are a weather assistant.', instructions='Be terse.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'{city}: sunny'

    result = await agent.run('What is the weather in Paris?')
    assert result.output == 'It is sunny in Paris.'
    model_response = result.all_messages()[1]
    assert isinstance(model_response, ModelResponse)
    assert model_response.parts == snapshot(
        [
            ThinkingPart(content='thinking', signature='U0lH', provider_name='google'),
            ToolCallPart(tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='call_0'),
        ]
    )
    assert model_response.usage == snapshot(RequestUsage(input_tokens=12, cache_read_tokens=4, output_tokens=5))
    assert model_response.model_name == 'gemini-2.5-flash-001'
    assert model_response.provider_name == 'google'
    assert model_response.provider_url == 'https://generativelanguage.googleapis.com/'
    assert model_response.provider_response_id == 'resp_1'
    # Gemini reports `STOP` for a function-calling turn too, which is `stop` on the native model as well.
    assert model_response.finish_reason == 'stop'
    second_call = generate.call_args_list[1].kwargs
    assert second_call['config']['system_instruction'] == snapshot(
        {'parts': [{'text': 'You are a weather assistant.'}, {'text': 'Be terse.'}]}
    )
    assert second_call['contents'] == snapshot(
        [
            {'role': 'user', 'parts': [{'text': 'What is the weather in Paris?'}]},
            {
                'role': 'model',
                'parts': [
                    {'text': 'thinking', 'thought': True, 'thought_signature': 'U0lH'},
                    {'function_call': {'name': 'get_weather', 'args': {'city': 'Paris'}}},
                ],
            },
            {
                'role': 'user',
                'parts': [{'function_response': {'name': 'get_weather', 'response': {'return_value': 'Paris: sunny'}}}],
            },
        ]
    )


async def test_no_system_instruction(allow_model_requests: None, mocker: Any):
    model = make_model()
    generate = mocker.patch.object(model.client.aio.models, 'generate_content', return_value=response([{'text': 'hi'}]))
    await Agent(model).run('hello')
    assert generate.call_args.kwargs['config'].get('system_instruction') is None


async def _chunks(chunks: list[GenerateContentResponse]) -> AsyncIterator[GenerateContentResponse]:
    for chunk in chunks:
        yield chunk


async def test_stream(allow_model_requests: None, mocker: Any):
    model = make_model()
    mocker.patch.object(
        model.client.aio.models,
        'generate_content_stream',
        return_value=_chunks(
            [
                response([{'text': 'It is '}], usage=False, create_time=datetime(2024, 1, 1, tzinfo=timezone.utc)),
                response([{'text': 'sunny.'}], usage=False, response_id=None),
                response([{'function_call': {'name': 'f', 'args': {'a': 1}}}]),
            ]
        ),
    )
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as streamed:
        assert isinstance(streamed, BabelGeminiStreamedResponse)
        events = [event async for event in streamed]
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
    assert streamed.get().parts == snapshot(
        [TextPart(content='It is sunny.'), ToolCallPart(tool_name='f', args='{"a":1}', tool_call_id=IsStr())]
    )
    assert streamed.provider_response_id == 'resp_1'
    assert streamed.provider_details == {'timestamp': datetime(2024, 1, 1, tzinfo=timezone.utc)}
    assert streamed.finish_reason == 'stop'
    assert streamed.usage == snapshot(RequestUsage(input_tokens=12, cache_read_tokens=4, output_tokens=5))


async def test_stream_without_create_time(allow_model_requests: None, mocker: Any):
    model = make_model()
    mocker.patch.object(
        model.client.aio.models,
        'generate_content_stream',
        return_value=_chunks([response([{'text': 'hi'}])]),
    )
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as streamed:
        _ = [event async for event in streamed]
    assert streamed.provider_details is None
    assert streamed.get().parts == [TextPart(content='hi')]
