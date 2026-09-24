from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import FinalResultEvent, PartDeltaEvent, PartEndEvent, PartStartEvent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ...conftest import IsStr, try_import

with try_import() as imports_successful:
    from google.genai import errors
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
    cached: bool = True,
    response_id: str | None = 'resp_1',
    create_time: datetime | None = None,
    candidate: dict[str, Any] | None = None,
    traffic_type: str | None = None,
    **extra: Any,
) -> GenerateContentResponse:
    data: dict[str, Any] = {
        'candidates': [
            {
                'content': {'role': 'model', 'parts': parts},
                'finish_reason': finish_reason,
                'index': 0,
                **(candidate or {}),
            }
        ],
        'response_id': response_id,
        'model_version': 'gemini-2.5-flash-001',
        'create_time': create_time,
        **extra,
    }
    if usage:
        data['usage_metadata'] = {
            'prompt_token_count': 12,
            'candidates_token_count': 5,
            'total_token_count': 17,
            **({'cached_content_token_count': 4} if cached else {}),
            **({'traffic_type': traffic_type} if traffic_type else {}),
        }
    return GenerateContentResponse.model_validate(data)


def safety_ratings(details: dict[str, Any] | None) -> list[tuple[str, str, bool | None]]:
    """The (category, probability, blocked) of each recorded rating; the dumps carry SDK enums and unset fields."""
    assert details is not None
    return [(r['category'], r['probability'], r.get('blocked')) for r in details.pop('safety_ratings')]


BLOCKED_PROMPT = {
    'block_reason': 'PROHIBITED_CONTENT',
    'block_reason_message': 'The prompt was blocked.',
    'safety_ratings': [{'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'HIGH', 'blocked': True}],
}
SAFETY_RATING = {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'LOW'}


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
            ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id='call_0'),
        ]
    )
    assert model_response.usage == snapshot(
        RequestUsage(details={'cached_content_tokens': 4}, input_tokens=12, cache_read_tokens=4, output_tokens=5)
    )
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
    assert streamed.provider_details == {
        'timestamp': datetime(2024, 1, 1, tzinfo=timezone.utc),
        'finish_reason': 'STOP',
    }
    assert streamed.finish_reason == 'stop'
    assert streamed.usage == snapshot(
        RequestUsage(details={'cached_content_tokens': 4}, input_tokens=12, cache_read_tokens=4, output_tokens=5)
    )


async def test_stream_keeps_usage_a_later_chunk_omits(allow_model_requests: None, mocker: Any):
    # Gemini reports cumulative usage per chunk, but a gateway can drop a count from a later chunk.
    model = make_model()
    mocker.patch.object(
        model.client.aio.models,
        'generate_content_stream',
        return_value=_chunks([response([{'text': 'a'}]), response([{'text': 'b'}], cached=False)]),
    )
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as streamed:
        _ = [event async for event in streamed]
    assert streamed.usage.cache_read_tokens == 4
    assert streamed.get().parts == [TextPart(content='ab')]


async def test_stream_api_error_is_mapped(allow_model_requests: None, mocker: Any):
    async def failing_chunks() -> AsyncIterator[GenerateContentResponse]:
        yield response([{'text': 'partial'}])
        raise errors.APIError(503, {'error': {'code': 503, 'message': 'overloaded', 'status': 'UNAVAILABLE'}})

    model = make_model()
    mocker.patch.object(model.client.aio.models, 'generate_content_stream', return_value=failing_chunks())
    with pytest.raises(ModelHTTPError) as exc_info:
        async with model.request_stream(
            [ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()
        ) as streamed:
            _ = [event async for event in streamed]
    assert exc_info.value.status_code == 503
    assert 'overloaded' in str(exc_info.value.body)


async def test_stream_without_create_time(allow_model_requests: None, mocker: Any):
    model = make_model()
    mocker.patch.object(
        model.client.aio.models,
        'generate_content_stream',
        return_value=_chunks([response([{'text': 'hi'}])]),
    )
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as streamed:
        _ = [event async for event in streamed]
    assert streamed.provider_details == {'finish_reason': 'STOP'}
    assert streamed.get().parts == [TextPart(content='hi')]


def test_response_details_are_read_from_the_raw_response(allow_model_requests: None):
    model = make_model()
    raw = response(
        [{'text': 'hi'}],
        'MAX_TOKENS',
        candidate={
            'safety_ratings': [SAFETY_RATING],
            'logprobs_result': {'chosen_candidates': [{'token': 'hi', 'log_probability': -0.1, 'token_id': 1}]},
            'avg_logprobs': -0.1,
        },
        sdk_http_response={'headers': {'x-gemini-service-tier': 'Priority'}},
        create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        traffic_type='ON_DEMAND',
    )
    model_response = model._process_response(raw)  # pyright: ignore[reportPrivateUsage]
    assert model_response.finish_reason == 'length'
    details = model_response.provider_details
    assert safety_ratings(details) == [('HARM_CATEGORY_HARASSMENT', 'LOW', None)]
    assert details is not None
    # The logprobs are the SDK object's dump, unset fields included, as the native model records them.
    assert details.pop('logprobs')['chosen_candidates'] == [{'log_probability': -0.1, 'token': 'hi', 'token_id': 1}]
    assert details == snapshot(
        {
            'service_tier': 'priority',
            'traffic_type': 'ON_DEMAND',
            'finish_reason': 'MAX_TOKENS',
            'timestamp': datetime(2024, 1, 1, tzinfo=timezone.utc),
            'avg_logprobs': -0.1,
        }
    )


def test_blocked_prompt_has_no_parts_and_a_content_filter(allow_model_requests: None):
    raw = GenerateContentResponse.model_validate({'response_id': 'resp_2', 'prompt_feedback': BLOCKED_PROMPT})
    model_response = make_model()._process_response(raw)  # pyright: ignore[reportPrivateUsage]
    assert model_response.parts == []
    assert model_response.finish_reason == 'content_filter'
    details = model_response.provider_details
    assert safety_ratings(details) == [('HARM_CATEGORY_DANGEROUS_CONTENT', 'HIGH', True)]
    assert details == {'block_reason': 'PROHIBITED_CONTENT', 'block_reason_message': 'The prompt was blocked.'}


def test_empty_response_without_feedback_has_no_parts(allow_model_requests: None):
    raw = GenerateContentResponse.model_validate({'response_id': 'resp_3'})
    model_response = make_model()._process_response(raw)  # pyright: ignore[reportPrivateUsage]
    assert model_response.parts == []
    assert model_response.finish_reason is None
    assert model_response.provider_details is None


async def test_stream_blocked_prompt_keeps_the_content_filter(allow_model_requests: None, mocker: Any):
    model = make_model()
    blocked = GenerateContentResponse.model_validate(
        {'response_id': 'resp_2', 'prompt_feedback': {'block_reason': 'PROHIBITED_CONTENT'}}
    )
    # A usage-only chunk has no candidates and no feedback; a candidate that follows (a proxy may send
    # one) must not replace the content filter.
    usage_only = GenerateContentResponse.model_validate({'usage_metadata': {'prompt_token_count': 1}})
    mocker.patch.object(
        model.client.aio.models,
        'generate_content_stream',
        return_value=_chunks(
            [
                blocked,
                usage_only,
                response([{'text': ''}], response_id=None, candidate={'safety_ratings': [SAFETY_RATING]}),
            ]
        ),
    )
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as streamed:
        _ = [event async for event in streamed]
    assert streamed.finish_reason == 'content_filter'
    assert streamed.provider_details == {'block_reason': 'PROHIBITED_CONTENT'}
    assert streamed.provider_response_id == 'resp_2'


async def test_stream_records_safety_ratings_and_service_tier(allow_model_requests: None, mocker: Any):
    model = make_model()
    chunk = response(
        [{'text': 'hi'}],
        candidate={'safety_ratings': [SAFETY_RATING]},
        sdk_http_response={'headers': {'x-gemini-service-tier': 'FLEX'}},
    )
    mocker.patch.object(model.client.aio.models, 'generate_content_stream', return_value=_chunks([chunk]))
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as streamed:
        _ = [event async for event in streamed]
    details = streamed.provider_details
    assert safety_ratings(details) == [('HARM_CATEGORY_HARASSMENT', 'LOW', None)]
    assert details == {'service_tier': 'flex', 'finish_reason': 'STOP'}
