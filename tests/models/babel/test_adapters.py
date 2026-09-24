"""The boundary between Pydantic AI messages and babel's IR, exercised without any provider SDK."""

from __future__ import annotations as _annotations

from datetime import datetime, timezone
from typing import Any, cast, get_args
from unittest.mock import AsyncMock

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    CachePoint,
    CompactionPart,
    FilePart,
    InstructionPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    PartStartEvent,
    TextContent,
    ToolAvailabilityDeltaPart,
    UploadedFile,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestStreamedResponse as _StubStreamedResponse
from pydantic_ai.usage import RequestUsage

from ...conftest import IsNow, IsStr, try_import

with try_import() as imports_successful:
    from llm_transform.ir import StopReason
    from llm_transform.registry import decode_response, encode, stream_step

    from pydantic_ai.models.babel import fold_stream_emits, ir_to_model_response, messages_to_ir
    from pydantic_ai.models.babel._adapters import (
        _FINISH_REASON,  # pyright: ignore[reportPrivateUsage]
        download_url_media,
        gemini_rest_to_sdk,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='llm-babel not installed'),
    pytest.mark.anyio,
]


def test_finish_reason_map_is_exhaustive_over_babel_stop_reasons():
    assert set(_FINISH_REASON) == set(get_args(StopReason))


def test_system_prompt_and_instructions_become_system_segments():
    ir = messages_to_ir(
        [ModelRequest(parts=[SystemPromptPart(content='be nice')])],
        model_name='gpt-4o',
        instruction_parts=[InstructionPart(content='and terse')],
    )
    assert ir == snapshot(
        {
            'messages': [],
            'model': 'gpt-4o',
            'system': [{'kind': 'text', 'text': 'be nice'}, {'kind': 'text', 'text': 'and terse'}],
        }
    )


def test_user_content_kinds():
    ir = messages_to_ir(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'look:',
                            TextContent(content='typed'),
                            ImageUrl(url='https://x/y.png', vendor_metadata={'detail': 'high'}),
                            AudioUrl(url='https://x/y.mp3'),
                            VideoUrl(url='https://x/y.mp4'),
                            DocumentUrl(url='https://x/y.pdf'),
                            BinaryContent(data=b'\x89PNG', media_type='image/png', vendor_metadata={'detail': 'low'}),
                            BinaryContent(data=b'%PDF', media_type='application/pdf'),
                            UploadedFile(file_id='file-1', provider_name='openai', media_type='audio/mpeg'),
                            UploadedFile(file_id='file-2', provider_name='openai', media_type='video/mp4'),
                        ]
                    )
                ]
            )
        ]
    )
    assert ir['messages'][0]['content'] == snapshot(
        [
            {'kind': 'text', 'text': 'look:'},
            {'kind': 'text', 'text': 'typed'},
            {
                'kind': 'file',
                'data': 'https://x/y.png',
                'source': 'url',
                'media_kind': 'image',
                'provider_ext': {'openai-chat': {'detail': 'high'}},
            },
            {'kind': 'file', 'data': 'https://x/y.mp3', 'source': 'url', 'media_kind': 'audio'},
            {'kind': 'file', 'data': 'https://x/y.mp4', 'source': 'url', 'media_kind': 'video'},
            {'kind': 'file', 'data': 'https://x/y.pdf', 'source': 'url', 'media_kind': 'document'},
            {
                'kind': 'file',
                'data': 'iVBORw==',
                'source': 'base64',
                'media_kind': 'image',
                'media_type': 'image/png',
                'provider_ext': {'openai-chat': {'detail': 'low'}},
            },
            {
                'kind': 'file',
                'data': 'JVBERg==',
                'source': 'base64',
                'media_kind': 'document',
                'media_type': 'application/pdf',
            },
            {'kind': 'file', 'data': 'file-1', 'source': 'file_id', 'media_kind': 'audio'},
            {'kind': 'file', 'data': 'file-2', 'source': 'file_id', 'media_kind': 'video'},
        ]
    )


def test_cache_point_attaches_to_the_preceding_part():
    ir = messages_to_ir([ModelRequest(parts=[UserPromptPart(content=['ctx', CachePoint(ttl='1h')])])])
    assert ir['messages'][0]['content'] == snapshot(
        [
            {
                'kind': 'text',
                'text': 'ctx',
                'provider_ext': {'anthropic-messages': {'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}},
            }
        ]
    )
    # The breakpoint is namespaced to Anthropic: encoding to another wire leaves no trace of it.
    assert 'cache_control' not in str(encode('gemini', {**ir, 'params': {}}))


def test_cache_point_without_preceding_content_raises():
    with pytest.raises(UserError, match='CachePoint cannot be the first content'):
        messages_to_ir([ModelRequest(parts=[UserPromptPart(content=[CachePoint()])])])


def test_tool_return_and_retry_parts():
    ir = messages_to_ir(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='f', content={'temp': 18}, tool_call_id='c1'),
                    RetryPromptPart(content='bad args', tool_name='g', tool_call_id='c2'),
                    RetryPromptPart(content='try again'),
                ]
            )
        ]
    )
    assert ir['messages'] == snapshot(
        [
            {
                'role': 'tool',
                'content': [
                    {'kind': 'tool_result', 'content': {'temp': 18}, 'id': 'c1', 'name': 'f'},
                    {
                        'kind': 'tool_result',
                        'content': 'bad args\n\nFix the errors and try again.',
                        'id': 'c2',
                        'name': 'g',
                        'is_error': True,
                    },
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'kind': 'text', 'text': 'Validation feedback:\ntry again\n\nFix the errors and try again.'}
                ],
            },
        ]
    )


def test_tool_return_with_media_trails_as_a_user_message():
    ir = messages_to_ir(
        [
            ModelRequest(
                parts=[ToolReturnPart(tool_name='f', content=ImageUrl(url='https://x/y.png'), tool_call_id='c1')]
            )
        ]
    )
    messages = cast(list[dict[str, Any]], ir['messages'])
    assert [message['role'] for message in messages] == ['tool', 'user']
    assert isinstance(messages[0]['content'][0]['content'], str)
    # The media is framed by the same text markers the native models add around a tool's files.
    trailing = messages[1]['content']
    assert [part['kind'] for part in trailing] == ['text', 'file', 'text']
    assert trailing[1] == {'kind': 'file', 'data': 'https://x/y.png', 'source': 'url', 'media_kind': 'image'}


def test_unsupported_request_part_raises():
    with pytest.raises(UserError, match='`ToolAvailabilityDeltaPart` is not supported by babel models'):
        messages_to_ir([ModelRequest(parts=[ToolAvailabilityDeltaPart()])])


def test_response_parts():
    ir = messages_to_ir(
        [
            ModelResponse(
                parts=[
                    TextPart(content='ok'),
                    ThinkingPart(content='hmm', signature='S', provider_name='anthropic'),
                    ThinkingPart(content='unknown provider keeps no signature', signature='S'),
                    ThinkingPart(id='redacted_thinking', content='', signature='ENC==', provider_name='anthropic'),
                    ToolCallPart(tool_name='f', args='{"a": 1', tool_call_id='c1'),
                    CompactionPart(content='## summary', provider_name='anthropic'),
                    CompactionPart(content='plain', provider_name='openai'),
                    FilePart(content=BinaryContent(data=b'img', media_type='image/png')),
                ]
            )
        ],
        provider_name='anthropic',
    )
    assert ir['messages'][0]['content'] == snapshot(
        [
            {'kind': 'text', 'text': 'ok'},
            {'kind': 'reasoning', 'text': 'hmm', 'provider_ext': {'anthropic-messages': {'signature': 'S'}}},
            {'kind': 'reasoning', 'text': 'unknown provider keeps no signature'},
            {'kind': 'reasoning', 'text': 'ENC==', 'redacted': True},
            {'kind': 'tool_call', 'name': 'f', 'input': {'INVALID_JSON': '{"a": 1'}, 'id': 'c1'},
            {'kind': 'text', 'text': '## summary', 'provider_ext': {'anthropic-messages': {'block': 'compaction'}}},
            {'kind': 'text', 'text': 'plain'},
        ]
    )


def test_native_tool_parts_replay_only_for_the_same_provider():
    parts = [
        NativeToolCallPart(tool_name='web_search', args={'q': 'x'}, tool_call_id='s1', provider_name='anthropic'),
        NativeToolReturnPart(
            tool_name='web_search', content=[{'url': 'u'}], tool_call_id='s1', provider_name='anthropic'
        ),
        NativeToolCallPart(tool_name='web_search', args={'q': 'y'}, tool_call_id='s2', provider_name='openai'),
        NativeToolReturnPart(tool_name='web_search', content='foreign', tool_call_id='s2', provider_name='openai'),
    ]
    ir = messages_to_ir([ModelResponse(parts=parts)], provider_name='anthropic')
    assert ir['messages'][0]['content'] == snapshot(
        [
            {'kind': 'tool_call', 'name': 'web_search', 'input': {'q': 'x'}, 'id': 's1', 'provider_executed': True},
            {
                'kind': 'tool_result',
                'content': [{'url': 'u'}],
                'id': 's1',
                'name': 'web_search',
                'provider_executed': True,
            },
        ]
    )
    # A turn made only of another provider's server-side tool parts renders nothing at all.
    assert messages_to_ir([ModelResponse(parts=parts[2:])], provider_name='anthropic')['messages'] == []


def test_unsupported_response_part_raises():
    with pytest.raises(UserError, match='`ToolAvailabilityDeltaPart` is not supported by babel models'):
        messages_to_ir([ModelResponse(parts=[ToolAvailabilityDeltaPart()])])  # type: ignore[list-item]


def test_thinking_signature_never_replays_to_another_provider():
    ir = messages_to_ir([ModelResponse(parts=[ThinkingPart(content='t', signature='GEMINI', provider_name='google')])])
    block = encode('anthropic-messages', {**ir, 'params': {'max_tokens': 8}})['messages'][0]['content'][0]
    assert 'signature' not in block


def _response_ir(**overrides: Any) -> dict[str, Any]:
    ir: dict[str, Any] = {
        'model': 'gpt-4o',
        'candidates': [
            {
                'content': [
                    {'kind': 'text', 'text': ''},
                    {'kind': 'text', 'text': 'hello'},
                    {'kind': 'reasoning', 'text': 'because', 'provider_ext': {'openai-chat': {'x': 1}}},
                    {'kind': 'tool_call', 'name': 'g', 'input': {'x': 2}, 'id': 'c9'},
                ],
                'stop_reason': 'tool_use',
            }
        ],
        'usage': {'input_tokens': 7, 'output_tokens': 3, 'cache_read_tokens': 1},
        'provider_ext': {'openai-chat': {'id': 'resp_42', 'created': 1704067200}},
    }
    ir.update(overrides)
    return ir


def test_ir_to_model_response_openai():
    response = ir_to_model_response(
        _response_ir(), fmt='openai-chat', provider_name='openai', provider_url='https://api.openai.com/v1'
    )
    assert response.parts == snapshot(
        [
            TextPart(content='hello'),
            ThinkingPart(content='because', provider_name='openai'),
            ToolCallPart(tool_name='g', args='{"x":2}', tool_call_id='c9'),
        ]
    )
    assert response.usage == snapshot(RequestUsage(input_tokens=7, cache_read_tokens=1, output_tokens=3))
    assert response.model_name == 'gpt-4o'
    assert response.finish_reason == 'tool_call'
    assert response.provider_response_id == 'resp_42'
    assert response.provider_details == snapshot(
        {'finish_reason': 'tool_calls', 'timestamp': datetime(2024, 1, 1, tzinfo=timezone.utc)}
    )


def test_ir_to_model_response_fallbacks():
    ir = _response_ir(provider_ext={}, model=None)
    ir['candidates'][0]['stop_reason'] = 'other'
    response = ir_to_model_response(ir, fmt='openai-chat', provider_name='openai', provider_url='u')
    assert response.provider_details is None
    assert response.provider_response_id is None
    assert response.model_name is None
    assert response.finish_reason is None
    # Bedrock's response body carries neither a model name nor a response id, so its model passes them.
    response = ir_to_model_response(
        ir,
        fmt='bedrock-converse',
        provider_name='bedrock',
        provider_url='u',
        model_name='m',
        provider_response_id='req_1',
    )
    assert response.model_name == 'm'
    assert response.provider_response_id == 'req_1'


def test_ir_to_model_response_anthropic_usage_and_signatures():
    ir = {
        'candidates': [
            {
                'content': [
                    {
                        'kind': 'reasoning',
                        'text': 'because',
                        'provider_ext': {'anthropic-messages': {'signature': 'S'}},
                    },
                    {'kind': 'reasoning', 'text': 'ENC==', 'redacted': True},
                    {
                        'kind': 'text',
                        'text': '## summary',
                        'provider_ext': {'anthropic-messages': {'block': 'compaction'}},
                    },
                    {
                        'kind': 'tool_call',
                        'name': 'web_search',
                        'input': {'q': 'x'},
                        'id': 's1',
                        'provider_executed': True,
                    },
                    {
                        'kind': 'tool_result',
                        'content': [{'url': 'u'}],
                        'id': 's1',
                        'name': 'web_search',
                        'provider_executed': True,
                    },
                    {'kind': 'source', 'url': 'u'},
                ],
                'stop_reason': 'end_turn',
            }
        ],
        'usage': {'input_tokens': 25, 'output_tokens': 9, 'cache_read_tokens': 50, 'cache_write_tokens': 100},
        'provider_ext': {'anthropic-messages': {'id': 'msg_1'}},
    }
    response = ir_to_model_response(ir, fmt='anthropic-messages', provider_name='anthropic', provider_url='u')
    assert response.parts == snapshot(
        [
            ThinkingPart(content='because', signature='S', provider_name='anthropic'),
            ThinkingPart(content='', id='redacted_thinking', signature='ENC==', provider_name='anthropic'),
            CompactionPart(content='## summary', provider_name='anthropic'),
            NativeToolCallPart(tool_name='web_search', args='{"q":"x"}', tool_call_id='s1', provider_name='anthropic'),
            NativeToolReturnPart(
                tool_name='web_search',
                content=[{'url': 'u'}],
                tool_call_id='s1',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
            ),
        ]
    )
    # Anthropic reports uncached input apart from the cache reads and writes; `RequestUsage` counts them all as input.
    assert response.usage == snapshot(
        RequestUsage(input_tokens=175, cache_write_tokens=100, cache_read_tokens=50, output_tokens=9)
    )
    assert response.provider_response_id == 'msg_1'
    assert response.provider_details is None


def test_ir_to_model_response_takes_a_usage_override():
    usage = RequestUsage(input_tokens=1)
    response = ir_to_model_response(_response_ir(), fmt='openai-chat', provider_name='o', provider_url='u', usage=usage)
    assert response.usage is usage


def test_decoded_numbers_stay_byte_faithful_in_tool_args():
    wire = {
        'id': 'r1',
        'model': 'gpt-4o',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'tool_calls',
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'c1',
                            'type': 'function',
                            'function': {'name': 'f', 'arguments': '{"n":5,"pi":3.14,"big":10000000000000000001}'},
                        }
                    ],
                },
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
    }
    response = ir_to_model_response(
        decode_response('openai-chat', wire), fmt='openai-chat', provider_name='o', provider_url='u'
    )
    assert response.parts == snapshot(
        [ToolCallPart(tool_name='f', args='{"big":10000000000000000001,"n":5,"pi":3.14}', tool_call_id='c1')]
    )


def _streamed_response() -> _StubStreamedResponse:
    return _StubStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        _model_name='test',
        _structured_response=ModelResponse(parts=[]),
        _messages=[],
        _provider_name='test',
    )


def test_fold_stream_emits_through_the_real_parts_manager():
    parts_manager = ModelResponsePartsManager(ModelRequestParameters())
    response = _streamed_response()
    seen_usage: list[dict[str, Any]] = []
    events = list(
        fold_stream_emits(
            [
                {'kind': 'text', 'text': 'It is '},
                {'kind': 'text', 'text': 'sunny.'},
                {'kind': 'reasoning', 'text': 'because', 'signature': 'sig'},
                {'kind': 'tool_call_start', 'id': 'c1', 'name': 'get_weather'},
                {'kind': 'tool_call', 'id': 'c1', 'name': 'get_weather', 'input': {'city': 'Paris'}},
                {'kind': 'tool_call_start', 'name': 'no_id'},
                {'kind': 'tool_call', 'name': 'no_id', 'input': {}},
                {'kind': 'source', 'url': 'u'},
                {'kind': 'usage', 'input_tokens': 3},
                {'kind': 'meta', 'stop_reason': 'tool_use'},
            ],
            parts_manager,
            response,
            provider_name='anthropic',
            on_usage=seen_usage.append,
        )
    )
    assert [type(event).__name__ for event in events] == snapshot(
        [
            'PartStartEvent',
            'PartDeltaEvent',
            'PartStartEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartStartEvent',
            'PartDeltaEvent',
        ]
    )
    assert parts_manager.get_parts() == snapshot(
        [
            TextPart(content='It is sunny.'),
            ThinkingPart(content='because', signature='sig', provider_name='anthropic'),
            ToolCallPart(tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='c1'),
            ToolCallPart(tool_name='no_id', args='{}', tool_call_id=IsStr()),
        ]
    )
    assert seen_usage == [{'kind': 'usage', 'input_tokens': 3}]
    assert response.finish_reason == 'tool_call'


def test_fold_stream_emits_ignores_usage_without_a_callback():
    parts_manager = ModelResponsePartsManager(ModelRequestParameters())
    assert list(fold_stream_emits([{'kind': 'usage', 'input_tokens': 1}], parts_manager, _streamed_response())) == []


def test_fold_stream_emits_skips_events_the_parts_manager_withholds():
    parts_manager = ModelResponsePartsManager(ModelRequestParameters())
    # Arguments for a call that was never announced stay a delta the parts manager does not emit yet.
    emits: list[dict[str, Any]] = [{'kind': 'tool_call', 'id': 'c1', 'name': 'f', 'input': {}}]
    assert list(fold_stream_emits(emits, parts_manager, _streamed_response())) == []


def test_fold_stream_emits_via_stream_step():
    chunks: list[dict[str, Any]] = [
        {
            'choices': [
                {'delta': {'tool_calls': [{'index': 0, 'id': 'c1', 'function': {'name': 'f', 'arguments': ''}}]}}
            ]
        },
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"n":5,"pi":3.14}'}}]}}]},
        {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]},
    ]
    parts_manager = ModelResponsePartsManager(ModelRequestParameters())
    state: Any = {}
    events: list[Any] = []
    for chunk in chunks:
        result = stream_step('openai-chat', state, chunk)
        state = result['state']
        events.extend(fold_stream_emits(result['emit'], parts_manager, _streamed_response()))
    assert [type(event) for event in events] == [PartStartEvent, PartDeltaEvent]
    assert parts_manager.get_parts() == snapshot(
        [ToolCallPart(tool_name='f', args='{"n":5,"pi":3.14}', tool_call_id='c1')]
    )


async def test_download_url_media(mocker: Any):
    download = mocker.patch(
        'pydantic_ai.models.babel._adapters.download_item',
        AsyncMock(return_value={'data': b'bytes', 'data_type': 'audio/mpeg'}),
    )
    request = ModelRequest(
        parts=[
            SystemPromptPart(content='s'),
            UserPromptPart(content='plain'),
            UserPromptPart(
                content=[
                    ImageUrl(url='https://x/a.png'),
                    ImageUrl(url='https://x/b.png', force_download=True, vendor_metadata={'detail': 'low'}),
                    AudioUrl(url='https://x/c.mp3'),
                ]
            ),
        ]
    )
    untouched = ModelRequest(parts=[UserPromptPart(content=[ImageUrl(url='https://x/d.png')])])
    response = ModelResponse(parts=[TextPart(content='x')])
    messages = await download_url_media([request, response, untouched], frozenset({'image'}))
    assert messages[1] is response
    assert messages[2] is untouched
    assert messages[0].parts[0] is request.parts[0]
    assert messages[0].parts[1] is request.parts[1]
    downloaded = messages[0].parts[2]
    assert isinstance(downloaded, UserPromptPart)
    assert downloaded.content == snapshot(
        [
            ImageUrl(url='https://x/a.png'),
            BinaryContent(data=b'bytes', media_type='audio/mpeg', vendor_metadata={'detail': 'low'}),
            BinaryContent(data=b'bytes', media_type='audio/mpeg'),
        ]
    )
    assert download.await_count == 2


def test_gemini_rest_to_sdk():
    node: dict[str, Any] = {
        'systemInstruction': {'parts': [{'text': 'hi'}]},
        'contents': [
            {
                'role': 'model',
                'parts': [{'functionCall': {'name': 'f', 'args': {'cityName': 'Paris', 'nested': {'someKey': 1}}}}],
            },
            {'role': 'user', 'parts': [{'functionResponse': {'name': 'f', 'response': 'plain string'}}]},
            {'role': 'user', 'parts': [{'functionResponse': {'name': 'f', 'response': {'already': 'dict'}}}]},
        ],
        'responseSchema': {'type': 'OBJECT', 'properties': {'someKey': {}}},
    }
    assert gemini_rest_to_sdk(node) == snapshot(
        {
            'system_instruction': {'parts': [{'text': 'hi'}]},
            'contents': [
                {
                    'role': 'model',
                    'parts': [
                        {'function_call': {'name': 'f', 'args': {'cityName': 'Paris', 'nested': {'someKey': 1}}}}
                    ],
                },
                {
                    'role': 'user',
                    'parts': [{'function_response': {'name': 'f', 'response': {'return_value': 'plain string'}}}],
                },
                {'role': 'user', 'parts': [{'function_response': {'name': 'f', 'response': {'already': 'dict'}}}]},
            ],
            'response_schema': {'type': 'OBJECT', 'properties': {'someKey': {}}},
        }
    )
    assert gemini_rest_to_sdk('scalar') == 'scalar'
