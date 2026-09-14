from __future__ import annotations

import base64
from collections.abc import AsyncIterator

import pytest
from httpx2 import Timeout

from pydantic_ai import NativeToolCallPart, NativeToolReturnPart, TextPart, UnexpectedModelBehavior, VideoUrl
from pydantic_ai.agent import Agent
from pydantic_ai.messages import PartStartEvent
from pydantic_ai.usage import RequestUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from google.genai.types import GenerateContentResponse, Part

    from pydantic_ai import _utils
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import (
        GeminiStreamedResponse,
        GoogleModel,
        _content_model_response,  # pyright: ignore[reportPrivateUsage]
        _GoogleMediaProcessingCodec,  # pyright: ignore[reportPrivateUsage]
        _process_response_from_parts,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
async def test_agentic_video_processing(
    stream: bool,
    allow_model_requests: None,
    gemini_api_key: str,
    request_capture: RequestCapture,
) -> None:
    """Agentic video requests expose Google's processing trace and preserve replayable history."""
    request_capture.client.timeout = Timeout(60)
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.client)
    agent = Agent(GoogleModel('gemini-3.7-flash', provider=provider))
    prompt = [
        'In one sentence, which animals appear and roughly when?',
        VideoUrl(
            url='https://www.youtube.com/watch?v=lCdaVNyHtjU',
            vendor_metadata={'media_processing': 'AGENTIC'},
        ),
    ]

    if stream:
        async with agent.run_stream(prompt) as result:
            await result.get_output()
            messages = result.all_messages()
    else:
        result = await agent.run(prompt)
        messages = result.all_messages()

    first_request_contents = request_capture.body()['contents']
    assert isinstance(first_request_contents, list)
    assert first_request_contents == snapshot(
        [
            {
                'parts': [
                    {'text': 'In one sentence, which animals appear and roughly when?'},
                    {
                        'fileData': {
                            'fileUri': 'https://www.youtube.com/watch?v=lCdaVNyHtjU',
                            'mimeType': 'video/mp4',
                        },
                        'mediaProcessing': 'AGENTIC',
                    },
                ],
                'role': 'user',
            }
        ]
    )
    response_parts = messages[-1].parts
    assert isinstance(response_parts[-1], TextPart)
    processing_parts = response_parts[:-1]
    assert processing_parts
    assert len(processing_parts) % 2 == 0
    for tool_call, tool_return in zip(processing_parts[::2], processing_parts[1::2]):
        assert isinstance(tool_call, NativeToolCallPart)
        assert isinstance(tool_return, NativeToolReturnPart)
        assert tool_call.tool_name == tool_return.tool_name == 'media_processing'
        assert tool_call.tool_call_id == tool_return.tool_call_id
        assert tool_call.provider_details == {
            'thought_signature': IsStr(),
            'media_processing_wire_part': {'tool_call': {'id': tool_call.tool_call_id}},
        }
        assert tool_return.provider_details == {
            'thought_signature': IsStr(),
            'media_processing_wire_part': {'tool_response': {'id': tool_return.tool_call_id}},
        }

    follow_up = 'What happens immediately before the first visible scene change?'
    await agent.run(follow_up, message_history=messages)
    follow_up_contents = request_capture.body(index=1)['contents']
    assert isinstance(follow_up_contents, list)
    assert follow_up_contents[:1] == first_request_contents
    assert follow_up_contents[-1] == {'parts': [{'text': follow_up}], 'role': 'user'}
    replayed_content = follow_up_contents[-2]
    assert isinstance(replayed_content, dict)
    replayed_parts = replayed_content['parts']
    assert isinstance(replayed_parts, list)
    assert len(replayed_parts) == len(response_parts)
    for index, (tool_call, tool_return) in enumerate(zip(processing_parts[::2], processing_parts[1::2])):
        assert isinstance(tool_call, NativeToolCallPart)
        assert isinstance(tool_return, NativeToolReturnPart)
        assert tool_call.provider_details is not None
        assert tool_return.provider_details is not None
        replayed_call = replayed_parts[index * 2]
        replayed_return = replayed_parts[index * 2 + 1]
        assert isinstance(replayed_call, dict)
        assert isinstance(replayed_return, dict)
        assert replayed_call['toolCall'] == {'id': tool_call.tool_call_id}
        assert replayed_return['toolResponse'] == {'id': tool_return.tool_call_id}
        replayed_call_signature = replayed_call['thoughtSignature']
        replayed_return_signature = replayed_return['thoughtSignature']
        assert isinstance(replayed_call_signature, str)
        assert isinstance(replayed_return_signature, str)
        assert base64.urlsafe_b64decode(replayed_call_signature) == base64.b64decode(
            tool_call.provider_details['thought_signature']
        )
        assert base64.urlsafe_b64decode(replayed_return_signature) == base64.b64decode(
            tool_return.provider_details['thought_signature']
        )
    assert replayed_parts[-1] == {'text': IsStr(), 'thoughtSignature': IsStr()}


def test_explicit_media_processing_parts_preserve_their_wire_shape() -> None:
    response = _process_response_from_parts(
        parts=[
            Part.model_validate({'thought_signature': b'call', 'tool_call': {'tool_type': 'MEDIA_PROCESSING'}}),
            Part.model_validate({'thought_signature': b'return', 'tool_response': {'tool_type': 'MEDIA_PROCESSING'}}),
        ],
        grounding_metadata=None,
        model_name='gemini-3.7-flash',
        provider_name='google-gla',
        provider_url='https://generativelanguage.googleapis.com/',
        usage=RequestUsage(),
        provider_response_id='response-id',
    )

    tool_call, tool_return = response.parts
    assert isinstance(tool_call, NativeToolCallPart)
    assert isinstance(tool_return, NativeToolReturnPart)
    assert tool_call.tool_call_id == tool_return.tool_call_id
    assert _content_model_response(response, frozenset({'google-gla'}), supports_tool_combination=True) == {
        'role': 'model',
        'parts': [
            {'thought_signature': b'call', 'tool_call': {'tool_type': 'MEDIA_PROCESSING'}},
            {'thought_signature': b'return', 'tool_response': {'tool_type': 'MEDIA_PROCESSING'}},
        ],
    }


def test_vertex_bare_processing_signatures_are_exposed_but_not_replayed() -> None:
    response = _process_response_from_parts(
        parts=[Part(thought_signature=b'call'), Part(thought_signature=b'return'), Part(text='done')],
        grounding_metadata=None,
        model_name='gemini-3.7-flash',
        provider_name='google-vertex',
        provider_url='https://aiplatform.googleapis.com/',
        usage=RequestUsage(),
        provider_response_id='response-id',
        media_processing=_GoogleMediaProcessingCodec(enabled=True),
    )

    assert len(response.parts) == 3
    tool_call, tool_return, text = response.parts
    assert isinstance(tool_call, NativeToolCallPart)
    assert isinstance(tool_return, NativeToolReturnPart)
    assert tool_call.tool_call_id == tool_return.tool_call_id
    assert tool_call.provider_details == {'thought_signature': 'Y2FsbA=='}
    assert tool_return.provider_details == {'thought_signature': 'cmV0dXJu'}
    assert text == TextPart(content='done')

    assert _content_model_response(response, frozenset({'google-vertex'}), supports_tool_combination=True) == {
        'role': 'model',
        'parts': [{'text': 'done'}],
    }


async def test_vertex_bare_processing_signatures_streaming_are_exposed() -> None:
    async def stream() -> AsyncIterator[GenerateContentResponse]:
        for parts in [[{'thought_signature': b'call'}], [{'thought_signature': b'return'}], [{'text': 'done'}]]:
            yield GenerateContentResponse.model_validate(
                {'candidates': [{'content': {'role': 'model', 'parts': parts}}]}
            )

    streamed = GeminiStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        _model_name='gemini-3.7-flash',
        _response=_utils.PeekableAsyncStream(stream()),
        _provider_name='google-vertex',
        _model_id_namespace='google',
        _provider_url='https://aiplatform.googleapis.com/',
        _media_processing=_GoogleMediaProcessingCodec(enabled=True),
    )

    events = [event async for event in streamed]
    started_parts = [event.part for event in events if isinstance(event, PartStartEvent)]
    assert isinstance(started_parts[0], NativeToolCallPart)
    assert isinstance(started_parts[1], NativeToolReturnPart)
    assert isinstance(started_parts[2], TextPart)

    parts = streamed.get().parts
    assert len(parts) == 3
    assert isinstance(parts[0], NativeToolCallPart)
    assert isinstance(parts[1], NativeToolReturnPart)
    assert parts[0].tool_call_id == parts[1].tool_call_id
    assert parts[2] == TextPart(content='done')


def test_empty_native_tool_part_without_signature_is_rejected() -> None:
    with pytest.raises(UnexpectedModelBehavior, match='Missing tool_type on native tool part'):
        _process_response_from_parts(
            parts=[Part.model_validate({'tool_call': {'id': 'unknown'}})],
            grounding_metadata=None,
            model_name='gemini-3.7-flash',
            provider_name='google-gla',
            provider_url='https://generativelanguage.googleapis.com/',
            usage=RequestUsage(),
            provider_response_id='response-id',
            media_processing=_GoogleMediaProcessingCodec(enabled=True),
        )
