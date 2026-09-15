from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pydantic_ai import NativeToolCallPart, NativeToolReturnPart, TextPart, UnexpectedModelBehavior, VideoUrl
from pydantic_ai.agent import Agent
from pydantic_ai.usage import RequestUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from google.genai.types import GenerateContentResponse, Part, ToolType

    from pydantic_ai import _utils
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import (
        GeminiStreamedResponse,
        GoogleModel,
        _process_response_from_parts,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.google import GoogleProvider

    from .test_native_tools import _process_response  # pyright: ignore[reportPrivateUsage]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
]


# Each media-processing step Gemini reported in the recording, keyed by `stream`.
EXPECTED_PARTS: dict[bool, list[NativeToolCallPart | NativeToolReturnPart | TextPart]] = {
    False: snapshot(
        [
            NativeToolCallPart(
                tool_name='media_processing',
                tool_call_id='call_3095045',
                provider_name='google',
                provider_details={'thought_signature': IsStr()},
            ),
            NativeToolReturnPart(
                tool_name='media_processing',
                content=None,
                tool_call_id='call_3095045',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_details={'thought_signature': IsStr()},
            ),
            TextPart(
                content=IsStr(),
                provider_name='google',
                provider_details={'thought_signature': IsStr()},
            ),
        ]
    ),
    True: snapshot(
        [
            NativeToolCallPart(
                tool_name='media_processing',
                tool_call_id='call_3627313',
                provider_name='google',
                provider_details={'thought_signature': IsStr()},
            ),
            NativeToolReturnPart(
                tool_name='media_processing',
                content=None,
                tool_call_id='call_3627313',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_details={'thought_signature': IsStr()},
            ),
            NativeToolCallPart(
                tool_name='media_processing',
                tool_call_id='call_3627373',
                provider_name='google',
                provider_details={'thought_signature': IsStr()},
            ),
            NativeToolReturnPart(
                tool_name='media_processing',
                content=None,
                tool_call_id='call_3627373',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_details={'thought_signature': IsStr()},
            ),
            TextPart(
                content=IsStr(),
                provider_name='google',
                provider_details={'thought_signature': IsStr()},
            ),
        ]
    ),
}


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
async def test_agentic_video_processing(
    stream: bool,
    allow_model_requests: None,
    gemini_api_key: str,
    request_capture: RequestCapture,
) -> None:
    """Each processing step becomes a `media_processing` call/return pair; the follow-up replays only the final text.

    The replayed shape asserted at the end is the one Gemini accepted in the recorded live follow-up request.
    """
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=60))
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

    assert request_capture.body()['contents'] == snapshot(
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
    assert messages[-1].parts == EXPECTED_PARTS[stream]

    await agent.run('What happens immediately before the first visible scene change?', message_history=messages)
    follow_up_contents = request_capture.body(index=1)['contents']
    assert isinstance(follow_up_contents, list)
    replayed_model_content = follow_up_contents[-2]
    assert isinstance(replayed_model_content, dict)
    replayed_model_parts = replayed_model_content['parts']
    assert isinstance(replayed_model_parts, list)
    assert replayed_model_parts == snapshot([{'text': IsStr(), 'thoughtSignature': IsStr()}])


def test_typed_media_processing_parts_map_to_the_same_tool() -> None:
    """Unit test: the API doesn't set `tool_type` on these parts today, so a `MEDIA_PROCESSING` value can't be recorded."""
    response = _process_response_from_parts(
        parts=[
            Part.model_validate(
                {'tool_call': {'id': 'typed-call', 'tool_type': ToolType.MEDIA_PROCESSING}, 'thought_signature': b'a'}
            ),
            Part.model_validate(
                {
                    'tool_response': {'id': 'typed-call', 'tool_type': ToolType.MEDIA_PROCESSING},
                    'thought_signature': b'b',
                }
            ),
        ],
        grounding_metadata=None,
        model_name='gemini-3.7-flash',
        provider_name='google-gla',
        provider_url='https://generativelanguage.googleapis.com/',
        usage=RequestUsage(),
        provider_response_id='response-id',
    )

    assert response.parts == snapshot(
        [
            NativeToolCallPart(
                tool_name='media_processing',
                tool_call_id='typed-call',
                provider_name='google-gla',
                provider_details={'thought_signature': 'YQ=='},
            ),
            NativeToolReturnPart(
                tool_name='media_processing',
                content=None,
                tool_call_id='typed-call',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'thought_signature': 'Yg=='},
            ),
        ]
    )


def test_untyped_tool_call_with_a_payload_is_not_media_processing() -> None:
    """Unit test: pins that only the observed id-only shape is inferred; no model produces this input today."""
    with pytest.raises(UnexpectedModelBehavior, match='Missing tool_type on native tool part'):
        _process_response_from_parts(
            parts=[Part.model_validate({'tool_call': {'id': 'other', 'args': {'query': 'x'}}})],
            grounding_metadata=None,
            model_name='gemini-3.7-flash',
            provider_name='google-gla',
            provider_url='https://generativelanguage.googleapis.com/',
            usage=RequestUsage(),
            provider_response_id='response-id',
        )


@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
async def test_signature_only_parts_are_dropped(stream: bool) -> None:
    """Unit test: Vertex AI emits these for agentic video, but no Vertex cassette exists; the shape comes from the issue report."""
    parts = [
        Part(thought_signature=b'a'),
        Part(thought_signature=b'b'),
        Part(text='done', thought_signature=b'c'),
    ]

    if stream:

        async def response_stream() -> AsyncIterator[GenerateContentResponse]:
            yield GenerateContentResponse.model_validate(
                {'candidates': [{'content': {'role': 'model', 'parts': parts}}]}
            )

        streamed = GeminiStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            _model_name='gemini-3.7-flash',
            _response=_utils.PeekableAsyncStream(response_stream()),
            _provider_name='google-vertex',
            _model_id_namespace='google',
            _provider_url='https://aiplatform.googleapis.com/',
        )
        _ = [event async for event in streamed]
        response_parts = streamed.get().parts
    else:
        response_parts = _process_response_from_parts(
            parts=parts,
            grounding_metadata=None,
            model_name='gemini-3.7-flash',
            provider_name='google-vertex',
            provider_url='https://aiplatform.googleapis.com/',
            usage=RequestUsage(),
            provider_response_id='response-id',
        ).parts

    assert response_parts == snapshot(
        [
            TextPart(
                content='done',
                provider_name='google-vertex',
                provider_details={'thought_signature': 'Yw=='},
            )
        ]
    )


def test_media_processing_parts_do_not_suppress_metadata_reconstruction() -> None:
    """Unit test: agentic video combined with a metadata-delivered builtin tool has no recording."""
    response = _process_response(
        [
            {'tool_call': {'id': 'media-call'}, 'thought_signature': b'a'},
            {'tool_response': {'id': 'media-call'}, 'thought_signature': b'b'},
        ],
        grounding={
            'web_search_queries': ['Pydantic AI'],
            'grounding_chunks': [
                {'web': {'uri': 'https://ai.pydantic.dev', 'title': 'Pydantic AI'}},
            ],
        },
    )

    assert response.parts == snapshot(
        [
            NativeToolCallPart(
                tool_name='web_search',
                args={'queries': ['Pydantic AI']},
                tool_call_id=IsStr(),
                provider_name='google-gla',
            ),
            NativeToolReturnPart(
                tool_name='web_search',
                content=[{'uri': 'https://ai.pydantic.dev', 'title': 'Pydantic AI', 'domain': None}],
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                provider_name='google-gla',
            ),
            NativeToolCallPart(
                tool_name='media_processing',
                tool_call_id='media-call',
                provider_name='google-gla',
                provider_details={'thought_signature': 'YQ=='},
            ),
            NativeToolReturnPart(
                tool_name='media_processing',
                content=None,
                tool_call_id='media-call',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'thought_signature': 'Yg=='},
            ),
        ]
    )
