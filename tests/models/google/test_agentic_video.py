from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Literal

import pytest
from cassetter import Cassette, RecordMode

from pydantic_ai import NativeToolCallPart, NativeToolReturnPart, TextPart, UnexpectedModelBehavior, VideoUrl
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, ModelResponsePart
from pydantic_ai.usage import RequestUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from google.auth import default as google_auth_default  # pyright: ignore[reportUnknownVariableType]
    from google.auth.credentials import Credentials
    from google.genai.errors import ClientError
    from google.genai.types import Content, FileData, GenerateContentResponse, Part, ToolType

    from pydantic_ai import _utils
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import (
        GeminiStreamedResponse,
        GoogleModel,
        _process_response_from_parts,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

    from .test_native_tools import _process_response  # pyright: ignore[reportPrivateUsage]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
]

_VIDEO_URL = 'https://www.youtube.com/watch?v=lCdaVNyHtjU'
_VIDEO_MODEL = 'gemini-3.7-flash'
_VIDEO_INITIAL_PROMPT = 'In one sentence, which animals appear and roughly when?'
_VIDEO_HELD_OUT_QUESTION = 'What filename is open in the active editor tab?'
_VIDEO_HELD_OUT_FILENAME = 'browserRouter.tsx'


def _default_vertex_credentials() -> Credentials:
    credentials, _ = google_auth_default(  # pyright: ignore[reportUnknownVariableType]
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    assert isinstance(credentials, Credentials)
    return credentials


async def _stream_response_parts(
    chunks: Sequence[Sequence[Mapping[str, object]]],
) -> list[ModelResponsePart]:
    async def response_chunks() -> AsyncIterator[GenerateContentResponse]:
        for parts in chunks:
            yield GenerateContentResponse.model_validate({'candidates': [{'content': {'parts': parts}}]})

    streamed = GeminiStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        _model_name=_VIDEO_MODEL,
        _response=_utils.PeekableAsyncStream(response_chunks()),
        _provider_name='google-gla',
        _model_id_namespace='google',
        _provider_url='https://generativelanguage.googleapis.com/',
    )
    _ = [event async for event in streamed]
    return list(streamed.get().parts)


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
async def test_agentic_video_processing(
    stream: bool,
    allow_model_requests: None,
    gemini_api_key: str,
    request_capture: RequestCapture,
) -> None:
    """Each processing step becomes a `media_processing` call/return pair; the follow-up omits those parts."""
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=60))
    agent = Agent(GoogleModel(_VIDEO_MODEL, provider=provider))
    prompt = [
        _VIDEO_INITIAL_PROMPT,
        VideoUrl(
            url=_VIDEO_URL,
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
                    {'text': _VIDEO_INITIAL_PROMPT},
                    {
                        'fileData': {
                            'file_uri': _VIDEO_URL,
                            'mime_type': 'video/mp4',
                        },
                        'mediaProcessing': 'AGENTIC',
                    },
                ],
                'role': 'user',
            }
        ]
    )
    assert isinstance(messages[-1], ModelResponse)
    media_calls = [
        part
        for part in messages[-1].parts
        if isinstance(part, NativeToolCallPart) and part.tool_name == 'media_processing'
    ]
    media_returns = [
        part
        for part in messages[-1].parts
        if isinstance(part, NativeToolReturnPart) and part.tool_name == 'media_processing'
    ]
    assert media_calls
    assert len(media_calls) == len(media_returns)
    assert [part.tool_call_id for part in media_calls] == [part.tool_call_id for part in media_returns]
    assert all(
        part.provider_details is not None and isinstance(part.provider_details.get('thought_signature'), str)
        for part in media_calls
    )
    assert all(
        part.provider_details is not None and isinstance(part.provider_details.get('thought_signature'), str)
        for part in media_returns
    )
    text_parts = [part for part in messages[-1].parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    initial_text = ''.join(part.content for part in messages[-1].parts if isinstance(part, TextPart))
    assert _VIDEO_HELD_OUT_FILENAME.casefold() not in initial_text.casefold()
    final_text = text_parts[0]
    assert isinstance(final_text, TextPart)
    assert final_text.provider_details is not None
    final_signature = final_text.provider_details.get('thought_signature')
    assert isinstance(final_signature, str)

    messages = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].parts[-1].provider_details == {'thought_signature': final_signature}

    if stream:
        async with agent.run_stream(_VIDEO_HELD_OUT_QUESTION, message_history=messages) as result:
            follow_up_output = await result.get_output()
    else:
        result = await agent.run(_VIDEO_HELD_OUT_QUESTION, message_history=messages)
        follow_up_output = result.output
    assert _VIDEO_HELD_OUT_FILENAME.casefold() in follow_up_output.casefold()
    follow_up_contents = request_capture.body(index=1)['contents']
    assert isinstance(follow_up_contents, list)
    replayed_model_content = follow_up_contents[-2]
    assert isinstance(replayed_model_content, dict)
    replayed_model_parts = replayed_model_content['parts']
    assert isinstance(replayed_model_parts, list)
    assert len(replayed_model_parts) == 1
    replayed_text_part = replayed_model_parts[0]
    assert isinstance(replayed_text_part, dict)
    assert replayed_text_part['text'] == final_text.content
    replayed_signature = replayed_text_part['thoughtSignature']
    assert isinstance(replayed_signature, str)
    assert base64.urlsafe_b64decode(replayed_signature) == base64.urlsafe_b64decode(final_signature)


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


@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
@pytest.mark.parametrize('typed', [False, True], ids=['inferred-empty', 'typed'])
async def test_media_processing_parts_without_ids_share_the_call_id(stream: bool, typed: bool) -> None:
    """IDs are optional on these SDK parts, so pin pairing for typed and id-only media-processing responses."""
    tool: dict[str, str] = {'tool_type': 'MEDIA_PROCESSING'} if typed else {}
    parts: list[dict[str, object]] = [{'tool_call': tool}, {'tool_response': tool}]

    if stream:
        response_parts = await _stream_response_parts([[parts[0]], [parts[1]]])
    else:
        response = _process_response_from_parts(
            parts=[Part.model_validate(part) for part in parts],
            grounding_metadata=None,
            model_name='gemini-3.7-flash',
            provider_name='google-gla',
            provider_url='https://generativelanguage.googleapis.com/',
            usage=RequestUsage(),
            provider_response_id='response-id',
        )
        response_parts = response.parts

    call, tool_response = response_parts
    assert isinstance(call, NativeToolCallPart)
    assert isinstance(tool_response, NativeToolReturnPart)
    assert call.tool_call_id == tool_response.tool_call_id


async def test_streaming_media_processing_pairs_reset_between_calls() -> None:
    chunks: list[list[dict[str, object]]] = [
        [{'tool_call': {'tool_type': 'MEDIA_PROCESSING'}}],
        [{'tool_response': {'tool_type': 'MEDIA_PROCESSING'}}],
        [{'tool_call': {'tool_type': 'MEDIA_PROCESSING'}}],
        [{'tool_response': {'tool_type': 'MEDIA_PROCESSING'}}],
    ]
    response_parts = await _stream_response_parts(chunks)

    first_call, first_return, second_call, second_return = response_parts
    assert isinstance(first_call, NativeToolCallPart)
    assert isinstance(first_return, NativeToolReturnPart)
    assert isinstance(second_call, NativeToolCallPart)
    assert isinstance(second_return, NativeToolReturnPart)
    assert first_call.tool_call_id == first_return.tool_call_id
    assert second_call.tool_call_id == second_return.tool_call_id
    assert first_call.tool_call_id != second_call.tool_call_id


@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
async def test_signature_only_parts_are_skipped(stream: bool) -> None:
    parts: list[dict[str, object]] = [{'thought_signature': b'a'}]
    if stream:
        response_parts = await _stream_response_parts([parts])
    else:
        response_parts = _process_response_from_parts(
            parts=[Part.model_validate(part) for part in parts],
            grounding_metadata=None,
            model_name='gemini-3.7-flash',
            provider_name='google-gla',
            provider_url='https://generativelanguage.googleapis.com/',
            usage=RequestUsage(),
            provider_response_id='response-id',
        ).parts

    assert response_parts == []


@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
async def test_signed_unsupported_parts_keep_the_unsupported_error(stream: bool) -> None:
    part: dict[str, object] = {
        'file_data': {'file_uri': 'https://example.com/video.mp4', 'mime_type': 'video/mp4'},
        'thought_signature': b'a',
    }

    if stream:
        with pytest.raises(AssertionError, match='Unexpected part'):
            await _stream_response_parts([[part]])
    else:
        with pytest.raises(UnexpectedModelBehavior, match='Unsupported response from Gemini'):
            _process_response_from_parts(
                parts=[Part.model_validate(part)],
                grounding_metadata=None,
                model_name='gemini-3.7-flash',
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                usage=RequestUsage(),
                provider_response_id='response-id',
            )


@pytest.mark.parametrize(
    ('call', 'tool_response', 'expected_id'),
    [
        ({'id': 'explicit-call'}, {}, 'explicit-call'),
        ({}, {'id': 'explicit-response'}, 'explicit-response'),
    ],
    ids=['missing-response-id', 'explicit-response-id'],
)
@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
async def test_media_processing_pairing_preserves_explicit_ids(
    call: dict[str, str], tool_response: dict[str, str], expected_id: str, stream: bool
) -> None:
    parts: list[dict[str, object]] = [
        {'tool_call': {'tool_type': 'MEDIA_PROCESSING', **call}},
        {'tool_response': {'tool_type': 'MEDIA_PROCESSING', **tool_response}},
    ]
    if stream:
        response_parts = await _stream_response_parts([[parts[0]], [parts[1]]])
    else:
        response = _process_response_from_parts(
            parts=[Part.model_validate(part) for part in parts],
            grounding_metadata=None,
            model_name='gemini-3.7-flash',
            provider_name='google-gla',
            provider_url='https://generativelanguage.googleapis.com/',
            usage=RequestUsage(),
            provider_response_id='response-id',
        )
        response_parts = response.parts

    call_part, response_part = response_parts
    assert isinstance(call_part, NativeToolCallPart)
    assert isinstance(response_part, NativeToolReturnPart)
    assert response_part.tool_call_id == expected_id
    if call:
        assert call_part.tool_call_id == response_part.tool_call_id


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


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['non-streaming', 'streaming'])
async def test_agentic_video_processing_vertex(
    stream: bool,
    allow_model_requests: None,
    vertex_provider_auth: None,
    request_capture: RequestCapture,
    vcr: Cassette,
) -> None:
    """Vertex AI reports separate signature-only parts, which are omitted from follow-up history."""
    if not os.getenv('CI', False):
        pytest.skip('Requires properly configured local google vertex config to pass')
    if vcr.record_mode != RecordMode.NONE:
        # The auth fixture substitutes a fake token in CI playback; recordings use real ADC explicitly.
        credentials = _default_vertex_credentials()
        provider = GoogleCloudProvider(
            credentials=credentials,
            project=os.getenv('GOOGLE_PROJECT', 'pydantic-ai'),
            location='global',
            http_client=request_capture.http_client(timeout=60),
        )
    else:
        provider = GoogleCloudProvider(
            project=os.getenv('GOOGLE_PROJECT', 'pydantic-ai'),
            location='global',
            http_client=request_capture.http_client(timeout=60),
        )
    agent = Agent(GoogleModel('gemini-3.7-flash', provider=provider))
    prompt = [
        _VIDEO_INITIAL_PROMPT,
        VideoUrl(
            url=_VIDEO_URL,
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
                    {'text': _VIDEO_INITIAL_PROMPT},
                    {
                        'fileData': {
                            'file_uri': _VIDEO_URL,
                            'mime_type': 'video/mp4',
                        },
                        'mediaProcessing': 'AGENTIC',
                    },
                ],
                'role': 'user',
            }
        ]
    )
    assert isinstance(messages[-1], ModelResponse)
    text_parts = [part for part in messages[-1].parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    initial_text = ''.join(part.content for part in text_parts)
    assert _VIDEO_HELD_OUT_FILENAME.casefold() not in initial_text.casefold()
    final_text = text_parts[0]
    assert final_text.provider_name == 'google-cloud'
    assert final_text.provider_details is not None
    final_signature = final_text.provider_details.get('thought_signature')
    assert isinstance(final_signature, str)

    messages = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].parts[-1].provider_details == {'thought_signature': final_signature}

    if stream:
        async with agent.run_stream(_VIDEO_HELD_OUT_QUESTION, message_history=messages) as result:
            follow_up_output = await result.get_output()
    else:
        result = await agent.run(_VIDEO_HELD_OUT_QUESTION, message_history=messages)
        follow_up_output = result.output
    assert _VIDEO_HELD_OUT_FILENAME.casefold() in follow_up_output.casefold()
    follow_up_contents = request_capture.body(index=1)['contents']
    assert isinstance(follow_up_contents, list)
    replayed_model_content = follow_up_contents[-2]
    assert isinstance(replayed_model_content, dict)
    replayed_model_parts = replayed_model_content['parts']
    assert isinstance(replayed_model_parts, list)
    assert len(replayed_model_parts) == 1
    replayed_text_part = replayed_model_parts[0]
    assert isinstance(replayed_text_part, dict)
    assert replayed_text_part['text'] == final_text.content
    replayed_signature = replayed_text_part['thoughtSignature']
    assert isinstance(replayed_signature, str)
    assert base64.urlsafe_b64decode(replayed_signature) == base64.urlsafe_b64decode(final_signature)


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(
    reason='compare the rejected full response with its retry after omitting media-processing parts'
)
@pytest.mark.parametrize('route', ['developer-api', 'vertex'], ids=['developer-api', 'vertex'])
async def test_agentic_video_context_replay(
    route: Literal['developer-api', 'vertex'],
    allow_model_requests: None,
    gemini_api_key: str,
    vertex_provider_auth: None,
    request_capture: RequestCapture,
    vcr: Cassette,
) -> None:
    """Compare replaying the SDK response with omitting media-processing parts."""
    if route == 'developer-api':
        provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=60))
    else:
        if not os.getenv('CI', False):
            pytest.skip('Requires properly configured local google vertex config to pass')
        if vcr.record_mode != RecordMode.NONE:
            # The auth fixture substitutes a fake token in CI playback; recordings use real ADC explicitly.
            credentials = _default_vertex_credentials()
            provider = GoogleCloudProvider(
                credentials=credentials,
                project=os.getenv('GOOGLE_PROJECT', 'pydantic-ai'),
                location='global',
                http_client=request_capture.http_client(timeout=60),
            )
        else:
            provider = GoogleCloudProvider(
                project=os.getenv('GOOGLE_PROJECT', 'pydantic-ai'),
                location='global',
                http_client=request_capture.http_client(timeout=60),
            )

    user_content: Content = Content(
        role='user',
        parts=[
            Part(text=_VIDEO_INITIAL_PROMPT),
            Part(file_data=FileData(file_uri=_VIDEO_URL, mime_type='video/mp4'), media_processing='AGENTIC'),
        ],
    )
    initial_response = await provider.client.aio.models.generate_content(model=_VIDEO_MODEL, contents=[user_content])
    assert initial_response.candidates
    assert initial_response.candidates[0].content is not None
    returned_content = initial_response.candidates[0].content
    assert returned_content.parts

    with pytest.raises(ClientError) as error:
        await provider.client.aio.models.generate_content(
            model=_VIDEO_MODEL,
            contents=[
                user_content,
                returned_content,
                Content(role='user', parts=[Part(text=_VIDEO_HELD_OUT_QUESTION)]),
            ],
        )
    assert error.value.code == 400

    replay_body = request_capture.body(index=1)
    replay_contents = replay_body['contents']
    assert isinstance(replay_contents, list)
    replayed_model_content = replay_contents[-2]
    assert isinstance(replayed_model_content, dict)
    replayed_model_parts = replayed_model_content['parts']
    assert isinstance(replayed_model_parts, list)
    assert replayed_model_parts == returned_content.model_dump(mode='json', by_alias=True, exclude_none=True)['parts']

    if route == 'developer-api':
        retained_parts = [
            part
            for part in returned_content.parts
            if not (
                part.tool_call is not None
                and (
                    part.tool_call.tool_type == ToolType.MEDIA_PROCESSING
                    or (part.tool_call.tool_type is None and not part.tool_call.args)
                )
                or part.tool_response is not None
                and (
                    part.tool_response.tool_type == ToolType.MEDIA_PROCESSING
                    or (part.tool_response.tool_type is None and not part.tool_response.response)
                )
            )
        ]
    else:
        retained_parts = [
            part
            for part in returned_content.parts
            if part.model_dump(exclude_none=True).keys() != {'thought_signature'}
        ]
    omitted_content = Content(role='model', parts=retained_parts)
    follow_up_response = await provider.client.aio.models.generate_content(
        model=_VIDEO_MODEL,
        contents=[user_content, omitted_content, Content(role='user', parts=[Part(text=_VIDEO_HELD_OUT_QUESTION)])],
    )
    assert _VIDEO_HELD_OUT_FILENAME.casefold() in (follow_up_response.text or '').casefold()

    omitted_body = request_capture.body(index=2)
    omitted_contents = omitted_body['contents']
    assert isinstance(omitted_contents, list)
    omitted_model_content = omitted_contents[-2]
    assert isinstance(omitted_model_content, dict)
    omitted_model_parts = omitted_model_content['parts']
    assert isinstance(omitted_model_parts, list)
    assert omitted_model_parts == omitted_content.model_dump(mode='json', by_alias=True, exclude_none=True)['parts']


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
        ]
    )
