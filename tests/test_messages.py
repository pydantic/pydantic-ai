import json
import re
import sys
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, cast, get_args, get_origin

import pytest
from pydantic import TypeAdapter

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    FilePart,
    ImageUrl,
    InstructionPart,
    InstrumentationSettings,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    MultiModalContent,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    RequestUsage,
    RetryPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    INVALID_JSON_KEY,
    MULTI_MODAL_CONTENT_TYPES,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ToolReturnContent,
    is_multi_modal_content,
    narrow_message_parts,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsNow, IsStr, message, message_part


def test_image_url():
    image_url = ImageUrl(url='https://example.com/image.jpg')
    assert image_url.media_type == 'image/jpeg'
    assert image_url.format == 'jpeg'

    image_url = ImageUrl(url='https://example.com/image', media_type='image/jpeg')
    assert image_url.media_type == 'image/jpeg'
    assert image_url.format == 'jpeg'


def test_video_url():
    video_url = VideoUrl(url='https://example.com/video.mp4')
    assert video_url.media_type == 'video/mp4'
    assert video_url.format == 'mp4'

    video_url = VideoUrl(url='https://example.com/video', media_type='video/mp4')
    assert video_url.media_type == 'video/mp4'
    assert video_url.format == 'mp4'


@pytest.mark.parametrize(
    'url,is_youtube',
    [
        pytest.param('https://youtu.be/lCdaVNyHtjU', True, id='youtu.be'),
        pytest.param('https://www.youtube.com/lCdaVNyHtjU', True, id='www.youtube.com'),
        pytest.param('https://youtube.com/lCdaVNyHtjU', True, id='youtube.com'),
        pytest.param('https://dummy.com/video.mp4', False, id='dummy.com'),
    ],
)
def test_youtube_video_url(url: str, is_youtube: bool):
    video_url = VideoUrl(url=url)
    assert video_url.is_youtube is is_youtube
    assert video_url.media_type == 'video/mp4'
    assert video_url.format == 'mp4'


@pytest.mark.parametrize(
    'url, expected_data_type',
    [
        ('https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/docs/help.md', 'text/markdown'),
        ('https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/docs/help.txt', 'text/plain'),
        ('https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/docs/help.pdf', 'application/pdf'),
        ('https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/docs/help.rtf', 'application/rtf'),
        (
            'https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/docs/help.asciidoc',
            'text/x-asciidoc',
        ),
    ],
)
def test_document_url_other_types(url: str, expected_data_type: str) -> None:
    document_url = DocumentUrl(url=url)
    assert document_url.media_type == expected_data_type


def test_document_url():
    document_url = DocumentUrl(url='https://example.com/document.pdf')
    assert document_url.media_type == 'application/pdf'
    assert document_url.format == 'pdf'

    document_url = DocumentUrl(url='https://example.com/document', media_type='application/pdf')
    assert document_url.media_type == 'application/pdf'
    assert document_url.format == 'pdf'


def test_text_content():
    text_content = TextContent(content='Pydantic AI!', metadata={'foo': 'bar'})
    assert text_content.content == 'Pydantic AI!'
    assert text_content.metadata == {'foo': 'bar'}


@pytest.mark.parametrize(
    'media_type, format',
    [
        ('audio/wav', 'wav'),
        ('audio/mpeg', 'mp3'),
    ],
)
def test_binary_content_audio(media_type: str, format: str):
    binary_content = BinaryContent(data=b'Hello, world!', media_type=media_type)
    assert binary_content.is_audio
    assert binary_content.format == format


@pytest.mark.parametrize(
    'media_type, format',
    [
        ('image/jpeg', 'jpeg'),
        ('image/png', 'png'),
        ('image/gif', 'gif'),
        ('image/webp', 'webp'),
    ],
)
def test_binary_content_image(media_type: str, format: str):
    binary_content = BinaryContent(data=b'Hello, world!', media_type=media_type)
    assert binary_content.is_image
    assert binary_content.format == format


def test_binary_image_requires_image_media_type():
    # Valid image media type should work
    img = BinaryImage(data=b'test', media_type='image/png')
    assert img.is_image

    # Non-image media type should raise
    with pytest.raises(ValueError, match='`BinaryImage` must have a media type that starts with "image/"'):
        BinaryImage(data=b'test', media_type='text/plain')


@pytest.mark.parametrize(
    'media_type, format',
    [
        ('video/x-matroska', 'mkv'),
        ('video/quicktime', 'mov'),
        ('video/mp4', 'mp4'),
        ('video/webm', 'webm'),
        ('video/x-flv', 'flv'),
        ('video/mpeg', 'mpeg'),
        ('video/x-ms-wmv', 'wmv'),
        ('video/3gpp', 'three_gp'),
    ],
)
def test_binary_content_video(media_type: str, format: str):
    binary_content = BinaryContent(data=b'Hello, world!', media_type=media_type)
    assert binary_content.is_video
    assert binary_content.format == format


@pytest.mark.parametrize(
    'media_type, format',
    [
        ('application/pdf', 'pdf'),
        ('text/plain', 'txt'),
        ('text/csv', 'csv'),
        ('application/msword', 'doc'),
        ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'docx'),
        ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xlsx'),
        ('text/html', 'html'),
        ('text/markdown', 'md'),
        ('application/vnd.ms-excel', 'xls'),
    ],
)
def test_binary_content_document(media_type: str, format: str):
    binary_content = BinaryContent(data=b'Hello, world!', media_type=media_type)
    assert binary_content.is_document
    assert binary_content.format == format


@pytest.mark.parametrize(
    'audio_url,media_type,format',
    [
        pytest.param(AudioUrl('foobar.mp3'), 'audio/mpeg', 'mp3', id='mp3'),
        pytest.param(AudioUrl('foobar.wav'), 'audio/wav', 'wav', id='wav'),
        pytest.param(AudioUrl('foobar.oga'), 'audio/ogg', 'oga', id='oga'),
        pytest.param(AudioUrl('foobar.flac'), 'audio/flac', 'flac', id='flac'),
        pytest.param(AudioUrl('foobar.aiff'), 'audio/aiff', 'aiff', id='aiff'),
        pytest.param(AudioUrl('foobar.aac'), 'audio/aac', 'aac', id='aac'),
        pytest.param(AudioUrl('foobar', media_type='audio/mpeg'), 'audio/mpeg', 'mp3', id='mp3'),
    ],
)
def test_audio_url(audio_url: AudioUrl, media_type: str, format: str):
    assert audio_url.media_type == media_type
    assert audio_url.format == format


def test_audio_url_invalid():
    with pytest.raises(ValueError, match=re.escape('Could not infer media type from audio URL: foobar.potato')):
        AudioUrl('foobar.potato').media_type


@pytest.mark.parametrize(
    'image_url,media_type,format',
    [
        pytest.param(ImageUrl('foobar.jpg'), 'image/jpeg', 'jpeg', id='jpg'),
        pytest.param(ImageUrl('foobar.jpeg'), 'image/jpeg', 'jpeg', id='jpeg'),
        pytest.param(ImageUrl('foobar.png'), 'image/png', 'png', id='png'),
        pytest.param(ImageUrl('foobar.gif'), 'image/gif', 'gif', id='gif'),
        pytest.param(ImageUrl('foobar.webp'), 'image/webp', 'webp', id='webp'),
    ],
)
def test_image_url_formats(image_url: ImageUrl, media_type: str, format: str):
    assert image_url.media_type == media_type
    assert image_url.format == format


def test_image_url_invalid():
    with pytest.raises(ValueError, match=re.escape('Could not infer media type from image URL: foobar.potato')):
        ImageUrl('foobar.potato').media_type

    with pytest.raises(ValueError, match=re.escape('Could not infer media type from image URL: foobar.potato')):
        ImageUrl('foobar.potato').format


_url_formats = [
    pytest.param(DocumentUrl('foobar.pdf'), 'application/pdf', 'pdf', id='pdf'),
    pytest.param(DocumentUrl('foobar.txt'), 'text/plain', 'txt', id='txt'),
    pytest.param(DocumentUrl('foobar.csv'), 'text/csv', 'csv', id='csv'),
    pytest.param(DocumentUrl('foobar.doc'), 'application/msword', 'doc', id='doc'),
    pytest.param(
        DocumentUrl('foobar.docx'),
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'docx',
        id='docx',
    ),
    pytest.param(
        DocumentUrl('foobar.xlsx'),
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xlsx',
        id='xlsx',
    ),
    pytest.param(DocumentUrl('foobar.html'), 'text/html', 'html', id='html'),
    pytest.param(DocumentUrl('foobar.xls'), 'application/vnd.ms-excel', 'xls', id='xls'),
]
if sys.version_info > (3, 11):  # pragma: no branch
    # This solves an issue with MIMEType on MacOS + python < 3.12. mimetypes.py added the text/markdown in 3.12, but on
    # versions of linux the knownfiles include text/markdown so it isn't an issue. The .md test is only consistent
    # independent of OS on > 3.11.
    _url_formats.append(pytest.param(DocumentUrl('foobar.md'), 'text/markdown', 'md', id='md'))


@pytest.mark.parametrize('document_url,media_type,format', _url_formats)
def test_document_url_formats(document_url: DocumentUrl, media_type: str, format: str):
    assert document_url.media_type == media_type
    assert document_url.format == format


def test_document_url_invalid():
    with pytest.raises(ValueError, match=re.escape('Could not infer media type from document URL: foobar.potato')):
        DocumentUrl('foobar.potato').media_type

    with pytest.raises(ValueError, match='Unknown document media type: text/x-python'):
        DocumentUrl('foobar.py').format


def test_binary_content_unknown_media_type():
    with pytest.raises(ValueError, match='Unknown media type: application/custom'):
        binary_content = BinaryContent(data=b'Hello, world!', media_type='application/custom')
        binary_content.format


def test_binary_content_is_methods():
    # Test that is_X returns False for non-matching media types
    audio_content = BinaryContent(data=b'Hello, world!', media_type='audio/wav')
    assert audio_content.is_audio is True
    assert audio_content.is_image is False
    assert audio_content.is_video is False
    assert audio_content.is_document is False
    assert audio_content.format == 'wav'

    audio_content = BinaryContent(data=b'Hello, world!', media_type='audio/wrong')
    assert audio_content.is_audio is True
    assert audio_content.is_image is False
    assert audio_content.is_video is False
    assert audio_content.is_document is False
    with pytest.raises(ValueError, match='Unknown media type: audio/wrong'):
        audio_content.format

    audio_content = BinaryContent(data=b'Hello, world!', media_type='image/wrong')
    assert audio_content.is_audio is False
    assert audio_content.is_image is True
    assert audio_content.is_video is False
    assert audio_content.is_document is False
    with pytest.raises(ValueError, match='Unknown media type: image/wrong'):
        audio_content.format

    image_content = BinaryContent(data=b'Hello, world!', media_type='image/jpeg')
    assert image_content.is_audio is False
    assert image_content.is_image is True
    assert image_content.is_video is False
    assert image_content.is_document is False
    assert image_content.format == 'jpeg'

    video_content = BinaryContent(data=b'Hello, world!', media_type='video/mp4')
    assert video_content.is_audio is False
    assert video_content.is_image is False
    assert video_content.is_video is True
    assert video_content.is_document is False
    assert video_content.format == 'mp4'

    video_content = BinaryContent(data=b'Hello, world!', media_type='video/wrong')
    assert video_content.is_audio is False
    assert video_content.is_image is False
    assert video_content.is_video is True
    assert video_content.is_document is False
    with pytest.raises(ValueError, match='Unknown media type: video/wrong'):
        video_content.format

    document_content = BinaryContent(data=b'Hello, world!', media_type='application/pdf')
    assert document_content.is_audio is False
    assert document_content.is_image is False
    assert document_content.is_video is False
    assert document_content.is_document is True
    assert document_content.format == 'pdf'


def test_binary_content_base64():
    bc = BinaryContent(data=b'Hello, world!', media_type='image/png')
    assert bc.base64 == 'SGVsbG8sIHdvcmxkIQ=='
    assert not bc.base64.startswith('data:')
    assert bc.data_uri == 'data:image/png;base64,SGVsbG8sIHdvcmxkIQ=='


def test_from_data_uri_base64():
    bc = BinaryContent.from_data_uri('data:image/png;base64,SGVsbG8sIHdvcmxkIQ==')
    assert bc.data == b'Hello, world!'
    assert bc.media_type == 'image/png'


def test_from_data_uri_non_base64():
    with pytest.raises(ValueError, match='must be base64-encoded'):
        BinaryContent.from_data_uri('data:text/plain,Hello%20World')


@pytest.mark.xdist_group(name='url_formats')
@pytest.mark.parametrize(
    'video_url,media_type,format',
    [
        pytest.param(VideoUrl('foobar.mp4'), 'video/mp4', 'mp4', id='mp4'),
        pytest.param(VideoUrl('foobar.mov'), 'video/quicktime', 'mov', id='mov'),
        pytest.param(VideoUrl('foobar.mkv'), 'video/x-matroska', 'mkv', id='mkv'),
        pytest.param(VideoUrl('foobar.webm'), 'video/webm', 'webm', id='webm'),
        pytest.param(VideoUrl('foobar.flv'), 'video/x-flv', 'flv', id='flv'),
        pytest.param(VideoUrl('foobar.mpeg'), 'video/mpeg', 'mpeg', id='mpeg'),
        pytest.param(VideoUrl('foobar.wmv'), 'video/x-ms-wmv', 'wmv', id='wmv'),
        pytest.param(VideoUrl('foobar.three_gp'), 'video/3gpp', 'three_gp', id='three_gp'),
    ],
)
def test_video_url_formats(video_url: VideoUrl, media_type: str, format: str):
    assert video_url.media_type == media_type
    assert video_url.format == format


def test_video_url_invalid():
    with pytest.raises(ValueError, match=re.escape('Could not infer media type from video URL: foobar.potato')):
        VideoUrl('foobar.potato').media_type


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="'Python 3.10's mimetypes module does not support query parameters'"
)
def test_url_with_query_parameters() -> None:
    """Test that Url types correctly infer media type from URLs with query parameters"""
    video_url = VideoUrl('https://example.com/video.mp4?query=param')
    assert video_url.media_type == 'video/mp4'
    assert video_url.format == 'mp4'


def test_thinking_part_delta_apply_to_thinking_part_delta():
    """Test lines 768-775: Apply ThinkingPartDelta to another ThinkingPartDelta."""
    original_delta = ThinkingPartDelta(
        content_delta='original',
        signature_delta='sig1',
        provider_name='original_provider',
        provider_details={'foo': 'bar', 'baz': 'qux'},
    )

    # Test applying delta with no content or signature - should raise error
    empty_delta = ThinkingPartDelta()
    with pytest.raises(ValueError, match='Cannot apply ThinkingPartDelta with no content or signature'):
        empty_delta.apply(original_delta)

    # Test applying delta with content_delta
    content_delta = ThinkingPartDelta(content_delta=' new_content')
    result = content_delta.apply(original_delta)
    assert isinstance(result, ThinkingPartDelta)
    assert result.content_delta == 'original new_content'

    # Test applying delta with signature_delta
    sig_delta = ThinkingPartDelta(signature_delta='new_sig')
    result = sig_delta.apply(original_delta)
    assert isinstance(result, ThinkingPartDelta)
    assert result.signature_delta == 'new_sig'

    # Test applying delta with provider_name
    content_delta = ThinkingPartDelta(content_delta='', provider_name='new_provider')
    result = content_delta.apply(original_delta)
    assert isinstance(result, ThinkingPartDelta)
    assert result.provider_name == 'new_provider'

    # Test applying delta with provider_details
    provider_details_delta = ThinkingPartDelta(
        content_delta='', provider_details={'finish_reason': 'STOP', 'foo': 'qux'}
    )
    result = provider_details_delta.apply(original_delta)
    assert isinstance(result, ThinkingPartDelta)
    assert result.provider_details == {'foo': 'qux', 'baz': 'qux', 'finish_reason': 'STOP'}

    # Test chaining callable provider_details in delta-to-delta
    delta1 = ThinkingPartDelta(
        content_delta='first',
        provider_details=lambda d: {**(d or {}), 'first': 1},
    )
    delta2 = ThinkingPartDelta(
        content_delta=' second',
        provider_details=lambda d: {**(d or {}), 'second': 2},
    )
    chained = delta2.apply(delta1)
    assert isinstance(chained, ThinkingPartDelta)
    assert callable(chained.provider_details)
    # Apply chained delta to actual ThinkingPart to verify both callables ran
    part = ThinkingPart(content='')
    result_part = chained.apply(part)
    assert result_part.provider_details == {'first': 1, 'second': 2}

    # Test applying dict delta to callable delta (dict should merge with callable result)
    delta_callable = ThinkingPartDelta(
        content_delta='callable',
        provider_details=lambda d: {**(d or {}), 'from_callable': 'yes'},
    )
    delta_dict = ThinkingPartDelta(
        content_delta=' dict',
        provider_details={'from_dict': 'also'},
    )
    chained = delta_dict.apply(delta_callable)
    assert isinstance(chained, ThinkingPartDelta)
    assert callable(chained.provider_details)
    part = ThinkingPart(content='')
    result_part = chained.apply(part)
    assert result_part.provider_details == {'from_callable': 'yes', 'from_dict': 'also'}


def test_thinking_part_delta_callable_provider_details_serializable():
    # Reproduce the real streaming path: OpenAI's gpt-oss raw-CoT handler passes a callable
    # `provider_details` to `handle_thinking_delta`, which emits it verbatim inside a `PartDeltaEvent`
    # (see `_make_raw_content_updater` in models/openai.py). Such an event must still serialize, e.g.
    # when crossing a Temporal activity boundary in durable execution.
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    list(manager.handle_thinking_delta(vendor_part_id='t', content='reasoning', provider_details={'raw_content': ['']}))

    def update_details(existing: dict[str, Any] | None) -> dict[str, Any]:
        details = {**(existing or {})}
        details['raw_content'] = [*details.get('raw_content', []), 'tok']
        return details

    events = list(manager.handle_thinking_delta(vendor_part_id='t', content=' more', provider_details=update_details))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PartDeltaEvent)
    assert isinstance(event.delta, ThinkingPartDelta)
    assert callable(event.delta.provider_details)

    adapter: TypeAdapter[AgentStreamEvent] = TypeAdapter(AgentStreamEvent)

    # The callable merge callback can't be JSON-serialized, so it is emitted as `null` instead of raising.
    serialized = adapter.dump_json(event)
    assert json.loads(serialized)['delta']['provider_details'] is None
    # The serialized event round-trips back into an `AgentStreamEvent`.
    assert isinstance(adapter.validate_json(serialized), PartDeltaEvent)

    # Serialization is scoped to JSON mode, so Python-mode `model_dump()` keeps the callable intact.
    assert callable(adapter.dump_python(event)['delta']['provider_details'])

    # A plain dict `provider_details` is preserved as-is.
    dict_event = PartDeltaEvent(
        index=0,
        delta=ThinkingPartDelta(content_delta='dict', provider_details={'provider': 'detail'}),
    )
    assert json.loads(adapter.dump_json(dict_event))['delta']['provider_details'] == {'provider': 'detail'}


def test_pre_usage_refactor_messages_deserializable():
    # https://github.com/pydantic/pydantic-ai/pull/2378 changed the `ModelResponse` fields,
    # but we as tell people to store those in the DB we want to be very careful not to break deserialization.
    data = [
        {
            'parts': [
                {
                    'content': 'What is the capital of Mexico?',
                    'timestamp': datetime.now(tz=timezone.utc),
                    'part_kind': 'user-prompt',
                }
            ],
            'instructions': None,
            'kind': 'request',
        },
        {
            'parts': [{'content': 'Mexico City.', 'part_kind': 'text'}],
            'usage': {
                'requests': 1,
                'request_tokens': 13,
                'response_tokens': 76,
                'total_tokens': 89,
                'details': None,
            },
            'model_name': 'gpt-5-2025-08-07',
            'timestamp': datetime.now(tz=timezone.utc),
            'kind': 'response',
            'vendor_details': {
                'finish_reason': 'STOP',
            },
            'vendor_id': 'chatcmpl-CBpEXeCfDAW4HRcKQwbqsRDn7u7C5',
        },
    ]
    messages = ModelMessagesTypeAdapter.validate_python(data)
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Mexico?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
            ),
            ModelResponse(
                parts=[TextPart(content='Mexico City.')],
                usage=RequestUsage(
                    input_tokens=13,
                    output_tokens=76,
                    details={},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsNow(tz=timezone.utc),
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='chatcmpl-CBpEXeCfDAW4HRcKQwbqsRDn7u7C5',
            ),
        ]
    )


@pytest.mark.anyio
async def test_legacy_vendor_message_history_replays_through_agent():
    """1.x message history serialized with `vendor_details` / `vendor_id` keys still routes through `agent.run(message_history=...)`.

    Backstop for the V2-RULES rule 4 (cross-history-replay): the deprecated `vendor_*` read properties
    are gone in v2, but the validation aliases on `provider_details` / `provider_response_id` stay so
    stored histories load.
    """
    legacy_history: list[dict[str, Any]] = [
        {
            'parts': [{'content': 'Hi', 'part_kind': 'user-prompt'}],
            'kind': 'request',
        },
        {
            'parts': [{'content': 'Hello!', 'part_kind': 'text'}],
            'kind': 'response',
            'model_name': 'gpt-5',
            'provider_name': 'openai',
            'vendor_details': {'finish_reason': 'stop'},
            'vendor_id': 'chatcmpl-legacy',
        },
    ]
    message_history = ModelMessagesTypeAdapter.validate_python(legacy_history)
    response = next(m for m in message_history if isinstance(m, ModelResponse))
    assert response.provider_details == {'finish_reason': 'stop'}
    assert response.provider_response_id == 'chatcmpl-legacy'

    agent = Agent(TestModel())
    result = await agent.run('And now?', message_history=message_history)

    replayed_response = next(
        m for m in result.all_messages() if isinstance(m, ModelResponse) and m.model_name == 'gpt-5'
    )
    assert replayed_response.provider_details == {'finish_reason': 'stop'}
    assert replayed_response.provider_response_id == 'chatcmpl-legacy'


def test_file_part_has_content():
    filepart = FilePart(content=BinaryContent(data=b'', media_type='application/pdf'))
    assert not filepart.has_content()

    filepart.content.data = b'not empty'
    assert filepart.has_content()


@pytest.mark.parametrize(
    'args',
    [
        {'key': 'value'},
        {'key': 0},
        {'key': False},
        {'key': ''},
        {'key': []},
        {'key': {}},
        '{"key": "value"}',
        '0',
    ],
)
def test_tool_call_part_has_content(args: dict[str, object] | str):
    part = ToolCallPart(tool_name='test_tool', args=args)
    assert part.has_content()


@pytest.mark.parametrize(
    'args',
    [
        {},
        '',
        None,
    ],
)
def test_tool_call_part_has_content_empty(args: dict[str, object] | str | None):
    part = ToolCallPart(tool_name='test_tool', args=args)
    assert not part.has_content()


@pytest.mark.parametrize(
    'args',
    [
        {'key': 'value'},
        {'key': 0},
        {'key': False},
    ],
)
def test_builtin_tool_call_part_has_content(args: dict[str, object] | str | None):
    part = NativeToolCallPart(tool_name='web_search', args=args)
    assert part.has_content()


@pytest.mark.parametrize(
    'args',
    [
        {},
        None,
    ],
)
def test_builtin_tool_call_part_has_content_empty(args: dict[str, object] | str | None):
    part = NativeToolCallPart(tool_name='web_search', args=args)
    assert not part.has_content()


def test_file_part_serialization_roundtrip():
    # Verify that a serialized BinaryImage doesn't come back as a BinaryContent.
    messages: list[ModelMessage] = [
        ModelResponse(parts=[FilePart(content=BinaryImage(data=b'fake', media_type='image/jpeg'))])
    ]
    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode='json')
    assert serialized == snapshot(
        [
            {
                'parts': [
                    {
                        'content': {
                            'data': 'ZmFrZQ==',
                            'media_type': 'image/jpeg',
                            'identifier': 'c053ec',
                            'vendor_metadata': None,
                            'kind': 'binary',
                        },
                        'id': None,
                        'provider_name': None,
                        'part_kind': 'file',
                        'provider_details': None,
                    }
                ],
                'usage': {
                    'input_tokens': 0,
                    'cache_write_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 0,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'details': {},
                },
                'model_name': None,
                'timestamp': IsStr(),
                'kind': 'response',
                'provider_name': None,
                'provider_url': None,
                'provider_details': None,
                'provider_response_id': None,
                'finish_reason': None,
                'run_id': None,
                'conversation_id': None,
                'metadata': None,
                'state': 'complete',
            }
        ]
    )
    deserialized = ModelMessagesTypeAdapter.validate_python(serialized)
    assert deserialized == messages


def test_model_messages_type_adapter_preserves_run_id():
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content='Hi there', timestamp=datetime.now(tz=timezone.utc))],
            run_id='run-123',
            metadata={'key': 'value'},
        ),
        ModelResponse(parts=[TextPart(content='Hello!')], run_id='run-123', metadata={'key': 'value'}),
    ]

    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode='python')
    deserialized = ModelMessagesTypeAdapter.validate_python(serialized)

    assert [message.run_id for message in deserialized] == snapshot(['run-123', 'run-123'])


def test_model_messages_type_adapter_preserves_conversation_id():
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content='Hi there', timestamp=datetime.now(tz=timezone.utc))],
            conversation_id='conv-abc',
        ),
        ModelResponse(parts=[TextPart(content='Hello!')], conversation_id='conv-abc'),
    ]

    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode='python')
    deserialized = ModelMessagesTypeAdapter.validate_python(serialized)

    assert [message.conversation_id for message in deserialized] == snapshot(['conv-abc', 'conv-abc'])


def test_model_messages_type_adapter_back_compat_missing_conversation_id():
    """Histories serialized before the field existed should deserialize with conversation_id=None."""
    pre_pr_serialized = [
        {
            'kind': 'request',
            'parts': [{'part_kind': 'user-prompt', 'content': 'Hello'}],
            'run_id': 'run-123',
        },
        {
            'kind': 'response',
            'parts': [{'part_kind': 'text', 'content': 'Hi'}],
            'run_id': 'run-123',
        },
    ]
    deserialized = ModelMessagesTypeAdapter.validate_python(pre_pr_serialized)
    assert all(m.conversation_id is None for m in deserialized)


def test_model_messages_type_adapter_preserves_user_text_prompt_metadata():
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[TextContent(content='What is the weather like today?', metadata={'foo': 'bar'})],
                    timestamp=datetime.now(tz=timezone.utc),
                )
            ],
            run_id='run-123',
            metadata={'key': 'value'},
        )
    ]

    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode='python')
    deserialized = ModelMessagesTypeAdapter.validate_python(serialized)

    assert deserialized[0].parts[0].content[0].metadata == snapshot({'foo': 'bar'})  # type: ignore[reportUnknownMemberType]


def test_model_response_convenience_methods():
    response = ModelResponse(parts=[])
    assert response.text == snapshot(None)
    assert response.thinking == snapshot(None)
    assert response.files == snapshot([])
    assert response.images == snapshot([])
    assert response.tool_calls == snapshot([])
    assert response.native_tool_calls == snapshot([])

    response = ModelResponse(
        parts=[
            ThinkingPart(content="Let's generate an image"),
            ThinkingPart(content="And then, call the 'hello_world' tool"),
            TextPart(content="I'm going to"),
            TextPart(content=' generate an image'),
            NativeToolCallPart(tool_name='image_generation', args={}, tool_call_id='123'),
            FilePart(content=BinaryImage(data=b'fake', media_type='image/jpeg')),
            NativeToolReturnPart(tool_name='image_generation', content={}, tool_call_id='123'),
            TextPart(content="I'm going to call"),
            TextPart(content=" the 'hello_world' tool"),
            ToolCallPart(tool_name='hello_world', args={}, tool_call_id='123'),
        ]
    )
    assert response.text == snapshot("""\
I'm going to generate an image

I'm going to call the 'hello_world' tool\
""")
    assert response.thinking == snapshot("""\
Let's generate an image

And then, call the 'hello_world' tool\
""")
    assert response.files == snapshot([BinaryImage(data=b'fake', media_type='image/jpeg', identifier='c053ec')])
    assert response.images == snapshot([BinaryImage(data=b'fake', media_type='image/jpeg', identifier='c053ec')])
    assert response.tool_calls == snapshot([ToolCallPart(tool_name='hello_world', args={}, tool_call_id='123')])
    assert response.native_tool_calls == snapshot(
        [
            (
                NativeToolCallPart(tool_name='image_generation', args={}, tool_call_id='123'),
                NativeToolReturnPart(
                    tool_name='image_generation',
                    content={},
                    tool_call_id='123',
                    timestamp=IsDatetime(),
                ),
            )
        ]
    )


def test_image_url_validation_with_optional_identifier():
    image_url_ta = TypeAdapter(ImageUrl)
    image = image_url_ta.validate_python({'url': 'https://example.com/image.jpg'})
    assert image.url == snapshot('https://example.com/image.jpg')
    assert image.identifier == snapshot('39cfc4')
    assert image.media_type == snapshot('image/jpeg')
    assert image_url_ta.dump_python(image) == snapshot(
        {
            'url': 'https://example.com/image.jpg',
            'force_download': False,
            'vendor_metadata': None,
            'kind': 'image-url',
            'media_type': 'image/jpeg',
            'identifier': '39cfc4',
        }
    )

    image = image_url_ta.validate_python(
        {'url': 'https://example.com/image.jpg', 'identifier': 'foo', 'media_type': 'image/png'}
    )
    assert image.url == snapshot('https://example.com/image.jpg')
    assert image.identifier == snapshot('foo')
    assert image.media_type == snapshot('image/png')
    assert image_url_ta.dump_python(image) == snapshot(
        {
            'url': 'https://example.com/image.jpg',
            'force_download': False,
            'vendor_metadata': None,
            'kind': 'image-url',
            'media_type': 'image/png',
            'identifier': 'foo',
        }
    )


def test_binary_content_validation_with_optional_identifier():
    binary_content_ta = TypeAdapter(BinaryContent)
    binary_content = binary_content_ta.validate_python({'data': b'fake', 'media_type': 'image/jpeg'})
    assert binary_content.data == b'fake'
    assert binary_content.identifier == snapshot('c053ec')
    assert binary_content.media_type == snapshot('image/jpeg')
    assert binary_content_ta.dump_python(binary_content) == snapshot(
        {
            'data': b'fake',
            'vendor_metadata': None,
            'kind': 'binary',
            'media_type': 'image/jpeg',
            'identifier': 'c053ec',
        }
    )

    binary_content = binary_content_ta.validate_python(
        {'data': b'fake', 'identifier': 'foo', 'media_type': 'image/png'}
    )
    assert binary_content.data == b'fake'
    assert binary_content.identifier == snapshot('foo')
    assert binary_content.media_type == snapshot('image/png')
    assert binary_content_ta.dump_python(binary_content) == snapshot(
        {
            'data': b'fake',
            'vendor_metadata': None,
            'kind': 'binary',
            'media_type': 'image/png',
            'identifier': 'foo',
        }
    )


def test_binary_content_from_path(tmp_path: Path):
    # test normal file
    test_xml_file = tmp_path / 'test.xml'
    test_xml_file.write_text('<think>about trains</think>', encoding='utf-8')
    binary_content = BinaryContent.from_path(test_xml_file)
    assert binary_content == snapshot(BinaryContent(data=b'<think>about trains</think>', media_type='application/xml'))

    # test non-existent file
    non_existent_file = tmp_path / 'non-existent.txt'
    with pytest.raises(FileNotFoundError, match='File not found:'):
        BinaryContent.from_path(non_existent_file)

    # test file with unknown media type
    test_unknown_file = tmp_path / 'test.unknownext'
    test_unknown_file.write_text('some content', encoding='utf-8')
    binary_content = BinaryContent.from_path(test_unknown_file)
    assert binary_content == snapshot(BinaryContent(data=b'some content', media_type='application/octet-stream'))

    # test string path
    test_txt_file = tmp_path / 'test.txt'
    test_txt_file.write_text('just some text', encoding='utf-8')
    string_path = test_txt_file.as_posix()
    binary_content = BinaryContent.from_path(string_path)  # pyright: ignore[reportArgumentType]
    assert binary_content == snapshot(BinaryContent(data=b'just some text', media_type='text/plain'))

    # test image file
    test_jpg_file = tmp_path / 'test.jpg'
    test_jpg_file.write_bytes(b'\xff\xd8\xff\xe0' + b'0' * 100)  # minimal JPEG header + padding
    binary_content = BinaryContent.from_path(test_jpg_file)
    assert binary_content == snapshot(
        BinaryImage(data=b'\xff\xd8\xff\xe0' + b'0' * 100, media_type='image/jpeg', _identifier='bc8d49')
    )

    # test yaml file
    test_yaml_file = tmp_path / 'config.yaml'
    test_yaml_file.write_text('key: value', encoding='utf-8')
    binary_content = BinaryContent.from_path(test_yaml_file)
    assert binary_content == snapshot(BinaryContent(data=b'key: value', media_type='application/yaml'))

    # test yml file (alternative extension)
    test_yml_file = tmp_path / 'docker-compose.yml'
    test_yml_file.write_text('version: "3"', encoding='utf-8')
    binary_content = BinaryContent.from_path(test_yml_file)
    assert binary_content == snapshot(BinaryContent(data=b'version: "3"', media_type='application/yaml'))

    # test toml file
    test_toml_file = tmp_path / 'pyproject.toml'
    test_toml_file.write_text('[project]\nname = "test"', encoding='utf-8')
    binary_content = BinaryContent.from_path(test_toml_file)
    assert binary_content == snapshot(BinaryContent(data=b'[project]\nname = "test"', media_type='application/toml'))


def test_uploaded_file_identifier_property():
    """Test that UploadedFile.identifier hashes the file_id."""
    # Test basic identifier (should be hashed)
    uploaded_file = UploadedFile(file_id='file-abc123', provider_name='anthropic')
    assert uploaded_file.identifier == snapshot('3a1a6c')

    # Test with custom identifier
    uploaded_file_with_id = UploadedFile(file_id='file-xyz789', provider_name='anthropic', identifier='my-custom-id')
    assert uploaded_file_with_id.identifier == 'my-custom-id'

    # Test with URL file_id (should still be hashed)
    uploaded_file_url = UploadedFile(
        file_id='https://generativelanguage.googleapis.com/v1beta/files/abc123',
        provider_name='google',
    )
    assert uploaded_file_url.identifier == snapshot('d8d637')


def test_uploaded_file_format():
    """Test UploadedFile.format property for different media types."""
    # Test with no media_type - defaults to 'application/octet-stream' which has no format
    uploaded_file = UploadedFile(file_id='file-abc123', provider_name='anthropic')
    assert uploaded_file.media_type == 'application/octet-stream'
    with pytest.raises(ValueError, match='Unknown media type'):
        uploaded_file.format

    # Test with image media_type
    uploaded_file = UploadedFile(file_id='file-abc123', provider_name='anthropic', media_type='image/png')
    assert uploaded_file.format == 'png'

    # Test with video media_type
    uploaded_file = UploadedFile(file_id='file-abc123', provider_name='anthropic', media_type='video/mp4')
    assert uploaded_file.format == 'mp4'

    # Test with audio media_type
    uploaded_file = UploadedFile(file_id='file-abc123', provider_name='anthropic', media_type='audio/wav')
    assert uploaded_file.format == 'wav'

    # Test with document media_type
    uploaded_file = UploadedFile(file_id='file-abc123', provider_name='anthropic', media_type='application/pdf')
    assert uploaded_file.format == 'pdf'

    # Test with unknown media_type - should raise ValueError
    uploaded_file = UploadedFile(file_id='file-abc123', provider_name='anthropic', media_type='application/custom')
    with pytest.raises(ValueError, match='Unknown media type'):
        uploaded_file.format


def test_uploaded_file_in_otel_message_parts():
    """Test that UploadedFile is handled correctly in otel message parts conversion.

    Per OTel GenAI spec, UploadedFile maps to FilePart with type='file', modality, and file_id.
    See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json
    """
    # Test with file ID (OTel FilePart format) - no media_type defaults to 'application/octet-stream'
    part = UserPromptPart(
        content=['text before', UploadedFile(file_id='file-abc123', provider_name='anthropic'), 'text after']
    )
    settings = InstrumentationSettings(include_content=True)
    otel_parts = part.otel_message_parts(settings)
    assert otel_parts == snapshot(
        [
            {'type': 'text', 'content': 'text before'},
            {'type': 'file', 'modality': 'document', 'file_id': 'file-abc123', 'mime_type': 'application/octet-stream'},
            {'type': 'text', 'content': 'text after'},
        ]
    )

    # Test with URL file_id (still uses file_id field per spec) - no extension defaults to 'application/octet-stream'
    part_url = UserPromptPart(
        content=[
            'analyze this',
            UploadedFile(
                file_id='https://generativelanguage.googleapis.com/v1beta/files/abc123',
                provider_name='google',
            ),
        ]
    )
    otel_parts_url = part_url.otel_message_parts(settings)
    assert otel_parts_url == snapshot(
        [
            {'type': 'text', 'content': 'analyze this'},
            {
                'type': 'file',
                'modality': 'document',
                'file_id': 'https://generativelanguage.googleapis.com/v1beta/files/abc123',
                'mime_type': 'application/octet-stream',
            },
        ]
    )

    # Test with S3 URL and media_type - should include modality and mime_type
    part_s3 = UserPromptPart(
        content=[
            'process this',
            UploadedFile(file_id='s3://my-bucket/my-file.pdf', provider_name='bedrock', media_type='application/pdf'),
        ]
    )
    otel_parts_s3 = part_s3.otel_message_parts(settings)
    assert otel_parts_s3 == snapshot(
        [
            {'type': 'text', 'content': 'process this'},
            {
                'type': 'file',
                'modality': 'document',
                'file_id': 's3://my-bucket/my-file.pdf',
                'mime_type': 'application/pdf',
            },
        ]
    )

    # Test with image media_type - should have image modality
    part_image = UserPromptPart(
        content=[UploadedFile(file_id='img-123', provider_name='openai', media_type='image/png')]
    )
    otel_parts_image = part_image.otel_message_parts(settings)
    assert otel_parts_image == snapshot(
        [{'type': 'file', 'modality': 'image', 'file_id': 'img-123', 'mime_type': 'image/png'}]
    )

    # Test with audio media_type - should have audio modality
    part_audio = UserPromptPart(
        content=[UploadedFile(file_id='audio-123', provider_name='openai', media_type='audio/mp3')]
    )
    otel_parts_audio = part_audio.otel_message_parts(settings)
    assert otel_parts_audio == snapshot(
        [{'type': 'file', 'modality': 'audio', 'file_id': 'audio-123', 'mime_type': 'audio/mp3'}]
    )

    # Test with video media_type - should have video modality
    part_video = UserPromptPart(
        content=[UploadedFile(file_id='video-123', provider_name='openai', media_type='video/mp4')]
    )
    otel_parts_video = part_video.otel_message_parts(settings)
    assert otel_parts_video == snapshot(
        [{'type': 'file', 'modality': 'video', 'file_id': 'video-123', 'mime_type': 'video/mp4'}]
    )

    # Test without include_content (should have type, modality, and mime_type but not file_id)
    settings_no_content = InstrumentationSettings(include_content=False)
    otel_parts_no_content = part.otel_message_parts(settings_no_content)
    assert otel_parts_no_content == snapshot(
        [
            {'type': 'text'},
            {'type': 'file', 'modality': 'document', 'mime_type': 'application/octet-stream'},
            {'type': 'text'},
        ]
    )


def test_uploaded_file_serialization_roundtrip():
    """Verify that UploadedFile survives a ModelMessagesTypeAdapter serialization roundtrip.

    UploadedFile uses `exclude=True` on private fields (`_media_type`, `_identifier`) and exposes
    them via computed fields — this test ensures those computed values are preserved through
    serialization and deserialization.
    """
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'analyze this file',
                        UploadedFile(file_id='file-abc123', provider_name='anthropic', media_type='application/pdf'),
                    ]
                )
            ]
        )
    ]
    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode='json')
    deserialized = ModelMessagesTypeAdapter.validate_python(serialized)
    assert deserialized == messages


def test_uploaded_file_custom_identifier_and_media_type_roundtrip():
    """Verify that custom `identifier` and `media_type` survive serialization roundtrip."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        UploadedFile(
                            file_id='file-abc123',
                            provider_name='anthropic',
                            media_type='image/png',
                            identifier='my-id',
                        ),
                    ]
                )
            ]
        )
    ]
    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode='json')
    deserialized = ModelMessagesTypeAdapter.validate_python(serialized)
    part = message_part(deserialized, UserPromptPart)
    uploaded = part.content[0]
    assert isinstance(uploaded, UploadedFile)
    assert uploaded.identifier == 'my-id'
    assert uploaded.media_type == 'image/png'
    assert deserialized == messages


def test_tool_return_content_with_url_field_not_coerced_to_image_url():
    """Test that dicts with 'url' keys are not incorrectly coerced to ImageUrl.

    Regression test for: https://github.com/pydantic/pydantic-ai/issues/4190

    Without a discriminator on MultiModalContent union, Pydantic would incorrectly
    match any dict containing a 'url' key against ImageUrl (first union member),
    causing data loss.
    """

    serialized_history = r"""[
      {
        "parts": [{"content": "Hello", "timestamp": "2026-02-03T22:25:50Z", "part_kind": "user-prompt"}],
        "kind": "request"
      },
      {
        "parts": [{"tool_name": "my_tool", "args": "{}", "tool_call_id": "call_1", "part_kind": "tool-call"}],
        "model_name": "test",
        "timestamp": "2026-02-03T22:26:39Z",
        "kind": "response"
      },
      {
        "parts": [
          {
            "tool_name": "my_tool",
            "content": {
              "items": [{"name": "Example", "url": "/some/path/12345"}]
            },
            "tool_call_id": "call_1",
            "timestamp": "2026-02-03T22:27:32Z",
            "part_kind": "tool-return"
          }
        ],
        "kind": "request"
      }
    ]
    """

    # Deserialize - the dict with 'url' should remain as a dict, not become ImageUrl
    deserialized = ModelMessagesTypeAdapter.validate_json(serialized_history)

    tool_return_part = message_part(deserialized, ToolReturnPart, message_index=2)

    # The content should be preserved as a dict, not coerced to ImageUrl
    expected_content = {'items': [{'name': 'Example', 'url': '/some/path/12345'}]}
    assert tool_return_part.content == expected_content

    # Round-trip should work without errors
    reserialized = ModelMessagesTypeAdapter.dump_json(deserialized)
    reloaded = ModelMessagesTypeAdapter.validate_json(reserialized)

    reloaded_tool_return = message_part(reloaded, ToolReturnPart, message_index=2)
    assert reloaded_tool_return.content == expected_content


def test_tool_return_content_with_explicit_image_url():
    """Test that ImageUrl with explicit 'kind' discriminator is correctly deserialized."""
    from pydantic_ai.messages import ToolReturnPart

    serialized_history = r"""[
      {
        "parts": [{"content": "Hello", "timestamp": "2026-02-03T22:25:50Z", "part_kind": "user-prompt"}],
        "kind": "request"
      },
      {
        "parts": [
          {
            "tool_name": "image_tool",
            "content": {
              "url": "https://example.com/image.png",
              "kind": "image-url"
            },
            "tool_call_id": "call_1",
            "timestamp": "2026-02-03T22:27:32Z",
            "part_kind": "tool-return"
          }
        ],
        "kind": "request"
      }
    ]
    """

    deserialized = ModelMessagesTypeAdapter.validate_json(serialized_history)

    tool_return_part = message_part(deserialized, ToolReturnPart, message_index=1)

    # Content with explicit kind: "image-url" should become ImageUrl
    assert isinstance(tool_return_part.content, ImageUrl)
    assert tool_return_part.content.url == 'https://example.com/image.png'


def test_tool_return_content_nested_multimodal():
    """Test that nested MultiModalContent types with explicit discriminators work."""
    from pydantic_ai.messages import ToolReturnPart

    serialized_history = r"""[
      {
        "parts": [
          {
            "tool_name": "mixed_tool",
            "content": {
              "images": [
                {"url": "https://example.com/img1.jpg", "kind": "image-url"},
                {"url": "https://example.com/img2.png", "kind": "image-url"}
              ],
              "documents": [
                {"url": "https://example.com/doc.pdf", "kind": "document-url"}
              ],
              "regular_data": [
                {"url": "/api/path", "id": 123, "name": "test"}
              ]
            },
            "tool_call_id": "call_1",
            "timestamp": "2026-02-03T22:27:32Z",
            "part_kind": "tool-return"
          }
        ],
        "kind": "request"
      }
    ]
    """

    deserialized = ModelMessagesTypeAdapter.validate_json(serialized_history)
    tool_return_part = message_part(deserialized, ToolReturnPart)

    # `ToolReturnPart`'s typed `ToolSearchReturnPart` subclass narrows `content` to a
    # `TypedDict`; cast back to a plain dict so we can probe arbitrary keys here.
    content = cast('dict[str, Any]', tool_return_part.content)
    assert isinstance(content, dict)

    # Items with kind: "image-url" should be ImageUrl
    assert isinstance(content['images'][0], ImageUrl)
    assert isinstance(content['images'][1], ImageUrl)

    # Items with kind: "document-url" should be DocumentUrl
    assert isinstance(content['documents'][0], DocumentUrl)

    # Items without kind should remain as dicts
    assert content['regular_data'] == [{'url': '/api/path', 'id': 123, 'name': 'test'}]

    # Round-trip should preserve types
    reserialized = ModelMessagesTypeAdapter.dump_json(deserialized)
    reloaded = ModelMessagesTypeAdapter.validate_json(reserialized)
    reloaded_tool_return = message_part(reloaded, ToolReturnPart)
    reloaded_content = cast('dict[str, Any]', reloaded_tool_return.content)
    assert isinstance(reloaded_content, dict)

    assert isinstance(reloaded_content['images'][0], ImageUrl)
    assert isinstance(reloaded_content['documents'][0], DocumentUrl)
    assert reloaded_content['regular_data'] == [{'url': '/api/path', 'id': 123, 'name': 'test'}]


def test_multi_modal_content_types_matches_union():
    """Validate that MULTI_MODAL_CONTENT_TYPES matches the MultiModalContent union members,
    and that is_multi_modal_content correctly narrows types."""
    # Unwrap any `Annotated` wrappers (e.g. `BinaryContent` carries an `AfterValidator` that narrows
    # image content to `BinaryImage`) so the comparison is against the underlying content types.
    union_members = {
        get_args(m)[0] if get_origin(m) is Annotated else m for m in get_args(get_args(MultiModalContent)[0])
    }
    assert set(MULTI_MODAL_CONTENT_TYPES) == union_members

    # Positive cases: each multimodal type is recognized
    assert is_multi_modal_content(ImageUrl(url='https://example.com/image.png'))
    assert is_multi_modal_content(AudioUrl(url='https://example.com/audio.mp3'))
    assert is_multi_modal_content(DocumentUrl(url='https://example.com/doc.pdf'))
    assert is_multi_modal_content(VideoUrl(url='https://example.com/video.mp4'))
    assert is_multi_modal_content(BinaryContent(data=b'\x89PNG', media_type='image/png'))

    # Negative cases: non-multimodal types
    assert not is_multi_modal_content('a string')
    assert not is_multi_modal_content({'key': 'value'})
    assert not is_multi_modal_content(42)


@pytest.mark.parametrize('mode', ['json', 'python'])
def test_binary_image_narrowed_wherever_multimodal_content_is_validated(mode: str):
    """An image `BinaryContent` narrows to `BinaryImage` on validation of any `MultiModalContent`
    (here via `UserPromptPart`), not just `FilePart.content`; non-image `BinaryContent` is left as-is.
    """
    image = BinaryContent(data=b'\x89PNG', media_type='image/png')
    audio = BinaryContent(data=b'\x00\x01', media_type='audio/mpeg')
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=[image, audio])])]

    if mode == 'json':
        loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    else:
        loaded = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.dump_python(messages, mode='json'))

    part = message_part(loaded, UserPromptPart)
    assert isinstance(part.content, list)
    reloaded_image, reloaded_audio = part.content
    assert type(reloaded_image) is BinaryImage
    assert reloaded_image.data == image.data and reloaded_image.media_type == image.media_type
    # Non-image content is not narrowed.
    assert type(reloaded_audio) is BinaryContent


def test_every_multimodal_type_rehydrates_as_tool_return_content():
    """Every `MultiModalContent` type, dumped as scalar `ToolReturnPart.content`, must rehydrate to
    its own subclass through `ModelMessagesTypeAdapter` — not collapse to a plain dict.

    Guards the `ToolReturnContent` discriminator's type-specific-field gate (`_MULTIMODAL_FIELDS`):
    if a future `MultiModalContent` type serialized without a `url`/`media_type`/`file_id` key, the
    gate would route its dumped dict to the `mapping` branch and silently stop rehydrating it. The
    factory must cover exactly `MULTI_MODAL_CONTENT_TYPES`, so a new type forces a deliberate update.
    `BinaryContent` uses a non-image media type so it isn't narrowed to `BinaryImage`.
    """
    samples: dict[type, MultiModalContent] = {
        ImageUrl: ImageUrl(url='https://example.com/a.png'),
        AudioUrl: AudioUrl(url='https://example.com/a.mp3'),
        VideoUrl: VideoUrl(url='https://example.com/a.mp4'),
        DocumentUrl: DocumentUrl(url='https://example.com/a.pdf'),
        BinaryContent: BinaryContent(data=b'x', media_type='application/pdf'),
        UploadedFile: UploadedFile(file_id='f1', provider_name='openai', media_type='image/png'),
    }
    assert set(samples) == set(MULTI_MODAL_CONTENT_TYPES)

    for cls, instance in samples.items():
        messages: list[ModelMessage] = [
            ModelRequest(parts=[ToolReturnPart(tool_name='t', content=instance, tool_call_id='c')])
        ]
        reloaded = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.dump_python(messages, mode='json'))
        part = message_part(reloaded, ToolReturnPart)
        assert type(part.content) is cls, (
            f'{cls.__name__} did not rehydrate through the discriminator gate '
            f'(got {type(part.content).__name__}) — a `_MULTIMODAL_FIELDS` mismatch would cause this'
        )


def test_tool_return_part_binary_content_serialization():
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
    binary_content = BinaryContent(png_data, media_type='image/png')
    tool_return = ToolReturnPart(tool_name='test_tool', content=binary_content, tool_call_id='test_call_123')
    assert tool_return.model_response_object() == snapshot({})


@pytest.mark.parametrize('case_id', ['scalar', 'list-with-binary', 'dict-with-nested-binary'])
def test_tool_return_part_binary_content_round_trip(case_id: str, tiny_audio: BinaryContent):
    """`ToolReturnPart.content` containing `BinaryContent` (scalar, in a list, or in a dict)
    must round-trip via `ModelMessagesTypeAdapter` in both `validate_json` (the wire path)
    and `validate_python` (the replay path used by UI adapters that already parsed JSON).

    Without the explicit `Discriminator` on `ToolReturnContent`, smart-union resolution picks
    `Mapping`/`Sequence`/`Any` over the discriminated `MultiModalContent` branch in
    `validate_python`, leaving binary leaves as plain dicts.

    Uses `tiny_audio` (non-image `BinaryContent`) to focus on rehydration, not the
    `BinaryImage` narrowing applied by UI adapters.
    """
    contents: dict[str, ToolReturnContent] = {
        'scalar': tiny_audio,
        'list-with-binary': ['hello', tiny_audio],
        'dict-with-nested-binary': {'caption': 'see audio', 'attachment': tiny_audio},
    }
    content = contents[case_id]
    messages: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content=content, tool_call_id='c')])
    ]

    json_loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    json_part = message_part(json_loaded, ToolReturnPart)
    assert json_part.content == content

    python_loaded = ModelMessagesTypeAdapter.validate_python(
        ModelMessagesTypeAdapter.dump_python(messages, mode='json')
    )
    python_part = message_part(python_loaded, ToolReturnPart)
    assert python_part.content == content


@pytest.mark.parametrize(
    'content',
    [
        pytest.param({'kind': 'binary', 'label': 'foo'}, id='kind-binary-no-media-type'),
        pytest.param({'kind': 'image-url', 'note': 'not a real url part'}, id='kind-url-no-media-type'),
    ],
)
def test_tool_return_dict_reusing_kind_without_type_field_stays_mapping(content: dict[str, str]):
    """A user dict that reuses one of our `kind` values but lacks a type-specific field
    (`media_type`/`file_id`) is left as a plain mapping rather than forced through
    `MultiModalContent` validation (which would raise a hard `ValidationError`).

    The discriminator is wired into core `ToolReturnContent`, so this guards every
    `ModelMessagesTypeAdapter` round trip, not just the UI adapters.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content=content, tool_call_id='c')])
    ]

    loaded = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.dump_python(messages, mode='json'))
    part = message_part(loaded, ToolReturnPart)
    assert part.content == content


@pytest.mark.parametrize(
    'content',
    [
        # Reserved `kind` + a type-specific field, but not a valid instance of that type:
        pytest.param({'kind': 'binary', 'media_type': 'text/plain', 'text': 'hello'}, id='binary-without-data'),
        pytest.param(
            {'kind': 'uploaded-file', 'file_id': 'abc', 'status': 'ready'}, id='uploaded-file-without-provider'
        ),
        pytest.param({'kind': 'image-url', 'media_type': 'image/png', 'note': 'x'}, id='image-url-without-url'),
    ],
)
@pytest.mark.parametrize('mode', ['json', 'python'])
def test_tool_return_dict_reusing_kind_with_type_field_stays_mapping(content: dict[str, str], mode: str):
    """A user dict that reuses a `kind` value AND carries a type field (`media_type`/`url`/`file_id`)
    but isn't a valid instance of that type must stay a plain mapping, not raise.

    The discriminator gates such a dict into the `multimodal` branch on the `kind`+field heuristic;
    `_validate_multimodal_or_passthrough` falls back to the raw dict when `MultiModalContent` validation
    fails, and `_serialize_multimodal_or_passthrough` dumps it without a spurious serializer warning —
    together matching the pre-discriminator behavior where these fell through to the `Any` arm.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content=content, tool_call_id='c')])
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # a `PydanticSerializationUnexpectedValue` warning would fail here
        if mode == 'json':
            loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
        else:
            loaded = ModelMessagesTypeAdapter.validate_python(
                ModelMessagesTypeAdapter.dump_python(messages, mode='json')
            )

    part = message_part(loaded, ToolReturnPart)
    assert part.content == content


@pytest.mark.parametrize(
    'kind',
    [
        pytest.param([1, 2], id='kind-list'),
        pytest.param({'x': 'y'}, id='kind-dict'),
        pytest.param(bytearray(b'binary'), id='kind-bytearray'),
    ],
)
@pytest.mark.parametrize('nested', [False, True], ids=['top-level', 'nested-in-sequence'])
def test_tool_return_dict_unhashable_kind_stays_mapping(kind: object, nested: bool):
    """A client dict whose `kind` is unhashable must not crash the discriminator with a `TypeError`.

    The discriminator's `kind in _MULTIMODAL_KINDS` membership test raises `TypeError` on an unhashable
    `kind` (`list`/`dict`/`bytearray`); the `isinstance(kind, str)` guard routes it to the `mapping`
    branch instead, where it round-trips as a plain mapping — the same graceful handling of malformed
    client input as the `_js_binary_to_bytes` hardening.
    """
    inner: dict[str, Any] = {'kind': kind, 'media_type': 'image/png', 'data': 'YWJj'}
    content: Any = [inner] if nested else inner
    dumped = {
        'parts': [{'tool_name': 't', 'content': content, 'tool_call_id': 'c', 'part_kind': 'tool-return'}],
        'kind': 'request',
    }

    loaded = ModelMessagesTypeAdapter.validate_python([dumped])
    part = message_part(loaded, ToolReturnPart)
    assert part.content == content


def test_tool_return_part_list_structure_preserved():
    single_dict = {'result': 'found'}
    single_item_list = [{'result': 'found'}]
    multi_item_list = [{'a': 1}, {'b': 2}]

    tool_return_dict = ToolReturnPart(tool_name='test', content=single_dict, tool_call_id='tc1')
    assert tool_return_dict.model_response_object() == snapshot({'result': 'found'})
    assert tool_return_dict.model_response_str() == snapshot('{"result":"found"}')

    tool_return_single_list = ToolReturnPart(tool_name='test', content=single_item_list, tool_call_id='tc2')
    assert tool_return_single_list.model_response_object() == snapshot({'return_value': [{'result': 'found'}]})
    assert tool_return_single_list.model_response_str() == snapshot('[{"result":"found"}]')

    tool_return_multi_list = ToolReturnPart(tool_name='test', content=multi_item_list, tool_call_id='tc3')
    assert tool_return_multi_list.model_response_object() == snapshot({'return_value': [{'a': 1}, {'b': 2}]})
    assert tool_return_multi_list.model_response_str() == snapshot('[{"a":1},{"b":2}]')


@pytest.mark.parametrize(
    'outcome,expected_str,expected_object',
    [
        pytest.param('success', 'Disk full', {'return_value': 'Disk full'}, id='success'),
        pytest.param('denied', 'Disk full', {'return_value': 'Disk full'}, id='denied'),
        pytest.param('failed', '{"error":"Disk full"}', {'error': 'Disk full'}, id='failed'),
    ],
)
def test_tool_return_part_model_response_outcome(
    outcome: Literal['success', 'failed', 'denied'], expected_str: str, expected_object: dict[str, Any]
) -> None:
    """Public model-conversion helpers frame only failed results and let native error channels opt out."""
    part = ToolReturnPart(tool_name='tool', content='Disk full', tool_call_id='call_1', outcome=outcome)

    assert part.model_response_str() == expected_str
    assert part.model_response_object() == expected_object

    if outcome == 'failed':
        assert part.model_response_str(wrap_if_error=False) == 'Disk full'
        assert part.model_response_object(wrap_if_error=False) == {'return_value': 'Disk full'}

        structured = ToolReturnPart(
            tool_name='tool',
            content={'error': 'legitimate output'},
            tool_call_id='call_2',
            outcome='failed',
        )
        assert structured.model_response_str() == '{"error":"{\\"error\\":\\"legitimate output\\"}"}'
        assert structured.model_response_str(wrap_if_error=False) == '{"error":"legitimate output"}'
        assert structured.model_response_object() == {'error': '{"error":"legitimate output"}'}
        assert structured.model_response_object(wrap_if_error=False) == {'error': 'legitimate output'}


def test_tool_return_part_content_items():
    img = ImageUrl(url='https://example.com/img.png')
    binary = BinaryContent(data=b'\x89PNG', media_type='image/png')

    p_str = ToolReturnPart(tool_name='t', content='hello', tool_call_id='c1')
    assert p_str.content_items() == snapshot(['hello'])
    assert p_str.content_items(mode='raw') == snapshot(['hello'])
    assert p_str.content_items(mode='str') == snapshot(['hello'])
    assert p_str.content_items(mode='jsonable') == snapshot(['hello'])

    p_dict = ToolReturnPart(tool_name='t', content={'key': 'val'}, tool_call_id='c2')
    assert p_dict.content_items() == snapshot([{'key': 'val'}])
    assert p_dict.content_items(mode='str') == snapshot(['{"key":"val"}'])
    assert p_dict.content_items(mode='jsonable') == snapshot([{'key': 'val'}])

    p_int = ToolReturnPart(tool_name='t', content=42, tool_call_id='c3')
    assert p_int.content_items() == snapshot([42])
    assert p_int.content_items(mode='str') == snapshot(['42'])
    assert p_int.content_items(mode='jsonable') == snapshot([42])

    p_file = ToolReturnPart(tool_name='t', content=img, tool_call_id='c4')
    assert p_file.content_items(mode='str') == snapshot([ImageUrl(url='https://example.com/img.png')])
    assert p_file.content_items(mode='jsonable') == snapshot([ImageUrl(url='https://example.com/img.png')])

    p_mixed = ToolReturnPart(tool_name='t', content=['text result', img, binary], tool_call_id='c5')
    assert p_mixed.content_items() == snapshot(
        [
            'text result',
            ImageUrl(url='https://example.com/img.png'),
            BinaryContent(data=b'\x89PNG', media_type='image/png'),
        ]
    )
    assert p_mixed.content_items(mode='str') == snapshot(
        [
            'text result',
            ImageUrl(url='https://example.com/img.png'),
            BinaryContent(data=b'\x89PNG', media_type='image/png'),
        ]
    )
    assert p_mixed.content_items(mode='jsonable') == snapshot(
        [
            'text result',
            ImageUrl(url='https://example.com/img.png'),
            BinaryContent(data=b'\x89PNG', media_type='image/png'),
        ]
    )

    p_list = ToolReturnPart(tool_name='t', content=[{'a': 1}, {'b': 2}], tool_call_id='c6')
    assert p_list.content_items(mode='str') == snapshot(['{"a":1}', '{"b":2}'])
    assert p_list.content_items(mode='jsonable') == snapshot([{'a': 1}, {'b': 2}])


def test_tool_return_part_files_property():
    img = ImageUrl(url='https://example.com/img.png')
    audio = AudioUrl(url='https://example.com/audio.mp3')
    binary = BinaryContent(data=b'\x89PNG', media_type='image/png')

    p_str = ToolReturnPart(tool_name='t', content='hello', tool_call_id='c1')
    assert p_str.files == snapshot([])

    p_dict = ToolReturnPart(tool_name='t', content={'key': 'val'}, tool_call_id='c2')
    assert p_dict.files == snapshot([])

    p_file = ToolReturnPart(tool_name='t', content=img, tool_call_id='c3')
    assert p_file.files == snapshot([ImageUrl(url='https://example.com/img.png')])

    p_mixed = ToolReturnPart(tool_name='t', content=['text', img, {'data': 1}, audio, binary], tool_call_id='c4')
    assert p_mixed.files == snapshot(
        [
            ImageUrl(url='https://example.com/img.png'),
            AudioUrl(url='https://example.com/audio.mp3'),
            BinaryContent(data=b'\x89PNG', media_type='image/png'),
        ]
    )

    p_no_files = ToolReturnPart(tool_name='t', content=['a', 'b'], tool_call_id='c5')
    assert p_no_files.files == snapshot([])


def test_tool_return_part_response_methods_with_files():
    img = ImageUrl(url='https://example.com/img.png')

    p_text_file = ToolReturnPart(tool_name='t', content=['hello', img], tool_call_id='c1')
    assert p_text_file.model_response_str() == snapshot('hello')
    assert p_text_file.model_response_object() == snapshot({'return_value': 'hello'})

    p_dict_file = ToolReturnPart(tool_name='t', content=[{'key': 'val'}, img], tool_call_id='c2')
    assert p_dict_file.model_response_str() == snapshot('{"key":"val"}')
    assert p_dict_file.model_response_object() == snapshot({'key': 'val'})

    p_single_list = ToolReturnPart(tool_name='t', content=['hello'], tool_call_id='c3')
    assert p_single_list.model_response_str() == snapshot('["hello"]')
    assert p_single_list.model_response_object() == snapshot({'return_value': ['hello']})

    p_file_only = ToolReturnPart(tool_name='t', content=img, tool_call_id='c4')
    assert p_file_only.model_response_str() == snapshot('')
    assert p_file_only.model_response_object() == snapshot({})

    p_multi = ToolReturnPart(tool_name='t', content=['a', 'b', img], tool_call_id='c5')
    assert p_multi.model_response_str() == snapshot('["a","b"]')
    assert p_multi.model_response_object() == snapshot({'return_value': ['a', 'b']})


def test_tool_return_part_model_response_str_and_user_content():
    img = ImageUrl(url='https://example.com/img.png')

    # Scalar string, no files → fast path returns model_response_str
    p_no_files = ToolReturnPart(tool_name='t', content='hello', tool_call_id='c1')
    text, user_content = p_no_files.model_response_str_and_user_content()
    assert text == snapshot('hello')
    assert user_content == snapshot([])

    # Single-element list, no files → list structure preserved
    p_single_list = ToolReturnPart(tool_name='t', content=['hello'], tool_call_id='c1b')
    text, user_content = p_single_list.model_response_str_and_user_content()
    assert text == snapshot('["hello"]')
    assert user_content == snapshot([])

    # Single text + file → scalar text, not JSON array
    p_text_file = ToolReturnPart(tool_name='t', content=['hello', img], tool_call_id='c2')
    text, user_content = p_text_file.model_response_str_and_user_content()
    assert text == snapshot('["hello","See file d5a901."]')
    assert user_content == snapshot(['This is file d5a901:', ImageUrl(url='https://example.com/img.png')])

    # Multiple text items + file → JSON array preserves list structure
    p_multi = ToolReturnPart(tool_name='t', content=['text1', img, 'text2'], tool_call_id='c3')
    text, user_content = p_multi.model_response_str_and_user_content()
    assert text == snapshot('["text1","See file d5a901.","text2"]')
    assert user_content == snapshot(['This is file d5a901:', ImageUrl(url='https://example.com/img.png')])

    # File-only content
    p_file_only = ToolReturnPart(tool_name='t', content=img, tool_call_id='c4')
    text, user_content = p_file_only.model_response_str_and_user_content()
    assert text == snapshot('See file d5a901.')
    assert user_content == snapshot(['This is file d5a901:', ImageUrl(url='https://example.com/img.png')])

    # Failed content is framed without file references, while files still ride in a user message.
    failed_img = ImageUrl(url='https://example.com/failed.png', identifier='report')
    p_failed = ToolReturnPart(tool_name='t', content=['Disk full', failed_img], tool_call_id='c5', outcome='failed')
    text, user_content = p_failed.model_response_str_and_user_content()
    assert text == snapshot('{"error":"Disk full"}')
    assert user_content == snapshot(
        ['This is file report:', ImageUrl(url='https://example.com/failed.png', identifier='report')]
    )

    text, user_content = p_failed.model_response_str_and_user_content(wrap_if_error=False)
    assert text == snapshot('["Disk full","See file report."]')
    assert user_content == snapshot(
        ['This is file report:', ImageUrl(url='https://example.com/failed.png', identifier='report')]
    )


def test_args_as_dict_valid_json():
    """args_as_dict should return parsed dict for valid JSON args."""
    part = ToolCallPart(tool_name='test_tool', args='{"key": "value"}')
    assert part.args_as_dict() == {'key': 'value'}


def test_args_as_dict_dict_args():
    """args_as_dict should return the dict directly when args is already a dict."""
    part = ToolCallPart(tool_name='test_tool', args={'key': 'value'})
    assert part.args_as_dict() == {'key': 'value'}


def test_args_as_dict_malformed_json_returns_invalid_json_wrapper():
    """args_as_dict should return INVALID_JSON wrapper for malformed JSON by default."""
    malformed = '{"query": "bad", "ids":[4556]</parameter>\n<parameter name="limit": 8}'
    part = ToolCallPart(tool_name='test_tool', args=malformed)
    result = part.args_as_dict()
    assert result == {INVALID_JSON_KEY: malformed}


def test_args_as_dict_non_dict_json_returns_invalid_json_wrapper():
    """args_as_dict should return INVALID_JSON wrapper for valid JSON that's not a dict."""
    json_list = '[1, 2, 3]'
    part = ToolCallPart(tool_name='test_tool', args=json_list)
    assert part.args_as_dict() == {INVALID_JSON_KEY: json_list}


def test_args_as_dict_empty_args():
    """args_as_dict should return {} when args is None/empty."""
    part = ToolCallPart(tool_name='test_tool', args=None)
    assert part.args_as_dict() == {}


def test_args_as_dict_raise_if_invalid_malformed_json():
    """args_as_dict(raise_if_invalid=True) should raise ValueError on malformed JSON."""
    malformed = '{"query": "bad", "ids":[4556]</parameter>\n<parameter name="limit": 8}'
    part = ToolCallPart(tool_name='test_tool', args=malformed)
    with pytest.raises(ValueError):
        part.args_as_dict(raise_if_invalid=True)


def test_args_as_dict_raise_if_invalid_non_dict_json():
    """args_as_dict(raise_if_invalid=True) should raise AssertionError on non-dict JSON."""
    part = ToolCallPart(tool_name='test_tool', args='[1, 2, 3]')
    with pytest.raises(AssertionError):
        part.args_as_dict(raise_if_invalid=True)


def test_user_prompt_part_with_text_content():
    part = UserPromptPart(
        content=[
            'Hi there',
            TextContent(content='This is text content', metadata={'key': 'value'}),
        ]
    )
    assert part.content[0] == 'Hi there'
    assert part.content[1].metadata == snapshot({'key': 'value'})  # type: ignore[reportUnknownMemberType]


class TestInstructionParts:
    def test_join_helper(self):
        """InstructionPart.join produces the correct joined string."""
        parts = [
            InstructionPart(content='First'),
            InstructionPart(content='Second'),
        ]
        assert InstructionPart.join(parts) == 'First\n\nSecond'
        assert InstructionPart.join([]) is None

    def test_join_strips_whitespace(self):
        """InstructionPart.join strips leading/trailing whitespace."""
        parts = [InstructionPart(content='  Hello  ')]
        assert InstructionPart.join(parts) == 'Hello'

    def test_model_request_instructions_is_plain_string(self):
        """ModelRequest.instructions is a plain str | None field."""
        request = ModelRequest(parts=[], instructions='Hello world')
        assert request.instructions == 'Hello world'

    def test_model_request_instructions_default_none(self):
        request = ModelRequest(parts=[])
        assert request.instructions is None

    def test_serialization_round_trip(self):
        """Instructions string survives serialization and deserialization."""
        original = ModelRequest(parts=[UserPromptPart('test')], instructions='static part\n\ndynamic part')

        serialized = ModelMessagesTypeAdapter.dump_json([original])
        deserialized = ModelMessagesTypeAdapter.validate_json(serialized)

        msg = message(deserialized, ModelRequest)
        assert msg.instructions == 'static part\n\ndynamic part'

    def test_repr(self):
        """InstructionPart repr omits default values."""
        part = InstructionPart(content='hello')
        assert repr(part) == "InstructionPart(content='hello')"
        dynamic_part = InstructionPart(content='world', dynamic=True)
        assert repr(dynamic_part) == "InstructionPart(content='world', dynamic=True)"


def test_retry_prompt_strips_input_from_top_level_errors():
    """Top-level validation errors should not include `input` in model_response() since it duplicates the entire generated output."""
    part = RetryPromptPart(
        content=[
            {'type': 'missing', 'loc': ('required_field',), 'msg': 'Field required', 'input': {'wrong_field': 'value'}},
        ],
    )
    response = part.model_response()
    assert '"input"' not in response
    assert '"required_field"' in response


def test_retry_prompt_keeps_input_for_nested_errors():
    """Nested validation errors should keep `input` in model_response() to help the model locate the invalid part."""
    part = RetryPromptPart(
        content=[
            {'type': 'missing', 'loc': ('items', 0, 'sub_field'), 'msg': 'Field required', 'input': {'other': 'val'}},
        ],
    )
    response = part.model_response()
    assert '"input"' in response
    assert '"sub_field"' in response


def test_retry_prompt_mixed_top_level_and_nested_errors():
    """When both top-level and nested errors exist, only top-level input should be stripped."""
    part = RetryPromptPart(
        content=[
            {'type': 'missing', 'loc': ('root_field',), 'msg': 'Field required', 'input': {'root_key': 'root_val'}},
            {
                'type': 'missing',
                'loc': ('items', 0, 'nested_field'),
                'msg': 'Field required',
                'input': {'nested_key': 'nested_val'},
            },
        ],
    )
    response = part.model_response()
    # Nested error's input should be present
    assert '"nested_key"' in response
    # But root-level input should not
    assert '"root_key"' not in response


def test_retry_prompt_strips_input_from_top_level_type_errors():
    """Top-level type/value errors also have input stripped, even though it's a small scalar value."""
    part = RetryPromptPart(
        content=[
            {
                'type': 'int_parsing',
                'loc': ('age',),
                'msg': 'Input should be a valid integer, unable to parse string as an integer',
                'input': 'not_a_number',
            },
        ],
    )
    response = part.model_response()
    assert '"input"' not in response
    assert '"age"' in response


def test_retry_prompt_tool_call_keeps_input_at_top_level():
    """Tool-call retries (`tool_name` set) must preserve `input` so the model sees what args it sent."""
    part = RetryPromptPart(
        tool_name='evaluate_content',
        content=[
            {'type': 'missing', 'loc': ('content',), 'msg': 'Field required', 'input': {}},
        ],
    )
    response = part.model_response()
    assert '"input": {}' in response
    assert '"content"' in response


def test_retry_prompt_tool_call_keeps_input_for_nested_errors():
    """Tool-call retries preserve `input` for nested errors too, matching the existing NativeOutput nested behavior."""
    part = RetryPromptPart(
        tool_name='evaluate_content',
        content=[
            {
                'type': 'string_type',
                'loc': ('items', 0, 'name'),
                'msg': 'Input should be a valid string',
                'input': 42,
            },
        ],
    )
    response = part.model_response()
    assert '"input": 42' in response
    assert '"name"' in response


def test_narrow_type_leaves_claim_free_part_unchanged_on_invalid_data():
    """Best-effort: a kwarg `tool_kind` claim whose data doesn't validate against the typed
    subclass leaves the (claim-free) part untouched instead of raising.

    Not reachable as a unit through one public flow: each part class's lenient branch sits
    behind a different producer (dict-args providers for calls, UI adapters for returns),
    so the four classes are pinned directly here.
    """
    call = ToolCallPart(tool_name='load_capability', args={'name': 'oops'})
    assert ToolCallPart.narrow_type(call, tool_kind='capability-load') is call

    tool_return = ToolReturnPart(tool_name='load_capability', tool_call_id='c1', content='error text')
    assert ToolReturnPart.narrow_type(tool_return, tool_kind='capability-load') is tool_return

    native_call = NativeToolCallPart(tool_name='tool_search', args={'bad': 1})
    assert NativeToolCallPart.narrow_type(native_call, tool_kind='tool-search') is native_call

    native_return = NativeToolReturnPart(tool_name='tool_search', tool_call_id='c2', content='oops')
    assert NativeToolReturnPart.narrow_type(native_return, tool_kind='tool-search') is native_return


def test_narrow_type_strips_unsubstantiated_tool_kind_set_on_part():
    """A `tool_kind` set directly on a part whose data doesn't validate against the typed subclass
    is stripped (rather than left on a base part), across all four part classes.

    Counterpart to the kwarg case above: there the claim is never on the part, here it is, so the
    narrower must actively clear it.
    """
    call = ToolCallPart(tool_name='load_capability', args={'name': 'oops'}, tool_kind='capability-load')
    assert ToolCallPart.narrow_type(call) == replace(call, tool_kind=None)

    tool_return = ToolReturnPart(
        tool_name='load_capability', tool_call_id='c1', content='not-a-dict', tool_kind='capability-load'
    )
    assert ToolReturnPart.narrow_type(tool_return) == replace(tool_return, tool_kind=None)

    native_call = NativeToolCallPart(tool_name='tool_search', args={'bad': 1}, tool_kind='tool-search')
    assert NativeToolCallPart.narrow_type(native_call) == replace(native_call, tool_kind=None)

    native_return = NativeToolReturnPart(
        tool_name='tool_search', tool_call_id='c2', content='oops', tool_kind='tool-search'
    )
    assert NativeToolReturnPart.narrow_type(native_return) == replace(native_return, tool_kind=None)


def test_structured_content_returns_structured_json_or_none():
    """`structured_content` parses a JSON-string `content` into structured data (dict/list), returns
    already-structured content as-is, and yields `None` for anything that isn't structured JSON."""
    assert ToolReturnPart(tool_name='t', tool_call_id='c1', content='{"a": 1}').structured_content() == {'a': 1}
    assert ToolReturnPart(tool_name='t', tool_call_id='c2', content={'a': 1}).structured_content() == {'a': 1}
    assert ToolReturnPart(tool_name='t', tool_call_id='c3', content='[1, 2]').structured_content() == [1, 2]
    # A non-JSON string, a JSON scalar, and a bare scalar all lack structured JSON data.
    assert ToolReturnPart(tool_name='t', tool_call_id='c4', content='not json').structured_content() is None
    assert ToolReturnPart(tool_name='t', tool_call_id='c5', content='"just a string"').structured_content() is None
    assert ToolReturnPart(tool_name='t', tool_call_id='c6', content=42).structured_content() is None


def test_narrow_type_upgrades_json_string_content():
    """A typed return whose content arrives as a JSON string (as UI adapters transmit it) is parsed
    and promoted to its typed subclass with structured content, not left as a base part."""
    tool_return = ToolReturnPart(
        tool_name='load_capability',
        tool_call_id='c1',
        content='{"instructions": "hi"}',
        tool_kind='capability-load',
    )
    narrowed = ToolReturnPart.narrow_type(tool_return)
    assert type(narrowed) is LoadCapabilityReturnPart
    assert narrowed.content == {'instructions': 'hi'}


def test_stripped_tool_kind_part_survives_roundtrip():
    """A base part that kept an unvalidatable `tool_kind` would be routed back to the typed subclass
    by the discriminator and fail validation on reload; stripping it preserves the round-trip."""
    invalid = ToolReturnPart(
        tool_name='load_capability', tool_call_id='c1', content='not-a-dict', tool_kind='capability-load'
    )
    messages: list[ModelMessage] = [ModelRequest(parts=[ToolReturnPart.narrow_type(invalid)])]
    reloaded = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.dump_python(messages))
    assert type(reloaded[0].parts[0]) is ToolReturnPart


def test_narrow_message_parts_promotes_valid_claims_and_leaves_plain_parts():
    """`narrow_message_parts` promotes shape-valid claims to their typed subclass and leaves parts
    without a `tool_kind` untouched (same object), so callers can hand it a whole history."""
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='load_capability', tool_call_id='c1', args={'id': 'foo'}, tool_kind='capability-load'
                ),
                TextPart(content='hello'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='load_capability',
                    tool_call_id='c1',
                    content={'instructions': 'hi'},
                    tool_kind='capability-load',
                )
            ]
        ),
    ]
    narrowed = narrow_message_parts(messages)
    assert type(narrowed[0].parts[0]) is LoadCapabilityCallPart
    assert narrowed[0].parts[1] is messages[0].parts[1]
    assert type(narrowed[1].parts[0]) is LoadCapabilityReturnPart
