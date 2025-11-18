import sys
from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    TextPart,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    UserPromptPart,
    VideoUrl,
)

from .conftest import IsDatetime, IsNow, IsStr


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
    with pytest.raises(ValueError, match='Could not infer media type from audio URL: foobar.potato'):
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
    with pytest.raises(ValueError, match='Could not infer media type from image URL: foobar.potato'):
        ImageUrl('foobar.potato').media_type

    with pytest.raises(ValueError, match='Could not infer media type from image URL: foobar.potato'):
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
    with pytest.raises(ValueError, match='Could not infer media type from document URL: foobar.potato'):
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
    with pytest.raises(ValueError, match='Could not infer media type from video URL: foobar.potato'):
        VideoUrl('foobar.potato').media_type


def test_thinking_part_delta_apply_to_thinking_part_delta():
    """Test lines 768-775: Apply ThinkingPartDelta to another ThinkingPartDelta."""
    original_delta = ThinkingPartDelta(
        content_delta='original', signature_delta='sig1', provider_name='original_provider'
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
                ]
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


def test_file_part_has_content():
    filepart = FilePart(content=BinaryContent(data=b'', media_type='application/pdf'))
    assert not filepart.has_content()

    filepart.content.data = b'not empty'
    assert filepart.has_content()


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
                'provider_details': None,
                'provider_response_id': None,
                'finish_reason': None,
                'run_id': None,
                'metadata': None,
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


def test_model_response_convenience_methods():
    response = ModelResponse(parts=[])
    assert response.text == snapshot(None)
    assert response.thinking == snapshot(None)
    assert response.files == snapshot([])
    assert response.images == snapshot([])
    assert response.tool_calls == snapshot([])
    assert response.builtin_tool_calls == snapshot([])

    response = ModelResponse(
        parts=[
            ThinkingPart(content="Let's generate an image"),
            ThinkingPart(content="And then, call the 'hello_world' tool"),
            TextPart(content="I'm going to"),
            TextPart(content=' generate an image'),
            BuiltinToolCallPart(tool_name='image_generation', args={}, tool_call_id='123'),
            FilePart(content=BinaryImage(data=b'fake', media_type='image/jpeg')),
            BuiltinToolReturnPart(tool_name='image_generation', content={}, tool_call_id='123'),
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
    assert response.builtin_tool_calls == snapshot(
        [
            (
                BuiltinToolCallPart(tool_name='image_generation', args={}, tool_call_id='123'),
                BuiltinToolReturnPart(
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
