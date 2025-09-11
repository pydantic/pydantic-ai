# pyright: reportPrivateUsage=false, reportMissingImports=false, reportUnnecessaryTypeIgnoreComment=false
"""Tests for Magic* parts provider handling."""

from typing import Any, cast

import pytest

from pydantic_ai.messages import MagicBinaryContent, MagicDocumentUrl, UserPromptPart

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


async def test_openai_magic_binary_text_plain_inlined_to_text():
    part = UserPromptPart(
        content=[MagicBinaryContent(data=b'Hello world', media_type='text/plain', filename='note.txt')]
    )
    msg = await OpenAIChatModel._map_user_prompt(part)
    assert isinstance(msg['content'], list)
    content = cast(list[dict[str, Any]], msg['content'])
    assert content[0]['type'] == 'text'
    text = cast(str, content[0]['text'])
    assert text.startswith('-----BEGIN FILE filename="note.txt" type="text/plain"-----\nHello world')
    assert text.rstrip().endswith('-----END FILE-----')


async def test_openai_magic_binary_pdf_as_file_part():
    part = UserPromptPart(
        content=[MagicBinaryContent(data=b'%PDF-1.4', media_type='application/pdf', filename='file.pdf')]
    )
    msg = await OpenAIChatModel._map_user_prompt(part)
    assert isinstance(msg['content'], list)
    content = cast(list[dict[str, Any]], msg['content'])
    assert content[0]['type'] == 'file'


async def test_openai_magic_document_url_text_plain_inlined_to_text(monkeypatch: pytest.MonkeyPatch):
    # Mock download_item to avoid network
    async def fake_download_item(
        item: MagicDocumentUrl, data_format: str = 'text', type_format: str = 'extension'
    ) -> dict[str, str]:
        return {'data': 'Hello from URL', 'data_type': 'txt'}

    monkeypatch.setattr('pydantic_ai.models.openai.download_item', fake_download_item)

    part = UserPromptPart(content=[MagicDocumentUrl(url='https://example.com/file.txt', filename='from-url.txt')])
    msg = await OpenAIChatModel._map_user_prompt(part)
    content = cast(list[dict[str, Any]], msg['content'])
    text = cast(str, content[0]['text'])
    assert text.startswith('-----BEGIN FILE filename="from-url.txt" type="text/plain"-----\nHello from URL')
    assert text.rstrip().endswith('-----END FILE-----')


async def test_openai_magic_document_url_pdf_as_file_part(monkeypatch: pytest.MonkeyPatch):
    async def fake_download_item(
        item: MagicDocumentUrl, data_format: str = 'base64_uri', type_format: str = 'extension'
    ) -> dict[str, str]:
        return {'data': 'data:application/pdf;base64,AAA', 'data_type': 'pdf'}

    monkeypatch.setattr('pydantic_ai.models.openai.download_item', fake_download_item)

    part = UserPromptPart(content=[MagicDocumentUrl(url='https://example.com/file.pdf')])
    msg = await OpenAIChatModel._map_user_prompt(part)
    assert isinstance(msg['content'], list)
    content = cast(list[dict[str, Any]], msg['content'])
    assert content[0]['type'] == 'file'


@pytest.mark.parametrize(
    'media_type,expected_type,body',
    [
        ('text/csv', 'text', 'a,b\n1,2'),
        ('application/json', 'text', '{"a":1}'),
        ('application/x-yaml', 'text', 'a: 1'),
        ('application/xml', 'text', '<a>1</a>'),
        ('text/plain', 'text', 'hello'),
    ],
)
async def test_openai_magic_binary_text_like_media_types_inlined(media_type: str, expected_type: str, body: str):
    part = UserPromptPart(content=[MagicBinaryContent(data=body.encode(), media_type=media_type, filename='f')])
    msg = await OpenAIChatModel._map_user_prompt(part)
    content = cast(list[dict[str, Any]], msg['content'])
    assert content[0]['type'] == expected_type
    text = cast(str, content[0]['text'])
    assert f'type="{media_type}"' in text


async def test_openai_map_single_item_unknown_returns_empty():
    parts = await OpenAIChatModel._map_single_item(object())
    assert parts == []


async def test_openai_magic_binary_audio_maps_to_input_audio():
    parts = await OpenAIChatModel._map_single_item(MagicBinaryContent(data=b'abc', media_type='audio/mpeg'))
    assert len(parts) == 1
    assert parts[0]['type'] == 'input_audio'


async def test_openai_binary_unsupported_returns_empty():
    from pydantic_ai.messages import BinaryContent

    parts = await OpenAIChatModel._map_single_item(BinaryContent(data=b'data', media_type='application/octet-stream'))
    assert parts == []


async def test_openai_document_url_json_inlined_text(monkeypatch: pytest.MonkeyPatch):
    from pydantic_ai.messages import DocumentUrl

    async def fake_download_item(
        item: DocumentUrl, data_format: str = 'base64_uri', type_format: str = 'extension'
    ) -> dict[str, str]:
        return {'data': 'data:application/json;base64,ewogICJhIjogMQp9', 'data_type': 'json'}

    monkeypatch.setattr('pydantic_ai.models.openai.download_item', fake_download_item)

    parts = await OpenAIChatModel._map_single_item(
        MagicDocumentUrl(url='https://example.com/file.json', media_type='application/json')
    )
    assert len(parts) == 1
    assert parts[0]['type'] == 'text'


async def test_openai_regular_binary_image_maps_to_image_url():
    from pydantic_ai.messages import BinaryContent

    parts = await OpenAIChatModel._map_single_item(BinaryContent(data=b'\x89PNG', media_type='image/png'))
    assert len(parts) == 1
    assert parts[0]['type'] == 'image_url'


async def test_openai_audio_url_maps_to_input_audio(monkeypatch: pytest.MonkeyPatch):
    from pydantic_ai.messages import AudioUrl

    async def fake_download_item(
        item: AudioUrl, data_format: str = 'base64', type_format: str = 'extension'
    ) -> dict[str, str]:
        return {'data': 'AAA=', 'data_type': 'mp3'}

    monkeypatch.setattr('pydantic_ai.models.openai.download_item', fake_download_item)

    parts = await OpenAIChatModel._map_single_item(AudioUrl(url='https://example.com/file.mp3'))
    assert len(parts) == 1
    assert parts[0]['type'] == 'input_audio'


async def test_openai_document_url_pdf_file_path(monkeypatch: pytest.MonkeyPatch):
    from pydantic_ai.messages import DocumentUrl

    async def fake_download_item(
        item: DocumentUrl, data_format: str = 'base64_uri', type_format: str = 'extension'
    ) -> dict[str, str]:
        return {'data': 'data:application/pdf;base64,AAA', 'data_type': 'pdf'}

    monkeypatch.setattr('pydantic_ai.models.openai.download_item', fake_download_item)

    parts = await OpenAIChatModel._map_single_item(DocumentUrl(url='https://example.com/file.pdf'))
    assert len(parts) == 1
    assert parts[0]['type'] == 'file'


async def test_openai_video_url_raises_not_implemented():
    from pydantic_ai.messages import VideoUrl

    with pytest.raises(NotImplementedError):
        await OpenAIChatModel._map_single_item(VideoUrl(url='https://example.com/file.mp4'))


async def test_openai_image_url_maps_to_image_url_part():
    # Functional: verify that an ImageUrl input maps to an image_url content part
    from pydantic_ai.messages import ImageUrl

    part = UserPromptPart(content=[ImageUrl(url='https://example.com/picture.png')])
    msg = await OpenAIChatModel._map_user_prompt(part)
    content = cast(list[dict[str, Any]], msg['content'])
    assert content[0]['type'] == 'image_url'


async def test_openai_magic_binary_image_maps_to_image_url():
    # Functional: MagicBinaryContent with image/* should map to image_url content part
    part = UserPromptPart(content=[MagicBinaryContent(data=b'\x89PNG', media_type='image/png')])
    msg = await OpenAIChatModel._map_user_prompt(part)
    content = cast(list[dict[str, Any]], msg['content'])
    assert content[0]['type'] == 'image_url'
