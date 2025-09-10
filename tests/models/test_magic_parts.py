import pytest

from pydantic_ai.messages import MagicBinaryContent, MagicDocumentUrl, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel


pytestmark = [pytest.mark.anyio]


async def test_openai_magic_binary_text_plain_inlined_to_text():
    part = UserPromptPart(
        content=[MagicBinaryContent(data=b'Hello world', media_type='text/plain', filename='note.txt')]
    )
    msg = await OpenAIChatModel._map_user_prompt(part)
    assert isinstance(msg["content"], list)
    assert msg["content"][0]["type"] == "text"
    text = msg["content"][0]["text"]
    assert text.startswith('-----BEGIN FILE filename="note.txt" type="text/plain"-----\nHello world')
    assert text.rstrip().endswith('-----END FILE-----')


async def test_openai_magic_binary_pdf_as_file_part():
    part = UserPromptPart(
        content=[MagicBinaryContent(data=b'%PDF-1.4', media_type='application/pdf', filename='file.pdf')]
    )
    msg = await OpenAIChatModel._map_user_prompt(part)
    assert isinstance(msg["content"], list)
    assert msg["content"][0]["type"] == "file"


async def test_openai_magic_document_url_text_plain_inlined_to_text(monkeypatch: pytest.MonkeyPatch):
    # Mock download_item to avoid network
    async def fake_download_item(item: DocumentUrl, data_format: str = 'text', type_format: str = 'extension'):
        return {'data': 'Hello from URL', 'data_type': 'txt'}

    monkeypatch.setattr('pydantic_ai.models.openai.download_item', fake_download_item)

    part = UserPromptPart(content=[MagicDocumentUrl(url='https://example.com/file.txt', filename='from-url.txt')])
    msg = await OpenAIChatModel._map_user_prompt(part)
    text = msg["content"][0]["text"]
    assert text.startswith('-----BEGIN FILE filename="from-url.txt" type="text/plain"-----\nHello from URL')
    assert text.rstrip().endswith('-----END FILE-----')


async def test_openai_magic_document_url_pdf_as_file_part(monkeypatch: pytest.MonkeyPatch):
    async def fake_download_item(item, data_format='base64_uri', type_format='extension'):
        return {'data': 'data:application/pdf;base64,AAA', 'data_type': 'pdf'}

    monkeypatch.setattr('pydantic_ai.models.openai.download_item', fake_download_item)

    part = UserPromptPart(content=[MagicDocumentUrl(url='https://example.com/file.pdf')])
    msg = await OpenAIChatModel._map_user_prompt(part)
    assert isinstance(msg["content"], list)
    assert msg["content"][0]["type"] == "file"

