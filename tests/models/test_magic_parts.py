"""Tests for Magic* parts provider handling.

pyright: reportPrivateUsage=false
"""

import pytest
from typing import Any, cast

from pydantic_ai.messages import MagicBinaryContent, MagicDocumentUrl, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel


pytestmark = [pytest.mark.anyio]


async def test_openai_magic_binary_text_plain_inlined_to_text():
    part = UserPromptPart(
        content=[MagicBinaryContent(data=b'Hello world', media_type='text/plain', filename='note.txt')]
    )
    msg = await OpenAIChatModel._map_user_prompt(part)
    assert isinstance(msg["content"], list)
    content = cast(list[dict[str, Any]], msg["content"])  # type: ignore[assignment]
    assert content[0]["type"] == "text"
    text = cast(str, content[0]["text"])  # type: ignore[index]
    assert text.startswith('-----BEGIN FILE filename="note.txt" type="text/plain"-----\nHello world')
    assert text.rstrip().endswith('-----END FILE-----')


async def test_openai_magic_binary_pdf_as_file_part():
    part = UserPromptPart(
        content=[MagicBinaryContent(data=b'%PDF-1.4', media_type='application/pdf', filename='file.pdf')]
    )
    msg = await OpenAIChatModel._map_user_prompt(part)
    assert isinstance(msg["content"], list)
    content = cast(list[dict[str, Any]], msg["content"])  # type: ignore[assignment]
    assert content[0]["type"] == "file"


async def test_openai_magic_document_url_text_plain_inlined_to_text(monkeypatch: pytest.MonkeyPatch):
    # Mock download_item to avoid network
    async def fake_download_item(
        item: MagicDocumentUrl, data_format: str = 'text', type_format: str = 'extension'
    ) -> dict[str, str]:
        return {'data': 'Hello from URL', 'data_type': 'txt'}

    monkeypatch.setattr('pydantic_ai.models.openai.download_item', fake_download_item)

    part = UserPromptPart(content=[MagicDocumentUrl(url='https://example.com/file.txt', filename='from-url.txt')])
    msg = await OpenAIChatModel._map_user_prompt(part)
    content = cast(list[dict[str, Any]], msg["content"])  # type: ignore[assignment]
    text = cast(str, content[0]["text"])  # type: ignore[index]
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
    assert isinstance(msg["content"], list)
    content = cast(list[dict[str, Any]], msg["content"])  # type: ignore[assignment]
    assert content[0]["type"] == "file"

