"""Tests for `XaiModel` reusing the xAI file IDs of documents it already uploaded."""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Any, cast

import httpx2
import pytest

from pydantic_ai import Agent, BinaryContent, DocumentUrl, UploadedFile

from ...conftest import try_import
from ..conftest import mock_downloads
from ..mock_xai import MockXai, create_response, get_mock_chat_create_kwargs

with try_import() as imports_successful:
    from xai_sdk import AsyncClient

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.vcr,
]

XAI_NON_REASONING_MODEL = 'grok-4-fast-non-reasoning'


@dataclass
class CountingMockXai(MockXai):
    """A MockXai that records document uploads and gives each one a distinct file ID (`file-1`, `file-2`, ...)."""

    upload_calls: list[tuple[bytes, str]] = field(default_factory=list[tuple[bytes, str]])

    async def files_upload(self, data: bytes, filename: str) -> Any:
        self.upload_calls.append((data, filename))
        return type('UploadedFile', (), {'id': f'file-{len(self.upload_calls)}'})()


def _sent_file_ids(client: AsyncClient) -> list[list[str]]:
    """The file IDs each chat request referenced, in message order."""
    return [
        [
            content['file']['file_id']
            for message in kwargs['messages']
            for content in message['content']
            if 'file' in content
        ]
        for kwargs in get_mock_chat_create_kwargs(client)
    ]


async def test_binary_content_document_uploaded_once_per_instance(allow_model_requests: None):
    """The same BinaryContent document is uploaded only once per model instance."""
    response = create_response(content='ok')
    counting_client = CountingMockXai(responses=[response, response, response])
    mock_client = cast(AsyncClient, counting_client)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)
    document_content = BinaryContent(data=b'%PDF-1.4\nTest document content', media_type='application/pdf')

    result = await agent.run(['What is in this document?', document_content])
    assert result.output == 'ok'
    assert counting_client.upload_calls == [(document_content.data, document_content.identifier)]

    for prompt in ['Summarize it again.', 'And once more.']:
        result = await agent.run(prompt, message_history=result.all_messages())
        assert result.output == 'ok'
    assert counting_client.upload_calls == [(document_content.data, document_content.identifier)]
    assert _sent_file_ids(mock_client) == [['file-1'], ['file-1'], ['file-1']]

    # Another model instance (possibly another xAI account) does not reuse this instance's file ID.
    other_client = CountingMockXai(responses=[response])
    other_model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=cast(AsyncClient, other_client)))
    result = await Agent(other_model).run('Summarize it again.', message_history=result.all_messages())
    assert result.output == 'ok'
    assert other_client.upload_calls == [(document_content.data, document_content.identifier)]


@pytest.mark.vcr(ignore_hosts=['93.184.215.14'])
async def test_document_url_uploaded_once_per_instance(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch):
    """A DocumentUrl replayed across requests is downloaded every time but uploaded only once per model instance.

    The download runs on every request so each one is checked against its own `force_download` permission;
    the upload is skipped because the downloaded bytes match an earlier upload.
    """
    response = create_response(content='ok')
    counting_client = CountingMockXai(responses=[response, response, response])
    mock_client = cast(AsyncClient, counting_client)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)
    download_calls: list[str] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        download_calls.append(str(request.url))
        return httpx2.Response(
            200, content=b'%PDF-1.4\ndownloaded content', headers={'content-type': 'application/pdf'}, request=request
        )

    mock_downloads(monkeypatch, handle_request)
    document_url = DocumentUrl(url='https://93.184.215.14/report.pdf')

    result = await agent.run(['What is in this document?', document_url])
    assert result.output == 'ok'
    assert counting_client.upload_calls == [(b'%PDF-1.4\ndownloaded content', f'{document_url.identifier}.pdf')]

    for prompt in ['Summarize it again.', 'And once more.']:
        result = await agent.run(prompt, message_history=result.all_messages())
        assert result.output == 'ok'
    assert counting_client.upload_calls == [(b'%PDF-1.4\ndownloaded content', f'{document_url.identifier}.pdf')]
    assert download_calls == ['https://93.184.215.14/report.pdf'] * 3
    assert _sent_file_ids(mock_client) == [['file-1'], ['file-1'], ['file-1']]


@pytest.mark.vcr(ignore_hosts=['127.0.0.1'])
async def test_document_url_cache_does_not_bypass_download_permissions(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
):
    """A local document uploaded under `force_download='allow-local'` is not reused by a request without that permission."""
    counting_client = CountingMockXai(responses=[create_response(content='ok')])
    mock_client = cast(AsyncClient, counting_client)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, content=b'%PDF-1.4\ninternal content', headers={'content-type': 'application/pdf'}, request=request
        )

    mock_downloads(monkeypatch, handle_request)

    allowed = DocumentUrl(url='http://127.0.0.1/private.pdf', force_download='allow-local', identifier='private')
    result = await agent.run(['What is in this document?', allowed])
    assert result.output == 'ok'
    assert counting_client.upload_calls == [(b'%PDF-1.4\ninternal content', 'private.pdf')]

    with pytest.raises(ValueError, match=r'Access to private/internal IP address \(127\.0\.0\.1\) is blocked'):
        await agent.run(['What is in this document?', DocumentUrl(url='http://127.0.0.1/private.pdf')])
    assert counting_client.upload_calls == [(b'%PDF-1.4\ninternal content', 'private.pdf')]
    assert _sent_file_ids(mock_client) == [['file-1']]


async def test_uploaded_file_passthrough_never_uploads(allow_model_requests: None):
    """A user-supplied UploadedFile is passed through without uploading."""
    response = create_response(content='ok')
    counting_client = CountingMockXai(responses=[response])
    mock_client = cast(AsyncClient, counting_client)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run(['What is in this document?', UploadedFile(file_id='file-abc123', provider_name='xai')])
    assert result.output == 'ok'
    assert counting_client.upload_calls == []


async def test_distinct_documents_upload_separately(allow_model_requests: None):
    """Distinct BinaryContent documents are uploaded separately."""
    response = create_response(content='ok')
    counting_client = CountingMockXai(responses=[response, response])
    mock_client = cast(AsyncClient, counting_client)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)
    document_a = BinaryContent(data=b'%PDF-1.4\nDocument A', media_type='application/pdf')
    document_b = BinaryContent(data=b'%PDF-1.4\nDocument B', media_type='application/pdf')

    result = await agent.run(['What is in this document?', document_a])
    assert result.output == 'ok'
    assert counting_client.upload_calls == [(document_a.data, document_a.identifier)]

    result = await agent.run(['And the other one?', document_b], message_history=result.new_messages())
    assert result.output == 'ok'
    assert counting_client.upload_calls == [
        (document_a.data, document_a.identifier),
        (document_b.data, document_b.identifier),
    ]
    assert _sent_file_ids(mock_client) == [['file-1'], ['file-1', 'file-2']]
