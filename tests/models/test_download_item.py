import re
from collections.abc import Callable
from unittest.mock import patch

import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, AudioUrl, DocumentUrl, ImageUrl, VideoUrl
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters, UserError, download_item
from pydantic_ai.models.batch import BatchRequest

from ..conftest import IsInstance, IsStr, try_import

with try_import() as openai_imports_successful:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI, completion_message
    from .test_openai import text_chunk

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider


def _mock_downloads(
    monkeypatch: pytest.MonkeyPatch, handle_request: Callable[[httpx2.Request], httpx2.Response]
) -> None:
    def create_http_client(*, timeout: int) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle_request))

    monkeypatch.setattr('pydantic_ai._ssrf.create_async_httpx2_client', create_http_client)


@pytest.mark.parametrize(
    ('url', 'protocol'),
    (
        pytest.param(AudioUrl(url='gs://pydantic-ai-dev/openai-alloy.wav', force_download=True), 'gs', id='gs-audio'),
        pytest.param(
            DocumentUrl(url='gs://pydantic-ai-dev/Gemini_1_5_Pro_Technical_Report_Arxiv_1805.pdf', force_download=True),
            'gs',
            id='gs-document',
        ),
        pytest.param(
            ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png', force_download=True), 'gs', id='gs-image'
        ),
        pytest.param(
            VideoUrl(url='gs://pydantic-ai-dev/grepit-tiny-video.mp4', force_download=True), 'gs', id='gs-video'
        ),
        pytest.param(AudioUrl(url='s3://my-bucket/audio.wav', force_download=True), 's3', id='s3-audio'),
        pytest.param(DocumentUrl(url='s3://my-bucket/document.pdf', force_download=True), 's3', id='s3-document'),
        pytest.param(ImageUrl(url='s3://my-bucket/image.png', force_download=True), 's3', id='s3-image'),
        pytest.param(VideoUrl(url='s3://my-bucket/video.mp4', force_download=True), 's3', id='s3-video'),
        pytest.param(DocumentUrl(url='file:///etc/passwd', force_download=True), 'file', id='file-document'),
        pytest.param(ImageUrl(url='ftp://ftp.example.com/image.png', force_download=True), 'ftp', id='ftp-image'),
    ),
)
async def test_download_item_raises_user_error_with_unsupported_protocol(
    url: AudioUrl | DocumentUrl | ImageUrl | VideoUrl,
    protocol: str,
) -> None:
    with pytest.raises(ValueError, match=f'URL protocol "{protocol}" is not allowed'):
        _ = await download_item(url, data_format='bytes')


async def test_download_item_raises_user_error_with_youtube_url() -> None:
    with pytest.raises(UserError, match=re.escape('Downloading YouTube videos is not supported.')):
        _ = await download_item(VideoUrl(url='https://youtu.be/lCdaVNyHtjU'), data_format='bytes')


@pytest.mark.parametrize(
    'url',
    (
        ImageUrl(url='https://93.184.215.14/image.png', media_type='image/png'),
        DocumentUrl(url='https://93.184.215.14/doc.pdf', media_type='application/pdf'),
        VideoUrl(url='https://93.184.215.14/video.mp4', media_type='video/mp4'),
        AudioUrl(url='https://93.184.215.14/audio.mp3', media_type='audio/mpeg'),
    ),
)
async def test_download_item_rejects_oversized_body(
    url: AudioUrl | DocumentUrl | ImageUrl | VideoUrl,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File URL downloads reject response bodies larger than the 50 MiB default limit."""

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=b'x' * 1024,
            headers={'content-type': url.media_type},
            request=request,
        )

    _mock_downloads(monkeypatch, handle_request)

    with (
        patch('pydantic_ai.models._MAX_FILE_URL_DOWNLOAD_BYTES', 512),
        pytest.raises(ValueError, match='maximum size of 512 bytes'),
    ):
        await download_item(url, data_format='bytes')


def _serve_text_files(monkeypatch: pytest.MonkeyPatch, size: int) -> list[str]:
    """Serve a `size`-byte text file at every URL, and return the URLs downloaded, in order."""
    downloaded: list[str] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        downloaded.append(str(request.url))
        return httpx2.Response(200, content=b'x' * size, headers={'content-type': 'text/plain'}, request=request)

    _mock_downloads(monkeypatch, handle_request)
    return downloaded


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.parametrize('stream', [False, True])
async def test_file_url_downloads_share_one_total_per_model_request(
    stream: bool, allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model request refuses the file URL that takes its downloads past the per-request total.

    `OpenAIChatModel` downloads every document URL, so each file here is fetched while the request is built.
    """
    downloaded = _serve_text_files(monkeypatch, 1024)
    client = (
        MockOpenAI.create_mock_stream([text_chunk('Done.')])
        if stream
        else MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='Done.', role='assistant')))
    )
    agent = Agent(OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client)))
    prompt = ['Summarize these.', *(DocumentUrl(url=f'https://93.184.215.14/doc{i}.txt') for i in range(3))]

    with (
        patch('pydantic_ai.models._MAX_REQUEST_FILE_URL_DOWNLOAD_BYTES', 2560),
        pytest.raises(
            ValueError,
            match=r'^File URL downloads for one model request exceed the total of 2560 bytes; pass large files as `BinaryContent` instead\.$',
        ),
    ):
        if stream:
            async with agent.run_stream(prompt):
                pass
        else:
            await agent.run(prompt)

    assert downloaded == snapshot(
        ['https://93.184.215.14/doc0.txt', 'https://93.184.215.14/doc1.txt', 'https://93.184.215.14/doc2.txt']
    )


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_file_url_over_the_per_file_limit_in_a_model_request(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside a model request, a file over the per-file limit raises the per-file error, not the total's."""
    _serve_text_files(monkeypatch, 1024)
    client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='Done.', role='assistant')))
    agent = Agent(OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client)))

    with (
        patch('pydantic_ai.models._MAX_FILE_URL_DOWNLOAD_BYTES', 512),
        pytest.raises(ValueError, match=r'^Download exceeds the maximum size of 512 bytes\.$'),
    ):
        await agent.run(['Summarize this.', DocumentUrl(url='https://93.184.215.14/doc.txt')])


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_each_model_request_has_its_own_download_total(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second request of a run downloads its history's file URLs again, within a total of its own."""
    downloaded = _serve_text_files(monkeypatch, 1024)
    client = MockOpenAI.create_mock(
        [
            completion_message(
                ChatCompletionMessage(
                    content=None,
                    role='assistant',
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id='1', function=Function(arguments='{}', name='get_date'), type='function'
                        )
                    ],
                )
            ),
            completion_message(ChatCompletionMessage(content='Done.', role='assistant')),
        ]
    )
    agent = Agent(OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client)))

    @agent.tool_plain
    def get_date() -> str:
        return '2026-11-09'

    prompt = ['Summarize these.', *(DocumentUrl(url=f'https://93.184.215.14/doc{i}.txt') for i in range(2))]
    with patch('pydantic_ai.models._MAX_REQUEST_FILE_URL_DOWNLOAD_BYTES', 2048):
        result = await agent.run(prompt)

    assert result.output == 'Done.'
    assert len(downloaded) == 4


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.parametrize('method', ['count_tokens', 'batch_encode', 'compact_messages'])
async def test_counting_batching_or_compacting_shares_one_download_total(
    method: str, allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`count_tokens`, `batch_encode` and `compact_messages` each build a request, whose downloads share one total too.

    `AnthropicModel` downloads `text/plain` document URLs; the request fails while it is built, before any API call.
    """
    downloaded = _serve_text_files(monkeypatch, 1024)
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(api_key='mock-api-key'))
    urls = [DocumentUrl(url=f'https://93.184.215.14/doc{i}.txt', media_type='text/plain') for i in range(3)]
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['Summarize these.', *urls])])]

    with (
        patch('pydantic_ai.models._MAX_REQUEST_FILE_URL_DOWNLOAD_BYTES', 2560),
        pytest.raises(
            ValueError,
            match=r'^File URL downloads for one model request exceed the total of 2560 bytes; pass large files as `BinaryContent` instead\.$',
        ),
    ):
        if method == 'count_tokens':
            await model.count_tokens(messages, None, ModelRequestParameters())
        elif method == 'batch_encode':
            await model.batch_encode(BatchRequest(custom_id='item', messages=messages))
        else:
            await model.compact_messages(
                ModelRequestContext(
                    model=model,
                    messages=messages,
                    model_settings=None,
                    model_request_parameters=ModelRequestParameters(),
                )
            )

    assert len(downloaded) == 3


async def test_download_item_outside_a_model_request_has_no_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `download_item` call made outside a model request is limited per file only."""
    _serve_text_files(monkeypatch, 1024)

    with patch('pydantic_ai.models._MAX_REQUEST_FILE_URL_DOWNLOAD_BYTES', 1024):
        for i in range(3):
            downloaded = await download_item(DocumentUrl(url=f'https://93.184.215.14/doc{i}.txt'), data_format='bytes')
            assert len(downloaded['data']) == 1024


def _round_trip(item: ImageUrl) -> ImageUrl:
    """Pass `item` through message history serialization, which stores its inferred `media_type`."""
    messages = ModelMessagesTypeAdapter.validate_json(
        ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart(content=[item])])])
    )
    request = messages[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart) and not isinstance(part.content, str)
    round_tripped = part.content[0]
    assert isinstance(round_tripped, ImageUrl)
    return round_tripped


@pytest.mark.parametrize(
    ('url', 'headers', 'expected_media_type'),
    (
        pytest.param(
            ImageUrl(url='https://93.184.215.14/receipt-123', media_type='image/png'),
            {'content-type': 'binary/octet-stream'},
            'image/png',
            id='explicit-over-binary-octet-stream',
        ),
        pytest.param(
            ImageUrl(url='https://93.184.215.14/image.jpg', media_type='image/png'),
            {'content-type': 'image/jpeg'},
            'image/png',
            id='explicit-over-specific-header',
        ),
        pytest.param(
            ImageUrl(url='https://93.184.215.14/image.jpg', media_type='image/jpeg'),
            {'content-type': 'image/webp'},
            'image/webp',
            id='explicit-matching-url-under-specific-header',
        ),
        pytest.param(
            _round_trip(ImageUrl(url='https://93.184.215.14/image.jpg')),
            {'content-type': 'image/webp'},
            'image/webp',
            id='round-tripped-inferred-under-specific-header',
        ),
        pytest.param(
            _round_trip(ImageUrl(url='https://93.184.215.14/receipt-123', media_type='image/png')),
            {'content-type': 'binary/octet-stream'},
            'image/png',
            id='round-tripped-explicit-over-binary-octet-stream',
        ),
        pytest.param(
            ImageUrl(url='https://93.184.215.14/image.png', media_type=''),
            {'content-type': 'image/jpeg'},
            'image/jpeg',
            id='empty-explicit-ignored',
        ),
        pytest.param(
            ImageUrl(url='https://93.184.215.14/image.png'),
            {'content-type': 'Binary/Octet-Stream '},
            'image/png',
            id='binary-octet-stream-mixed-case-inferred',
        ),
        pytest.param(
            ImageUrl(url='https://93.184.215.14/image.png'),
            {'content-type': 'binary/octet-stream'},
            'image/png',
            id='binary-octet-stream-inferred',
        ),
        pytest.param(
            ImageUrl(url='https://93.184.215.14/image.png'),
            {'content-type': 'application/octet-stream; charset=binary'},
            'image/png',
            id='application-octet-stream-inferred',
        ),
        pytest.param(ImageUrl(url='https://93.184.215.14/image.png'), {}, 'image/png', id='missing-header-inferred'),
        pytest.param(
            ImageUrl(url='https://93.184.215.14/image.png'),
            {'content-type': 'image/jpeg; charset=binary'},
            'image/jpeg',
            id='header-over-inferred',
        ),
    ),
)
async def test_download_item_media_type_precedence(
    url: ImageUrl,
    headers: dict[str, str],
    expected_media_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit `media_type` wins over the `content-type` header, which wins over URL inference unless it's generic.

    A mock transport serves each header, since a recording can't pair every server header with every URL.
    """

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b'\x89PNG', headers=headers, request=request)

    _mock_downloads(monkeypatch, handle_request)

    downloaded_item = await download_item(url, data_format='base64_uri')
    assert downloaded_item == {
        'data': f'data:{expected_media_type};base64,iVBORw==',
        'data_type': expected_media_type,
    }


@pytest.mark.vcr()
async def test_download_item_application_octet_stream(disable_ssrf_protection_for_vcr: None) -> None:
    downloaded_item = await download_item(
        VideoUrl(
            url='https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/tests/assets/small_video.mp4'
        ),
        data_format='bytes',
    )
    assert downloaded_item['data_type'] == 'video/mp4'
    assert downloaded_item['data'] == IsInstance(bytes)


@pytest.mark.vcr()
async def test_download_item_audio_mpeg(disable_ssrf_protection_for_vcr: None) -> None:
    downloaded_item = await download_item(
        AudioUrl(url='https://smokeshow.helpmanual.io/4l1l1s0s6q4741012x1w/common_voice_en_537507.mp3'),
        data_format='bytes',
    )
    assert downloaded_item['data_type'] == 'audio/mpeg'
    assert downloaded_item['data'] == IsInstance(bytes)


@pytest.mark.vcr()
async def test_download_item_no_content_type(disable_ssrf_protection_for_vcr: None) -> None:
    downloaded_item = await download_item(
        DocumentUrl(url='https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/docs/help.md'),
        data_format='text',
    )
    assert downloaded_item['data_type'] == 'text/markdown'
    assert downloaded_item['data'] == IsStr()
