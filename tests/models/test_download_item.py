import re
from unittest.mock import patch

import httpx
import pytest

from pydantic_ai import AudioUrl, DocumentUrl, ImageUrl, VideoUrl
from pydantic_ai.models import UserError, download_item

from ..conftest import IsInstance, IsStr

pytestmark = [pytest.mark.anyio]

# Minimal byte fixtures: the image sniffer only inspects the leading magic-byte signature.
_HTML_BYTES = b'<!doctype html><html><body><h1>not an image</h1></body></html>'
_PNG_BYTES = b'\x89PNG\r\n\x1a\n' + b'\x00' * 16
_JPEG_BYTES = b'\xff\xd8\xff\xe0' + b'\x00' * 16
_GIF_BYTES = b'GIF89a' + b'\x00' * 16
_WEBP_BYTES = b'RIFF' + b'\x00\x00\x00\x00' + b'WEBP' + b'\x00' * 8


def _install_mock_download(monkeypatch: pytest.MonkeyPatch, *, content: bytes, content_type: str | None) -> None:
    """Route `download_item`'s HTTP client at a mock returning `content` and `content_type`."""

    def handle_request(request: httpx.Request) -> httpx.Response:
        headers = {'content-type': content_type} if content_type is not None else {}
        return httpx.Response(200, content=content, headers=headers, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))

    def create_http_client(*, timeout: int) -> httpx.AsyncClient:
        return http_client

    monkeypatch.setattr('pydantic_ai._ssrf.create_async_http_client', create_http_client)


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

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'x' * 1024,
            headers={'content-type': url.media_type},
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))

    def create_http_client(*, timeout: int) -> httpx.AsyncClient:
        return http_client

    monkeypatch.setattr('pydantic_ai._ssrf.create_async_http_client', create_http_client)

    with (
        patch('pydantic_ai.models._MAX_FILE_URL_DOWNLOAD_BYTES', 512),
        pytest.raises(ValueError, match='maximum size of 512 bytes'),
    ):
        await download_item(url, data_format='bytes')


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


@pytest.mark.parametrize(
    ('content', 'content_type', 'expected_media_type'),
    (
        pytest.param(_PNG_BYTES, 'image/png', 'image/png', id='honest-png'),
        pytest.param(_JPEG_BYTES, 'image/jpeg', 'image/jpeg', id='honest-jpeg'),
        pytest.param(_GIF_BYTES, 'image/gif', 'image/gif', id='honest-gif'),
        pytest.param(_WEBP_BYTES, 'image/webp', 'image/webp', id='honest-webp'),
        # The bytes are authoritative: a mislabeled but genuine image resolves to its real type.
        pytest.param(_PNG_BYTES, 'image/jpeg', 'image/png', id='mislabeled-genuine-image'),
        # `application/octet-stream` carrying a genuine image is accepted with the sniffed type.
        pytest.param(_PNG_BYTES, 'application/octet-stream', 'image/png', id='octet-stream-genuine-image'),
        # An unfamiliar `image/*` header on genuine image bytes still resolves to the sniffed type.
        pytest.param(_PNG_BYTES, 'image/avif', 'image/png', id='unfamiliar-header-genuine-image'),
        # Header whitespace/casing must not affect the outcome for a genuine image.
        pytest.param(_PNG_BYTES, '  IMAGE/PNG ; charset=binary', 'image/png', id='messy-header-genuine-image'),
        # No content-type header, but the bytes are a genuine image.
        pytest.param(_PNG_BYTES, None, 'image/png', id='missing-header-genuine-image'),
    ),
)
async def test_download_image_accepts_verified_image(
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    content_type: str | None,
    expected_media_type: str,
) -> None:
    """A downloaded `ImageUrl` is accepted only when its bytes match a known image signature."""
    _install_mock_download(monkeypatch, content=content, content_type=content_type)
    item = ImageUrl(url='https://93.184.215.14/image.png', media_type='image/png')
    downloaded_item = await download_item(item, data_format='bytes')
    assert downloaded_item['data'] == content
    assert downloaded_item['data_type'] == expected_media_type


@pytest.mark.parametrize(
    ('content', 'content_type'),
    (
        # Honest HTML served for a `.png` URL must not be forwarded as an image.
        pytest.param(_HTML_BYTES, 'text/html', id='honest-html'),
        # A spoofed `image/png` header carrying HTML bytes must be rejected.
        pytest.param(_HTML_BYTES, 'image/png', id='spoofed-image-png-header'),
        # `application/octet-stream` falling back to the URL's `.png` type must not smuggle HTML.
        pytest.param(_HTML_BYTES, 'application/octet-stream', id='octet-stream-html'),
        # An unfamiliar `image/*` header cannot vouch for non-image bytes.
        pytest.param(_HTML_BYTES, 'image/avif', id='unfamiliar-header-non-image'),
        # A missing content-type falling back to the URL's `.png` type must not smuggle HTML.
        pytest.param(_HTML_BYTES, None, id='missing-header-html'),
        # Whitespace/casing must not smuggle an unverified declared type past the check.
        pytest.param(_HTML_BYTES, ' IMAGE/PNG ', id='messy-spoofed-header'),
    ),
)
async def test_download_image_rejects_unverified_bytes(
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    content_type: str | None,
) -> None:
    """A downloaded `ImageUrl` whose bytes are not a recognized image is rejected."""
    _install_mock_download(monkeypatch, content=content, content_type=content_type)
    item = ImageUrl(url='https://93.184.215.14/image.png', media_type='image/png')
    with pytest.raises(UserError, match='is not a recognized image'):
        await download_item(item, data_format='bytes')


async def test_download_non_image_does_not_sniff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-image file URLs are out of scope for image-signature validation and still download."""
    _install_mock_download(monkeypatch, content=b'%PDF-1.7 ...', content_type='application/pdf')
    item = DocumentUrl(url='https://93.184.215.14/doc.pdf', media_type='application/pdf')
    downloaded_item = await download_item(item, data_format='bytes')
    assert downloaded_item['data_type'] == 'application/pdf'
