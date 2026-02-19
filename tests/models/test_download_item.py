import pytest

from pydantic_ai import AudioUrl, DocumentUrl, ImageUrl, VideoUrl
from pydantic_ai.models import UserError, download_item

from ..conftest import IsInstance, IsStr

pytestmark = [pytest.mark.anyio]


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
    with pytest.raises(UserError, match='Downloading YouTube videos is not supported.'):
        _ = await download_item(VideoUrl(url='https://youtu.be/lCdaVNyHtjU'), data_format='bytes')


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
