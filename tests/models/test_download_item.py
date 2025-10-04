import pytest

from pydantic_ai import AudioUrl, DocumentUrl, ImageUrl, VideoUrl
from pydantic_ai.models import UserError, download_item

from ..conftest import IsInstance, IsStr

pytestmark = [pytest.mark.anyio]


@pytest.mark.parametrize(
    'url',
    (
        pytest.param(AudioUrl(url='gs://pydantic-ai-dev/openai-alloy.wav')),
        pytest.param(DocumentUrl(url='gs://pydantic-ai-dev/Gemini_1_5_Pro_Technical_Report_Arxiv_1805.pdf')),
        pytest.param(ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png')),
        pytest.param(VideoUrl(url='gs://pydantic-ai-dev/grepit-tiny-video.mp4')),
    ),
)
async def test_download_item_raises_user_error_with_gs_uri(
    url: AudioUrl | DocumentUrl | ImageUrl | VideoUrl,
) -> None:
    with pytest.raises(UserError, match='Downloading from protocol "gs://" is not supported.'):
        _ = await download_item(url, data_format='bytes')


async def test_download_item_raises_user_error_with_youtube_url() -> None:
    with pytest.raises(UserError, match='Downloading YouTube videos is not supported.'):
        _ = await download_item(VideoUrl(url='https://youtu.be/lCdaVNyHtjU'), data_format='bytes')


@pytest.mark.vcr()
async def test_download_item_application_octet_stream() -> None:
    downloaded_item = await download_item(
        VideoUrl(
            url='https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/tests/assets/small_video.mp4'
        ),
        data_format='bytes',
    )
    assert downloaded_item['data_type'] == 'video/mp4'
    assert downloaded_item['data'] == IsInstance(bytes)


@pytest.mark.vcr()
async def test_download_item_audio_mpeg() -> None:
    downloaded_item = await download_item(
        AudioUrl(url='https://smokeshow.helpmanual.io/4l1l1s0s6q4741012x1w/common_voice_en_537507.mp3'),
        data_format='bytes',
    )
    assert downloaded_item['data_type'] == 'audio/mpeg'
    assert downloaded_item['data'] == IsInstance(bytes)


@pytest.mark.vcr()
async def test_download_item_no_content_type() -> None:
    downloaded_item = await download_item(
        DocumentUrl(url='https://raw.githubusercontent.com/pydantic/pydantic-ai/refs/heads/main/docs/help.md'),
        data_format='text',
    )
    assert downloaded_item['data_type'] == 'text/markdown'
    assert downloaded_item['data'] == IsStr()
