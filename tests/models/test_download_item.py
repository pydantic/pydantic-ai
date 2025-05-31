from typing import Union

import pytest

from pydantic_ai.models import AudioUrl, DocumentUrl, ImageUrl, UserError, VideoUrl, download_item

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
    url: Union[AudioUrl, DocumentUrl, ImageUrl, VideoUrl],
) -> None:
    with pytest.raises(UserError, match='Downloading from protocol "gs://" is not supported.'):
        _ = await download_item(url, data_format='bytes')


async def test_download_item_raises_user_error_with_youtube_url() -> None:
    with pytest.raises(UserError, match='Downloading YouTube videos is not supported.'):
        _ = await download_item(VideoUrl(url='https://youtu.be/lCdaVNyHtjU'), data_format='bytes')
