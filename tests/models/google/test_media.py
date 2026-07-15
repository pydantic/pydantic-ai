"""Tests for `GoogleModel`'s mapping of media/file inputs into request Parts.

These assert the pre-request Part shape directly (through `_map_user_prompt` /
`_map_file_to_function_response_part`) rather than via a cassette: these are
request-body fields, and the VCR matchers are body-insensitive, so a cassette
would replay green even if a field were dropped or renamed — the internal-shape
assertion is what pins the mapping. Live acceptance of `media_resolution`
(image `ULTRA_HIGH`, document `HIGH`) is verified separately against Vertex; see
https://github.com/pydantic/pydantic-ai/issues/6524.
"""

from __future__ import annotations as _annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    TextContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import UploadedFile

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
]


@pytest.fixture
def mapping_model() -> GoogleModel:
    """A `GoogleModel` used only to exercise the request-mapping helpers.

    No network request is made, so the model name and API key are arbitrary.
    """
    return GoogleModel('gemini-1.5-flash', provider=GoogleProvider(api_key='test-key'))


# =============================================================================
# Per-Part `media_resolution` forwarding via `vendor_metadata`
# =============================================================================


@dataclass
class MediaResolutionCase:
    id: str
    content: BinaryContent | ImageUrl | DocumentUrl
    expected: dict[str, object]
    system: str | None = None
    """When set, patch `GoogleModel.system` (e.g. 'google-cloud' for gs:// URIs)."""


MEDIA_RESOLUTION_CASES = [
    MediaResolutionCase(
        id='binary_media_resolution_only',
        content=BinaryContent(
            data=b'\x00\x00\x00\x00',
            media_type='video/mp4',
            vendor_metadata={'media_resolution': {'level': 'MEDIA_RESOLUTION_ULTRA_HIGH'}},
        ),
        expected={
            'inline_data': {'data': b'\x00\x00\x00\x00', 'mime_type': 'video/mp4'},
            'media_resolution': {'level': 'MEDIA_RESOLUTION_ULTRA_HIGH'},
        },
    ),
    MediaResolutionCase(
        id='binary_media_resolution_and_video_metadata',
        content=BinaryContent(
            data=b'\x00\x00\x00\x00',
            media_type='video/mp4',
            vendor_metadata={
                'media_resolution': {'level': 'MEDIA_RESOLUTION_ULTRA_HIGH'},
                'start_offset': '2s',
                'end_offset': '10s',
            },
        ),
        expected={
            'inline_data': {'data': b'\x00\x00\x00\x00', 'mime_type': 'video/mp4'},
            'media_resolution': {'level': 'MEDIA_RESOLUTION_ULTRA_HIGH'},
            'video_metadata': {'start_offset': '2s', 'end_offset': '10s'},
        },
    ),
    MediaResolutionCase(
        id='binary_no_media_resolution_unchanged',
        content=BinaryContent(
            data=b'\x00\x00\x00\x00',
            media_type='video/mp4',
            vendor_metadata={'start_offset': '2s', 'end_offset': '10s'},
        ),
        expected={
            'inline_data': {'data': b'\x00\x00\x00\x00', 'mime_type': 'video/mp4'},
            'video_metadata': {'start_offset': '2s', 'end_offset': '10s'},
        },
    ),
    MediaResolutionCase(
        id='image_url_media_resolution',
        content=ImageUrl(
            url='gs://bucket/image.png',
            vendor_metadata={'media_resolution': {'level': 'MEDIA_RESOLUTION_ULTRA_HIGH'}},
        ),
        expected={
            'file_data': {'file_uri': 'gs://bucket/image.png', 'mime_type': 'image/png'},
            'media_resolution': {'level': 'MEDIA_RESOLUTION_ULTRA_HIGH'},
        },
        system='google-cloud',
    ),
    MediaResolutionCase(
        id='document_url_media_resolution',
        content=DocumentUrl(
            url='gs://bucket/report.pdf',
            vendor_metadata={'media_resolution': {'level': 'MEDIA_RESOLUTION_HIGH'}},
        ),
        expected={
            'file_data': {'file_uri': 'gs://bucket/report.pdf', 'mime_type': 'application/pdf'},
            'media_resolution': {'level': 'MEDIA_RESOLUTION_HIGH'},
        },
        system='google-cloud',
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in MEDIA_RESOLUTION_CASES])
async def test_media_resolution_forwarding(
    case: MediaResolutionCase, mapping_model: GoogleModel, mocker: MockerFixture
):
    """`vendor_metadata['media_resolution']` is lifted to the per-Part `media_resolution`
    field for every file type, remaining keys still route to `video_metadata`, and the
    user's `vendor_metadata` dict is never mutated (the mapper works on a copy).
    """
    if case.system is not None:
        mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value=case.system)

    original_vendor_metadata = deepcopy(case.content.vendor_metadata)

    content = await mapping_model._map_user_prompt(UserPromptPart(content=[case.content]))  # pyright: ignore[reportPrivateUsage]

    assert content == [case.expected]
    assert case.content.vendor_metadata == original_vendor_metadata


# =============================================================================
# `UploadedFile` mapping
# =============================================================================


async def test_uploaded_file_mapping(mapping_model: GoogleModel):
    """Test that UploadedFile is correctly mapped to file_data in Google model."""
    file_uri = 'https://generativelanguage.googleapis.com/v1beta/files/abc123'
    content = await mapping_model._map_user_prompt(  # pyright: ignore[reportPrivateUsage]
        UserPromptPart(content=['Analyze this file', UploadedFile(file_id=file_uri, provider_name='google')])
    )

    assert len(content) == 2
    assert content[0] == {'text': 'Analyze this file'}
    assert content[1] == {'file_data': {'file_uri': file_uri, 'mime_type': 'application/octet-stream'}}


async def test_uploaded_file_mapping_with_media_type(mapping_model: GoogleModel):
    """Test that UploadedFile with media_type is correctly mapped."""
    file_uri = 'https://generativelanguage.googleapis.com/v1beta/files/xyz789'
    content = await mapping_model._map_user_prompt(  # pyright: ignore[reportPrivateUsage]
        UserPromptPart(content=[UploadedFile(file_id=file_uri, provider_name='google', media_type='application/pdf')])
    )

    assert len(content) == 1
    assert content[0] == {'file_data': {'file_uri': file_uri, 'mime_type': 'application/pdf'}}


async def test_uploaded_file_wrong_provider(allow_model_requests: None, mapping_model: GoogleModel):
    """Test that UploadedFile with wrong provider raises an error in GoogleModel."""
    agent = Agent(mapping_model)

    with pytest.raises(UserError, match=r"provider_name='anthropic'.*cannot be used with GoogleModel"):
        await agent.run(['Analyze this file', UploadedFile(file_id='file-abc123', provider_name='anthropic')])


async def test_uploaded_file_invalid_file_id(allow_model_requests: None, mapping_model: GoogleModel):
    """Test that UploadedFile with a non-URI file_id raises an error in GoogleModel."""
    agent = Agent(mapping_model)

    with pytest.raises(UserError, match='must use a file URI from the Google Files API'):
        await agent.run(['Analyze this file', UploadedFile(file_id='file-abc123', provider_name='google')])


async def test_uploaded_file_vertex_requires_gs_uri(mapping_model: GoogleModel, mocker: MockerFixture):
    """Vertex `UploadedFile` must use a gs:// URI (not Files API https URLs)."""
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-cloud')

    https_files_api = 'https://generativelanguage.googleapis.com/v1beta/files/abc123'
    with pytest.raises(UserError, match='must use a GCS URI'):
        await mapping_model._map_user_prompt(  # pyright: ignore[reportPrivateUsage]
            UserPromptPart(
                content=[UploadedFile(file_id=https_files_api, provider_name='google-cloud')],
            )
        )


async def test_uploaded_file_with_vendor_metadata(mapping_model: GoogleModel):
    """Test that UploadedFile with vendor_metadata includes video_metadata."""
    file_uri = 'https://generativelanguage.googleapis.com/v1beta/files/video123'
    content = await mapping_model._map_user_prompt(  # pyright: ignore[reportPrivateUsage]
        UserPromptPart(
            content=[
                UploadedFile(
                    file_id=file_uri,
                    provider_name='google',
                    media_type='video/mp4',
                    vendor_metadata={'start_offset': '10s', 'end_offset': '30s'},
                )
            ]
        )
    )

    assert len(content) == 1
    assert content[0] == {
        'file_data': {'file_uri': file_uri, 'mime_type': 'video/mp4'},
        'video_metadata': {'start_offset': '10s', 'end_offset': '30s'},
    }


async def test_youtube_video_url_without_vendor_metadata(mapping_model: GoogleModel):
    """Test that YouTube VideoUrl without vendor_metadata doesn't include video_metadata."""
    video = VideoUrl(url='https://youtu.be/dQw4w9WgXcQ', media_type='video/mp4')
    content = await mapping_model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]

    assert len(content) == 1
    assert 'video_metadata' not in content[0]
    assert content[0] == {'file_data': {'file_uri': 'https://youtu.be/dQw4w9WgXcQ', 'mime_type': 'video/mp4'}}


# =============================================================================
# GCS VideoUrl mapping for google-cloud (Vertex)
#
# GCS URIs (gs://...) with vendor_metadata (video offsets) only work on
# google-cloud because Vertex AI can access GCS buckets directly.
# Regression test for https://github.com/pydantic/pydantic-ai/issues/3805
# =============================================================================


async def test_gcs_video_url_with_vendor_metadata_on_google_cloud(mapping_model: GoogleModel, mocker: MockerFixture):
    """GCS URIs use file_uri with video_metadata on google-cloud (Vertex).

    This is the main fix - GCS URIs were previously falling through to FileUrl
    handling which doesn't pass vendor_metadata as video_metadata.
    """
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-cloud')

    video = VideoUrl(
        url='gs://bucket/video.mp4',
        vendor_metadata={'start_offset': '300s', 'end_offset': '330s'},
    )
    content = await mapping_model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]

    assert len(content) == 1
    assert content[0] == {
        'file_data': {'file_uri': 'gs://bucket/video.mp4', 'mime_type': 'video/mp4'},
        'video_metadata': {'start_offset': '300s', 'end_offset': '330s'},
    }


async def test_gcs_video_url_raises_error_on_google(mapping_model: GoogleModel):
    """GCS URIs on the Gemini API (google) fall through to FileUrl and raise a clear error.

    The Gemini API cannot access GCS buckets, so attempting to use gs:// URLs
    should fail with a helpful error message rather than a cryptic API error.
    SSRF protection now catches non-http(s) protocols first.
    """
    # GoogleProvider with api_key targets the Gemini API; assert it explicitly.
    assert mapping_model.system == 'google'

    video = VideoUrl(url='gs://bucket/video.mp4')

    with pytest.raises(ValueError, match='URL protocol "gs" is not allowed'):
        await mapping_model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]


# =============================================================================
# HTTP VideoUrl fallback (not YouTube, not GCS)
#
# HTTP VideoUrls fall through to FileUrl handling, which is provider-specific:
# - google (Gemini API): downloads the video and sends inline_data
# - google-cloud (Vertex): uses file_uri directly (no download)
# =============================================================================


async def test_http_video_url_downloads_on_google(mapping_model: GoogleModel, mocker: MockerFixture):
    """HTTP VideoUrls are downloaded on the Gemini API (google) with video_metadata preserved."""
    mock_download = mocker.patch(
        'pydantic_ai.models.google.download_item',
        return_value={'data': b'fake video data', 'data_type': 'video/mp4'},
    )

    video = VideoUrl(
        url='https://example.com/video.mp4',
        vendor_metadata={'start_offset': '10s', 'end_offset': '20s'},
    )
    content = await mapping_model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]

    mock_download.assert_called_once()
    assert len(content) == 1
    assert 'inline_data' in content[0]
    assert 'file_data' not in content[0]
    assert content[0].get('video_metadata') == {'start_offset': '10s', 'end_offset': '20s'}


async def test_http_video_url_uses_file_uri_on_google_cloud(mapping_model: GoogleModel, mocker: MockerFixture):
    """HTTP VideoUrls use file_uri directly on google-cloud (Vertex) with video_metadata."""
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-cloud')

    video = VideoUrl(
        url='https://example.com/video.mp4',
        vendor_metadata={'start_offset': '10s', 'end_offset': '20s'},
    )
    content = await mapping_model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]

    assert len(content) == 1
    assert content[0] == {
        'file_data': {'file_uri': 'https://example.com/video.mp4', 'mime_type': 'video/mp4'},
        'video_metadata': {'start_offset': '10s', 'end_offset': '20s'},
    }


# =============================================================================
# `_map_file_to_function_response_part` for tool returns on Vertex
#
# Covers the FunctionResponsePartDict mapping for Gemini 3+ native tool returns
# on google-cloud (Vertex), which uses file_data for URLs instead of downloading
# (unlike `_map_file_to_part`, which is for user prompts).
# =============================================================================


@pytest.mark.parametrize(
    'file_url,expected',
    [
        pytest.param(
            VideoUrl(url='https://youtu.be/lCdaVNyHtjU'),
            {'file_data': {'file_uri': 'https://youtu.be/lCdaVNyHtjU', 'mime_type': 'video/mp4'}},
            id='youtube',
        ),
        pytest.param(
            VideoUrl(url='gs://bucket/video.mp4'),
            {'file_data': {'file_uri': 'gs://bucket/video.mp4', 'mime_type': 'video/mp4'}},
            id='gcs',
        ),
        pytest.param(
            ImageUrl(url='https://example.com/image.png'),
            {'file_data': {'file_uri': 'https://example.com/image.png', 'mime_type': 'image/png'}},
            id='http_file_url',
        ),
    ],
)
async def test_file_url_in_tool_return_on_vertex(
    mocker: MockerFixture, file_url: VideoUrl | ImageUrl, expected: dict[str, object]
):
    """Test file URLs use file_data (not download) in tool returns on Vertex."""
    model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='test-key'))
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-cloud')

    result = await model._map_file_to_function_response_part(file_url)  # pyright: ignore[reportPrivateUsage]

    assert result == expected


async def test_map_user_prompt_with_text_content(mapping_model: GoogleModel, mocker: MockerFixture):
    """Test that _map_user_prompt correctly handles a mix of text content and str."""
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google')

    user_prompt_part = UserPromptPart(
        content=['Hi', TextContent(content='This is some context', metadata={'source': 'user'})]
    )
    content = await mapping_model._map_user_prompt(user_prompt_part)  # pyright: ignore[reportPrivateUsage]

    assert content == snapshot([{'text': 'Hi'}, {'text': 'This is some context'}])
