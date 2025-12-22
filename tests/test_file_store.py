from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai import BinaryContent
from pydantic_ai._run_context import RunContext
from pydantic_ai.file_store import S3Error, S3FileStore, file_store_processor, generate_file_key
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .conftest import TestEnv

pytestmark = pytest.mark.anyio


class TestGenerateFileKey:
    """Tests for the generate_file_key utility function."""

    @pytest.mark.parametrize(
        'media_type,expected_format',
        [
            ('image/png', 'png'),
            ('image/jpeg', 'jpeg'),
            ('video/mp4', 'mp4'),
            ('audio/mpeg', 'mp3'),
            ('application/pdf', 'pdf'),
        ],
    )
    def test_generate_file_key_formats(self, media_type: str, expected_format: str):
        content = BinaryContent(data=b'test', media_type=media_type)
        key = generate_file_key(content)
        assert key == f'{content.identifier}.{expected_format}'
        assert key.endswith(f'.{expected_format}')


class TestS3FileStoreConstruction:
    """Tests for S3FileStore construction and validation."""

    def test_explicit_params(self):
        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='us-east-1',
        )
        assert store.bucket == 'test-bucket'
        assert store.endpoint == 'https://s3.us-east-1.amazonaws.com'
        assert store.access_key_id == 'test-key'
        assert store.secret_access_key == 'test-secret'
        assert store.region == 'us-east-1'

    def test_env_var_fallback(self, env: TestEnv):
        env.set('S3_ENDPOINT', 'https://endpoint.example.com')
        env.set('S3_ACCESS_KEY_ID', 'env-key')
        env.set('S3_SECRET_ACCESS_KEY', 'env-secret')
        env.set('S3_REGION', 'us-west-2')

        store = S3FileStore(bucket='test-bucket')
        assert store.endpoint == 'https://endpoint.example.com'
        assert store.access_key_id == 'env-key'
        assert store.secret_access_key == 'env-secret'
        assert store.region == 'us-west-2'

    def test_missing_endpoint_raises(self, env: TestEnv):
        env.remove('S3_ENDPOINT')
        with pytest.raises(ValueError, match='endpoint is required'):
            S3FileStore(
                bucket='test-bucket',
                access_key_id='key',
                secret_access_key='secret',
                region='us-east-1',
            )

    def test_missing_access_key_id_raises(self, env: TestEnv):
        env.remove('S3_ACCESS_KEY_ID')
        with pytest.raises(ValueError, match='access_key_id is required'):
            S3FileStore(
                bucket='test-bucket',
                endpoint='https://s3.example.com',
                secret_access_key='secret',
                region='us-east-1',
            )

    def test_missing_secret_access_key_raises(self, env: TestEnv):
        env.remove('S3_SECRET_ACCESS_KEY')
        with pytest.raises(ValueError, match='secret_access_key is required'):
            S3FileStore(
                bucket='test-bucket',
                endpoint='https://s3.example.com',
                access_key_id='key',
                region='us-east-1',
            )

    def test_missing_region_raises(self, env: TestEnv):
        env.remove('S3_REGION')
        with pytest.raises(ValueError, match='region is required'):
            S3FileStore(
                bucket='test-bucket',
                endpoint='https://s3.example.com',
                access_key_id='key',
                secret_access_key='secret',
            )

    def test_endpoint_trailing_slash_removed(self):
        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.example.com/',
            access_key_id='key',
            secret_access_key='secret',
            region='us-east-1',
        )
        assert store.endpoint == 'https://s3.example.com'

    def test_public_url_trailing_slash_removed(self):
        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.example.com',
            access_key_id='key',
            secret_access_key='secret',
            region='us-east-1',
            public_url='https://cdn.example.com/',
        )
        assert store.public_url == 'https://cdn.example.com'


class TestS3FileStoreOperations:
    """Tests for S3FileStore CRUD operations with mocked HTTP."""

    @pytest.fixture
    def store(self) -> S3FileStore:
        return S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='us-east-1',
        )

    async def test_store_sends_put_with_content_type(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(200, content=b'')
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        key = await store.store('test.png', b'image data')

        assert key == 'test.png'
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == 'PUT'
        assert call_args[1]['headers']['content-type'] == 'image/png'
        assert call_args[1]['content'] == b'image data'

    async def test_store_unknown_extension_content_type(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(200, content=b'')
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        await store.store('file.unknownextension', b'data')

        call_args = mock_client.request.call_args
        assert call_args[1]['headers']['content-type'] == 'application/octet-stream'

    async def test_retrieve_returns_data(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(200, content=b'retrieved data')
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        data = await store.retrieve('test.png')

        assert data == b'retrieved data'
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == 'GET'

    async def test_exists_returns_true_for_200(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(200)
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        exists = await store.exists('test.png')

        assert exists is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == 'HEAD'

    async def test_exists_returns_false_for_404(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(404)
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        exists = await store.exists('test.png')

        assert exists is False

    async def test_delete_returns_true_for_204(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(204)
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        deleted = await store.delete('test.png')

        assert deleted is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == 'DELETE'

    async def test_delete_returns_true_for_404(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(404)
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        deleted = await store.delete('nonexistent.png')

        assert deleted is True

    async def test_verify_access_returns_true_for_200(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(200)
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        result = await store.verify_access()

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == 'HEAD'

    async def test_verify_access_raises_for_404(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(404)
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        with pytest.raises(S3Error, match="Bucket 'test-bucket' not found"):
            await store.verify_access()

    async def test_verify_access_raises_for_403(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(403)
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        with pytest.raises(S3Error, match="Access denied to bucket 'test-bucket'"):
            await store.verify_access()

    async def test_store_raises_on_error(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(500, text='Internal Server Error')
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        with pytest.raises(S3Error, match='S3 Error 500'):
            await store.store('test.png', b'data')

    async def test_retrieve_raises_on_error(self, store: S3FileStore, mocker: Any):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_response = httpx.Response(404, text='Not Found')
        mock_client.request = AsyncMock(return_value=mock_response)

        mocker.patch.object(store, '_get_client', return_value=mock_client)

        with pytest.raises(S3Error, match='S3 Error 404'):
            await store.retrieve('nonexistent.png')


class TestS3FileStoreDownloadURI:
    """Tests for get_download_uri with different configurations."""

    def test_public_url_returns_cdn_url(self):
        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='us-east-1',
            public_url='https://cdn.example.com',
        )

        url = store.get_download_uri('images/test.png')

        assert url == 'https://cdn.example.com/images/test.png'

    def test_public_url_strips_leading_slash_from_key(self):
        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='us-east-1',
            public_url='https://cdn.example.com',
        )

        url = store.get_download_uri('/images/test.png')

        assert url == 'https://cdn.example.com/images/test.png'

    def test_private_bucket_returns_presigned_url(self):
        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='us-east-1',
        )

        url = store.get_download_uri('test.png')

        assert 'X-Amz-Algorithm=AWS4-HMAC-SHA256' in url
        assert 'X-Amz-Credential=' in url
        assert 'X-Amz-Date=' in url
        assert 'X-Amz-Expires=' in url
        assert 'X-Amz-Signature=' in url
        assert 'test.png' in url

    def test_custom_ttl_in_presigned_url(self):
        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='us-east-1',
            ttl=7200,
        )

        url = store.get_download_uri('test.png')

        assert 'X-Amz-Expires=7200' in url

    def test_custom_download_uri_callback(self):
        def custom_uri(key: str) -> str:
            return f'https://custom.example.com/{key}?token=abc123'

        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='us-east-1',
            custom_download_uri=custom_uri,
        )

        url = store.get_download_uri('test.png')

        assert url == 'https://custom.example.com/test.png?token=abc123'

    def test_custom_download_uri_takes_precedence_over_public_url(self):
        def custom_uri(key: str) -> str:
            return f'https://custom.example.com/{key}'

        store = S3FileStore(
            bucket='test-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='test-key',
            secret_access_key='test-secret',
            region='us-east-1',
            public_url='https://cdn.example.com',
            custom_download_uri=custom_uri,
        )

        url = store.get_download_uri('test.png')

        assert url == 'https://custom.example.com/test.png'


@pytest.fixture(scope='module')
def vcr_config():
    return {
        'ignore_localhost': True,
        'filter_headers': ['authorization', 'x-api-key', 'x-amz-content-sha256', 'host'],
        'filter_query_parameters': [
            'X-Amz-Algorithm',
            'X-Amz-Credential',
            'X-Amz-Date',
            'X-Amz-Expires',
            'X-Amz-SignedHeaders',
            'X-Amz-Signature',
        ],
        'decode_compressed_response': True,
        'match_on': ['method', 'scheme', 'port', 'path', 'query'],
    }


class TestS3FileStoreLiveIntegration:
    """Integration tests against live S3-compatible service (R2)."""

    @pytest.fixture
    async def live_store(self) -> AsyncIterator[S3FileStore]:
        store = S3FileStore(
            bucket=os.getenv('S3_BUCKET_NAME', 'pydantic-ai-tests'),
            endpoint=os.getenv('S3_ENDPOINT', 'https://accountid.r2.cloudflarestorage.com'),
            access_key_id=os.getenv('S3_ACCESS_KEY_ID', 'test-access-key'),
            secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY', 'test-secret-key'),
            region=os.getenv('S3_REGION', 'auto'),
        )
        yield store
        await store.close()

    @pytest.mark.vcr()
    async def test_round_trip_store_exists_retrieve_delete(self, live_store: S3FileStore):
        test_key = 'test-files/integration-test.png'
        test_data = b'test image data for integration test'

        try:
            stored_key = await live_store.store(test_key, test_data)
            assert stored_key == test_key

            exists = await live_store.exists(test_key)
            assert exists is True

            retrieved_data = await live_store.retrieve(test_key)
            assert retrieved_data == test_data

            deleted = await live_store.delete(test_key)
            assert deleted is True

            exists_after = await live_store.exists(test_key)
            assert exists_after is False
        finally:
            await live_store.delete(test_key)

    @pytest.mark.vcr()
    async def test_presigned_url_is_downloadable(self, live_store: S3FileStore):
        test_key = 'test-files/presigned-test.txt'
        test_data = b'test data for presigned URL'

        try:
            await live_store.store(test_key, test_data)

            presigned_url = live_store.get_download_uri(test_key)

            assert 'X-Amz-Algorithm=AWS4-HMAC-SHA256' in presigned_url
            assert 'X-Amz-Signature=' in presigned_url

            async with httpx.AsyncClient() as client:
                response = await client.get(presigned_url)
                assert response.status_code == 200
                assert response.content == test_data
        finally:
            await live_store.delete(test_key)

    @pytest.mark.vcr()
    async def test_verify_access_succeeds(self, live_store: S3FileStore):
        result = await live_store.verify_access()
        assert result is True

    @pytest.mark.vcr()
    async def test_store_with_various_content_types(self, live_store: S3FileStore):
        test_cases = [
            ('test-files/image.png', b'PNG data'),
            ('test-files/video.mp4', b'MP4 data'),
            ('test-files/audio.mp3', b'MP3 data'),
            ('test-files/document.pdf', b'PDF data'),
        ]

        try:
            for key, data in test_cases:
                await live_store.store(key, data)
                assert await live_store.exists(key)
                retrieved = await live_store.retrieve(key)
                assert retrieved == data
        finally:
            for key, _ in test_cases:
                await live_store.delete(key)


class TestGetMediaCategory:
    """Tests for the _get_media_category helper function."""

    @pytest.mark.parametrize(
        'media_type,expected',
        [
            ('image/png', 'image'),
            ('image/jpeg', 'image'),
            ('image/gif', 'image'),
            ('audio/wav', 'audio'),
            ('audio/mpeg', 'audio'),
            ('audio/ogg', 'audio'),
            ('video/mp4', 'video'),
            ('video/webm', 'video'),
            ('video/quicktime', 'video'),
            ('application/pdf', 'document'),
            ('text/plain', 'document'),
            ('text/csv', 'document'),
            ('application/json', 'document'),
            ('unknown/type', None),
        ],
    )
    def test_media_category_detection(self, media_type: str, expected: str | None):
        from pydantic_ai._file_store import _get_media_category  # pyright: ignore[reportPrivateUsage]

        assert _get_media_category(media_type) == expected


class TestGetUrlSupport:
    """Tests for the _get_url_support helper function."""

    @pytest.fixture
    def openai_model(self):
        model = TestModel()
        model._system = 'openai'  # pyright: ignore[reportPrivateUsage]
        return model

    @pytest.fixture
    def anthropic_model(self):
        model = TestModel()
        model._system = 'anthropic'  # pyright: ignore[reportPrivateUsage]
        return model

    @pytest.fixture
    def google_vertex_model(self):
        model = TestModel()
        model._system = 'google-vertex'  # pyright: ignore[reportPrivateUsage]
        return model

    @pytest.fixture
    def google_gla_model(self):
        model = TestModel()
        model._system = 'google-gla'  # pyright: ignore[reportPrivateUsage]
        return model

    @pytest.fixture
    def unknown_model(self):
        model = TestModel()
        model._system = 'unknown-provider'  # pyright: ignore[reportPrivateUsage]
        return model

    def test_openai_supports_image_urls(self, openai_model: TestModel):
        from pydantic_ai._file_store import _get_url_support  # pyright: ignore[reportPrivateUsage]

        support = _get_url_support(openai_model)
        assert support == snapshot({'image': True, 'audio': False, 'video': False, 'document': False})

    def test_anthropic_supports_image_and_document_urls(self, anthropic_model: TestModel):
        from pydantic_ai._file_store import _get_url_support  # pyright: ignore[reportPrivateUsage]

        support = _get_url_support(anthropic_model)
        assert support == snapshot({'image': True, 'audio': False, 'video': False, 'document': True})

    def test_google_vertex_supports_all_urls(self, google_vertex_model: TestModel):
        from pydantic_ai._file_store import _get_url_support  # pyright: ignore[reportPrivateUsage]

        support = _get_url_support(google_vertex_model)
        assert support == snapshot({'image': True, 'audio': True, 'video': True, 'document': True})

    def test_google_gla_needs_bytes(self, google_gla_model: TestModel):
        from pydantic_ai._file_store import _get_url_support  # pyright: ignore[reportPrivateUsage]

        support = _get_url_support(google_gla_model)
        assert support == snapshot({'image': False, 'audio': False, 'video': False, 'document': False})

    def test_unknown_provider_is_conservative(self, unknown_model: TestModel):
        from pydantic_ai._file_store import _get_url_support  # pyright: ignore[reportPrivateUsage]

        support = _get_url_support(unknown_model)
        assert support == snapshot({'image': False, 'audio': False, 'video': False, 'document': False})


class TestFileStoreProcessor:
    """Tests for the file_store_processor function."""

    @pytest.fixture
    def mock_store(self) -> AsyncMock:
        store = AsyncMock()
        store.store = AsyncMock(side_effect=lambda key, data: key)  # pyright: ignore[reportUnknownLambdaType]
        store.get_download_uri = lambda key: f'https://cdn.example.com/{key}'  # pyright: ignore[reportUnknownLambdaType]
        return store

    @pytest.fixture
    def openai_model(self):
        model = TestModel()
        model._system = 'openai'  # pyright: ignore[reportPrivateUsage]
        return model

    @pytest.fixture
    def google_gla_model(self):
        model = TestModel()
        model._system = 'google-gla'  # pyright: ignore[reportPrivateUsage]
        return model

    async def test_uploads_file_part_and_converts_to_image_url(self, mock_store: AsyncMock, openai_model: TestModel):
        processor = file_store_processor(mock_store)

        content = BinaryContent(data=b'PNG image data', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        result = await processor(ctx, [response])

        mock_store.store.assert_called_once()
        call_key, call_data = mock_store.store.call_args[0]
        assert call_key == f'{content.identifier}.png'
        assert call_data == b'PNG image data'

        assert len(result) == 1
        processed_response = result[0]
        assert isinstance(processed_response, ModelResponse)
        assert len(processed_response.parts) == 1
        part = processed_response.parts[0]
        assert isinstance(part, ImageUrl)
        assert part.url == f'https://cdn.example.com/{content.identifier}.png'
        assert part.media_type == 'image/png'

    async def test_keeps_file_part_for_model_not_supporting_urls(
        self, mock_store: AsyncMock, google_gla_model: TestModel
    ):
        processor = file_store_processor(mock_store)

        content = BinaryContent(data=b'PNG image data', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx = RunContext(deps=None, model=google_gla_model, usage=RunUsage())

        result = await processor(ctx, [response])

        mock_store.store.assert_called_once()

        assert len(result) == 1
        processed_response = result[0]
        assert isinstance(processed_response, ModelResponse)
        assert len(processed_response.parts) == 1
        part = processed_response.parts[0]
        assert isinstance(part, FilePart)

    async def test_leaves_text_part_unchanged(self, mock_store: AsyncMock, openai_model: TestModel):
        processor = file_store_processor(mock_store)

        text_part = TextPart(content='Hello, world!')
        response = ModelResponse(parts=[text_part])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        result = await processor(ctx, [response])

        mock_store.store.assert_not_called()
        assert len(result) == 1
        processed_response = result[0]
        assert isinstance(processed_response, ModelResponse)
        assert processed_response.parts[0] is text_part

    async def test_passes_model_request_unchanged(self, mock_store: AsyncMock, openai_model: TestModel):
        processor = file_store_processor(mock_store)

        request = ModelRequest(parts=[])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        result = await processor(ctx, [request])

        mock_store.store.assert_not_called()
        assert result == [request]

    async def test_skips_file_part_with_empty_data(self, mock_store: AsyncMock, openai_model: TestModel):
        processor = file_store_processor(mock_store)

        content = BinaryContent(data=b'', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        result = await processor(ctx, [response])

        mock_store.store.assert_not_called()
        assert len(result) == 1
        assert result[0] is response

    async def test_handles_mixed_parts(self, mock_store: AsyncMock, openai_model: TestModel):
        processor = file_store_processor(mock_store)

        text_part = TextPart(content='Here is an image:')
        content = BinaryContent(data=b'PNG data', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[text_part, file_part])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        result = await processor(ctx, [response])

        mock_store.store.assert_called_once()
        assert len(result) == 1
        processed_response = result[0]
        assert isinstance(processed_response, ModelResponse)
        assert len(processed_response.parts) == 2
        assert processed_response.parts[0] is text_part
        assert isinstance(processed_response.parts[1], ImageUrl)

    @pytest.mark.parametrize(
        'media_type,expected_url_class',
        [
            ('image/png', ImageUrl),
            ('audio/wav', AudioUrl),
            ('video/mp4', VideoUrl),
            ('application/pdf', DocumentUrl),
        ],
    )
    async def test_converts_to_correct_url_type(self, mock_store: AsyncMock, media_type: str, expected_url_class: type):
        processor = file_store_processor(mock_store)

        model = TestModel()
        model._system = 'google-vertex'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'test data', media_type=media_type)
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx = RunContext(deps=None, model=model, usage=RunUsage())

        result = await processor(ctx, [response])

        assert len(result) == 1
        processed_response = result[0]
        assert isinstance(processed_response, ModelResponse)
        part = processed_response.parts[0]
        assert isinstance(part, expected_url_class)
        assert isinstance(part, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl))
        assert part.media_type == media_type


class TestFileStoreProcessorUrlRefresh:
    """Tests for URL refresh functionality in file_store_processor."""

    @pytest.fixture
    def mock_store_with_changing_urls(self) -> AsyncMock:
        store = AsyncMock()
        call_count = {'count': 0}

        def get_url(key: str) -> str:
            call_count['count'] += 1
            return f'https://cdn.example.com/{key}?sig={call_count["count"]}'

        store.store = AsyncMock(side_effect=lambda key, data: key)  # pyright: ignore[reportUnknownLambdaType]
        store.get_download_uri = get_url
        return store

    async def test_refreshes_url_for_tracked_file(self, mock_store_with_changing_urls: AsyncMock):
        processor = file_store_processor(mock_store_with_changing_urls)

        model = TestModel()
        model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'PNG data', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx = RunContext(deps=None, model=model, usage=RunUsage())

        result1 = await processor(ctx, [response])
        first_url_part = result1[0].parts[0]
        assert isinstance(first_url_part, ImageUrl)
        first_url = first_url_part.url

        result2 = await processor(ctx, result1)
        second_url_part = result2[0].parts[0]
        assert isinstance(second_url_part, ImageUrl)
        second_url = second_url_part.url

        assert first_url != second_url
        assert '?sig=1' in first_url
        assert '?sig=2' in second_url


class TestFileStoreProcessorAgentIntegration:
    """Integration tests for file_store_processor with Agent.

    These tests verify the processor works correctly when integrated with Agent's
    history_processors mechanism using RunContext directly.
    """

    @pytest.fixture
    def mock_store(self) -> AsyncMock:
        store = AsyncMock()
        store.store = AsyncMock(side_effect=lambda key, data: key)  # pyright: ignore[reportUnknownLambdaType]
        store.get_download_uri = lambda key: f'https://cdn.example.com/{key}'  # pyright: ignore[reportUnknownLambdaType]
        return store

    async def test_processor_in_multi_message_history(self, mock_store: AsyncMock):
        """Test processor handles a list with multiple messages (requests and responses)."""
        processor = file_store_processor(mock_store)

        model = TestModel()
        model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        request = ModelRequest(parts=[])
        content = BinaryContent(data=b'Generated image', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx = RunContext(deps=None, model=model, usage=RunUsage())

        result = await processor(ctx, [request, response])

        mock_store.store.assert_called_once()
        assert len(result) == 2
        assert result[0] is request
        assert isinstance(result[1], ModelResponse)
        part = result[1].parts[0]
        assert isinstance(part, ImageUrl)
        assert part.url.startswith('https://cdn.example.com/')

    async def test_processor_tracks_files_across_calls(self, mock_store: AsyncMock):
        """Test processor tracks uploaded files and refreshes URLs on subsequent calls."""
        processor = file_store_processor(mock_store)

        model = TestModel()
        model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'Image data', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx = RunContext(deps=None, model=model, usage=RunUsage())

        result1 = await processor(ctx, [response])
        assert mock_store.store.call_count == 1

        result2 = await processor(ctx, result1)
        assert mock_store.store.call_count == 1

        part1 = result1[0].parts[0]
        part2 = result2[0].parts[0]
        assert isinstance(part1, ImageUrl)
        assert isinstance(part2, ImageUrl)


class TestFileStoreProcessorLoadDirection:
    """Tests for load direction: FileUrl → FilePart when model needs bytes."""

    @pytest.fixture
    def mock_store(self) -> AsyncMock:
        store = AsyncMock()
        store.store = AsyncMock(side_effect=lambda key, data: key)  # pyright: ignore[reportUnknownLambdaType]
        store.retrieve = AsyncMock(return_value=b'retrieved data')
        store.get_download_uri = lambda key: f'https://cdn.example.com/{key}'  # pyright: ignore[reportUnknownLambdaType]
        return store

    async def test_loads_file_url_to_file_part_when_model_needs_bytes(self, mock_store: AsyncMock):
        """When switching to a model that needs bytes, FileUrl is converted back to FilePart."""
        processor = file_store_processor(mock_store)

        # First, store with OpenAI (supports image URLs)
        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'PNG image data', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx_openai = RunContext(deps=None, model=openai_model, usage=RunUsage())
        result1 = await processor(ctx_openai, [response])

        # Verify it was converted to ImageUrl
        assert isinstance(result1[0].parts[0], ImageUrl)
        image_url = result1[0].parts[0]
        assert isinstance(image_url, ImageUrl)

        # Now replay to Gemini (needs bytes for external URLs)
        gemini_model = TestModel()
        gemini_model._system = 'google-gla'  # pyright: ignore[reportPrivateUsage]

        mock_store.retrieve.return_value = b'PNG image data'

        ctx_gemini = RunContext(deps=None, model=gemini_model, usage=RunUsage())
        result2 = await processor(ctx_gemini, result1)

        # Verify it was converted back to FilePart
        mock_store.retrieve.assert_called_once()
        assert isinstance(result2[0].parts[0], FilePart)
        loaded_part = result2[0].parts[0]
        assert isinstance(loaded_part, FilePart)
        assert loaded_part.content.data == b'PNG image data'
        assert loaded_part.content.media_type == 'image/png'

    async def test_external_url_passes_through_unchanged(self, mock_store: AsyncMock):
        """External URLs (not uploaded by us) are passed through unchanged."""
        processor = file_store_processor(mock_store)

        gemini_model = TestModel()
        gemini_model._system = 'google-gla'  # pyright: ignore[reportPrivateUsage]

        # Create an ImageUrl that wasn't uploaded by the processor
        external_url = ImageUrl(url='https://external.example.com/image.png', media_type='image/png')
        response = ModelResponse(parts=[external_url])

        ctx = RunContext(deps=None, model=gemini_model, usage=RunUsage())
        result = await processor(ctx, [response])

        # Should pass through unchanged (not in uploaded_keys)
        mock_store.retrieve.assert_not_called()
        assert result[0].parts[0] is external_url


class TestFileStoreProcessorUserContent:
    """Tests for UserPromptPart content handling."""

    @pytest.fixture
    def mock_store(self) -> AsyncMock:
        store = AsyncMock()
        store.store = AsyncMock(side_effect=lambda key, data: key)  # pyright: ignore[reportUnknownLambdaType]
        store.retrieve = AsyncMock(return_value=b'retrieved data')
        store.get_download_uri = lambda key: f'https://cdn.example.com/{key}'  # pyright: ignore[reportUnknownLambdaType]
        return store

    async def test_stores_binary_content_in_user_prompt(self, mock_store: AsyncMock):
        """BinaryContent in UserPromptPart is stored and converted to ImageUrl."""
        processor = file_store_processor(mock_store)

        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'user uploaded image', media_type='image/png')
        user_part = UserPromptPart(content=['Here is my image:', content])
        request = ModelRequest(parts=[user_part])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())
        result = await processor(ctx, [request])

        mock_store.store.assert_called_once()
        assert len(result) == 1
        processed_request = result[0]
        assert isinstance(processed_request, ModelRequest)

        processed_user_part = processed_request.parts[0]
        assert isinstance(processed_user_part, UserPromptPart)
        assert isinstance(processed_user_part.content, list)
        assert processed_user_part.content[0] == 'Here is my image:'
        assert isinstance(processed_user_part.content[1], ImageUrl)

    async def test_keeps_binary_content_for_model_needing_bytes(self, mock_store: AsyncMock):
        """BinaryContent is kept as-is for models that need bytes."""
        processor = file_store_processor(mock_store)

        gemini_model = TestModel()
        gemini_model._system = 'google-gla'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'user uploaded image', media_type='image/png')
        user_part = UserPromptPart(content=[content])
        request = ModelRequest(parts=[user_part])

        ctx = RunContext(deps=None, model=gemini_model, usage=RunUsage())
        result = await processor(ctx, [request])

        # Still uploaded for storage
        mock_store.store.assert_called_once()

        processed_request = result[0]
        assert isinstance(processed_request, ModelRequest)
        processed_user_part = processed_request.parts[0]
        assert isinstance(processed_user_part, UserPromptPart)
        # But kept as BinaryContent since model needs bytes
        assert isinstance(processed_user_part.content, list)
        assert isinstance(processed_user_part.content[0], BinaryContent)

    async def test_string_content_passes_through(self, mock_store: AsyncMock):
        """String content in UserPromptPart passes through unchanged."""
        processor = file_store_processor(mock_store)

        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        user_part = UserPromptPart(content='Just a text message')
        request = ModelRequest(parts=[user_part])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())
        result = await processor(ctx, [request])

        mock_store.store.assert_not_called()
        assert result[0] is request


class TestFileStoreProcessorDeduplication:
    """Tests for deduplication: same content uploaded once."""

    @pytest.fixture
    def mock_store(self) -> AsyncMock:
        store = AsyncMock()
        store.store = AsyncMock(side_effect=lambda key, data: key)  # pyright: ignore[reportUnknownLambdaType]
        store.get_download_uri = lambda key: f'https://cdn.example.com/{key}'  # pyright: ignore[reportUnknownLambdaType]
        return store

    async def test_same_content_uploaded_once(self, mock_store: AsyncMock):
        """Same content appearing multiple times is only uploaded once."""
        processor = file_store_processor(mock_store)

        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        # Same data = same identifier = same key
        content1 = BinaryContent(data=b'same image data', media_type='image/png')
        content2 = BinaryContent(data=b'same image data', media_type='image/png')
        assert content1.identifier == content2.identifier

        response1 = ModelResponse(parts=[FilePart(content=content1)])
        response2 = ModelResponse(parts=[FilePart(content=content2)])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        # Process both responses
        await processor(ctx, [response1])
        await processor(ctx, [response2])

        # Only uploaded once due to deduplication
        assert mock_store.store.call_count == 1

    async def test_different_content_uploaded_separately(self, mock_store: AsyncMock):
        """Different content is uploaded separately."""
        processor = file_store_processor(mock_store)

        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content1 = BinaryContent(data=b'first image', media_type='image/png')
        content2 = BinaryContent(data=b'second image', media_type='image/png')
        assert content1.identifier != content2.identifier

        response1 = ModelResponse(parts=[FilePart(content=content1)])
        response2 = ModelResponse(parts=[FilePart(content=content2)])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        await processor(ctx, [response1])
        await processor(ctx, [response2])

        # Both uploaded
        assert mock_store.store.call_count == 2


class TestFileStoreProcessorKeyBasedTracking:
    """Tests for key-based tracking: works even when URLs change."""

    async def test_identifier_stored_in_file_url(self):
        """FileUrl.identifier contains the storage key for later lookup."""
        store = AsyncMock()
        store.store = AsyncMock(side_effect=lambda key, data: key)  # pyright: ignore[reportUnknownLambdaType]
        store.get_download_uri = lambda key: f'https://cdn.example.com/{key}?sig=abc'  # pyright: ignore[reportUnknownLambdaType]

        processor = file_store_processor(store)

        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'image data', media_type='image/png')
        expected_key = generate_file_key(content)
        response = ModelResponse(parts=[FilePart(content=content)])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())
        result = await processor(ctx, [response])

        image_url = result[0].parts[0]
        assert isinstance(image_url, ImageUrl)
        # The identifier should be the storage key
        assert image_url.identifier == expected_key

    async def test_tracking_works_with_changing_urls(self):
        """Key-based tracking works even when presigned URLs change."""
        call_count = {'count': 0}

        def get_changing_url(key: str) -> str:
            call_count['count'] += 1
            return f'https://cdn.example.com/{key}?sig={call_count["count"]}'

        store = AsyncMock()
        store.store = AsyncMock(side_effect=lambda key, data: key)  # pyright: ignore[reportUnknownLambdaType]
        store.retrieve = AsyncMock(return_value=b'image data')
        store.get_download_uri = get_changing_url

        processor = file_store_processor(store)

        # Store with OpenAI
        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'image data', media_type='image/png')
        response = ModelResponse(parts=[FilePart(content=content)])

        ctx_openai = RunContext(deps=None, model=openai_model, usage=RunUsage())
        result1 = await processor(ctx_openai, [response])

        # URL has ?sig=1
        url1 = result1[0].parts[0]
        assert isinstance(url1, ImageUrl)
        assert '?sig=1' in url1.url

        # Now switch to Gemini which needs bytes
        gemini_model = TestModel()
        gemini_model._system = 'google-gla'  # pyright: ignore[reportPrivateUsage]

        ctx_gemini = RunContext(deps=None, model=gemini_model, usage=RunUsage())
        result2 = await processor(ctx_gemini, result1)

        # Should successfully load even though URL changed
        # The key is tracked via identifier, not URL
        store.retrieve.assert_called_once()
        assert isinstance(result2[0].parts[0], FilePart)


class TestFileStoreProcessorLiveIntegration:
    """Live integration tests for file_store_processor with real S3.

    These tests verify the processor works correctly with actual S3 storage,
    testing both store direction (FilePart → upload → URL) and load direction
    (URL → download → FilePart).
    """

    @pytest.fixture
    async def live_store(self) -> AsyncIterator[S3FileStore]:
        store = S3FileStore(
            bucket=os.getenv('S3_BUCKET_NAME', 'pydantic-ai-tests'),
            endpoint=os.getenv('S3_ENDPOINT', 'https://accountid.r2.cloudflarestorage.com'),
            access_key_id=os.getenv('S3_ACCESS_KEY_ID', 'test-access-key'),
            secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY', 'test-secret-key'),
            region=os.getenv('S3_REGION', 'auto'),
        )
        yield store
        await store.close()

    @pytest.mark.vcr()
    async def test_processor_stores_and_converts_to_url(self, live_store: S3FileStore):
        """Test processor uploads FilePart to S3 and converts to ImageUrl."""
        processor = file_store_processor(live_store)

        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'live test image data', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        try:
            result = await processor(ctx, [response])

            assert len(result) == 1
            processed_response = result[0]
            assert isinstance(processed_response, ModelResponse)
            assert len(processed_response.parts) == 1

            part = processed_response.parts[0]
            assert isinstance(part, ImageUrl)
            assert part.media_type == 'image/png'
            assert part.identifier == generate_file_key(content)

            # Verify the URL is actually downloadable
            async with httpx.AsyncClient() as client:
                download_response = await client.get(part.url)
                assert download_response.status_code == 200
                assert download_response.content == b'live test image data'
        finally:
            await live_store.delete(generate_file_key(content))

    @pytest.mark.vcr()
    async def test_processor_load_direction_retrieves_from_s3(self, live_store: S3FileStore):
        """Test processor downloads from S3 when model needs bytes."""
        processor = file_store_processor(live_store)

        # First: store with OpenAI (supports URLs)
        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'load direction test data', media_type='image/png')
        file_part = FilePart(content=content)
        response = ModelResponse(parts=[file_part])

        ctx_openai = RunContext(deps=None, model=openai_model, usage=RunUsage())

        try:
            result1 = await processor(ctx_openai, [response])

            # Verify conversion to URL
            url_part = result1[0].parts[0]
            assert isinstance(url_part, ImageUrl)

            # Second: replay to Gemini (needs bytes)
            gemini_model = TestModel()
            gemini_model._system = 'google-gla'  # pyright: ignore[reportPrivateUsage]

            ctx_gemini = RunContext(deps=None, model=gemini_model, usage=RunUsage())
            result2 = await processor(ctx_gemini, result1)

            # Verify conversion back to FilePart
            loaded_part = result2[0].parts[0]
            assert isinstance(loaded_part, FilePart)
            assert loaded_part.content.data == b'load direction test data'
            assert loaded_part.content.media_type == 'image/png'
        finally:
            await live_store.delete(generate_file_key(content))

    @pytest.mark.vcr()
    async def test_processor_handles_user_content(self, live_store: S3FileStore):
        """Test processor handles BinaryContent in UserPromptPart."""
        processor = file_store_processor(live_store)

        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        content = BinaryContent(data=b'user uploaded image', media_type='image/png')
        user_part = UserPromptPart(content=['Check this image:', content])
        request = ModelRequest(parts=[user_part])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        try:
            result = await processor(ctx, [request])

            assert len(result) == 1
            processed_request = result[0]
            assert isinstance(processed_request, ModelRequest)

            processed_user_part = processed_request.parts[0]
            assert isinstance(processed_user_part, UserPromptPart)
            assert isinstance(processed_user_part.content, list)
            assert processed_user_part.content[0] == 'Check this image:'

            url_content = processed_user_part.content[1]
            assert isinstance(url_content, ImageUrl)
            assert url_content.media_type == 'image/png'

            # Verify URL is downloadable
            async with httpx.AsyncClient() as client:
                download_response = await client.get(url_content.url)
                assert download_response.status_code == 200
                assert download_response.content == b'user uploaded image'
        finally:
            await live_store.delete(generate_file_key(content))

    @pytest.mark.vcr()
    async def test_processor_deduplication_with_real_s3(self, live_store: S3FileStore):
        """Test processor deduplicates uploads with real S3."""
        processor = file_store_processor(live_store)

        openai_model = TestModel()
        openai_model._system = 'openai'  # pyright: ignore[reportPrivateUsage]

        # Same data = same identifier = same key
        content1 = BinaryContent(data=b'duplicate test data', media_type='image/png')
        content2 = BinaryContent(data=b'duplicate test data', media_type='image/png')
        assert content1.identifier == content2.identifier

        response1 = ModelResponse(parts=[FilePart(content=content1)])
        response2 = ModelResponse(parts=[FilePart(content=content2)])

        ctx = RunContext(deps=None, model=openai_model, usage=RunUsage())

        try:
            # Process both
            result1 = await processor(ctx, [response1])
            result2 = await processor(ctx, [response2])

            # Both should succeed and point to same URL
            url1 = result1[0].parts[0]
            url2 = result2[0].parts[0]
            assert isinstance(url1, ImageUrl)
            assert isinstance(url2, ImageUrl)
            assert url1.identifier == url2.identifier

            # File should exist only once in S3
            exists = await live_store.exists(generate_file_key(content1))
            assert exists is True
        finally:
            await live_store.delete(generate_file_key(content1))
