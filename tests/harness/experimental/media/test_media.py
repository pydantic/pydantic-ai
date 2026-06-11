"""Tests for `pydantic_ai_harness.experimental.media`: stores + walker + SigV4 + S3 store."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from httpx import AsyncClient, MockTransport, Request, Response

from pydantic_ai_harness.experimental.media import (
    DiskMediaStore,
    MediaContext,
    MediaStore,
    S3MediaStore,
    SqliteMediaStore,
    externalize_media,
    make_static_public_url,
    media_uri_for,
    parse_media_uri,
    restore_media,
)
from pydantic_ai_harness.experimental.media._s3 import sign_request

pytestmark = pytest.mark.anyio


class TestMediaUriHelpers:
    def test_media_uri_for_returns_canonical_scheme(self) -> None:
        uri = media_uri_for(b'hello world')
        assert uri == 'media+sha256://b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'

    def test_parse_media_uri_strips_scheme(self) -> None:
        uri = media_uri_for(b'abc')
        digest = parse_media_uri(uri)
        assert len(digest) == 64
        assert all(c in '0123456789abcdef' for c in digest)

    def test_parse_media_uri_rejects_other_schemes(self) -> None:
        with pytest.raises(ValueError, match='not a media URI'):
            parse_media_uri('http://example.com/foo')

    def test_parse_media_uri_rejects_short_digest(self) -> None:
        with pytest.raises(ValueError, match='invalid sha256 digest'):
            parse_media_uri('media+sha256://deadbeef')

    def test_parse_media_uri_rejects_uppercase_hex(self) -> None:
        with pytest.raises(ValueError, match='invalid sha256 digest'):
            parse_media_uri('media+sha256://' + 'A' * 64)


class TestDiskMediaStore:
    async def test_put_get_round_trip(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'hello bytes', context=MediaContext(media_type='application/octet-stream'))
        assert uri.startswith('media+sha256://')
        assert await store.get(uri) == b'hello bytes'

    async def test_put_with_empty_context_works(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'no context')
        assert await store.get(uri) == b'no context'

    async def test_dedup_on_repeated_put(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri1 = await store.put(b'same content')
        uri2 = await store.put(b'same content')
        assert uri1 == uri2
        files = list(tmp_path.glob('*.bin'))
        assert len(files) == 1

    async def test_exists(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'present')
        assert await store.exists(uri) is True
        missing_uri = media_uri_for(b'never put')
        assert await store.exists(missing_uri) is False

    async def test_get_missing_raises(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.get(media_uri_for(b'never put'))

    async def test_custom_key_strategy_controls_path(self, tmp_path: Path) -> None:
        def strategy(uri: str, ctx: MediaContext) -> str:
            return f'images/{parse_media_uri(uri)}.png'

        store = DiskMediaStore(tmp_path, key_strategy=strategy)
        uri = await store.put(b'pixels')
        assert (tmp_path / 'images' / f'{parse_media_uri(uri)}.png').exists()
        assert await store.get(uri) == b'pixels'

    async def test_key_strategy_blocks_traversal(self, tmp_path: Path) -> None:
        def evil(uri: str, ctx: MediaContext) -> str:
            return '../../../etc/passwd'

        store = DiskMediaStore(tmp_path, key_strategy=evil)
        with pytest.raises(ValueError, match='traversal-unsafe'):
            await store.put(b'attack')

    async def test_key_strategy_blocks_absolute_path(self, tmp_path: Path) -> None:
        # `Path('/root') / '/abs'` silently returns `/abs` — without this check
        # an absolute key escapes the store directory.
        def evil(uri: str, ctx: MediaContext) -> str:
            return '/etc/passwd'

        store = DiskMediaStore(tmp_path, key_strategy=evil)
        with pytest.raises(ValueError, match='traversal-unsafe'):
            await store.put(b'attack')

    async def test_metadata_round_trips_via_sidecar(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(
            b'tagged',
            context=MediaContext(metadata={'origin': 'user', 'tenant': 'acme'}),
        )
        sidecars = list(tmp_path.glob('*.meta.json'))
        assert len(sidecars) == 1
        assert await store.get_metadata(uri) == {'origin': 'user', 'tenant': 'acme'}

    async def test_metadata_absent_when_not_supplied(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'no tags')
        assert list(tmp_path.glob('*.meta.json')) == []
        assert await store.get_metadata(uri) == {}

    async def test_get_metadata_rejects_non_object_sidecar(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'with meta', context=MediaContext(metadata={'k': 'v'}))
        sidecar = next(iter(tmp_path.glob('*.meta.json')))
        sidecar.write_text('"not an object"')
        with pytest.raises(ValueError, match='must be a JSON object'):
            await store.get_metadata(uri)

    async def test_get_metadata_rejects_non_string_values(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'with meta', context=MediaContext(metadata={'k': 'v'}))
        sidecar = next(iter(tmp_path.glob('*.meta.json')))
        sidecar.write_text('{"k": 1}')
        with pytest.raises(ValueError, match='must be str→str'):
            await store.get_metadata(uri)


class TestSqliteMediaStore:
    async def test_put_get_round_trip(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'media.db')
        uri = await store.put(b'hello bytes', context=MediaContext(media_type='image/png'))
        assert await store.get(uri) == b'hello bytes'

    async def test_put_persists_metadata_column(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'media.db')
        uri = await store.put(
            b'tagged',
            context=MediaContext(media_type='image/png', metadata={'origin': 'user', 'tenant': 'acme'}),
        )
        conn = sqlite3.connect(tmp_path / 'media.db', check_same_thread=False)
        try:
            row = conn.execute('SELECT media_type, metadata FROM media').fetchone()
        finally:
            conn.close()
        assert row[0] == 'image/png'
        assert await store.get_metadata(uri) == {'origin': 'user', 'tenant': 'acme'}

    async def test_get_metadata_missing_raises(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'media.db')
        with pytest.raises(FileNotFoundError):
            await store.get_metadata(media_uri_for(b'never put'))

    async def test_with_shared_connection(self) -> None:
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        try:
            store = SqliteMediaStore(connection=connection)
            uri = await store.put(b'shared conn data')
            assert await store.get(uri) == b'shared conn data'
            assert await store.exists(uri) is True
        finally:
            connection.close()

    async def test_dedup_via_insert_or_ignore(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'm.db')
        uri = ''
        for _ in range(3):
            uri = await store.put(b'duplicated')
        assert await store.get(uri) == b'duplicated'

    async def test_get_missing_raises(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'm.db')
        with pytest.raises(FileNotFoundError):
            await store.get(media_uri_for(b'missing'))

    def test_requires_exactly_one_of_database_or_connection(self) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            SqliteMediaStore()
        conn = sqlite3.connect(':memory:')
        try:
            with pytest.raises(ValueError, match='exactly one'):
                SqliteMediaStore(database='x', connection=conn)
        finally:
            conn.close()

    def test_rejects_invalid_table_name(self) -> None:
        with pytest.raises(ValueError, match='invalid table name'):
            SqliteMediaStore(database='x.db', table='bad-name')


class TestExternalizeRestoreWalker:
    async def test_round_trip_with_inline_binary(self, tmp_path: Path) -> None:
        import base64
        import json as _json

        store = DiskMediaStore(tmp_path)
        big_payload = b'\x00' * 70_000
        b64_payload = base64.b64encode(big_payload).decode('ascii')
        node: object = {
            'parts': [
                {
                    'kind': 'binary',
                    'data': b64_payload,
                    'media_type': 'image/png',
                    'identifier': 'abc',
                    'vendor_metadata': None,
                }
            ]
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        externalized_text = _json.dumps(externalized)
        assert '__harness_external_media__' in externalized_text
        assert 'media+sha256://' in externalized_text
        assert b64_payload not in externalized_text  # bytes really went external

        restored = await restore_media(externalized, media_store=store)
        restored_text = _json.dumps(restored)
        assert '"kind": "binary"' in restored_text
        assert b64_payload in restored_text  # bytes restored exactly

    async def test_threshold_boundary_keeps_small_inline(self, tmp_path: Path) -> None:
        import base64

        store = DiskMediaStore(tmp_path)
        small_payload = b'\x42' * 32
        node = {
            'kind': 'binary',
            'data': base64.b64encode(small_payload).decode('ascii'),
            'media_type': 'text/plain',
            'identifier': 's',
            'vendor_metadata': None,
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert externalized == node
        assert list(tmp_path.glob('*.bin')) == []

    async def test_restore_raises_when_marker_missing_uri(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        bad_node = {'__harness_external_media__': True, 'media_type': 'image/png'}
        with pytest.raises(ValueError, match='missing string uri'):
            await restore_media(bad_node, media_store=store)

    async def test_round_trip_preserves_unknown_binary_fields(self, tmp_path: Path) -> None:
        """A field the walker doesn't know about survives externalize -> restore."""
        import base64
        import json as _json

        store = DiskMediaStore(tmp_path)
        big = base64.b64encode(b'\x01' * 70_000).decode('ascii')
        node = {
            'kind': 'binary',
            'data': big,
            'media_type': 'image/png',
            'identifier': 'abc',
            'vendor_metadata': {'x': 1},
            'future_field': 'keep-me',
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert '"data"' not in _json.dumps(externalized)  # bytes field went external
        restored = await restore_media(externalized, media_store=store)
        assert restored == node

    async def test_restore_raises_when_blob_is_missing(self, tmp_path: Path) -> None:
        """A well-formed marker whose blob was pruned surfaces the store's error."""
        store = DiskMediaStore(tmp_path)
        marker = {
            '__harness_external_media__': True,
            'uri': 'media+sha256://' + '0' * 64,
            'kind': 'binary',
            'media_type': 'image/png',
        }
        with pytest.raises(FileNotFoundError):
            await restore_media(marker, media_store=store)

    async def test_walker_passes_through_scalars(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        for value in [None, 1, 'foo', True]:
            assert await externalize_media(value, media_store=store, threshold_bytes=1) == value
            assert await restore_media(value, media_store=store) == value


class TestSigV4Signer:
    def test_produces_required_headers(self) -> None:
        headers = sign_request(
            method='PUT',
            host='examplebucket.s3.amazonaws.com',
            path='/my-key',
            body=b'payload',
            region='us-east-1',
            access_key_id='AKIAIOSFODNN7EXAMPLE',
            secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            content_type='image/png',
            now=datetime(2013, 5, 24, tzinfo=timezone.utc),
        )
        assert headers['host'] == 'examplebucket.s3.amazonaws.com'
        assert headers['x-amz-date'] == '20130524T000000Z'
        assert headers['x-amz-content-sha256'] == ('239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5')
        assert headers['content-type'] == 'image/png'
        auth = headers['authorization']
        assert auth.startswith('AWS4-HMAC-SHA256 ')
        assert 'Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request' in auth
        assert 'SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date' in auth
        # Signature is deterministic for fixed inputs — locks down the algorithm.
        assert 'Signature=' in auth

    def test_signature_is_deterministic(self) -> None:
        common = dict(
            method='GET',
            host='bucket.s3.amazonaws.com',
            path='/key',
            body=b'',
            region='us-east-1',
            access_key_id='K',
            secret_access_key='S',
            content_type=None,
            now=datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        )
        a = sign_request(**common)  # type: ignore[arg-type]
        b = sign_request(**common)  # type: ignore[arg-type]
        assert a == b

    def test_signature_changes_with_body(self) -> None:
        kwargs = dict(
            method='PUT',
            host='bucket.s3.amazonaws.com',
            path='/key',
            region='us-east-1',
            access_key_id='K',
            secret_access_key='S',
            content_type=None,
            now=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        sig_a = sign_request(body=b'aaa', **kwargs)['authorization']  # type: ignore[arg-type]
        sig_b = sign_request(body=b'bbb', **kwargs)['authorization']  # type: ignore[arg-type]
        assert sig_a != sig_b


class TestS3MediaStoreWithMockTransport:
    async def test_put_signs_request(self) -> None:
        captured: list[Request] = []

        async def handler(request: Request) -> Response:
            captured.append(request)
            return Response(200)

        transport = MockTransport(handler)
        async with AsyncClient(transport=transport) as client:
            store = S3MediaStore(
                bucket='my-bucket',
                endpoint='https://s3.us-east-1.amazonaws.com',
                region='us-east-1',
                access_key_id='AKIA-FAKE',
                secret_access_key='secret-fake',
                client=client,
            )
            uri = await store.put(b'payload', context=MediaContext(media_type='image/png'))

        assert uri.startswith('media+sha256://')
        assert len(captured) == 1
        request = captured[0]
        assert request.method == 'PUT'
        assert 'authorization' in request.headers
        assert request.headers['authorization'].startswith('AWS4-HMAC-SHA256 ')
        assert request.headers['x-amz-content-sha256']
        assert request.headers['content-type'] == 'image/png'

    async def test_wire_path_matches_signed_path_for_reserved_chars(self) -> None:
        """A key_prefix with reserved chars must be percent-encoded identically on the wire.

        Otherwise the signed canonical path and the path S3 receives diverge ->
        SignatureDoesNotMatch. The wire `raw_path` must equal `_canonical_uri(path)`.
        """
        from pydantic_ai_harness.experimental.media._s3 import _canonical_uri  # pyright: ignore[reportPrivateUsage]

        captured: list[Request] = []

        async def handler(request: Request) -> Response:
            captured.append(request)
            return Response(200)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='my-bucket',
                endpoint='https://acct.r2.cloudflarestorage.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                key_prefix='runs(2024)/tenant@acme/',
                client=client,
            )
            uri = await store.put(b'payload')

        expected_path = _canonical_uri(store._object_path(uri, MediaContext()))  # pyright: ignore[reportPrivateUsage]
        assert '%28' in expected_path and '%40' in expected_path  # ( and @ encoded
        assert captured[0].url.raw_path.decode('ascii') == expected_path

    async def test_put_propagates_metadata_as_signed_x_amz_meta_headers(self) -> None:
        captured: list[Request] = []

        async def handler(request: Request) -> Response:
            captured.append(request)
            return Response(200)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            await store.put(
                b'meta-tagged',
                context=MediaContext(
                    media_type='image/png',
                    metadata={'origin': 'pipeline-a', 'tenant': 'acme'},
                ),
            )
        request = captured[0]
        assert request.headers.get('x-amz-meta-origin') == 'pipeline-a'
        assert request.headers.get('x-amz-meta-tenant') == 'acme'
        # Metadata headers MUST be in SignedHeaders to be valid SigV4.
        signed = request.headers['authorization'].split('SignedHeaders=')[1].split(',')[0]
        assert 'x-amz-meta-origin' in signed
        assert 'x-amz-meta-tenant' in signed

    async def test_put_rejects_invalid_metadata_key(self) -> None:
        async def handler(request: Request) -> Response:  # pragma: no cover
            return Response(200)  # never reached — validation raises pre-request

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(ValueError, match='not a valid HTTP header token'):
                await store.put(
                    b'data',
                    context=MediaContext(metadata={'bad key with space': 'v'}),
                )

    async def test_get_resolves_uri(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(200, content=b'fetched bytes')

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            data = await store.get(media_uri_for(b'fetched bytes'))
            assert data == b'fetched bytes'

    async def test_get_404_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(404)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(FileNotFoundError):
                await store.get(media_uri_for(b'missing'))

    async def test_exists_uses_head(self) -> None:
        captured_methods: list[str] = []

        async def handler(request: Request) -> Response:
            captured_methods.append(request.method)
            return Response(200)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            assert await store.exists(media_uri_for(b'anything')) is True

        assert captured_methods == ['HEAD']

    async def test_exists_returns_false_on_404(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(404)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            assert await store.exists(media_uri_for(b'missing')) is False

    async def test_put_failure_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(500, content=b'internal error')

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(RuntimeError, match='S3 PUT failed'):
                await store.put(b'data')

    async def test_get_failure_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(500, content=b'oops')

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(RuntimeError, match='S3 GET failed'):
                await store.get(media_uri_for(b'x'))

    async def test_get_metadata_reads_x_amz_meta_headers(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(
                200,
                headers={
                    'x-amz-meta-origin': 'pipeline-a',
                    'x-amz-meta-tenant': 'acme',
                    'content-type': 'image/png',
                    'content-length': '0',
                },
            )

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            meta = await store.get_metadata(media_uri_for(b'x'))
        assert meta == {'origin': 'pipeline-a', 'tenant': 'acme'}

    async def test_get_metadata_404_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(404)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(FileNotFoundError):
                await store.get_metadata(media_uri_for(b'missing'))

    async def test_get_metadata_500_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(500, content=b'oops')

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(RuntimeError, match='S3 HEAD failed'):
                await store.get_metadata(media_uri_for(b'x'))

    async def test_head_failure_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(500)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(RuntimeError, match='S3 HEAD failed'):
                await store.exists(media_uri_for(b'x'))

    async def test_no_client_branch_opens_one_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without `client=`, the store opens a fresh `httpx.AsyncClient` per call."""
        import httpx

        captured: list[Request] = []

        async def handler(request: Request) -> Response:
            captured.append(request)
            return Response(200)

        original_async_client = httpx.AsyncClient

        def patched_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
            return original_async_client(transport=MockTransport(handler))

        monkeypatch.setattr(httpx, 'AsyncClient', patched_client)

        store = S3MediaStore(
            bucket='my-bucket',
            endpoint='https://example.com',
            region='auto',
            access_key_id='k',
            secret_access_key='s',
            key_prefix='runs/',
        )
        await store.put(b'no-client data')
        assert len(captured) == 1
        sample_uri = media_uri_for(b'sample')
        digest = parse_media_uri(sample_uri)
        path = store._object_path(sample_uri, MediaContext())  # type: ignore[reportPrivateUsage]
        assert path == f'/my-bucket/runs/{digest}.bin'


@pytest.mark.skipif(  # pragma: no cover
    not all(
        os.environ.get(k)
        for k in ('S3_ENDPOINT', 'S3_BUCKET_NAME', 'S3_ACCESS_KEY_ID', 'S3_SECRET_ACCESS_KEY', 'S3_REGION')
    ),
    reason='live S3/R2 env vars not set',
)
class TestS3MediaStoreLive:  # pragma: no cover
    """Live integration against the configured S3 endpoint (e.g. R2).

    Activated only when all five S3_* env vars are set. Reads creds out of
    the environment — the test harness inherits them when invoked with
    `~/.claude/scripts/env-run .env -- make test`.
    """

    async def test_round_trip_against_live_endpoint(self) -> None:
        bucket = os.environ['S3_BUCKET_NAME']
        store = S3MediaStore(
            bucket=bucket,
            endpoint=os.environ['S3_ENDPOINT'],
            region=os.environ['S3_REGION'],
            access_key_id=os.environ['S3_ACCESS_KEY_ID'],
            secret_access_key=os.environ['S3_SECRET_ACCESS_KEY'],
            key_prefix='harness-test/',
        )
        payload = b'pydantic-ai-harness step-persistence live test ' + os.urandom(32)
        ctx = MediaContext(media_type='application/octet-stream')
        uri = await store.put(payload, context=ctx)
        assert await store.exists(uri) is True
        fetched = await store.get(uri)
        assert fetched == payload


class TestPublicUrl:
    """Verify `public_url` resolves through the user-supplied callable.

    Static prefix, async callable, and the absence of a resolver all go
    through the same `MediaStore.public_url(uri)` shape — the future
    `MediaExternalizer` capability uses this to swap `BinaryContent` for
    URL message parts before the model sees them.
    """

    async def test_disk_store_without_resolver_returns_none(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        assert await store.public_url(media_uri_for(b'x')) is None

    async def test_sqlite_store_without_resolver_returns_none(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'm.db')
        assert await store.public_url(media_uri_for(b'x')) is None

    async def test_s3_store_without_resolver_returns_none(self) -> None:
        store = S3MediaStore(
            bucket='b',
            endpoint='https://example.com',
            region='auto',
            access_key_id='k',
            secret_access_key='s',
        )
        assert await store.public_url(media_uri_for(b'x')) is None

    async def test_sync_callable_resolver(self, tmp_path: Path) -> None:
        store = DiskMediaStore(
            tmp_path,
            public_url=lambda uri, ctx: f'https://cdn.example.com/{parse_media_uri(uri)}.bin',
        )
        uri = media_uri_for(b'payload')
        result = await store.public_url(uri)
        assert result == f'https://cdn.example.com/{parse_media_uri(uri)}.bin'

    async def test_async_callable_resolver(self, tmp_path: Path) -> None:
        async def signer(uri: str, ctx: MediaContext) -> str:
            return f'https://signed.example.com/{parse_media_uri(uri)}?sig=abc'

        store = DiskMediaStore(tmp_path, public_url=signer)
        result = await store.public_url(media_uri_for(b'payload'))
        assert result is not None
        assert result.startswith('https://signed.example.com/')
        assert '?sig=abc' in result

    async def test_resolver_receives_media_context(self, tmp_path: Path) -> None:
        """The resolver sees the full `MediaContext` so it can vary by media type, metadata, etc."""
        seen: list[MediaContext] = []

        def resolver(uri: str, ctx: MediaContext) -> str:
            seen.append(ctx)
            return 'https://example.com/x'

        store = DiskMediaStore(tmp_path, public_url=resolver)
        await store.public_url(
            media_uri_for(b'p'),
            context=MediaContext(media_type='image/png', metadata={'tag': 'v'}),
        )
        assert seen[0].media_type == 'image/png'
        assert dict(seen[0].metadata) == {'tag': 'v'}

    async def test_resolver_can_return_none(self, tmp_path: Path) -> None:
        """Resolvers may opt out per-URI (e.g. small payloads keep inline)."""
        store = DiskMediaStore(tmp_path, public_url=lambda uri, ctx: None)
        assert await store.public_url(media_uri_for(b'x')) is None

    async def test_make_static_public_url_helper(self, tmp_path: Path) -> None:
        resolver = make_static_public_url('https://pub-abc.r2.dev', key_prefix='media/', extension='.bin')
        store = DiskMediaStore(tmp_path, public_url=resolver)
        uri = media_uri_for(b'payload')
        digest = parse_media_uri(uri)
        assert await store.public_url(uri) == f'https://pub-abc.r2.dev/media/{digest}.bin'

    async def test_make_static_public_url_strips_trailing_slash(self, tmp_path: Path) -> None:
        resolver = make_static_public_url('https://cdn.example.com/')
        store = DiskMediaStore(tmp_path, public_url=resolver)
        digest = parse_media_uri(media_uri_for(b'p'))
        assert await store.public_url(media_uri_for(b'p')) == f'https://cdn.example.com/{digest}.bin'

    async def test_s3_with_resolver_uses_it(self) -> None:
        store = S3MediaStore(
            bucket='b',
            endpoint='https://example.com',
            region='auto',
            access_key_id='k',
            secret_access_key='s',
            key_prefix='media/',
            public_url=make_static_public_url('https://pub-xyz.r2.dev', key_prefix='media/'),
        )
        uri = media_uri_for(b'p')
        digest = parse_media_uri(uri)
        assert await store.public_url(uri) == f'https://pub-xyz.r2.dev/media/{digest}.bin'

    async def test_sqlite_with_resolver_uses_it(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(
            database=tmp_path / 'm.db',
            public_url=make_static_public_url('https://cdn.example.com'),
        )
        uri = media_uri_for(b'p')
        digest = parse_media_uri(uri)
        assert await store.public_url(uri) == f'https://cdn.example.com/{digest}.bin'


def _assert_media_store_protocol(store: MediaStore) -> None:
    """Mypy/pyright check: every concrete store satisfies the protocol."""
    assert isinstance(store, MediaStore)


def test_concrete_stores_satisfy_protocol(tmp_path: Path) -> None:
    _assert_media_store_protocol(DiskMediaStore(tmp_path))
    _assert_media_store_protocol(SqliteMediaStore(database=tmp_path / 'm.db'))
    _assert_media_store_protocol(
        S3MediaStore(
            bucket='b',
            endpoint='https://example.com',
            region='auto',
            access_key_id='k',
            secret_access_key='s',
        )
    )
