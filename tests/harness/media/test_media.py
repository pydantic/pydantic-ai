"""Tests for `pydantic_ai_harness.media`: stores + walker + SigV4 + S3 store."""

from __future__ import annotations

import base64
import functools
import json
import json as _json
import os
import sqlite3
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeGuard

import anyio
import httpx
import pytest
from httpx import AsyncClient, MockTransport, Request, Response

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    StructuredContent,
    TextContent,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.media import (
    DiskMediaStore,
    MediaContext,
    MediaStore,
    PostgresConnection,
    PostgresMediaStore,
    S3MediaStore,
    SqliteMediaStore,
    externalize_media,
    make_static_public_url,
    media_uri_for,
    parse_media_uri,
    restore_media,
)
from pydantic_ai_harness.media._s3 import (
    _canonical_uri,  # pyright: ignore[reportPrivateUsage]
    sign_request,
)


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
        with pytest.raises(ValueError, match='must map str to str'):
            await store.get_metadata(uri)

    async def test_get_metadata_missing_raises(self, tmp_path: Path) -> None:
        """A blob that was never stored raises `FileNotFoundError`, like every other store."""
        store = DiskMediaStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.get_metadata(media_uri_for(b'never put'))

    async def test_later_put_with_metadata_replaces_the_sidecar(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'tagged twice', context=MediaContext(metadata={'writer': 'first'}))
        await store.put(b'tagged twice', context=MediaContext(metadata={'writer': 'second'}))
        assert await store.get_metadata(uri) == {'writer': 'second'}

    async def test_concurrent_puts_of_the_same_bytes_all_succeed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writers racing on one digest each use their own temporary file.

        The first writer's `replace` runs a whole second `put`, which writes and
        renames its own temporary file, before the first writer renames its own.
        """
        store = DiskMediaStore(tmp_path)
        data = b'raced'
        context = MediaContext(metadata={'k': 'v'})
        real_replace = Path.replace
        interleaved: list[str] = []

        def replace_after_another_writer(self: Path, target: Path) -> Path:
            if not interleaved:
                interleaved.append('running')
                # `put` runs this on a worker thread, so the second writer goes back through the event loop.
                interleaved.append(anyio.from_thread.run(functools.partial(store.put, data, context=context)))
            return real_replace(self, target)

        monkeypatch.setattr(Path, 'replace', replace_after_another_writer)
        uri = await store.put(data, context=context)

        assert interleaved == ['running', uri]
        assert await store.get(uri) == data
        assert await store.get_metadata(uri) == {'k': 'v'}
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            f'{parse_media_uri(uri)}.bin',
            f'{parse_media_uri(uri)}.bin.meta.json',
        ]

    async def test_failed_write_leaves_no_temporary_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        store = DiskMediaStore(tmp_path)

        def fail_replace(self: Path, target: Path) -> Path:
            raise OSError('disk full')

        monkeypatch.setattr(Path, 'replace', fail_replace)
        with pytest.raises(OSError, match='disk full'):
            await store.put(b'never lands')
        assert list(tmp_path.iterdir()) == []

    async def test_rejects_a_malformed_uri_before_the_key_strategy(self, tmp_path: Path) -> None:
        """A URI read back from a snapshot is validated even when the strategy splices it in unparsed."""
        (tmp_path / 'images').mkdir()
        (tmp_path / 'images' / 'other.bin').write_bytes(b'not addressed by any digest')

        def splice(uri: str, ctx: MediaContext) -> str:
            return 'images/' + uri.removeprefix('media+sha256://') + '.bin'

        store = DiskMediaStore(tmp_path, key_strategy=splice)
        stored = await store.put(b'addressed by its digest')
        assert await store.get(stored) == b'addressed by its digest'
        forged = 'media+sha256://other'
        with pytest.raises(ValueError, match='invalid sha256 digest'):
            await store.get(forged)
        with pytest.raises(ValueError, match='invalid sha256 digest'):
            await store.exists(forged)
        with pytest.raises(ValueError, match='invalid sha256 digest'):
            await store.get_metadata(forged)


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

    async def test_shared_connection_put_is_committed(self, tmp_path: Path) -> None:
        """A put on a default-isolation connection commits, so it survives close and blocks no writer."""
        database = tmp_path / 'media.db'
        connection = sqlite3.connect(database, check_same_thread=False)
        try:
            uri = await SqliteMediaStore(connection=connection).put(b'committed')
            assert not connection.in_transaction
            other = sqlite3.connect(database, timeout=0.1)
            try:
                other.execute('CREATE TABLE unrelated (x)')
                other.commit()
            finally:
                other.close()
        finally:
            connection.close()
        assert await SqliteMediaStore(database=database).get(uri) == b'committed'

    async def test_shared_connection_put_leaves_a_caller_transaction_open(self, tmp_path: Path) -> None:
        """Inside a transaction the caller opened, the caller decides whether the row lands."""
        database = tmp_path / 'media.db'
        connection = sqlite3.connect(database, check_same_thread=False)
        try:
            store = SqliteMediaStore(connection=connection)
            await store.exists(media_uri_for(b'schema setup'))
            connection.execute('BEGIN')
            uri = await store.put(b'rolled back')
            assert connection.in_transaction
            connection.rollback()
            assert await store.exists(uri) is False
        finally:
            connection.close()

    async def test_first_call_inside_a_caller_transaction_commits_none_of_it(self, tmp_path: Path) -> None:
        """Creating the table on the first call leaves the caller's pending work to the caller."""
        database = tmp_path / 'media.db'
        connection = sqlite3.connect(database, check_same_thread=False)
        try:
            connection.execute('CREATE TABLE app (x)')
            connection.commit()
            connection.execute('INSERT INTO app VALUES (1)')
            store = SqliteMediaStore(connection=connection)
            # The first call creates the table, inside the caller's transaction.
            assert await store.exists(media_uri_for(b'first call')) is False
            uri = await store.put(b'first call')
            assert connection.in_transaction
            connection.rollback()
            assert connection.execute('SELECT count(*) FROM app').fetchone() == (0,)
            # The rollback took the media table with it, and the store creates it again.
            assert await store.exists(uri) is False
            assert await store.get(await store.put(b'again')) == b'again'
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


def _is_marker_dict(node: object) -> TypeGuard[dict[str, object]]:
    """Narrow the walker's `object` return to the JSON dict it produced."""
    return isinstance(node, dict)


def _is_object_list(node: object) -> TypeGuard[list[object]]:
    """Narrow a JSON array without propagating `json.loads`'s `Any`."""
    return isinstance(node, list)


async def _restore_as_pre_pr_reader(node: dict[str, object], store: MediaStore) -> dict[str, object]:
    """Behavior of `_restore_external` before this PR added text markers.

    The previous reader treats every marker as binary. Its missing-`uri` guard
    is left out because the markers under test always carry one.
    """

    uri_value = node.get('uri')
    assert isinstance(uri_value, str)
    raw = await store.get(uri_value)
    restored = {key: value for key, value in node.items() if key not in ('__harness_external_media__', 'uri')}
    restored['data'] = base64.b64encode(raw).decode('ascii')
    return restored


async def _restore_as_pre_escaping_reader(node: dict[str, object], store: MediaStore) -> dict[str, object]:
    """Behavior of `_restore_external` at the merge-base commit, before the escaping format.

    That reader knows `_URI_KEY` and text markers: it takes the reference from
    `_URI_KEY`, drops the `uri` mirror, re-inlines a text marker's payload as
    `content`, and keeps every key it does not recognize, so a marker written by
    the escaping format restores with the stash and the stamp still in the
    payload. It reads the blob under the same `MediaContext` that reader built,
    so a context-dependent `key_strategy` resolves the same key on the way back.
    Three of its branches are left out because no marker under test reaches
    them: the plain-`uri` fallback, since the writer always emits `_URI_KEY`;
    the caller-owned `uri` that is not a mirror, since these source nodes carry
    no `uri`; and the recursion into preserved fields, since none of these
    markers nests another.
    """
    dropped = {
        '__harness_external_media__',
        '__harness_external_text__',
        '__harness_external_uri__',
        'uri',
    }
    uri_value = node['__harness_external_uri__']
    assert isinstance(uri_value, str)
    is_text = node.get('__harness_external_text__') is True
    if is_text:
        context = MediaContext(media_type='text/plain')
    else:
        media_type = node.get('media_type')
        assert media_type is None or isinstance(media_type, str)
        context = MediaContext(media_type=media_type)
    raw = await store.get(uri_value, context=context)
    restored = {key: value for key, value in node.items() if key not in dropped}
    if is_text:
        restored['content'] = raw.decode('utf-8', errors='surrogatepass')
    else:
        restored['data'] = base64.b64encode(raw).decode('ascii')
    return restored


class TestExternalizeRestoreWalker:
    async def test_round_trip_with_inline_binary(self, tmp_path: Path) -> None:

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

    async def test_binary_round_trip_accepts_pydantic_urlsafe_base64(self, tmp_path: Path) -> None:
        payload = bytes(range(256)) * 500
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[UserPromptPart(content=[BinaryContent(data=payload, media_type='application/octet-stream')])]
            )
        ]
        node: object = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
        store = DiskMediaStore(tmp_path)

        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        restored = await restore_media(externalized, media_store=store)
        [restored_message] = ModelMessagesTypeAdapter.validate_python(restored)
        assert isinstance(restored_message, ModelRequest)
        [restored_part] = restored_message.parts
        assert isinstance(restored_part, UserPromptPart)
        assert not isinstance(restored_part.content, str)
        [restored_binary] = restored_part.content
        assert isinstance(restored_binary, BinaryContent)

        assert restored_binary.data == payload

    async def test_binary_restore_reuses_write_context(self, tmp_path: Path) -> None:
        """A context-dependent storage key is found with the binary media type."""

        seen_contexts: list[MediaContext] = []

        def key_strategy(uri: str, context: MediaContext) -> str:
            seen_contexts.append(context)
            return f'{context.media_type or "none"}/{parse_media_uri(uri)}'

        store = DiskMediaStore(tmp_path, key_strategy=key_strategy)
        node = {
            'kind': 'binary',
            'data': base64.b64encode(b'\x00' * 128).decode('ascii'),
            'media_type': 'image/png',
        }

        externalized = await externalize_media(node, media_store=store, threshold_bytes=64)
        assert await restore_media(externalized, media_store=store) == node
        assert seen_contexts == [MediaContext(media_type='image/png'), MediaContext(media_type='image/png')]

    async def test_threshold_boundary_keeps_small_inline(self, tmp_path: Path) -> None:

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

    async def test_binary_threshold_boundary_is_inclusive(self, tmp_path: Path) -> None:
        """Exactly `threshold_bytes` externalizes; one byte under stays inline."""

        store = DiskMediaStore(tmp_path)
        threshold = 256

        def binary_node(size: int) -> dict[str, object]:
            return {'kind': 'binary', 'data': base64.b64encode(b'\x07' * size).decode('ascii')}

        under = binary_node(threshold - 1)
        assert await externalize_media(under, media_store=store, threshold_bytes=threshold) == under
        assert list(tmp_path.glob('*.bin')) == []

        at = binary_node(threshold)
        externalized = await externalize_media(at, media_store=store, threshold_bytes=threshold)
        assert externalized != at
        assert await restore_media(externalized, media_store=store) == at

    async def test_text_threshold_boundary_is_inclusive(self, tmp_path: Path) -> None:
        """Exactly `threshold_bytes` externalizes; one byte under stays inline."""
        store = DiskMediaStore(tmp_path)
        threshold = 256

        under = {'content': 'y' * (threshold - 1), 'part_kind': 'text'}
        assert await externalize_media(under, media_store=store, threshold_bytes=threshold) == under
        assert list(tmp_path.glob('*.bin')) == []

        at = {'content': 'y' * threshold, 'part_kind': 'text'}
        externalized = await externalize_media(at, media_store=store, threshold_bytes=threshold)
        assert externalized != at
        assert await restore_media(externalized, media_store=store) == at

    async def test_restore_raises_when_marker_missing_uri(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        bad_node = {'__harness_external_media__': True, 'media_type': 'image/png'}
        with pytest.raises(ValueError, match='missing string uri'):
            await restore_media(bad_node, media_store=store)

    async def test_round_trip_preserves_unknown_binary_fields(self, tmp_path: Path) -> None:
        """A field the walker doesn't know about survives externalize -> restore."""

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
            '__harness_external_uri__': 'media+sha256://' + '0' * 64,
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

    async def test_large_text_part_round_trips(self, tmp_path: Path) -> None:
        """A part carrying a large string `content` externalizes to text and restores.

        Closes #440: large tool-return strings would otherwise stay inline and
        push a snapshot past MongoDB's 16MB document cap.
        """

        store = DiskMediaStore(tmp_path)
        big_text = 'x' * 70_000
        node: object = {
            'parts': [{'content': big_text, 'tool_name': 'scrape', 'tool_call_id': 'c1', 'part_kind': 'tool-return'}]
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        externalized_text = _json.dumps(externalized)
        assert '__harness_external_text__' in externalized_text
        assert 'media+sha256://' in externalized_text
        assert big_text not in externalized_text  # text really went external
        assert list(tmp_path.glob('*.bin'))  # bytes on disk

        restored = await restore_media(externalized, media_store=store)
        assert restored == node  # content re-inlined, every other field preserved

    async def test_text_restore_reuses_write_context(self, tmp_path: Path) -> None:
        """A context-dependent storage key is found with `text/plain`."""
        seen_contexts: list[MediaContext] = []

        def key_strategy(uri: str, context: MediaContext) -> str:
            seen_contexts.append(context)
            return f'{context.media_type or "none"}/{parse_media_uri(uri)}'

        store = DiskMediaStore(tmp_path, key_strategy=key_strategy)
        node = {'content': 'x' * 128, 'part_kind': 'text'}

        externalized = await externalize_media(node, media_store=store, threshold_bytes=64)
        assert await restore_media(externalized, media_store=store) == node
        assert seen_contexts == [MediaContext(media_type='text/plain'), MediaContext(media_type='text/plain')]

    async def test_small_text_part_stays_inline(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        node = {'content': 'short', 'part_kind': 'text', 'id': None}
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert externalized == node
        assert list(tmp_path.glob('*.bin')) == []

    async def test_non_string_content_is_not_externalized(self, tmp_path: Path) -> None:
        """A structured (non-string) `content` is walked, not treated as text."""
        store = DiskMediaStore(tmp_path)
        node = {'content': ['a', 'b'], 'part_kind': 'user-prompt'}
        externalized = await externalize_media(node, media_store=store, threshold_bytes=1)
        assert externalized == node
        assert list(tmp_path.glob('*.bin')) == []

    async def test_text_measured_in_utf8_bytes_not_characters(self, tmp_path: Path) -> None:
        """A multi-byte string below the char count but above the byte threshold externalizes."""

        store = DiskMediaStore(tmp_path)
        # 'é' is 2 UTF-8 bytes: 40 chars, 80 bytes.
        node = {'content': 'é' * 40, 'part_kind': 'text'}
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64)
        assert '__harness_external_text__' in _json.dumps(externalized)
        restored = await restore_media(externalized, media_store=store)
        assert restored == node

    async def test_non_part_dict_with_content_is_left_alone(self, tmp_path: Path) -> None:
        """A dict with a string `content` but no `part_kind` is not a message part.

        Its own `uri` field must survive untouched -- treating it as a part
        would overwrite `uri` with a media URI and drop it on restore.
        """
        store = DiskMediaStore(tmp_path)
        node = {'content': 'x' * 70_000, 'uri': 'source-id'}
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert externalized == node
        assert list(tmp_path.glob('*.bin')) == []

    async def test_text_part_externalizes_nested_binary(self, tmp_path: Path) -> None:
        """A large text part with a nested large binary externalizes both and round-trips."""

        store = DiskMediaStore(tmp_path)
        big_text = 'x' * 70_000
        big_bin_b64 = base64.b64encode(b'\x01' * 70_000).decode('ascii')
        node = {
            'part_kind': 'tool-return',
            'content': big_text,
            'metadata': {'kind': 'binary', 'data': big_bin_b64, 'media_type': 'image/png'},
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        dumped = _json.dumps(externalized)
        assert big_text not in dumped
        assert big_bin_b64 not in dumped  # nested binary went external too
        assert len(list(tmp_path.glob('*.bin'))) == 2  # one text blob, one binary blob

        restored = await restore_media(externalized, media_store=store)
        assert restored == node  # both fields re-inlined

    async def test_colliding_uri_field_round_trips(self, tmp_path: Path) -> None:
        """A lookalike mapping carrying its own `uri` keeps it across the round-trip.

        `ToolReturnContent` permits arbitrary `Mapping[str, ...]`, so a genuine
        tool-return payload can carry `part_kind` + a large string `content` +
        its own `uri` (e.g. `source://original`). The walker externalizes the
        large `content`, but the reference URI is stored under a namespaced key,
        so the payload's own `uri` must survive externalize -> restore intact.
        """

        store = DiskMediaStore(tmp_path)
        node = {
            'part_kind': 'tool-return',
            'content': 'x' * 70_000,
            'uri': 'source://original',
            'tool_name': 'scrape',
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        dumped = _json.dumps(externalized)
        assert '__harness_external_text__' in dumped  # large content went external
        assert 'source://original' in dumped  # original uri untouched, not overwritten
        restored = await restore_media(externalized, media_store=store)
        assert restored == node  # original uri survives, content re-inlined

    async def test_namespaced_marker_fields_round_trip(self, tmp_path: Path) -> None:
        """Caller-owned marker metadata is escaped before the marker writes its values."""
        store = DiskMediaStore(tmp_path)
        node = {
            'kind': 'binary',
            'data': base64.b64encode(b'\x01' * 70_000).decode('ascii'),
            '__harness_external_media__': 'caller-media-marker',
            '__harness_external_text__': True,
            '__harness_external_uri__': 'caller-uri',
            '__harness_external_escaped_keys__': {'caller': 'escaped-keys'},
            '__harness_external_marker_format__': 'caller-format',
            '__harness_external_field__': 'caller-field',
        }

        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert _is_marker_dict(externalized)
        assert externalized['__harness_external_media__'] is True
        assert externalized['__harness_external_marker_format__'] == 'escaped-keys-v1'
        assert externalized['__harness_external_uri__'] != 'caller-uri'

        assert await restore_media(externalized, media_store=store) == node

    async def test_text_marker_field_round_trips(self, tmp_path: Path) -> None:
        """A text part can use the text marker key without losing its value."""
        store = DiskMediaStore(tmp_path)
        node = {
            'part_kind': 'tool-return',
            'content': 'x' * 70_000,
            '__harness_external_text__': 'caller-text-marker',
        }

        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert _is_marker_dict(externalized)
        assert externalized['__harness_external_text__'] is True
        assert await restore_media(externalized, media_store=store) == node

    @pytest.mark.parametrize(
        'marker_metadata',
        [
            pytest.param(
                {
                    '__harness_external_marker_format__': 'escaped-keys-v1',
                    '__harness_external_escaped_keys__': ['not', 'a', 'mapping'],
                },
                id='stamped-but-stash-is-not-a-mapping',
            ),
            pytest.param(
                {
                    '__harness_external_marker_format__': 'escaped-keys-v1',
                    '__harness_external_escaped_keys__': {},
                },
                id='stamped-but-stash-is-empty',
            ),
            pytest.param(
                {
                    '__harness_external_marker_format__': 'escaped-keys-v1',
                    '__harness_external_escaped_keys__': {'caller': 'value'},
                },
                id='stamped-but-stash-has-keys-we-never-write',
            ),
            pytest.param(
                {'__harness_external_escaped_keys__': {'__harness_external_uri__': 'caller-uri'}},
                id='stash-shaped-like-ours-but-unstamped',
            ),
            pytest.param(
                {
                    '__harness_external_marker_format__': 'caller-format',
                    '__harness_external_escaped_keys__': {'__harness_external_uri__': 'caller-uri'},
                },
                id='stash-shaped-like-ours-but-stamped-by-the-caller',
            ),
        ],
    )
    async def test_marker_metadata_the_writer_never_produced_is_caller_data(
        self, tmp_path: Path, marker_metadata: dict[str, object]
    ) -> None:
        """A marker that does not carry both halves of the escape format keeps them as the payload's own.

        Both keys are collidable, so a marker written before the escaping format
        existed can carry either or both as caller data. Reading them as an
        escape stash would drop them, and rejecting the marker would make a
        snapshot that reads today unreadable with no recovery path. It takes the
        stash's shape and a version stamp from the format's own namespace
        together; short of both, it leaves them alone.
        """
        store = DiskMediaStore(tmp_path)
        payload = b'\x03' * 16
        uri = await store.put(payload)
        marker: dict[str, object] = {
            '__harness_external_media__': True,
            '__harness_external_uri__': uri,
            'kind': 'binary',
            **marker_metadata,
        }

        restored = await restore_media(marker, media_store=store)

        assert restored == {
            'kind': 'binary',
            'data': base64.b64encode(payload).decode('ascii'),
            **marker_metadata,
        }

    async def test_unsupported_escaped_keys_format_raises(self, tmp_path: Path) -> None:
        """A stash with the writer's shape but a version this reader does not know fails loudly.

        The shape says the harness wrote it, so silently ignoring the stash would
        strip the caller's values. This reader is the only one that can reject a
        later format, so it has to.
        """
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'payload')
        marker = {
            '__harness_external_media__': True,
            '__harness_external_uri__': uri,
            '__harness_external_marker_format__': 'escaped-keys-v2',
            '__harness_external_escaped_keys__': {'__harness_external_uri__': 'caller-uri'},
            'kind': 'binary',
        }

        with pytest.raises(ValueError, match='unsupported escaped-keys format'):
            await restore_media(marker, media_store=store)

    async def test_escaped_marker_restores_on_a_pre_escaping_reader(self, tmp_path: Path) -> None:
        """A reader that predates the escaping format still re-inlines an escaped marker.

        Compatibility in this direction is upgrade-only, and this pins what the
        merge-base reader gets on both marker kinds: the externalized field
        intact, the caller's escaped value still sitting in the stash rather
        than back under its own key.
        """
        store = DiskMediaStore(tmp_path)
        b64_payload = base64.b64encode(b'\x04' * 70_000).decode('ascii')
        node: dict[str, object] = {
            'kind': 'binary',
            'media_type': 'image/png',
            'data': b64_payload,
            '__harness_external_uri__': 'caller-uri',
        }

        marker = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert _is_marker_dict(marker)

        rolled_back = await _restore_as_pre_escaping_reader(marker, store)
        assert '__harness_external_uri__' not in rolled_back
        assert rolled_back['data'] == b64_payload
        assert rolled_back['__harness_external_escaped_keys__'] == {'__harness_external_uri__': 'caller-uri'}
        assert rolled_back['__harness_external_marker_format__'] == 'escaped-keys-v1'

        # The current reader puts the caller's value back where it belongs.
        assert await restore_media(marker, media_store=store) == node

        # The same downgrade claim holds for a text marker: that reader knows
        # `_TEXT_MARKER`, so it re-inlines `content` rather than `data`.
        text_payload = 'x' * 70_000
        text_node: dict[str, object] = {
            'part_kind': 'tool-return',
            'content': text_payload,
            '__harness_external_text__': 'caller-text-marker',
        }

        text_marker = await externalize_media(text_node, media_store=store, threshold_bytes=64 * 1024)
        assert _is_marker_dict(text_marker)

        text_rolled_back = await _restore_as_pre_escaping_reader(text_marker, store)
        assert text_rolled_back['content'] == text_payload
        assert text_rolled_back['__harness_external_escaped_keys__'] == {
            '__harness_external_text__': 'caller-text-marker'
        }
        assert text_rolled_back['__harness_external_marker_format__'] == 'escaped-keys-v1'

        assert await restore_media(text_marker, media_store=store) == text_node

    async def test_legacy_uri_marker_restores(self, tmp_path: Path) -> None:
        """A marker in the pre-`_URI_KEY` format (blob ref under plain `uri`) restores.

        Snapshots persisted by an earlier build stored the reference under `uri`.
        Restore falls back to it when `_URI_KEY` is absent, so those snapshots
        stay readable after the key rename.
        """
        store = DiskMediaStore(tmp_path)
        big_text = 'x' * 70_000
        uri = await store.put(big_text.encode('utf-8'), context=MediaContext(media_type='text/plain'))
        legacy_marker = {
            '__harness_external_media__': True,
            '__harness_external_text__': True,
            'uri': uri,  # legacy: the blob reference, not a genuine field
            'tool_name': 'scrape',
            'part_kind': 'tool-return',
        }
        restored = await restore_media(legacy_marker, media_store=store)
        assert restored == {'content': big_text, 'tool_name': 'scrape', 'part_kind': 'tool-return'}

        # A genuine `uri` on a normal (non-marker) part is never consumed as a
        # reference: restore only treats `uri` as the blob ref inside an
        # `_EXTERNAL_MARKER` node that lacks `_URI_KEY`.
        normal = {'part_kind': 'tool-return', 'content': 'small', 'uri': 'source://original'}
        assert await restore_media(normal, media_store=store) == normal

    async def test_sub_threshold_surrogate_passes_through(self, tmp_path: Path) -> None:
        """A below-threshold string with an unpaired surrogate passes through, not raising.

        A JSON string can decode to a lone surrogate (`"\\ud800"`), which strict
        UTF-8 refuses to encode. Below-threshold content must pass through
        untouched rather than raising on an eager encode.
        """
        store = DiskMediaStore(tmp_path)
        node = {'content': '\ud800' * 4, 'part_kind': 'text'}
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert externalized == node
        assert list(tmp_path.glob('*.bin')) == []

    async def test_above_threshold_surrogate_round_trips(self, tmp_path: Path) -> None:
        """A large string with an unpaired surrogate externalizes and restores exactly."""
        store = DiskMediaStore(tmp_path)
        content = '\ud800' + 'x' * 70_000
        node = {'content': content, 'part_kind': 'tool-return'}
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert externalized != node  # went external
        assert list(tmp_path.glob('*.bin'))  # bytes on disk
        restored = await restore_media(externalized, media_store=store)
        assert restored == node  # lone surrogate survives the round-trip

    async def test_text_content_element_round_trips_through_real_messages(self, tmp_path: Path) -> None:
        """A large `TextContent` inside a part's content sequence externalizes and restores.

        `TextContent` serializes `kind: 'text-content'` and no `part_kind`, so
        a discriminator check keyed only on `part_kind` leaves it inline -- and
        a multi-megabyte tool return delivered as `[TextContent(...)]` would
        then be written whole into the snapshot record.
        """

        store = DiskMediaStore(tmp_path)
        big_text = 'q' * 70_000
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=[TextContent(content=big_text)]),
                    ToolReturnPart(tool_name='scrape', tool_call_id='c1', content=[TextContent(content=big_text)]),
                ]
            )
        ]
        dumped = _json.loads(ModelMessagesTypeAdapter.dump_json(messages))
        externalized = await externalize_media(dumped, media_store=store, threshold_bytes=64 * 1024)
        assert big_text not in _json.dumps(externalized)

        restored = await restore_media(externalized, media_store=store)
        assert restored == dumped
        # Compared against the un-externalized payload's own validation: a direct
        # comparison with `messages` would trip over the timezone object pydantic
        # rebuilds from the JSON timestamp, which has nothing to do with the walker.
        assert ModelMessagesTypeAdapter.validate_python(restored) == ModelMessagesTypeAdapter.validate_python(dumped)

    async def test_structured_content_text_round_trips_through_real_messages(self, tmp_path: Path) -> None:
        """A large `StructuredContent` externalizes its rendered text and restores it byte for byte.

        It serializes `kind: 'structured-content'`, so matching only `'text-content'` would leave its
        text inline. Its `value` isn't a string payload, so it stays inline.
        """
        store = DiskMediaStore(tmp_path)
        item = StructuredContent({'rows': ['q' * 10] * 7_000}, tag='report')
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=[item])])]
        dumped = _json.loads(ModelMessagesTypeAdapter.dump_json(messages))
        externalized = await externalize_media(dumped, media_store=store, threshold_bytes=64 * 1024)
        # The JSON path to the item: messages[0].parts[0].content[0].
        assert _is_object_list(externalized)
        message = externalized[0]
        assert _is_marker_dict(message)
        parts = message['parts']
        assert _is_object_list(parts)
        part = parts[0]
        assert _is_marker_dict(part)
        content = part['content']
        assert _is_object_list(content)
        [node] = content
        assert _is_marker_dict(node)
        assert 'content' not in node
        assert node['value'] == item.value

        restored = await restore_media(externalized, media_store=store)
        assert restored == dumped

    @pytest.mark.parametrize('kind', [[], {}, ['text-content']], ids=['empty-list', 'dict', 'list'])
    async def test_unhashable_kind_with_string_content_is_left_alone(self, tmp_path: Path, kind: object) -> None:
        """Tool-return data can be any mapping, so a list or dict `kind` is not a text item, and doesn't raise."""
        store = DiskMediaStore(tmp_path)
        node = {'kind': kind, 'content': 'x' * 70_000}
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert externalized == node
        assert list(tmp_path.glob('*.bin')) == []

    async def test_unknown_kind_with_string_content_is_left_alone(self, tmp_path: Path) -> None:
        """A `kind` that is not a discriminator we know does not make a dict a text part."""
        store = DiskMediaStore(tmp_path)
        node = {'kind': 'note', 'content': 'x' * 70_000, 'uri': 'source://original'}
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert externalized == node
        assert list(tmp_path.glob('*.bin')) == []

    async def test_marker_mirrors_reference_to_uri_for_pre_uri_key_readers(self, tmp_path: Path) -> None:
        """A marker written now still restores on a reader that only knows plain `uri`."""

        store = DiskMediaStore(tmp_path)
        b64_payload = base64.b64encode(b'\x02' * 70_000).decode('ascii')
        node: dict[str, object] = {'kind': 'binary', 'data': b64_payload, 'media_type': 'image/png'}
        marker = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert _is_marker_dict(marker)
        assert marker['uri'] == marker['__harness_external_uri__']
        assert '__harness_external_marker_format__' not in marker
        assert '__harness_external_escaped_keys__' not in marker

        rolled_back = await _restore_as_pre_pr_reader(marker, store)
        # The old reader keeps keys it does not know; pydantic ignores the extra
        # field on validation, so the part is intact apart from it.
        assert {k: v for k, v in rolled_back.items() if k != '__harness_external_uri__'} == node

        # The current reader drops the mirror, so the marker still round-trips exactly.
        assert await restore_media(marker, media_store=store) == node

    async def test_text_marker_requires_current_reader_after_downgrade(self, tmp_path: Path) -> None:
        """The pre-PR binary reader cannot reconstruct externally stored text."""

        store = DiskMediaStore(tmp_path)
        original: object = _json.loads(
            ModelMessagesTypeAdapter.dump_json([ModelResponse(parts=[TextPart(content='x' * 128)])])
        )
        marker: object = await externalize_media(original, media_store=store, threshold_bytes=64)
        assert _is_object_list(marker)
        message: object = marker[0]
        assert _is_marker_dict(message)
        parts = message['parts']
        assert _is_object_list(parts)
        text_marker: object = parts[0]
        assert _is_marker_dict(text_marker)

        parts[0] = await _restore_as_pre_pr_reader(text_marker, store)
        with pytest.raises(ValueError, match='Field required'):
            ModelMessagesTypeAdapter.validate_python(marker)

    async def test_genuine_uri_suppresses_the_compat_mirror(self, tmp_path: Path) -> None:
        """A node with its own `uri` gets no mirror, so rollback fails loudly instead of lying."""
        store = DiskMediaStore(tmp_path)
        node: dict[str, object] = {'part_kind': 'tool-return', 'content': 'x' * 70_000, 'uri': 'source://original'}
        marker = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert _is_marker_dict(marker)
        assert marker['uri'] == 'source://original'
        assert marker['__harness_external_uri__'] != marker['uri']

        with pytest.raises(ValueError, match='not a media URI'):
            await _restore_as_pre_pr_reader(marker, store)

        assert await restore_media(marker, media_store=store) == node

    async def test_unusable_uri_key_does_not_fall_back_to_a_genuine_uri(self, tmp_path: Path) -> None:
        """A corrupt `_URI_KEY` raises; the node's own `uri` is never read as a reference."""
        store = DiskMediaStore(tmp_path)
        marker = {
            '__harness_external_media__': True,
            '__harness_external_uri__': None,
            'uri': 'source://original',
            'kind': 'binary',
        }
        with pytest.raises(ValueError, match='missing string uri'):
            await restore_media(marker, media_store=store)

    async def test_rejects_trees_that_are_not_json(self, tmp_path: Path) -> None:
        """A message object or a `dump_python()` tree raises instead of passing through unexternalized."""
        store = DiskMediaStore(tmp_path)
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=[BinaryContent(data=b'\x00' * 128, media_type='image/png')])])
        ]
        with pytest.raises(TypeError, match='takes a JSON tree, got ModelRequest'):
            await externalize_media(messages, media_store=store, threshold_bytes=64)
        with pytest.raises(TypeError, match='takes a JSON tree, got bytes'):
            await externalize_media(
                ModelMessagesTypeAdapter.dump_python(messages), media_store=store, threshold_bytes=64
            )
        assert list(tmp_path.iterdir()) == []

        json_mode = ModelMessagesTypeAdapter.dump_python(messages, mode='json')
        externalized = await externalize_media(json_mode, media_store=store, threshold_bytes=64)
        assert '__harness_external_media__' in _json.dumps(externalized)

    @pytest.mark.parametrize(
        'content',
        [
            pytest.param({'__harness_external_media__': True, 'status': 'ok'}, id='marker-key'),
            pytest.param(
                {
                    '__harness_external_media__': True,
                    '__harness_external_text__': True,
                    'uri': media_uri_for(b'another blob'),
                },
                id='marker-key-pointing-at-a-stored-blob',
            ),
            pytest.param({'__harness_external_literal__': {'nested': True}}, id='literal-key'),
            pytest.param({'outer': {'__harness_external_media__': True, 'x': [1, 2]}}, id='nested-marker-key'),
        ],
    )
    async def test_tool_return_with_reserved_keys_round_trips(self, tmp_path: Path, content: dict[str, object]) -> None:
        """A tool return that carries the marker's own keys restores as the tool returned it, never as a marker."""
        store = DiskMediaStore(tmp_path)
        await store.put(b'another blob')
        messages: list[ModelMessage] = [
            ModelRequest(parts=[ToolReturnPart(tool_name='fetch', tool_call_id='c1', content=content)])
        ]
        dumped: object = _json.loads(ModelMessagesTypeAdapter.dump_json(messages))

        externalized = await externalize_media(dumped, media_store=store, threshold_bytes=64)
        assert await restore_media(externalized, media_store=store) == dumped

    async def test_externalizing_a_lean_tree_takes_one_restore_per_pass(self, tmp_path: Path) -> None:
        """A marker in the input is escaped like caller data, never trusted as a marker.

        Trusting it would let a tool return forge one. Each `externalize_media`
        pass therefore needs its own `restore_media`.
        """
        store = DiskMediaStore(tmp_path)
        node = {'part_kind': 'text', 'content': 'x' * 128}
        once = await externalize_media(node, media_store=store, threshold_bytes=64)
        twice = await externalize_media(once, media_store=store, threshold_bytes=64)
        assert await restore_media(twice, media_store=store) == once
        assert await restore_media(once, media_store=store) == node

    async def test_externalized_part_inside_a_wrapped_dict_restores(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        node = {
            '__harness_external_media__': 'caller-value',
            'image': {'kind': 'binary', 'data': base64.b64encode(b'\x06' * 128).decode('ascii')},
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64)
        assert _is_marker_dict(externalized)
        assert externalized.keys() == {'__harness_external_literal__', '__harness_external_marker_format__'}
        assert len(list(tmp_path.glob('*.bin'))) == 1
        assert await restore_media(externalized, media_store=store) == node

    @pytest.mark.parametrize(
        'data',
        [
            pytest.param('hello', id='not-base64'),
            pytest.param('aGVsbG8gd29ybGQ=\n', id='trailing-newline'),
            pytest.param('ab-_ab+/', id='mixed-alphabets'),
            pytest.param('QR==', id='non-canonical-padding-bits'),
            pytest.param('é', id='non-ascii'),
        ],
    )
    async def test_binary_shaped_dict_without_canonical_base64_stays_inline(self, tmp_path: Path, data: str) -> None:
        """A tool dict shaped like `BinaryContent` whose `data` is not canonical base64 is left as it is."""
        store = DiskMediaStore(tmp_path)
        node = {'kind': 'binary', 'data': data}
        assert await externalize_media(node, media_store=store, threshold_bytes=1) == node
        assert list(tmp_path.iterdir()) == []

    async def test_urlsafe_and_standard_base64_restore_in_their_own_alphabet(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        raw = bytes([0xFB, 0xFF, 0xBF]) * 64
        urlsafe = {'kind': 'binary', 'data': base64.urlsafe_b64encode(raw).decode('ascii')}
        standard = {'kind': 'binary', 'data': base64.b64encode(raw).decode('ascii')}
        assert urlsafe != standard

        urlsafe_marker = await externalize_media(urlsafe, media_store=store, threshold_bytes=64)
        standard_marker = await externalize_media(standard, media_store=store, threshold_bytes=64)
        assert _is_marker_dict(urlsafe_marker) and _is_marker_dict(standard_marker)
        assert urlsafe_marker['__harness_external_urlsafe__'] == 'base64-urlsafe-v1'
        assert '__harness_external_urlsafe__' not in standard_marker

        assert await restore_media(urlsafe_marker, media_store=store) == urlsafe
        assert await restore_media(standard_marker, media_store=store) == standard

    async def test_real_binary_content_restores_to_the_identical_dump(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[UserPromptPart(content=[BinaryContent(data=bytes(range(256)) * 4, media_type='image/png')])]
            )
        ]
        dumped: object = _json.loads(ModelMessagesTypeAdapter.dump_json(messages))
        externalized = await externalize_media(dumped, media_store=store, threshold_bytes=64)
        assert '__harness_external_urlsafe__' in _json.dumps(externalized)
        assert await restore_media(externalized, media_store=store) == dumped

    async def test_pre_wrapper_data_under_the_literal_key_is_not_unwrapped(self, tmp_path: Path) -> None:
        """Only a stamped wrapper is unwrapped, so a snapshot from before the wrapper keeps its shape."""
        store = DiskMediaStore(tmp_path)
        legacy = {'__harness_external_literal__': {'caller': 'data'}}
        assert await restore_media(legacy, media_store=store) == legacy
        stamped_by_caller = {
            '__harness_external_literal__': {'caller': 'data'},
            '__harness_external_marker_format__': 'v0',
        }
        assert await restore_media(stamped_by_caller, media_store=store) == stamped_by_caller
        not_a_mapping = {'__harness_external_literal__': ['caller'], '__harness_external_marker_format__': 'literal-v1'}
        assert await restore_media(not_a_mapping, media_store=store) == not_a_mapping

    async def test_unsupported_literal_format_raises(self, tmp_path: Path) -> None:
        """A wrapper from a later format version fails loudly instead of restoring as caller data."""
        store = DiskMediaStore(tmp_path)
        wrapper = {'__harness_external_literal__': {'k': 'v'}, '__harness_external_marker_format__': 'literal-v2'}
        with pytest.raises(ValueError, match='unsupported format'):
            await restore_media(wrapper, media_store=store)

    async def test_pre_stamp_marker_keeps_a_caller_urlsafe_field(self, tmp_path: Path) -> None:
        """A marker written before the URL-safe stamp keeps a caller field of that name and the standard alphabet."""
        store = DiskMediaStore(tmp_path)
        raw = bytes([0xFB, 0xFF, 0xBF]) * 8
        uri = await store.put(raw)
        marker = {
            '__harness_external_media__': True,
            '__harness_external_uri__': uri,
            'kind': 'binary',
            '__harness_external_urlsafe__': True,
        }
        assert await restore_media(marker, media_store=store) == {
            'kind': 'binary',
            '__harness_external_urlsafe__': True,
            'data': base64.b64encode(raw).decode('ascii'),
        }

    async def test_caller_owned_urlsafe_key_is_escaped(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        node = {
            'kind': 'binary',
            'data': base64.b64encode(b'\x08' * 128).decode('ascii'),
            '__harness_external_urlsafe__': True,
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64)
        assert _is_marker_dict(externalized)
        assert '__harness_external_urlsafe__' not in externalized
        assert await restore_media(externalized, media_store=store) == node


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
            now=datetime(2013, 5, 24, tzinfo=UTC),
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
            now=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
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
            now=datetime(2024, 1, 1, tzinfo=UTC),
        )
        sig_a = sign_request(body=b'aaa', **kwargs)['authorization']  # type: ignore[arg-type]
        sig_b = sign_request(body=b'bbb', **kwargs)['authorization']  # type: ignore[arg-type]
        assert sig_a != sig_b

    @pytest.mark.parametrize('value', ['two  spaces', ' leading', 'trailing ', ' all  three '])
    def test_header_values_are_canonicalized_like_aws(self, value: str) -> None:
        """SigV4 trims each header value and collapses whitespace runs before signing.

        S3 canonicalizes the headers it receives the same way, so the signature
        covers the trimmed, collapsed value while the header is sent as given.
        """

        def sign(meta: str) -> dict[str, str]:
            return sign_request(
                method='PUT',
                host='s3.amazonaws.com',
                path='/bucket/key.bin',
                body=b'payload',
                region='us-east-1',
                access_key_id='AKID',
                secret_access_key='SECRET',
                extra_signed_headers={'x-amz-meta-note': meta},
                now=datetime(2026, 10, 19, 12, tzinfo=UTC),
            )

        signed = sign(value)
        assert signed['authorization'] == sign(' '.join(value.split()))['authorization']
        assert signed['x-amz-meta-note'] == value  # sent as given


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

    async def test_put_sends_metadata_keys_lowercased(self) -> None:
        """A mixed-case metadata key is sent, and so read back, lowercased."""
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
            await store.put(b'mixed-case', context=MediaContext(metadata={'Tenant': 'acme'}))
        meta_names = [name for name, _ in captured[0].headers.raw if name.startswith(b'x-amz-meta-')]
        assert meta_names == [b'x-amz-meta-tenant']

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


@pytest.mark.skipif(  # pragma: lax no cover
    not all(
        os.environ.get(k)
        for k in ('S3_ENDPOINT', 'S3_BUCKET_NAME', 'S3_ACCESS_KEY_ID', 'S3_SECRET_ACCESS_KEY', 'S3_REGION')
    ),
    reason='live S3/R2 env vars not set',
)
class TestS3MediaStoreLive:  # pragma: lax no cover
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


class _UnusedPostgresPool:
    """Satisfies `PostgresPool` structurally; a protocol check never acquires."""

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        raise AssertionError('the pool must not be used')  # pragma: no cover


def test_concrete_stores_satisfy_protocol(tmp_path: Path) -> None:
    _assert_media_store_protocol(DiskMediaStore(tmp_path))
    _assert_media_store_protocol(SqliteMediaStore(database=tmp_path / 'm.db'))
    _assert_media_store_protocol(PostgresMediaStore(_UnusedPostgresPool()))
    _assert_media_store_protocol(
        S3MediaStore(
            bucket='b',
            endpoint='https://example.com',
            region='auto',
            access_key_id='k',
            secret_access_key='s',
        )
    )
