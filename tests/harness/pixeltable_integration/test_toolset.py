"""Tests for `PixeltableToolset` against a real Pixeltable catalog."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pixeltable as pxt
import pytest
from PIL import Image

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai_harness.pixeltable import Pixeltable, PixeltableToolset

from .support import DIM, create_table, get_table, insert_rows, tiny_embed, tiny_image_embed


@pytest.fixture
def catalog(root: str) -> str:
    chunks = create_table(
        f'{root}.chunks',
        {'text': pxt.String, 'pos': pxt.Int, 'status': pxt.String, 'vec': pxt.Array[(8,), pxt.Float]},
        comment='chunk store',
    )
    other = create_table(f'{root}.other', {'text': pxt.String})
    zero = np.zeros(DIM, dtype=np.float32)
    insert_rows(
        chunks,
        [
            {'text': 'cats sit on mats', 'pos': 0, 'status': 'open', 'vec': zero},
            {'text': 'dogs run in parks', 'pos': 1, 'status': 'closed', 'vec': zero},
            {'text': 'birds fly in sky', 'pos': 2, 'status': 'open', 'vec': zero},
        ],
    )
    insert_rows(other, [{'text': 'secret handbook'}])
    return root


def _tools(tables: list[str], *, max_rows: int = 20, max_chars: int = 8000) -> PixeltableToolset[None]:
    return PixeltableToolset(tables=tables, max_rows=max_rows, max_chars=max_chars)


def _chunks(root: str, *, max_rows: int = 20, max_chars: int = 8000) -> PixeltableToolset[None]:
    return _tools([f'{root}.chunks'], max_rows=max_rows, max_chars=max_chars)


class TestPixeltableToolsetAllowlist:
    def test_allowlist_exact_and_prefix(self, catalog: str) -> None:
        exact = _chunks(catalog)
        assert exact.list_tables()['tables'] == [f'{catalog}.chunks']
        prefix = _tools([catalog])
        assert prefix.list_tables()['tables'] == [f'{catalog}.chunks', f'{catalog}.other']
        whole = _tools(['*']).list_tables()['tables']
        assert {f'{catalog}.chunks', f'{catalog}.other'} <= set(whole)

        with pytest.raises(ModelRetry, match='allowlist'):
            exact.query_table(f'{catalog}.other')
        with pytest.raises(ModelRetry, match='allowlist'):
            exact.describe_table(f'{catalog}.other')
        with pytest.raises(ModelRetry, match='allowlist'):
            exact.similarity_search(f'{catalog}.other', 'secret', 'text')

    def test_slash_paths_match_the_allowlist(self, catalog: str) -> None:
        tools = _tools([f'{catalog}/chunks'])
        assert tools.query_table(f'{catalog}/chunks', columns=['pos'], where={'pos': 1})['rows'] == [{'pos': 1}]

    def test_allowlist_folds_case_like_pixeltable(self, catalog: str) -> None:
        # Pixeltable lowercases identifiers, so a differently cased entry names the same table.
        tools = _tools([f'{catalog.upper()}.Chunks'])
        assert tools.list_tables()['tables'] == [f'{catalog}.chunks']
        assert tools.describe_table(f'{catalog}.CHUNKS')['table'] == f'{catalog}.chunks'
        with pytest.raises(ModelRetry, match='allowlist'):
            tools.describe_table(f'{catalog.upper()}.OTHER')

    def test_version_handles_are_refused(self, catalog: str) -> None:
        # An old version still holds rows deleted and columns dropped since (e.g. for privacy).
        for tables in ([f'{catalog}.chunks'], [catalog], ['*']):
            with pytest.raises(ModelRetry, match='allowlist'):
                _tools(tables).query_table(f'{catalog}.chunks:1', columns=['text'])

    def test_missing_table_under_a_prefix(self, catalog: str) -> None:
        with pytest.raises(ModelRetry, match='Cannot open table'):
            _tools([catalog]).query_table(f'{catalog}.missing')

    def test_infrastructure_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unavailable() -> list[str]:
            raise pxt.ServiceUnavailableError(pxt.ErrorCode.DATABASE_UNAVAILABLE, 'database unavailable')

        monkeypatch.setattr(pxt, 'list_tables', unavailable)
        with pytest.raises(pxt.ServiceUnavailableError, match='database unavailable'):
            _tools(['*']).list_tables()

    def test_moved_table_is_not_served_under_its_old_name(self, catalog: str) -> None:
        tools = _tools([f'{catalog}.other'])
        assert tools.query_table(f'{catalog}.other')['rows']
        pxt.move(f'{catalog}.other', f'{catalog}.moved')
        with pytest.raises(ModelRetry, match='Cannot open table'):
            tools.query_table(f'{catalog}.other')


class TestPixeltableToolsetDescribe:
    def test_describe_table(self, catalog: str) -> None:
        get_table(f'{catalog}.chunks').add_embedding_index('text', string_embed=tiny_embed, idx_name='text_idx')
        described = _chunks(catalog).describe_table(f'{catalog}.chunks')
        assert described['table'] == f'{catalog}.chunks'
        assert described['kind'] == 'table'
        assert described['comment'] == 'chunk store'
        assert [column['name'] for column in described['columns']] == ['text', 'pos', 'status', 'vec']
        assert described['columns'][0] == {'name': 'text', 'type': 'String', 'is_computed': False, 'is_stored': True}
        assert {'name': 'text_idx', 'index_type': 'embedding', 'columns': ['text']} in described['indexes']

    def test_views_work_as_catalog_targets(self, catalog: str) -> None:
        chunks = get_table(f'{catalog}.chunks')
        pxt.create_view(f'{catalog}.open_chunks', chunks.where(chunks.status == 'open'))
        tools = _tools([f'{catalog}.open_chunks'])
        assert tools.describe_table(f'{catalog}.open_chunks')['kind'] == 'view'
        assert tools.list_tables()['tables'] == [f'{catalog}.open_chunks']
        assert {row['status'] for row in tools.query_table(f'{catalog}.open_chunks')['rows']} == {'open'}


class TestPixeltableToolsetQuery:
    def test_equality_where_and_default_columns(self, catalog: str) -> None:
        result = _chunks(catalog).query_table(f'{catalog}.chunks', where={'status': 'open'})
        assert result == {
            'table': f'{catalog}.chunks',
            'rows': [
                {'text': 'cats sit on mats', 'pos': 0, 'status': 'open'},
                {'text': 'birds fly in sky', 'pos': 2, 'status': 'open'},
            ],
            'truncated': False,
        }

    def test_where_combines_clauses(self, catalog: str) -> None:
        result = _chunks(catalog).query_table(f'{catalog}.chunks', columns=['pos'], where={'status': 'open', 'pos': 2})
        assert result['rows'] == [{'pos': 2}]

    def test_duplicate_columns_are_deduplicated(self, catalog: str) -> None:
        result = _chunks(catalog).query_table(f'{catalog}.chunks', columns=['text', 'text'])
        assert len(result['rows']) == 3
        assert all(set(row) == {'text'} for row in result['rows'])

    def test_named_array_column_is_skipped(self, catalog: str) -> None:
        tools = _chunks(catalog)
        result = tools.query_table(f'{catalog}.chunks', columns=['vec', 'pos'], where={'pos': 0})
        assert result['rows'] == [{'pos': 0}]
        with pytest.raises(ModelRetry, match='No selectable columns'):
            tools.query_table(f'{catalog}.chunks', columns=['vec'])

    def test_named_media_column_is_rejected(self, catalog: str, tmp_path: Path) -> None:
        docs = create_table(f'{catalog}.docs', {'doc': pxt.Document, 'title': pxt.String})
        note = tmp_path / 'note.md'
        note.write_text('# note\n')
        insert_rows(docs, [{'doc': str(note), 'title': 'note'}])
        tools = _tools([f'{catalog}.docs'])

        assert tools.query_table(f'{catalog}.docs')['rows'] == [{'title': 'note'}]
        with pytest.raises(ModelRetry, match='local file path'):
            tools.query_table(f'{catalog}.docs', columns=['doc', 'title'])

    def test_row_bound(self, catalog: str) -> None:
        rows = _chunks(catalog, max_rows=2).query_table(f'{catalog}.chunks', limit=10)
        assert len(rows['rows']) == 2
        assert rows['truncated']
        assert not _chunks(catalog).query_table(f'{catalog}.chunks', limit=3)['truncated']
        assert _chunks(catalog).query_table(f'{catalog}.chunks', limit=2)['truncated']

    def test_character_bound(self, catalog: str) -> None:
        tiny = _chunks(catalog, max_chars=100).query_table(f'{catalog}.chunks', columns=['text'])
        assert tiny['truncated']
        assert len(json.dumps(tiny, ensure_ascii=False)) <= 100

        # A cap below the minimal envelope size cannot shrink the payload further.
        micro = _chunks(catalog, max_chars=1).query_table(f'{catalog}.chunks', columns=['text'])
        assert micro['rows'] == []
        assert micro['truncated']

    def test_oversized_cell_is_truncated_not_dropped(self, catalog: str) -> None:
        chunks = get_table(f'{catalog}.chunks')
        big = 'x' * 4000
        chunks.update({'text': big}, where=chunks.pos == 0)
        result = _chunks(catalog, max_chars=500).query_table(f'{catalog}.chunks', columns=['text'], where={'pos': 0})
        assert result['truncated']
        text = result['rows'][0]['text']
        assert isinstance(text, str)
        assert text.endswith('...')
        assert len(text) < len(big)
        assert len(json.dumps(result, ensure_ascii=False)) <= 500

    def test_escaped_text_is_cut_to_fit(self, catalog: str) -> None:
        # Quotes double in JSON, so the cut shrinks until the escaped text fits.
        chunks = get_table(f'{catalog}.chunks')
        chunks.update({'text': '"' * 1000}, where=chunks.pos == 0)
        result = _chunks(catalog, max_chars=200).query_table(f'{catalog}.chunks', columns=['text'], where={'pos': 0})
        text = result['rows'][0]['text']
        assert isinstance(text, str)
        assert text.strip('"') == '...'
        assert len(json.dumps(result, ensure_ascii=False)) <= 200

    def test_oversized_structured_cell_becomes_null(self, catalog: str) -> None:
        # A cut dict would read as real data, so it is dropped and `truncated` flags the gap.
        blobs = create_table(f'{catalog}.blobs', {'name': pxt.String, 'data': pxt.Json})
        insert_rows(blobs, [{'name': 'a', 'data': {'values': list(range(500))}}])
        result = _tools([f'{catalog}.blobs'], max_chars=200).query_table(f'{catalog}.blobs')
        assert result == {'table': f'{catalog}.blobs', 'rows': [{'name': 'a', 'data': None}], 'truncated': True}

    def test_cell_without_room_keeps_only_the_ellipsis(self, catalog: str) -> None:
        wide = create_table(f'{catalog}.wide', {'a': pxt.String, 'b': pxt.String})
        insert_rows(wide, [{'a': 'x' * 50, 'b': 'y' * 50}])
        envelope = len(json.dumps({'table': f'{catalog}.wide', 'rows': [], 'truncated': False}))
        # Room for the keys plus five characters per value: a quoted `...` and no kept text.
        cap = envelope + len('{"a": 0, "b": 0}') - 2 + 2 * len('"..."')
        result = _tools([f'{catalog}.wide'], max_chars=cap).query_table(f'{catalog}.wide')
        assert result['rows'] == [{'a': '...', 'b': '...'}]
        assert result['truncated']

    def test_values_are_json_friendly(self, catalog: str) -> None:
        key = uuid.uuid4()
        typed = create_table(
            f'{catalog}.typed',
            {'ts': pxt.Timestamp, 'd': pxt.Date, 'id': pxt.UUID, 'flag': pxt.Bool, 'x': pxt.Float, 'j': pxt.Json},
        )
        insert_rows(
            typed,
            [
                {
                    'ts': datetime(2024, 1, 2, 3, 4, 5),
                    'd': date(2024, 1, 2),
                    'id': key,
                    'flag': True,
                    'x': 1.5,
                    'j': {'k': [1, {'x': 2}]},
                }
            ],
        )
        (row,) = _tools([f'{catalog}.typed']).query_table(f'{catalog}.typed')['rows']
        assert isinstance(row['ts'], str)
        assert row['ts'].startswith('2024-01-02T')
        assert row['d'] == '2024-01-02'
        assert row['id'] == str(key)
        assert row['flag'] is True
        assert row['x'] == 1.5
        assert row['j'] == {'k': [1, {'x': 2}]}

    def test_argument_errors_are_retries(self, catalog: str) -> None:
        tools = _chunks(catalog)
        with pytest.raises(ModelRetry, match='Unknown column'):
            tools.query_table(f'{catalog}.chunks', columns=['nope'])
        with pytest.raises(ModelRetry, match='Unknown column'):
            tools.query_table(f'{catalog}.chunks', where={'nope': 1})
        with pytest.raises(ModelRetry, match='at least one column'):
            tools.query_table(f'{catalog}.chunks', columns=[])
        with pytest.raises(ModelRetry, match='at least 1'):
            tools.query_table(f'{catalog}.chunks', limit=0)

    def test_where_rejects_wrong_value_type(self, catalog: str) -> None:
        # A mistyped filter must fail loudly; returning [] would tell the model the data is absent.
        tools = _chunks(catalog)
        with pytest.raises(ModelRetry, match='must be Int, got str'):
            tools.query_table(f'{catalog}.chunks', where={'pos': 'open'})
        with pytest.raises(ModelRetry, match='must be Int, got bool'):
            tools.query_table(f'{catalog}.chunks', where={'pos': True})
        with pytest.raises(ModelRetry, match='must be String, got int'):
            tools.query_table(f'{catalog}.chunks', where={'status': 5})
        with pytest.raises(ModelRetry, match='must be String, got list'):
            tools.query_table(f'{catalog}.chunks', where={'status': ['open', 'closed']})  # pyright: ignore[reportArgumentType]
        with pytest.raises(ModelRetry, match='equality filters'):
            tools.query_table(f'{catalog}.chunks', where={'vec': 5})
        assert tools.query_table(f'{catalog}.chunks', columns=['pos'], where={'pos': 0})['rows'] == [{'pos': 0}]

    def test_where_accepts_bool_and_int_for_float(self, catalog: str) -> None:
        flags = create_table(f'{catalog}.flags', {'name': pxt.String, 'on': pxt.Bool, 'x': pxt.Float})
        insert_rows(flags, [{'name': 'a', 'on': True, 'x': 2.0}, {'name': 'b', 'on': False, 'x': 3.5}])
        tools = _tools([f'{catalog}.flags'])
        assert tools.query_table(f'{catalog}.flags', columns=['name'], where={'on': False})['rows'] == [{'name': 'b'}]
        assert tools.query_table(f'{catalog}.flags', columns=['name'], where={'x': 2})['rows'] == [{'name': 'a'}]

    def test_where_on_json_columns(self, catalog: str) -> None:
        # Json has no expected Python type, so pixeltable decides what it can compare.
        blobs = create_table(f'{catalog}.blobs', {'name': pxt.String, 'data': pxt.Json})
        insert_rows(blobs, [{'name': 'a', 'data': 'x'}])
        tools = _tools([f'{catalog}.blobs'])
        assert tools.query_table(f'{catalog}.blobs', columns=['name'], where={'data': 'x'})['rows'] == [{'name': 'a'}]
        with pytest.raises(ModelRetry, match="Cannot filter 'data'"):
            tools.query_table(f'{catalog}.blobs', where={'data': [1, 2]})  # pyright: ignore[reportArgumentType]

    def test_media_and_null_filters(self, catalog: str, tmp_path: Path) -> None:
        # Media columns never match an equality filter; None is still an IS NULL filter.
        note_file = tmp_path / 'note.md'
        note_file.write_text('x')
        docs = create_table(
            f'{catalog}.docs', {'title': pxt.String, 'note': pxt.String | None, 'doc': pxt.Document | None}
        )
        insert_rows(docs, [{'title': 'a', 'note': None, 'doc': str(note_file)}, {'title': 'b', 'note': 'n'}])
        tools = _tools([f'{catalog}.docs'])
        with pytest.raises(ModelRetry, match='equality filters'):
            tools.query_table(f'{catalog}.docs', where={'doc': 'x'})
        assert tools.query_table(f'{catalog}.docs', columns=['title'], where={'doc': None})['rows'] == [{'title': 'b'}]
        assert tools.query_table(f'{catalog}.docs', columns=['title'], where={'note': None})['rows'] == [{'title': 'a'}]

    def test_where_parses_iso_timestamp_date_and_uuid(self, catalog: str) -> None:
        # JSON has no timestamp, date, or UUID; unparsed strings would match nothing.
        key = uuid.uuid4()
        typed = create_table(
            f'{catalog}.typed', {'ts': pxt.Timestamp | None, 'd': pxt.Date, 'id': pxt.UUID, 'n': pxt.Int}
        )
        insert_rows(
            typed,
            [
                {'ts': datetime(2024, 1, 2, 3, 4, 5), 'd': date(2024, 1, 2), 'id': key, 'n': 1},
                {'ts': None, 'd': date(2025, 1, 1), 'id': uuid.uuid4(), 'n': 2},
            ],
        )
        tools = _tools([f'{catalog}.typed'])
        assert tools.query_table(f'{catalog}.typed', columns=['n'], where={'ts': '2024-01-02T03:04:05'})['rows'] == [
            {'n': 1}
        ]
        assert tools.query_table(f'{catalog}.typed', columns=['n'], where={'d': '2024-01-02'})['rows'] == [{'n': 1}]
        assert tools.query_table(f'{catalog}.typed', columns=['n'], where={'id': str(key)})['rows'] == [{'n': 1}]
        assert tools.query_table(f'{catalog}.typed', columns=['n'], where={'ts': None})['rows'] == [{'n': 2}]
        with pytest.raises(ModelRetry, match='ISO-format Timestamp'):
            tools.query_table(f'{catalog}.typed', where={'ts': 'yesterday'})

    def test_unstored_computed_columns_are_not_run(self, catalog: str) -> None:
        # Unstored computed columns recompute at query time; an LLM UDF would spend money.
        chunks = get_table(f'{catalog}.chunks')
        chunks.add_computed_column(virtual=chunks.pos * 2, stored=False)
        chunks.add_computed_column(materialized=chunks.pos * 3, stored=True)
        tools = _chunks(catalog)
        first = tools.query_table(f'{catalog}.chunks', where={'pos': 1})['rows'][0]
        assert 'virtual' not in first
        assert first['materialized'] == 3
        # Naming one would run its function for every fetched row.
        with pytest.raises(ModelRetry, match='computed on read'):
            tools.query_table(f'{catalog}.chunks', columns=['virtual'], where={'pos': 1})
        stored = {
            column['name']: column['is_stored'] for column in tools.describe_table(f'{catalog}.chunks')['columns']
        }
        assert stored['virtual'] is False
        assert stored['materialized'] is True
        # A filter would run the column for every row (and crash pixeltable with an AssertionError).
        with pytest.raises(ModelRetry, match='computed on read'):
            tools.query_table(f'{catalog}.chunks', where={'virtual': 0})


class TestPixeltableToolsetSimilarity:
    def test_image_similarity_omits_local_file_path(self, catalog: str, tmp_path: Path) -> None:
        images = create_table(f'{catalog}.images', {'image': pxt.Image, 'title': pxt.String})
        image_path = tmp_path / 'photo.png'
        Image.new('RGB', (2, 2), color='red').save(image_path)
        insert_rows(images, [{'image': str(image_path), 'title': 'photo'}])
        images.add_embedding_index('image', image_embed=tiny_image_embed, string_embed=tiny_embed)

        tools = _tools([f'{catalog}.images'])
        result = tools.similarity_search(f'{catalog}.images', 'red photo', 'image')
        assert result['rows'][0]['title'] == 'photo'
        assert set(result['rows'][0]) == {'title', 'score'}
        with pytest.raises(ModelRetry, match='local file path'):
            tools.similarity_search(f'{catalog}.images', 'red photo', 'image', columns=['image', 'title'])

    def test_similarity_ranks_exact_text(self, catalog: str) -> None:
        get_table(f'{catalog}.chunks').add_embedding_index('text', string_embed=tiny_embed)
        result = _chunks(catalog).similarity_search(f'{catalog}.chunks', 'cats sit on mats', 'text', limit=3)
        rows = result['rows']
        assert [row['text'] for row in rows][0] == 'cats sit on mats'
        assert set(rows[0]) == {'text', 'pos', 'status', 'score'}
        top, last = rows[0]['score'], rows[-1]['score']
        assert isinstance(top, float)
        assert isinstance(last, float)
        assert abs(top - 1.0) < 1e-6
        assert top >= last

    def test_similarity_column_is_added_to_named_columns(self, catalog: str) -> None:
        get_table(f'{catalog}.chunks').add_embedding_index('text', string_embed=tiny_embed)
        tools = _chunks(catalog, max_rows=1)
        result = tools.similarity_search(f'{catalog}.chunks', 'dogs run in parks', 'text', columns=['pos'], limit=5)
        assert result['truncated']
        assert list(result['rows'][0]) == ['text', 'pos', 'score']
        assert result['rows'][0]['pos'] == 1
        # Naming the score column keeps the similarity score rather than failing as unknown.
        named = tools.similarity_search(f'{catalog}.chunks', 'dogs run in parks', 'text', columns=['score', 'pos'])
        assert list(named['rows'][0]) == ['text', 'pos', 'score']

    def test_similarity_idx_disambiguation(self, catalog: str) -> None:
        t = get_table(f'{catalog}.chunks')
        t.add_embedding_index('text', string_embed=tiny_embed, idx_name='e_a')
        t.add_embedding_index('text', string_embed=tiny_embed, idx_name='e_b')
        tools = _chunks(catalog)
        with pytest.raises(ModelRetry, match='multiple embedding indices'):
            tools.similarity_search(f'{catalog}.chunks', 'cats sit on mats', 'text')
        result = tools.similarity_search(f'{catalog}.chunks', 'cats sit on mats', 'text', idx='e_a')
        assert result['rows'][0]['text'] == 'cats sit on mats'

    def test_similarity_score_does_not_shadow_real_columns(self, catalog: str) -> None:
        scored = create_table(
            f'{catalog}.scored', {'text': pxt.String, 'score': pxt.Float, 'similarity_score': pxt.Float}
        )
        insert_rows(scored, [{'text': 'cats sit on mats', 'score': 0.9, 'similarity_score': 0.1}])
        scored.add_embedding_index('text', string_embed=tiny_embed)
        result = _tools([f'{catalog}.scored']).similarity_search(f'{catalog}.scored', 'cats', 'text', limit=1)
        row = result['rows'][0]
        assert row['score'] == 0.9
        assert row['similarity_score'] == 0.1
        assert 'similarity_similarity_score' in row

    def test_similarity_errors_are_retries(self, catalog: str) -> None:
        tools = _chunks(catalog)
        with pytest.raises(ModelRetry, match='non-empty'):
            tools.similarity_search(f'{catalog}.chunks', '  ', 'text')
        with pytest.raises(ModelRetry, match='Unknown column'):
            tools.similarity_search(f'{catalog}.chunks', 'x', 'nope')
        with pytest.raises(ModelRetry, match='No embedding index'):
            tools.similarity_search(f'{catalog}.chunks', 'x', 'vec', columns=['text'])


class TestPixeltableToolsetCapability:
    def test_capability_toolset_honors_caps(self, catalog: str) -> None:
        tools = Pixeltable(tables=[f'{catalog}.chunks'], max_rows=1).get_toolset()
        assert isinstance(tools, PixeltableToolset)
        result = tools.query_table(f'{catalog}.chunks')
        assert len(result['rows']) == 1
        assert result['truncated']
