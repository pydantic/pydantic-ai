"""Tests for cassette_utils coverage gaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.cassette_utils import (
    CassetteContext,
    _get_cassette_bodies_from_yaml,  # pyright: ignore[reportPrivateUsage]
    _get_xai_cassette_request_bodies,  # pyright: ignore[reportPrivateUsage]
)

from ._inline_snapshot import snapshot


def _make_ctx(tmp_path: Path, provider: str = 'openai') -> CassetteContext:
    return CassetteContext(
        provider=provider,
        vcr=None,
        test_name='fake_test',
        test_module='fake_module',
        test_dir=tmp_path,
    )


class TestGetCassetteBodiesFromYaml:
    def test_dict_body(self, tmp_path: Path) -> None:
        cassette_data: dict[str, Any] = {
            'interactions': [
                {'request': {'parsed_body': {'model': 'gpt-4', 'messages': []}}},
            ]
        }
        path = tmp_path / 'cassette.yaml'
        path.write_text(yaml.dump(cassette_data), encoding='utf-8')
        bodies = _get_cassette_bodies_from_yaml(path)
        assert len(bodies) == 1
        assert '"model": "gpt-4"' in bodies[0]

    def test_string_body(self, tmp_path: Path) -> None:
        cassette_data: dict[str, Any] = {
            'interactions': [
                {'request': {'body': 'raw request body'}},
            ]
        }
        path = tmp_path / 'cassette.yaml'
        path.write_text(yaml.dump(cassette_data), encoding='utf-8')
        bodies = _get_cassette_bodies_from_yaml(path)
        assert bodies == ['raw request body']

    def test_none_and_empty_bodies_skipped(self, tmp_path: Path) -> None:
        cassette_data: dict[str, Any] = {
            'interactions': [
                {'request': {'body': None}},
                {'request': {'body': ''}},
                {'request': {}},
            ]
        }
        path = tmp_path / 'cassette.yaml'
        path.write_text(yaml.dump(cassette_data), encoding='utf-8')
        bodies = _get_cassette_bodies_from_yaml(path)
        assert bodies == []

    def test_list_body(self, tmp_path: Path) -> None:
        cassette_data: dict[str, Any] = {
            'interactions': [
                {'request': {'parsed_body': [{'role': 'user'}]}},
            ]
        }
        path = tmp_path / 'cassette.yaml'
        path.write_text(yaml.dump(cassette_data), encoding='utf-8')
        bodies = _get_cassette_bodies_from_yaml(path)
        assert len(bodies) == 1
        assert '"role": "user"' in bodies[0]


class TestCassetteContextGetBodies:
    def test_xai_no_cassette(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, provider='xai')
        assert ctx._get_bodies() == []  # pyright: ignore[reportPrivateUsage]

    def test_yaml_fallback(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        cassette_dir = tmp_path / 'cassettes' / 'fake_module'
        cassette_dir.mkdir(parents=True)
        cassette_data: dict[str, Any] = {
            'interactions': [
                {'request': {'parsed_body': {'key': 'value'}}},
            ]
        }
        (cassette_dir / 'fake_test.yaml').write_text(yaml.dump(cassette_data), encoding='utf-8')
        bodies = ctx._get_bodies()  # pyright: ignore[reportPrivateUsage]
        assert len(bodies) == 1
        assert '"key": "value"' in bodies[0]

    def test_no_cassette_file(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        assert ctx._get_bodies() == []  # pyright: ignore[reportPrivateUsage]


class TestVerifyWithEmptyBodies:
    @pytest.fixture()
    def ctx(self, tmp_path: Path) -> CassetteContext:
        return _make_ctx(tmp_path)

    def test_verify_contains_no_bodies(self, ctx: CassetteContext) -> None:
        ctx.verify_contains('anything')

    def test_verify_ordering_no_bodies(self, ctx: CassetteContext) -> None:
        ctx.verify_ordering('a', 'b', 'c')


class TestGetXaiCassetteRequestBodies:
    def test_sdk_unavailable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('tests.models.xai_proto_cassettes.xai_sdk_available', lambda: False)
        assert _get_xai_cassette_request_bodies(tmp_path / 'fake.yaml') == []

    def test_sdk_available_with_interactions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.models.xai_proto_cassettes import SampleInteraction, StreamInteraction, XaiProtoCassette

        sample = SampleInteraction(
            request_raw=b'',
            response_raw=b'',
            request_json={'model': 'grok'},
            response_json={'text': 'hi'},
        )
        stream = StreamInteraction(
            request_raw=b'',
            chunks_raw=[],
            request_json={'model': 'grok-stream'},
            chunks_json=[{'chunk': 'data'}],
        )
        cassette = XaiProtoCassette(interactions=[sample, stream])

        monkeypatch.setattr('tests.models.xai_proto_cassettes.xai_sdk_available', lambda: True)
        monkeypatch.setattr(XaiProtoCassette, 'load', staticmethod(lambda _path: cassette))  # pyright: ignore[reportUnknownArgumentType,reportUnknownLambdaType]

        bodies = _get_xai_cassette_request_bodies(tmp_path / 'fake.yaml')
        assert bodies == snapshot(
            ['{"model": "grok"}', '{"text": "hi"}', '{"model": "grok-stream"}', '{"chunk": "data"}']
        )
        assert '{"model": "grok"}' in bodies[0]
        assert '{"text": "hi"}' in bodies[1]
        assert '{"model": "grok-stream"}' in bodies[2]
        assert '{"chunk": "data"}' in bodies[3]
