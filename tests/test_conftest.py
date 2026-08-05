from __future__ import annotations

from pathlib import Path

import pytest
from cassetter import Cassette, RawRequest, RecordMode, SkipRecording

from .conftest import check_vcr_cassette_usage, skip_recording_oauth_tokens


def _raw_request(uri: str) -> RawRequest:
    return RawRequest('POST', uri, {}, b'{}')


def test_skip_recording_oauth_tokens_drops_google_token_exchanges() -> None:
    with pytest.raises(SkipRecording):
        skip_recording_oauth_tokens(_raw_request('https://oauth2.googleapis.com/token'))


def test_skip_recording_oauth_tokens_keeps_everything_else() -> None:
    request = _raw_request('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash')
    assert skip_recording_oauth_tokens(request) is request


def _cassette_with_two_interactions(tmp_path: Path) -> Cassette:
    cassette = Cassette(tmp_path / 'fake.yaml', record_mode=RecordMode.ALL)
    cassette.load()
    for name in ('one', 'two'):
        cassette.record(
            method='POST',
            uri=f'https://example.com/{name}',
            request_headers={},
            request_body=None,
            status=200,
            response_headers={},
            response_body=b'{}',
        )
    return cassette


def test_check_vcr_cassette_usage_allows_loaded_unused_cassette_by_default(tmp_path: Path) -> None:
    cassette = Cassette(tmp_path / 'fake.yaml', record_mode=RecordMode.NONE)
    cassette.load()

    check_vcr_cassette_usage(cassette, strict_usage=False)


def test_check_vcr_cassette_usage_reports_unused_interactions(tmp_path: Path) -> None:
    cassette = _cassette_with_two_interactions(tmp_path)
    cassette.play('POST', 'https://example.com/one', {}, None)

    with pytest.raises(pytest.fail.Exception, match=r'played 1/2; unused indexes: \[1\]'):
        check_vcr_cassette_usage(cassette, strict_usage=False)


def test_check_vcr_cassette_usage_allows_fully_used_cassette(tmp_path: Path) -> None:
    cassette = _cassette_with_two_interactions(tmp_path)
    cassette.play('POST', 'https://example.com/one', {}, None)
    cassette.play('POST', 'https://example.com/two', {}, None)

    check_vcr_cassette_usage(cassette, strict_usage=False)
