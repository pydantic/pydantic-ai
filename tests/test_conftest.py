from __future__ import annotations

from pathlib import Path

import pytest
from cassetter import Cassette, RawRequest, RecordMode, SkipRecording

from .conftest import check_vcr_cassette_usage, normalize_uri, scrub_request


def _raw_request(uri: str) -> RawRequest:
    return RawRequest('POST', uri, {}, b'{}')


def test_scrub_request_drops_google_token_exchanges() -> None:
    with pytest.raises(SkipRecording):
        scrub_request(_raw_request('https://oauth2.googleapis.com/token'))


def test_scrub_request_keeps_everything_else() -> None:
    request = _raw_request('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash')
    assert scrub_request(request) is request


def test_scrub_request_erases_the_aws_account_id_it_would_record() -> None:
    """`uri_normalizer` only rewrites the matching mirror, so the recorded URI needs this."""
    arn = 'arn:aws:bedrock:us-east-1:123456789012:inference-profile/x'
    request = _raw_request(
        f'https://bedrock-runtime.us-east-1.amazonaws.com/model/{arn.replace("123456789012", "999988887777")}/converse'
    )

    assert '999988887777' not in scrub_request(request).uri


def test_normalize_uri_erases_the_aws_account_id() -> None:
    uri = 'https://bedrock-runtime.us-east-1.amazonaws.com/model/arn:aws:bedrock:us-east-1:999988887777:x/converse'

    assert normalize_uri(uri) == (
        'https://bedrock-runtime.REGION.amazonaws.com/model/arn:aws:bedrock:us-east-1:123456789012:x/converse'
    )


def test_normalize_uri_erases_the_bedrock_region_from_the_host() -> None:
    east = normalize_uri('https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse')
    west = normalize_uri('https://bedrock-runtime.eu-west-3.amazonaws.com/model/m/converse')

    assert east == west == 'https://bedrock-runtime.REGION.amazonaws.com/model/m/converse'


def test_normalize_uri_erases_the_vertex_region_from_the_host() -> None:
    regional = normalize_uri('https://us-central1-aiplatform.googleapis.com/v1/publishers/google/models/m')

    assert regional == 'https://aiplatform.googleapis.com/v1/publishers/google/models/m'


def test_normalize_uri_erases_the_vertex_region_and_project_from_the_path() -> None:
    uri = 'https://aiplatform.googleapis.com/v1/projects/my-project/locations/europe-west4/publishers/google/models/m'

    assert normalize_uri(uri) == (
        'https://aiplatform.googleapis.com/v1/projects/PROJECT/locations/REGION/publishers/google/models/m'
    )


def test_normalize_uri_leaves_an_unremarkable_uri_alone() -> None:
    uri = 'https://api.openai.com/v1/chat/completions'

    assert normalize_uri(uri) == uri


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
