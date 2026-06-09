from __future__ import annotations

from collections.abc import Callable

import pytest
from vcr.cassette import Cassette
from vcr.record_mode import RecordMode
from vcr.request import Request

from .conftest import check_vcr_cassette_usage, pytest_recording_configure


class RecordingVCR:
    before_record_request: Callable[[Request], Request | None] | None = None

    def register_serializer(self, name: str, serializer: object) -> None:
        pass

    def register_matcher(self, name: str, matcher: Callable[[Request, Request], None]) -> None:
        pass


def test_pytest_recording_configure_drops_google_oauth_token_requests() -> None:
    vcr = RecordingVCR()
    pytest_recording_configure(None, vcr)  # pyright: ignore[reportArgumentType]

    before_record_request = vcr.before_record_request
    assert before_record_request is not None
    request = Request('POST', 'https://oauth2.googleapis.com/token', None, {})

    assert before_record_request(request) is None


def test_check_vcr_cassette_usage_allows_loaded_unused_cassette_by_default() -> None:
    cassette = Cassette('fake.yaml', record_mode=RecordMode.NONE)

    check_vcr_cassette_usage(cassette, strict_usage=False)


def test_check_vcr_cassette_usage_reports_unused_interactions() -> None:
    cassette = Cassette('fake.yaml', record_mode=RecordMode.NONE)
    cassette.append(Request('POST', 'https://example.com/one', b'{}', {}), {})  # pyright: ignore[reportUnknownMemberType]
    cassette.append(Request('POST', 'https://example.com/two', b'{}', {}), {})  # pyright: ignore[reportUnknownMemberType]
    cassette.play_counts[0] = 1  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(pytest.fail.Exception, match=r'played 1/2; unused indexes: \[1\]'):
        check_vcr_cassette_usage(cassette, strict_usage=False)
