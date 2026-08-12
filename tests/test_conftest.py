from __future__ import annotations

import io
import os
from collections.abc import Callable

import pytest
from blockbuster import BlockBuster, BlockingError
from coverage.python import get_python_source
from vcr.cassette import Cassette
from vcr.record_mode import RecordMode
from vcr.request import Request

from . import conftest
from .conftest import BLOCKBUSTER_EXEMPTIONS, check_vcr_cassette_usage, pytest_recording_configure


class RecordingVCR:
    before_record_request: Callable[[Request], Request | None] | None = None

    def register_serializer(self, name: str, serializer: object) -> None:
        pass

    def register_matcher(self, name: str, matcher: Callable[[Request, Request], None]) -> None:
        pass


def _blocking_stat() -> None:
    os.stat(__file__)


def test_pytest_recording_configure_drops_google_oauth_token_requests() -> None:
    vcr = RecordingVCR()
    pytest_recording_configure(None, vcr)  # pyright: ignore[reportArgumentType]

    before_record_request = vcr.before_record_request
    assert before_record_request is not None
    request = Request('POST', 'https://oauth2.googleapis.com/token', None, dict[str, str]())

    assert before_record_request(request) is None


def test_check_vcr_cassette_usage_allows_loaded_unused_cassette_by_default() -> None:
    cassette = Cassette('fake.yaml', record_mode=RecordMode.NONE)

    check_vcr_cassette_usage(cassette, strict_usage=False)


def test_check_vcr_cassette_usage_reports_unused_interactions() -> None:
    cassette = Cassette('fake.yaml', record_mode=RecordMode.NONE)
    cassette.append(Request('POST', 'https://example.com/one', b'{}', dict[str, str]()), {})  # pyright: ignore[reportUnknownMemberType]
    cassette.append(Request('POST', 'https://example.com/two', b'{}', dict[str, str]()), {})  # pyright: ignore[reportUnknownMemberType]
    cassette.play_counts[0] = 1  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(pytest.fail.Exception, match=r'played 1/2; unused indexes: \[1\]'):
        check_vcr_cassette_usage(cassette, strict_usage=False)


def test_check_vcr_cassette_usage_allows_fully_used_cassette() -> None:
    cassette = Cassette('fake.yaml', record_mode=RecordMode.NONE)
    cassette.append(Request('POST', 'https://example.com/one', b'{}', dict[str, str]()), {})  # pyright: ignore[reportUnknownMemberType]
    cassette.append(Request('POST', 'https://example.com/two', b'{}', dict[str, str]()), {})  # pyright: ignore[reportUnknownMemberType]
    cassette.play_counts[0] = 1  # pyright: ignore[reportUnknownMemberType]
    cassette.play_counts[1] = 1  # pyright: ignore[reportUnknownMemberType]

    check_vcr_cassette_usage(cassette, strict_usage=False)


@pytest.mark.anyio
async def test_blockbuster_exemption_contract() -> None:
    """The detector catches unapproved calls while coverage's source reads stay exempt."""
    bb = BlockBuster(['tests.test_conftest'])
    for func, filename, functions in BLOCKBUSTER_EXEMPTIONS:
        bb.functions[func].can_block_in(filename, functions)

    try:
        bb.activate()
        with pytest.raises(BlockingError):
            _blocking_stat()

        assert ('os.stat', 'coverage/python.py', 'get_python_source') in BLOCKBUSTER_EXEMPTIONS
        assert ('io.BufferedReader.read', 'coverage/python.py', 'read_python_source') in BLOCKBUSTER_EXEMPTIONS
        assert get_python_source(__file__) is not None
    finally:
        bb.deactivate()


def test_blockbuster_deactivates_when_exemption_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, 'BLOCKBUSTER_EXEMPTIONS', [('missing', 'test_conftest.py', 'test')])
    stat = os.stat
    buffered_read = io.BufferedReader.read
    fixture = conftest.blockbuster._fixture_function()  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(KeyError):
        next(fixture)

    assert os.stat is stat
    assert io.BufferedReader.read is buffered_read
