from __future__ import annotations

import io
import os
import subprocess
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from coverage.python import get_python_source
from vcr.cassette import Cassette
from vcr.record_mode import RecordMode
from vcr.request import Request

from . import conftest
from .conftest import BLOCKBUSTER_EXEMPTIONS, check_vcr_cassette_usage, pytest_recording_configure

if TYPE_CHECKING:
    from blockbuster import BlockBuster, BlockingError


@pytest.fixture
def blockbuster_enabled() -> bool:
    """Test the root fixture directly without activating its shared instance for this module."""
    return False


@pytest.fixture
def blockbuster_types() -> tuple[type[BlockBuster], type[BlockingError]]:
    if os.getenv('BLOCKBUSTER_ENABLED') == 'false':
        pytest.skip('BlockBuster is disabled in this CI lane')

    from blockbuster import BlockBuster, BlockingError

    return BlockBuster, BlockingError


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
async def test_blockbuster_exemption_contract(
    blockbuster_types: tuple[type[BlockBuster], type[BlockingError]],
) -> None:
    """The detector catches unapproved calls while coverage's source reads stay exempt."""
    BlockBuster, BlockingError = blockbuster_types
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


def test_blockbuster_does_not_activate_when_configuration_fails(
    blockbuster_types: tuple[type[BlockBuster], type[BlockingError]],
) -> None:
    stat = os.stat
    buffered_read = io.BufferedReader.read

    with pytest.raises(KeyError):
        conftest._configure_blockbuster([('missing', 'test_conftest.py', 'test')])  # pyright: ignore[reportPrivateUsage]

    assert os.stat is stat
    assert io.BufferedReader.read is buffered_read


def test_blockbuster_disabled_when_explicitly_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('BLOCKBUSTER_ENABLED', 'false')
    stat = os.stat
    fixture = conftest.blockbuster._fixture_function(True, ())  # pyright: ignore[reportPrivateUsage]

    assert next(fixture) is None
    assert os.stat is stat
    with pytest.raises(StopIteration):
        next(fixture)


def test_disabled_blockbuster_does_not_import_instrumentation() -> None:
    subprocess.run(
        [
            sys.executable,
            '-c',
            'import builtins, sys; before = builtins.dir; import tests.test_conftest; '
            'assert list(tests.test_conftest.conftest.blockbuster._fixture_function(True, ())) == [None]; '
            'assert builtins.dir is before; '
            "assert 'blockbuster' not in sys.modules; "
            "assert 'forbiddenfruit' not in sys.modules",
        ],
        check=True,
        env={**os.environ, 'BLOCKBUSTER_ENABLED': 'false'},
    )


def test_configured_blockbusters_are_cached_per_exclusion_set(
    blockbuster_types: tuple[type[BlockBuster], type[BlockingError]],
) -> None:
    BlockBuster, _ = blockbuster_types
    default = conftest._configured_blockbuster(())  # pyright: ignore[reportPrivateUsage]
    same_default = conftest._configured_blockbuster(())  # pyright: ignore[reportPrivateUsage]
    excluding_clai = conftest._configured_blockbuster(('clai',))  # pyright: ignore[reportPrivateUsage]

    assert isinstance(default, BlockBuster)
    assert default is same_default
    assert default is not excluding_clai
    assert not default.functions['os.stat'].activated
    assert not excluding_clai.functions['os.stat'].activated


def test_blockbuster_deactivates_when_a_test_fails(
    blockbuster_types: tuple[type[BlockBuster], type[BlockingError]],
) -> None:
    BlockBuster, _ = blockbuster_types
    bb = BlockBuster(['tests.test_conftest'])
    stat = os.stat

    with pytest.raises(RuntimeError), conftest._activated_blockbuster(bb):  # pyright: ignore[reportPrivateUsage]
        assert bb.functions['os.stat'].activated
        raise RuntimeError

    assert os.stat is stat


def test_blockbuster_deactivates_when_activation_fails(
    monkeypatch: pytest.MonkeyPatch,
    blockbuster_types: tuple[type[BlockBuster], type[BlockingError]],
) -> None:
    BlockBuster, _ = blockbuster_types
    bb = BlockBuster(['tests.test_conftest'])
    stat = os.stat

    def fail_after_partial_activation() -> None:
        bb.functions['os.stat'].activate()
        raise RuntimeError

    monkeypatch.setattr(bb, 'activate', fail_after_partial_activation)

    with pytest.raises(RuntimeError), conftest._activated_blockbuster(bb):  # pyright: ignore[reportPrivateUsage]
        pass

    assert os.stat is stat
