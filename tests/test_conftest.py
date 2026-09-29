from __future__ import annotations

import io
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from cassetter import Body, Cassette, RecordMode, WsFrame
from coverage.python import get_python_source

from . import conftest
from .conftest import BLOCKBUSTER_EXEMPTIONS, check_vcr_cassette_usage

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


def _blocking_stat() -> None:
    os.stat(__file__)


def _recorded_cassette(*uris: str) -> Cassette:
    cassette = Cassette('fake.yaml', record_mode=RecordMode.NONE)
    for uri in uris:
        cassette.record('POST', uri, {}, b'{}', 200, {}, b'{}')
    return cassette


def test_check_vcr_cassette_usage_allows_loaded_unused_cassette_by_default() -> None:
    cassette = Cassette('fake.yaml', record_mode=RecordMode.NONE)

    check_vcr_cassette_usage(cassette, strict_usage=False)


def test_check_vcr_cassette_usage_reports_unused_interactions() -> None:
    cassette = _recorded_cassette('https://example.com/one', 'https://example.com/two')
    cassette.play('POST', 'https://example.com/one', {}, b'{}')

    with pytest.raises(pytest.fail.Exception, match=r'unused HTTP indexes: \[1\]$'):
        check_vcr_cassette_usage(cassette, strict_usage=False)


def test_check_vcr_cassette_usage_numbers_each_protocol_separately() -> None:
    """HTTP, gRPC and WebSocket interactions are numbered separately, and only HTTP has play counts."""
    cassette = _recorded_cassette('https://example.com/one', 'https://example.com/two')
    cassette.record_ws('wss://example.com/ws', {}, [WsFrame('recv', 'text', Body('text', 'hi'), 0)])
    cassette.record_grpc(
        method='/pkg.Svc/Call', metadata={}, request_body=Body('binary', b'\x01'), response_body=Body('binary', b'\x02')
    )
    cassette.play('POST', 'https://example.com/one', {}, b'{}')
    cassette.play('POST', 'https://example.com/two', {}, b'{}')
    cassette.play_ws('wss://example.com/ws')

    with pytest.raises(pytest.fail.Exception, match=r'did not play all interactions: unused gRPC indexes: \[0\]$'):
        check_vcr_cassette_usage(cassette, strict_usage=False)


def test_check_vcr_cassette_usage_allows_fully_used_cassette() -> None:
    cassette = _recorded_cassette('https://example.com/one', 'https://example.com/two')
    cassette.play('POST', 'https://example.com/one', {}, b'{}')
    cassette.play('POST', 'https://example.com/two', {}, b'{}')

    check_vcr_cassette_usage(cassette, strict_usage=False)


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

        # `blockbuster` exempts coverage's source reads itself, so this read must not raise.
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
