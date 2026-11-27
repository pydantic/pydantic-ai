"""Tests for the UUIDv7 polyfill."""

from unittest.mock import patch

import pydantic_ai._uuid as uuid_mod
from pydantic_ai._uuid import uuid7


def test_uuid7_basic() -> None:
    u = uuid7()
    assert u.version == 7
    assert u.variant == 'specified in RFC 4122'
    assert len(str(u)) == 36


def test_uuid7_monotonic_within_millisecond() -> None:
    """Multiple uuid7() calls in the same millisecond produce monotonically increasing values."""
    ids = [uuid7() for _ in range(100)]
    for a, b in zip(ids, ids[1:]):
        assert a < b


def test_uuid7_clock_regression() -> None:
    """uuid7() handles system clock going backward (e.g. NTP adjustment)."""
    saved_ts = uuid_mod._last_timestamp_v7  # pyright: ignore[reportPrivateUsage]
    saved_counter = uuid_mod._last_counter_v7  # pyright: ignore[reportPrivateUsage]
    try:
        # Generate a UUID at a known timestamp
        with patch.object(uuid_mod.time, 'time_ns', return_value=2_000 * 1_000_000_000):
            u1 = uuid7()

        # Clock goes backward
        with patch.object(uuid_mod.time, 'time_ns', return_value=1_000 * 1_000_000_000):
            u2 = uuid7()

        assert u2 > u1
        assert u2.version == 7
    finally:
        uuid_mod._last_timestamp_v7 = saved_ts  # pyright: ignore[reportPrivateUsage]
        uuid_mod._last_counter_v7 = saved_counter  # pyright: ignore[reportPrivateUsage]


def test_uuid7_counter_overflow() -> None:
    """uuid7() advances the timestamp when the 42-bit counter overflows."""
    saved_ts = uuid_mod._last_timestamp_v7  # pyright: ignore[reportPrivateUsage]
    saved_counter = uuid_mod._last_counter_v7  # pyright: ignore[reportPrivateUsage]
    try:
        # Set counter to max so the next same-ms call overflows
        with patch.object(uuid_mod.time, 'time_ns', return_value=3_000 * 1_000_000_000):
            u1 = uuid7()

        uuid_mod._last_counter_v7 = 0x3FF_FFFF_FFFF  # pyright: ignore[reportPrivateUsage]

        with patch.object(uuid_mod.time, 'time_ns', return_value=3_000 * 1_000_000_000):
            u2 = uuid7()

        assert u2 > u1
        assert u2.version == 7
    finally:
        uuid_mod._last_timestamp_v7 = saved_ts  # pyright: ignore[reportPrivateUsage]
        uuid_mod._last_counter_v7 = saved_counter  # pyright: ignore[reportPrivateUsage]


def test_uuid7_same_millisecond_timestamp_preserved() -> None:
    """When calls occur in the same millisecond, the 48-bit timestamp ms prefix must not increment."""
    saved_ts = uuid_mod._last_timestamp_v7  # pyright: ignore[reportPrivateUsage]
    saved_counter = uuid_mod._last_counter_v7  # pyright: ignore[reportPrivateUsage]
    try:
        uuid_mod._last_timestamp_v7 = None  # pyright: ignore[reportPrivateUsage]
        uuid_mod._last_counter_v7 = 0  # pyright: ignore[reportPrivateUsage]
        with patch.object(uuid_mod.time, 'time_ns', return_value=5_000_000_000_000_000):
            u1 = uuid7()
            u2 = uuid7()
        assert u1.hex[:12] == u2.hex[:12]
    finally:
        uuid_mod._last_timestamp_v7 = saved_ts  # pyright: ignore[reportPrivateUsage]
        uuid_mod._last_counter_v7 = saved_counter  # pyright: ignore[reportPrivateUsage]


def test_uuid7_counter_boundary_no_premature_overflow() -> None:
    """When the 42-bit counter increments up to exactly 0x3FF_FFFF_FFFF, it must not overflow prematurely."""
    saved_ts = uuid_mod._last_timestamp_v7  # pyright: ignore[reportPrivateUsage]
    saved_counter = uuid_mod._last_counter_v7  # pyright: ignore[reportPrivateUsage]
    try:
        uuid_mod._last_timestamp_v7 = None  # pyright: ignore[reportPrivateUsage]
        uuid_mod._last_counter_v7 = 0  # pyright: ignore[reportPrivateUsage]
        with patch.object(uuid_mod.time, 'time_ns', return_value=6_000_000_000_000_000):
            u1 = uuid7()
            uuid_mod._last_counter_v7 = 0x3FF_FFFF_FFFF - 1  # pyright: ignore[reportPrivateUsage]
            u2 = uuid7()
            assert u1.hex[:12] == u2.hex[:12]
            counter_val = ((u2.int >> 64) & 0x0FFF) << 30 | ((u2.int >> 32) & 0x3FFF_FFFF)
            assert counter_val == 0x3FF_FFFF_FFFF
    finally:
        uuid_mod._last_timestamp_v7 = saved_ts  # pyright: ignore[reportPrivateUsage]
        uuid_mod._last_counter_v7 = saved_counter  # pyright: ignore[reportPrivateUsage]
