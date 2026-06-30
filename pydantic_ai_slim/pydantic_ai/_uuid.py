"""UUIDv7 polyfill for Python < 3.14.

Matches the CPython 3.14 implementation:
https://github.com/python/cpython/blob/main/Lib/uuid.py

Replace with `uuid.uuid7()` once Python 3.14 is the minimum supported version.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

# Global state for sub-millisecond monotonicity (Method 1, RFC 9562 §6.2).
_last_timestamp_v7: int | None = None
_last_counter_v7: int = 0  # 42-bit counter
_lock_v7 = threading.Lock()

# version = 0b0111, variant = 0b10
_RFC_4122_VERSION_7_FLAGS = 0x0000_0000_0000_7000_8000_0000_0000_0000


def _uuid7_get_counter_and_tail() -> tuple[int, int]:
    rand = int.from_bytes(os.urandom(10), 'big')
    # 42-bit counter with MSB set to 0
    counter = (rand >> 32) & 0x1FF_FFFF_FFFF
    # 32-bit random data
    tail = rand & 0xFFFF_FFFF
    return counter, tail


def uuid7() -> uuid.UUID:
    """Generate a UUIDv7 (time-sortable UUID) per RFC 9562.

    UUIDv7 objects feature monotonicity within a millisecond.
    """
    # --- 48 ---   -- 4 --   --- 12 ---   -- 2 --   --- 30 ---   - 32 -
    # unix_ts_ms | version | counter_hi | variant | counter_lo | random
    #
    # 'counter = counter_hi | counter_lo' is a 42-bit counter constructed
    # with Method 1 of RFC 9562, §6.2, and its MSB is set to 0.
    #
    # 'random' is a 32-bit random value regenerated for every new UUID.
    #
    # If multiple UUIDs are generated within the same millisecond, the LSB
    # of 'counter' is incremented by 1. When overflowing, the timestamp is
    # advanced and the counter is reset to a random 42-bit integer with MSB
    # set to 0.

    global _last_timestamp_v7
    global _last_counter_v7

    nanoseconds = time.time_ns()
    timestamp_ms = nanoseconds // 1_000_000

    with _lock_v7:
        if _last_timestamp_v7 is None or timestamp_ms > _last_timestamp_v7:
            counter, tail = _uuid7_get_counter_and_tail()
        else:
            if timestamp_ms < _last_timestamp_v7:
                timestamp_ms = _last_timestamp_v7 + 1
            # advance the 42-bit counter
            counter = _last_counter_v7 + 1
            if counter > 0x3FF_FFFF_FFFF:
                # advance the 48-bit timestamp
                timestamp_ms += 1
                counter, tail = _uuid7_get_counter_and_tail()
            else:
                # 32-bit random data
                tail = int.from_bytes(os.urandom(4), 'big')

        _last_timestamp_v7 = timestamp_ms
        _last_counter_v7 = counter

    unix_ts_ms = timestamp_ms & 0xFFFF_FFFF_FFFF
    counter_hi = (counter >> 30) & 0x0FFF
    counter_lo = counter & 0x3FFF_FFFF

    int_uuid_7 = unix_ts_ms << 80
    int_uuid_7 |= counter_hi << 64
    int_uuid_7 |= counter_lo << 32
    int_uuid_7 |= tail & 0xFFFF_FFFF
    int_uuid_7 |= _RFC_4122_VERSION_7_FLAGS

    return uuid.UUID(int=int_uuid_7)
