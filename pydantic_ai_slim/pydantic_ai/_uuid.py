"""UUIDv7 polyfill for Python < 3.14.

Replace with `uuid.uuid7()` once Python 3.14 is the minimum supported version.
"""

from __future__ import annotations

import os
import time
import uuid


def uuid7() -> uuid.UUID:
    """Generate a UUIDv7 (time-sortable UUID) per RFC 9562."""
    timestamp_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), 'big')
    # UUIDv7: 48-bit timestamp | 4-bit version(7) | 12-bit rand | 2-bit variant(10) | 62-bit rand
    value = (timestamp_ms & 0xFFFFFFFFFFFF) << 80
    value |= 7 << 76
    value |= ((rand >> 68) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= rand & 0x3FFFFFFFFFFFFFFF
    return uuid.UUID(int=value)
