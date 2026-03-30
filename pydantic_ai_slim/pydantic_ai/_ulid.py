from __future__ import annotations

import os
import time


def generate_ulid() -> str:
    """Generate a ULID as a 32-char hex string.

    128-bit identifier: 48-bit millisecond timestamp followed by 80 bits of randomness,
    making IDs time-sortable.
    """
    timestamp_ms = time.time_ns() // 1_000_000
    random_bits = int.from_bytes(os.urandom(10), 'big')
    return format((timestamp_ms << 80) | random_bits, '032x')
