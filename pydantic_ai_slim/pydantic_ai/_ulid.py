from __future__ import annotations

from collections.abc import Callable
from random import Random


def ulid(random: Random, ms_timestamp_generator: Callable[[], int]) -> int:
    """Generate an integer ULID compatible with UUID v4.

    ULIDs as defined by the [spec](https://github.com/ulid/spec) look like this:

     01AN4Z07BY      79KA1307SR9X4MV3
    |----------|    |----------------|
     Timestamp         Randomness
     48bits            80bits

    In the future it would be nice to make this compatible with a UUID,
    e.g. v4 UUIDs by setting the version and variant bits correctly.
    We can't currently do this because setting these bits would leave us with only 7 bytes of randomness,
    which isn't enough for the Python SDK's sampler that currently expects 8 bytes of randomness.
    In the future OTEL will probably adopt https://www.w3.org/TR/trace-context-2/#random-trace-id-flag
    which relies only on the lower 7 bytes of the trace ID, then all SDKs and tooling should be updated
    and leaving only 7 bytes of randomness should be fine.

    Right now we only care about:
    - Our SDK / Python SDK's in general.
    - The OTEL collector.

    And both behave properly with 8 bytes of randomness because trace IDs were originally 64 bits
    so to be compatible with old trace IDs nothing in OTEL can assume >8 bytes of randomness in trace IDs
    unless they generated the trace ID themselves (e.g. the Go SDK _does_ expect >8 bytes of randomness internally).
    """
    # Timestamp: first 6 bytes of the ULID (48 bits)
    # Note that it's not important that this timestamp is super precise or unique.
    # It just needs to be roughly monotonically increasing so that the ULID is sortable, at least for our purposes.
    timestamp = ms_timestamp_generator().to_bytes(6, byteorder='big')
    # Randomness: next 10 bytes of the ULID (80 bits)
    randomness = random.getrandbits(80).to_bytes(10, byteorder='big')
    # Convert to int and return
    return int.from_bytes(timestamp + randomness, byteorder='big')


def ulid_as_uuid(random: Random, ms_timestamp_generator: Callable[[], int]) -> str:
    """Generate a ULID string and return it as a hex string (without dashes) that can be used as a UUID."""
    ulid_int = ulid(random, ms_timestamp_generator)
    return f'{ulid_int:032x}'
